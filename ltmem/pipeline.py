import numpy as np
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, cast

from ltmem.perception import PerceptionStage
from ltmem.memory_system import MemorySystem
from ltmem.models import DeltaEvent, ObjectCluster, ObservationResult, SessionPerception
from ltmem.object_memory import ObjectFeatures, ObjectMemoryRAG
from ltmem.reid import (
    ReIDArbiter,
    ReIDScorer,
    run_reid_stage,
)
from ltmem.alignment_gate import (
    AlignmentGateStatus,
    AlignmentGateResult,
    AlignmentRequest,
    SessionAlignmentRegistry,
    observe_anchor_models,
    run_alignment_stage,
)
from ltmem.change_detection import (
    ChangeDetector,
    run_change_detection_stage,
)
from ltmem.memory_evolution import (
    UpdatePolicy,
    VolatilityPriorEstimator,
    run_update_stage,
    setup_observation_volatility,
)


class IntegratedPipeline:
    def __init__(
        self,
        data_root: str,
        rag_persist_dir: str | Path,
        hypothesis_merge_distance: float = 0.3,
        hypothesis_promotion_margin: float = 0.1,
        hypothesis_evidence_alpha: float = 0.5,
        hypothesis_promotion_sessions: int = 2,
        *,
        model_runtime: Any,
        session_dirs: Mapping[str, str],
        alignment_gate_enabled: bool = False,
    ):
        self.perception = PerceptionStage(data_root, session_dirs)

        self.reid_scorer = ReIDScorer()
        self.session_alignment = SessionAlignmentRegistry()
        self.alignment_gate_enabled = alignment_gate_enabled
        self.change_detector = ChangeDetector()
        self.update_policy = UpdatePolicy()
        self.model_runtime = model_runtime
        self.reid_arbiter = ReIDArbiter(self.model_runtime)
        self.volatility_prior = VolatilityPriorEstimator(self.model_runtime)
        self.memory = MemorySystem(
            hypothesis_merge_distance=hypothesis_merge_distance,
            hypothesis_promotion_margin=hypothesis_promotion_margin,
            hypothesis_evidence_alpha=hypothesis_evidence_alpha,
            hypothesis_promotion_sessions=hypothesis_promotion_sessions,
        )

        self.object_rag = ObjectMemoryRAG(
            persist_dir=str(rag_persist_dir),
        )

    def observe_object(
        self,
        label: str,
        session: SessionPerception,
        clusters: Sequence[ObjectCluster],
        *,
        track_id: Optional[str] = None,
    ) -> Optional[ObservationResult]:
        if self.is_session_held(session.session_id):
            return None
        if track_id is None:
            clusters = [max(clusters, key=lambda cluster: len(cluster.points))]
        observations = [cluster.observation(session.session_id, label) for cluster in clusters]
        for observation in observations:
            setup_observation_volatility(
                observation,
                meta=self.memory.meta,
                estimator=self.volatility_prior,
            )
        selection = run_reid_stage(
            observations,
            memory=self.memory,
            scorer=self.reid_scorer,
            arbiter=self.reid_arbiter,
            object_rag=self.object_rag,
            track_id=track_id,
        )
        if selection is None:
            return None
        index, decision = selection
        cluster = clusters[index]
        observation = observations[index]
        if decision.decision == "HOLD":
            event = self.memory.process_decision(decision, observation)
            return ObservationResult(observation, event, decision.track_id, state_held=True)
        change = run_change_detection_stage(
            observation,
            decision,
            memory=self.memory,
            detector=self.change_detector,
            observed_scene_points=session.scene_points,
            object_points=cluster.points,
        )
        event = run_update_stage(
            observation,
            decision,
            change,
            memory=self.memory,
            policy=self.update_policy,
        )
        result = ObservationResult(
            observation, event, decision.track_id, state_held=decision.update_action == "PENDING"
        )
        if result.track_id:
            self._index_observation(result, cluster.points)
        return result

    def _index_observation(self, result: ObservationResult, points: np.ndarray) -> None:
        observation = result.observation
        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
        extent = maximum - minimum
        self.object_rag.add_object(
            ObjectFeatures(
                object_id=observation.object_id,
                object_class=observation.semantic_label,
                centroid=list(observation.location),
                volume=cast(float, observation.volume),
                bounding_box={"min": minimum.tolist(), "max": maximum.tolist()},
                size={"x": float(extent[0]), "y": float(extent[1]), "z": float(extent[2])},
                confidence=observation.confidence,
                session_id=observation.session_id,
                num_points=len(points),
                track_id=result.track_id,
            )
        )

    def record_absence(
        self,
        track_id: str,
        session_id: str,
    ):
        if self.is_session_held(session_id):
            return None

        return self.memory.record_absence(
            track_id=track_id,
            timestamp=session_id,
        )

    def observe_structural_anchor_points(
        self,
        request: AlignmentRequest,
    ) -> Dict[str, tuple[np.ndarray, ...]]:
        session = self.perception.process_session(
            session_id=request.session_id,
            object_classes=list(request.structural_anchors),
            structural_alignment=True,
        )
        return session.anchor_points

    def _record_alignment_failure(
        self,
        session_id: str,
        result: AlignmentGateResult,
    ) -> None:
        if result.status == AlignmentGateStatus.GLOBAL_HOLD and not any(
            event.event_type == "ALIGNMENT_FAILURE" and event.timestamp == session_id
            for event in self.memory.delta.events
        ):
            self.memory.delta.events.append(
                DeltaEvent(
                    event_type="ALIGNMENT_FAILURE",
                    object_id="__structural_alignment__",
                    timestamp=session_id,
                    metadata={"distances_m": result.distances_m.copy()},
                )
            )

    def initialize_session_alignment(
        self,
        session_id: str,
        structural_anchors: tuple[str, ...],
    ) -> AlignmentGateResult:
        request = AlignmentRequest(session_id, structural_anchors)
        models = observe_anchor_models(request, self)
        result = self.session_alignment.initialize(session_id, models)
        self.memory.live.structural_anchor_centers.update(
            {
                label: center.copy()
                for label, center in self.session_alignment.reference_centers.items()
            }
        )
        self._record_alignment_failure(session_id, result)
        return result

    def prepare_session_alignment(
        self,
        session_id: str,
        structural_anchors: tuple[str, ...],
    ) -> AlignmentGateResult:
        result = run_alignment_stage(
            AlignmentRequest(session_id, structural_anchors),
            perception_port=self,
            registry=self.session_alignment,
            registered_centers=self.memory.live.structural_anchor_centers,
        )
        self._record_alignment_failure(session_id, result)
        return result

    def is_session_held(self, session_id: str) -> bool:
        return (
            self.alignment_gate_enabled
            and self.session_alignment.results[session_id].status == AlignmentGateStatus.GLOBAL_HOLD
        )

    def get_events(self) -> Tuple[DeltaEvent, ...]:
        return tuple(self.memory.delta.events)
