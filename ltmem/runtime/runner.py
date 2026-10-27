from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from ltmem.model_runtime import ModelRuntime
from ltmem.pipeline import IntegratedPipeline
from ltmem.runtime.config import RuntimeConfig, RuntimeObjectConfig

DEFAULT_RUNTIME_ARTIFACT_DIR = Path("output")
MIN_OBJECT_CLUSTER_POINTS = 50


def prepare_alignment_prepass(
    pipeline: IntegratedPipeline,
    sessions: Iterable[str],
    structural_anchors: tuple[str, ...],
) -> None:
    first_session, *remaining_sessions = sessions
    pipeline.initialize_session_alignment(first_session, structural_anchors)
    for session_id in remaining_sessions:
        pipeline.prepare_session_alignment(
            session_id=session_id,
            structural_anchors=structural_anchors,
        )


def process_object_pipeline(
    obj: RuntimeObjectConfig,
    pipeline: IntegratedPipeline,
    sessions: Iterable[str],
) -> Dict[str, np.ndarray]:
    positions: Dict[str, np.ndarray] = {}
    last_center: Optional[np.ndarray] = None
    active_track_id: Optional[str] = None

    sessions = tuple(sessions)
    for index, session_id in enumerate(sessions, 1):
        print(f"  [{index}/{len(sessions)}] {session_id}", flush=True)
        if pipeline.is_session_held(session_id):
            if last_center is not None:
                positions[session_id] = last_center
            continue

        session = pipeline.perception.process_session(
            session_id=session_id,
            object_classes=obj.object_classes,
        )
        clusters = [
            cluster
            for cluster in session.clusters
            if len(cluster.points) >= MIN_OBJECT_CLUSTER_POINTS
        ]
        result = (
            pipeline.observe_object(obj.name, session, clusters, track_id=active_track_id)
            if clusters
            else None
        )
        if result is None:
            last_center = None
            if active_track_id is not None:
                pipeline.record_absence(
                    track_id=active_track_id,
                    session_id=session_id,
                )
            continue

        if result.state_held:
            if last_center is not None:
                positions[session_id] = last_center
            continue

        if result.track_id is not None:
            active_track_id = result.track_id
        last_center = np.asarray(result.observation.location, dtype=float)
        positions[session_id] = last_center

    return positions


def execute_pipeline(
    runtime: RuntimeConfig,
    pipeline: IntegratedPipeline,
) -> Dict[str, Dict[str, np.ndarray]]:
    if runtime.alignment_gate_enabled:
        prepare_alignment_prepass(
            pipeline,
            runtime.sessions,
            runtime.structural_anchors,
        )
    positions = {}
    for index, obj in enumerate(runtime.objects, 1):
        print(f"[Object {index}/{len(runtime.objects)}] {obj.name}", flush=True)
        positions[obj.name] = process_object_pipeline(obj, pipeline, runtime.sessions)
    return positions


def run_pipeline(
    runtime: RuntimeConfig,
    *,
    runtime_artifact_dir: Path = DEFAULT_RUNTIME_ARTIFACT_DIR,
    rag_persist_dir: Optional[Path] = None,
) -> Tuple[IntegratedPipeline, Dict[str, Dict[str, np.ndarray]]]:
    if rag_persist_dir is None:
        memory_dir = runtime_artifact_dir / f"object_memory_{runtime.env}"
    else:
        memory_dir = rag_persist_dir
    memory_dir = memory_dir.expanduser().resolve()
    pipeline = IntegratedPipeline(
        model_runtime=ModelRuntime(),
        data_root=str(runtime.data_root),
        session_dirs=runtime.sessions,
        rag_persist_dir=memory_dir,
        alignment_gate_enabled=runtime.alignment_gate_enabled,
    )
    return pipeline, execute_pipeline(runtime, pipeline)
