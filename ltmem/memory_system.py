import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, replace

from ltmem.models import (
    DeltaEvent,
    HypothesisSupport,
    LocationHypothesis,
    Observation,
    ReIDDecision,
    TrackState,
)

EVENT_LABELS = ["MOVE", "APPEAR", "DISAPPEAR", "REAPPEAR", "NONE", "HOLD"]
CHANGE_EVENTS = ("MOVE", "APPEAR", "REAPPEAR", "DISAPPEAR")
CHANGE_EVIDENCE_WEIGHT = 0.6


def session_number(timestamp: str) -> int:
    return int(timestamp.removeprefix("Session"))


def categorize_volatility(volatility_score: float) -> str:
    if volatility_score <= 0.3:
        return "static"
    if volatility_score <= 0.5:
        return "semi-static"
    return "volatile"


@dataclass
class LiveMemory:
    tracks: Dict[str, TrackState] = field(default_factory=dict)
    structural_anchor_centers: Dict[str, np.ndarray] = field(default_factory=dict)

    def update_track(self, state: TrackState) -> None:
        self.tracks[state.track_id] = state

    def query_all_tracks(self) -> List[TrackState]:
        return list(self.tracks.values())


@dataclass
class DeltaMemory:
    events: List[DeltaEvent] = field(default_factory=list)


@dataclass
class MetaMemory:
    session_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    session_times: Dict[str, str] = field(default_factory=dict)
    volatility_score: Dict[str, float] = field(default_factory=dict)
    event_count: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def set_volatility_prior(self, object_id: str, prior: float) -> None:
        self.volatility_score[object_id] = prior
        self.event_count[object_id] = {event_type: 0 for event_type in EVENT_LABELS}

    def bayesian_update_volatility(
        self,
        object_id: str,
        observed_event: str,
    ) -> float:
        previous = self.volatility_score[object_id]

        if observed_event == "HOLD":
            self.event_count[object_id][observed_event] += 1
            return previous

        evidence_weight = (
            CHANGE_EVIDENCE_WEIGHT
            if observed_event in CHANGE_EVENTS
            else 1.0 - CHANGE_EVIDENCE_WEIGHT
        )
        numerator = evidence_weight * previous
        posterior = numerator / (numerator + (1.0 - evidence_weight) * (1.0 - previous))
        self.volatility_score[object_id] = posterior
        self.event_count[object_id][observed_event] += 1
        return posterior


@dataclass
class _TrackHistory:
    location: Optional[Tuple[float, float, float]] = None
    present: bool = True
    event_type: Optional[str] = None
    event_timestamp: Optional[str] = None
    presence_event_type: Optional[str] = None


@dataclass
class _HypothesisSelection:
    selected_location: Tuple[float, float, float]
    winner_supported: bool
    location_changed: bool
    resolved: bool = False
    selection_applied: bool = False


class MemorySystem:
    def __init__(
        self,
        hypothesis_merge_distance: float = 0.3,
        hypothesis_promotion_margin: float = 0.1,
        hypothesis_evidence_alpha: float = 0.5,
        hypothesis_promotion_sessions: int = 2,
    ):
        self.live = LiveMemory()
        self.delta = DeltaMemory()
        self.meta = MetaMemory()
        self.hypothesis_merge_distance = hypothesis_merge_distance
        self.hypothesis_promotion_margin = hypothesis_promotion_margin
        self.hypothesis_evidence_alpha = hypothesis_evidence_alpha
        self.hypothesis_promotion_sessions = hypothesis_promotion_sessions

    def record_event(
        self,
        event: DeltaEvent,
        volatility_key: str,
    ) -> DeltaEvent:
        for existing in reversed(self.delta.events):
            if (
                existing.timestamp == event.timestamp
                and existing.event_type in EVENT_LABELS
                and existing.object_id == event.object_id
            ):
                return existing

        before = self.meta.volatility_score[volatility_key]
        self.meta.bayesian_update_volatility(volatility_key, event.event_type)
        event.metadata.update(
            {
                "volatility_key": volatility_key,
                "volatility_before": before,
            }
        )
        self.delta.events.append(event)
        return event

    def record_absence(
        self,
        track_id: str,
        timestamp: str,
    ) -> DeltaEvent:
        track = self.live.tracks[track_id]

        for event in reversed(self.delta.events):
            if (
                event.track_id == track.track_id
                and event.timestamp == timestamp
                and event.event_type in EVENT_LABELS
            ):
                return event

        metadata = track.metadata
        semantic_label = metadata["semantic_label"]
        was_present = metadata["present"]
        event_type = "DISAPPEAR" if was_present else "NONE"

        event = DeltaEvent(
            event_type=event_type,
            object_id=track.object_id,
            track_id=track.track_id,
            from_location=track.current_location,
            to_location=None,
            timestamp=timestamp,
            detection_status="ND",
            metadata={
                "semantic_label": semantic_label,
                "last_seen_time": track.last_seen_time,
            },
        )
        self.record_event(event, volatility_key=semantic_label)

        metadata["present"] = False
        metadata["last_absent_time"] = timestamp
        return event

    def process_decision(
        self,
        decision: ReIDDecision,
        observation: Observation,
        *,
        change_detection_result: Optional[str] = None,
    ) -> Optional[DeltaEvent]:
        if decision.decision == "HOLD":
            track = self.live.tracks[decision.track_id]
            label = track.metadata["semantic_label"]
            return self.record_event(
                DeltaEvent(
                    event_type="HOLD",
                    object_id=track.object_id,
                    track_id=track.track_id,
                    timestamp=observation.time,
                    metadata={"semantic_label": label, "association_decision": "HOLD"},
                ),
                volatility_key=label,
            )
        if decision.decision == "MATCH":
            track = self.live.tracks[decision.track_id]
            current_session = session_number(observation.time)
            same_session_event = next(
                (
                    event
                    for event in reversed(self.delta.events)
                    if event.track_id == track.track_id
                    and session_number(event.timestamp) == current_session
                    and event.event_type in EVENT_LABELS
                ),
                None,
            )
            if (
                current_session <= session_number(track.last_seen_time)
                or same_session_event is not None
            ):
                if (
                    same_session_event is not None
                    and change_detection_result == "REAL_MOVE"
                    and decision.update_action != "MULTI_HYP"
                    and same_session_event.object_id == observation.object_id
                    and same_session_event.timestamp == observation.time
                ):
                    return same_session_event
                return None

        if decision.decision == "NEW_TRACK":
            event = self._create_new_track_event(decision, observation)
            return self._finalize_observation_event(event, observation, None)

        history = self._track_history(decision)
        action = decision.update_action
        event, revised_event = self._apply_update(
            decision, observation, history, action, change_detection_result
        )
        event = self._restore_presence(decision, observation, history, action, event)

        if event is None or event.event_type not in EVENT_LABELS:
            event = DeltaEvent(
                event_type="HOLD" if action == "PENDING" else "NONE",
                object_id=observation.object_id,
                track_id=decision.track_id,
                from_location=history.location,
                to_location=observation.location,
                timestamp=observation.time,
                detection_status=observation.detection_status,
                metadata=self._observation_metadata(observation),
            )
        return self._finalize_observation_event(event, observation, revised_event)

    def _track_history(self, decision: ReIDDecision) -> _TrackHistory:
        history = _TrackHistory()
        if decision.decision != "MATCH":
            return history
        track = self.live.tracks[decision.track_id]
        history.location = track.current_location
        history.present = track.metadata["present"]
        for event in reversed(self.delta.events):
            if event.track_id == decision.track_id:
                if history.event_type is None:
                    history.event_type = event.event_type
                    history.event_timestamp = event.timestamp
                if event.event_type in CHANGE_EVENTS:
                    history.presence_event_type = event.event_type
                    break
        return history

    def _apply_update(
        self,
        decision: ReIDDecision,
        observation: Observation,
        history: _TrackHistory,
        action: Optional[str],
        change_detection_result: Optional[str],
    ) -> Tuple[Optional[DeltaEvent], Optional[DeltaEvent]]:
        if action in {"HOLD", "PENDING"}:
            if decision.decision == "MATCH" and observation.detection_status != "ND":
                track = self.live.tracks[decision.track_id]
                track.last_seen_time = observation.time
        elif action == "MULTI_HYP":
            return self._process_multi_hypothesis(decision, observation, history.location)
        elif action == "OVERWRITE" and observation.detection_status != "ND":
            self._overwrite_confirmed_track(
                decision, observation, self.live.tracks[decision.track_id]
            )
            if change_detection_result == "REAL_MOVE":
                return DeltaEvent(
                    event_type=(
                        "REAPPEAR"
                        if not history.present or history.presence_event_type == "DISAPPEAR"
                        else "MOVE"
                    ),
                    object_id=observation.object_id,
                    track_id=decision.track_id,
                    from_location=history.location,
                    to_location=observation.location,
                    timestamp=observation.time,
                    detection_status=observation.detection_status,
                    metadata={"object_class": observation.semantic_label},
                ), None
        return None, None

    def _restore_presence(
        self,
        decision: ReIDDecision,
        observation: Observation,
        history: _TrackHistory,
        action: Optional[str],
        event: Optional[DeltaEvent],
    ) -> Optional[DeltaEvent]:
        if (
            decision.decision != "MATCH"
            or action == "PENDING"
            or history.present
            or observation.detection_status == "ND"
        ):
            return event
        track = self.live.tracks[decision.track_id]
        if action != "OVERWRITE":
            track.metadata = dict(track.metadata)
            track.metadata["present"] = True
        if history.event_type == "REAPPEAR" and history.event_timestamp == observation.time:
            return None
        if event is not None and event.event_type == "REAPPEAR":
            return event
        multi_hypothesis = action == "MULTI_HYP"
        return DeltaEvent(
            event_type="REAPPEAR",
            object_id=observation.object_id,
            track_id=decision.track_id,
            from_location=None if multi_hypothesis else history.location,
            to_location=track.current_location if multi_hypothesis else observation.location,
            timestamp=observation.time,
            detection_status="PD",
            metadata=self._observation_metadata(observation),
        )

    @staticmethod
    def _observation_metadata(observation: Observation) -> Dict[str, Any]:
        return {
            "confidence": observation.confidence,
            "occlusion_score": observation.occlusion_score,
            "fov_score": observation.fov_score,
        }

    def _locations_close(
        self,
        first: Tuple[float, float, float],
        second: Tuple[float, float, float],
    ) -> bool:
        squared_distance = sum(
            (first_value - second_value) ** 2 for first_value, second_value in zip(first, second)
        )
        return squared_distance <= self.hypothesis_merge_distance**2

    def _rank_hypotheses(
        self,
        track: TrackState,
        observation: Observation,
        *,
        association_confidence: float,
    ) -> Tuple[List[LocationHypothesis], _HypothesisSelection]:
        confirmed_location = track.current_location
        ranked = [replace(hypothesis) for hypothesis in track.hypotheses]
        if not ranked:
            ranked.append(
                LocationHypothesis(confirmed_location, track.confidence, track.last_seen_time)
            )

        observation_location = observation.location
        observation_time = observation.time
        matched_hypothesis = next(
            (
                hypothesis
                for hypothesis in ranked
                if self._locations_close(
                    hypothesis.location,
                    observation_location,
                )
            ),
            None,
        )
        if matched_hypothesis is None:
            matched_hypothesis = LocationHypothesis(
                observation_location, association_confidence, observation_time
            )
            ranked.append(matched_hypothesis)
        if (
            matched_hypothesis.support_start is None
            or session_number(matched_hypothesis.last_seen) + 1 != session_number(observation_time)
            or not track.metadata["present"]
        ):
            matched_hypothesis.support_start = HypothesisSupport(
                observation_time, observation.object_id
            )
        matched_hypothesis.last_seen = observation_time

        alpha = self.hypothesis_evidence_alpha
        for hypothesis in ranked:
            if hypothesis is not matched_hypothesis:
                hypothesis.support_start = None
            support = association_confidence if hypothesis is matched_hypothesis else 0.0
            hypothesis.confidence = (1.0 - alpha) * hypothesis.confidence + alpha * support

        ranked.sort(key=lambda hypothesis: hypothesis.confidence, reverse=True)
        ranked = ranked[:3]
        selected_location = ranked[0].location
        return ranked, _HypothesisSelection(
            selected_location=selected_location,
            winner_supported=ranked[0] is matched_hypothesis,
            location_changed=not self._locations_close(selected_location, confirmed_location),
        )

    def _revise_hypothesis_events(
        self,
        track_id: str,
        volatility_key: str,
        support_start: HypothesisSupport,
        confirmed_at: str,
        location: Tuple[float, float, float],
        *,
        moved: bool,
    ) -> Optional[DeltaEvent]:
        start_session = session_number(support_start.timestamp)
        confirmed_session = session_number(confirmed_at)
        if start_session >= confirmed_session:
            return None
        episode = sorted(
            (
                event
                for event in self.delta.events
                if event.track_id == track_id
                and start_session <= session_number(event.timestamp) < confirmed_session
            ),
            key=lambda event: session_number(event.timestamp),
        )
        if len(episode) != confirmed_session - start_session:
            return None
        for offset, event in enumerate(episode):
            if (
                session_number(event.timestamp) != start_session + offset
                or event.metadata["volatility_key"] != volatility_key
                or event.event_type
                not in (("NONE", "HOLD", "REAPPEAR") if offset == 0 else ("NONE", "HOLD"))
            ):
                return None
        anchor = episode[0]
        if anchor.object_id != support_start.object_id:
            return None
        for offset, event in enumerate(episode):
            if offset == 0:
                if event.event_type in {"NONE", "HOLD"}:
                    event.event_type = "MOVE" if moved else "NONE"
            else:
                event.event_type = "NONE"
                event.from_location = location
            event.to_location = location
        return anchor

    def _reconcile_volatility(
        self,
        key: str,
    ) -> None:
        events = [
            event
            for event in self.delta.events
            if event.event_type in EVENT_LABELS and event.metadata["volatility_key"] == key
        ]
        self.meta.set_volatility_prior(key, events[0].metadata["volatility_before"])
        for event in events:
            self.meta.bayesian_update_volatility(key, event.event_type)

    def _finalize_observation_event(
        self,
        event: DeltaEvent,
        observation: Observation,
        revised_event: Optional[DeltaEvent],
    ) -> DeltaEvent:
        if observation.observation_confidence is not None:
            event.metadata.update(
                {
                    "observation_confidence": observation.observation_confidence,
                    "observation_count": observation.observation_count,
                }
            )
        volatility_key = observation.semantic_label
        event = self.record_event(event, volatility_key=volatility_key)
        if revised_event is not None:
            self._reconcile_volatility(volatility_key)
        return event

    def _update_hypothesis_track(
        self,
        track: TrackState,
        observation: Observation,
        association_confidence: float,
    ) -> Tuple[_HypothesisSelection, Optional[DeltaEvent]]:
        hypotheses, selection = self._rank_hypotheses(
            track,
            observation,
            association_confidence=association_confidence,
        )
        track.hypotheses = hypotheses
        winner = hypotheses[0]
        competing = len(hypotheses) > 1
        dominant = (
            competing
            and selection.winner_supported
            and winner.confidence > hypotheses[1].confidence
            and winner.confidence >= (hypotheses[1].confidence + self.hypothesis_promotion_margin)
        )
        current_session = session_number(observation.time)
        for hypothesis in hypotheses:
            if hypothesis is not winner or not dominant:
                hypothesis.dominance_count = 0
                hypothesis.dominance_session = None
        if dominant:
            consecutive = (
                winner.dominance_session == current_session - 1 and track.metadata["present"]
            )
            winner.dominance_count = winner.dominance_count + 1 if consecutive else 1
            winner.dominance_session = current_session
        promoted = dominant and winner.dominance_count >= self.hypothesis_promotion_sessions
        selection.resolved = not competing or promoted
        if selection.resolved:
            track.confidence = winner.confidence
            track.hypotheses = []
        if promoted:
            track.current_location = winner.location
            track.center_point = np.asarray(winner.location, dtype=float)

        selection.selection_applied = promoted and selection.location_changed
        revised_event = None
        if promoted:
            revised_event = self._revise_hypothesis_events(
                track.track_id,
                observation.semantic_label,
                winner.support_start,
                observation.time,
                winner.location,
                moved=selection.location_changed,
            )
        track.last_seen_time = observation.time
        return selection, revised_event

    def _process_multi_hypothesis(
        self,
        decision: ReIDDecision,
        observation: Observation,
        previous_location: Optional[Tuple[float, float, float]],
    ) -> Tuple[DeltaEvent, Optional[DeltaEvent]]:
        track = self.live.tracks[decision.track_id]
        selection, revised_event = self._update_hypothesis_track(
            track,
            observation,
            decision.confidence,
        )
        return DeltaEvent(
            event_type=(
                "MOVE"
                if selection.selection_applied and revised_event is None
                else "NONE"
                if selection.resolved
                else "HOLD"
            ),
            object_id=observation.object_id,
            track_id=decision.track_id,
            from_location=(
                selection.selected_location if revised_event is not None else previous_location
            ),
            to_location=(
                selection.selected_location if selection.selection_applied else observation.location
            ),
            timestamp=observation.time,
            detection_status=observation.detection_status,
            metadata=self._observation_metadata(observation),
        ), revised_event

    def _create_new_track_event(
        self,
        decision: ReIDDecision,
        observation: Observation,
    ) -> DeltaEvent:
        track_id = decision.track_id

        metadata = {
            "session_id": observation.session_id,
            "semantic_label": observation.semantic_label,
            "present": True,
        }

        track_state = TrackState(
            track_id=track_id,
            object_id=observation.object_id,
            current_location=observation.location,
            last_seen_time=observation.time,
            confidence=decision.confidence,
            embedding=observation.embedding,
            center_point=observation.center_point,
            metadata=metadata,
        )
        self.live.update_track(track_state)
        return DeltaEvent(
            event_type="APPEAR",
            object_id=observation.object_id,
            track_id=track_id,
            from_location=None,
            to_location=observation.location,
            timestamp=observation.time,
            detection_status=observation.detection_status,
            metadata=self._observation_metadata(observation),
        )

    def _overwrite_confirmed_track(
        self,
        decision: ReIDDecision,
        observation: Observation,
        previous_track: TrackState,
    ) -> None:
        metadata = dict(previous_track.metadata)
        metadata.update(
            {
                "session_id": observation.session_id,
                "semantic_label": observation.semantic_label,
                "present": True,
            }
        )

        track_state = TrackState(
            track_id=decision.track_id,
            object_id=observation.object_id,
            current_location=observation.location,
            last_seen_time=observation.time,
            confidence=decision.confidence,
            embedding=observation.embedding,
            center_point=observation.center_point,
            metadata=metadata,
        )
        self.live.update_track(track_state)
