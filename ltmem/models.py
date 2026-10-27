from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class HypothesisSupport:
    timestamp: str
    object_id: str


@dataclass
class LocationHypothesis:
    location: Tuple[float, float, float]
    confidence: float
    last_seen: str
    support_start: Optional[HypothesisSupport] = None
    dominance_count: int = 0
    dominance_session: Optional[int] = None


@dataclass
class Observation:
    time: str
    object_id: str
    location: tuple[float, float, float]
    confidence: float = 1.0
    volume: Optional[float] = None
    occlusion_score: Optional[float] = None
    fov_score: Optional[float] = None
    center_point: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None
    session_id: Optional[str] = None
    volatility_category: Optional[str] = None
    detection_status: str = "PD"
    semantic_label: str = field(kw_only=True)
    observation_confidence: Optional[float] = field(default=None, kw_only=True)
    observation_count: int = field(default=0, kw_only=True)


@dataclass
class Candidate:
    track_id: str
    current_location: tuple[float, float, float]
    last_seen_time: str
    center_point: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None
    volatility_category: Optional[str] = None
    semantic_label: Optional[str] = None


@dataclass
class ReIDDecision:
    decision: str
    confidence: float
    update_action: Optional[str] = None
    track_id: Optional[str] = None


@dataclass
class TrackState:
    track_id: str
    object_id: str
    current_location: Tuple[float, float, float]
    last_seen_time: str
    confidence: float
    embedding: Optional[np.ndarray]
    center_point: Optional[np.ndarray]
    metadata: Dict[str, Any] = field(default_factory=dict)
    hypotheses: List[LocationHypothesis] = field(default_factory=list)

    def present_state(self) -> Tuple[Tuple[float, float, float], float]:
        if self.hypotheses:
            best = max(self.hypotheses, key=lambda item: item.confidence)
            return best.location, best.confidence
        return self.current_location, self.confidence


@dataclass
class DeltaEvent:
    event_type: str
    object_id: str
    timestamp: str
    metadata: Dict[str, Any]
    track_id: Optional[str] = None
    from_location: Optional[Tuple[float, float, float]] = None
    to_location: Optional[Tuple[float, float, float]] = None
    detection_status: Optional[str] = None


@dataclass
class ObjectCluster:
    points: np.ndarray
    observation_confidence: float
    observation_count: int
    embedding: Optional[np.ndarray]

    def observation(self, session_id: str, semantic_label: str) -> Observation:
        center = self.points.mean(axis=0)
        extent = self.points.max(axis=0) - self.points.min(axis=0)
        return Observation(
            time=session_id,
            object_id=f"{semantic_label}_{session_id}_0",
            location=tuple(center.tolist()),
            confidence=self.observation_confidence,
            observation_confidence=self.observation_confidence,
            observation_count=self.observation_count,
            volume=float(np.prod(extent)),
            center_point=np.asarray(center, dtype=float),
            embedding=self.embedding,
            session_id=session_id,
            semantic_label=semantic_label,
        )


@dataclass
class SessionPerception:
    session_id: str
    clusters: List[ObjectCluster]
    scene_points: np.ndarray
    anchor_points: Dict[str, Tuple[np.ndarray, ...]] = field(default_factory=dict)


@dataclass
class ObservationResult:
    observation: Observation
    event: Optional[DeltaEvent]
    track_id: Optional[str]
    state_held: bool = False
