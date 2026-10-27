from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from ltmem.models import Observation, ReIDDecision
from ltmem.memory_system import MemorySystem


DECISION_REAL_MOVE = "REAL_MOVE"
DECISION_UNCERTAIN = "UNCERTAIN"
DEFAULT_TAU_MIN = 0.30
DEFAULT_TAU_MAX = 0.70
OLD_LOCATION_OBSERVABILITY_RADIUS = 0.50


def get_move_threshold(
    previous_volatility: float,
    tau_min: float = DEFAULT_TAU_MIN,
    tau_max: float = DEFAULT_TAU_MAX,
) -> float:
    return tau_min + previous_volatility * (tau_max - tau_min)


def _centroid_displacement(
    previous_location: tuple[float, float, float],
    current_location: tuple[float, float, float],
) -> float:
    return float(
        np.linalg.norm(
            np.asarray(current_location, dtype=float) - np.asarray(previous_location, dtype=float)
        )
    )


def old_location_absence_evidence(
    previous_location: tuple[float, float, float],
    observed_scene_points: np.ndarray,
    current_object_points: np.ndarray,
    radius: float,
) -> Optional[bool]:
    center = np.asarray(previous_location, dtype=float)
    scene = np.asarray(observed_scene_points, dtype=float).reshape(-1, 3)
    detected = np.asarray(current_object_points, dtype=float).reshape(-1, 3)
    radius_sq = radius**2

    scene_observed = bool(np.any(np.sum((scene - center) ** 2, axis=1) <= radius_sq))
    if not scene_observed:
        return None

    object_surface_overlaps = bool(np.any(np.sum((detected - center) ** 2, axis=1) <= radius_sq))
    return not object_surface_overlaps


class ChangeDecision(Enum):
    REAL_MOVE = DECISION_REAL_MOVE
    NOISE = "NOISE"
    UNCERTAIN = DECISION_UNCERTAIN


@dataclass(frozen=True)
class ChangeDetectorInput:
    movement_distance: float
    previous_volatility: float
    old_location_absent: Optional[bool] = None


class ChangeDetector:
    def __init__(
        self,
        tau_min: float = DEFAULT_TAU_MIN,
        tau_max: float = DEFAULT_TAU_MAX,
    ) -> None:
        self.tau_min = tau_min
        self.tau_max = tau_max

    def decide(
        self,
        input_data: ChangeDetectorInput,
    ) -> ChangeDecision:
        threshold = get_move_threshold(
            input_data.previous_volatility,
            tau_min=self.tau_min,
            tau_max=self.tau_max,
        )
        if input_data.movement_distance <= threshold:
            return ChangeDecision.NOISE
        if input_data.old_location_absent is True:
            return ChangeDecision.REAL_MOVE
        return ChangeDecision.UNCERTAIN


@dataclass(frozen=True)
class ChangeDetectionResult:
    change_detection_result: Optional[str] = None


def run_change_detection_stage(
    observation: Observation,
    decision: ReIDDecision,
    *,
    memory: MemorySystem,
    detector: ChangeDetector,
    observed_scene_points: np.ndarray,
    object_points: Optional[np.ndarray],
) -> ChangeDetectionResult:
    if decision.decision != "MATCH":
        return ChangeDetectionResult()
    track = memory.live.tracks[decision.track_id]

    old_location_absent = None
    if object_points is not None and len(object_points):
        old_location_absent = old_location_absence_evidence(
            track.current_location,
            observed_scene_points,
            object_points,
            OLD_LOCATION_OBSERVABILITY_RADIUS,
        )
    distance = _centroid_displacement(track.current_location, observation.location)
    volatility = float(memory.meta.volatility_score[observation.semantic_label])
    change = detector.decide(
        ChangeDetectorInput(distance, volatility, old_location_absent),
    )
    return ChangeDetectionResult(
        change_detection_result=change.value,
    )
