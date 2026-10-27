from typing import Optional

from dataclasses import dataclass
from enum import Enum

from ltmem.change_detection import ChangeDetectionResult, DECISION_REAL_MOVE, DECISION_UNCERTAIN
from ltmem.memory_system import MemorySystem, MetaMemory, categorize_volatility
from ltmem.models import DeltaEvent, Observation, ReIDDecision


class VolatilityPriorEstimator:
    def __init__(self, model_runtime) -> None:
        self.model_runtime = model_runtime

    def estimate(self, object_label: str) -> float:
        label = object_label.lower().strip()
        prompt = f'''Estimate how likely the indoor object "{label}" is to change location between repeated visits. Use typical physical use of that object, not your uncertainty, and use the full numeric scale.

Calibration anchors:
- 0.10: built-in fixture or permanently mounted object
- 0.20: heavy appliance or furniture that is rarely relocated
- 0.50: object that is occasionally relocated
- 0.80: portable or wheeled object that is frequently moved
- 0.90: object that routinely moves itself

Return only one decimal number in [0.1, 0.9]. Do not use a percentage, fraction, x/10 rating, words, markdown, or explanation.

Decimal:'''
        choices = tuple(str(index / 10) for index in range(1, 10))
        return float(self.model_runtime.select_choice(prompt, choices))


def setup_observation_volatility(
    observation: Observation, *, meta: MetaMemory, estimator: VolatilityPriorEstimator
) -> None:
    key = observation.semantic_label
    if key not in meta.volatility_score:
        meta.set_volatility_prior(key, estimator.estimate(key))
    observation.volatility_category = categorize_volatility(meta.volatility_score[key])


class UpdateAction(Enum):
    OVERWRITE = "OVERWRITE"
    MULTI_HYPOTHESIS = "MULTI_HYP"
    HOLD = "HOLD"
    PENDING = "PENDING"


@dataclass
class UpdatePolicyInput:
    change_decision: Optional[str]
    association_confidence: float
    has_hypotheses: bool


class UpdatePolicy:
    def __init__(
        self,
        *,
        hold_confidence: float = 0.5,
    ) -> None:
        self.hold_confidence = hold_confidence

    def decide(self, input_data: UpdatePolicyInput) -> UpdateAction:
        if (
            input_data.change_decision is None
            or input_data.association_confidence < self.hold_confidence
        ):
            return UpdateAction.PENDING
        if input_data.change_decision == DECISION_REAL_MOVE:
            return UpdateAction.OVERWRITE
        if input_data.change_decision == DECISION_UNCERTAIN or input_data.has_hypotheses:
            return UpdateAction.MULTI_HYPOTHESIS

        return UpdateAction.HOLD


def run_update_stage(
    observation: Observation,
    decision: ReIDDecision,
    change: ChangeDetectionResult,
    *,
    memory: MemorySystem,
    policy: UpdatePolicy,
) -> Optional[DeltaEvent]:
    if decision.decision == "MATCH":
        track = memory.live.tracks[decision.track_id]
        decision.update_action = policy.decide(
            UpdatePolicyInput(
                change_decision=change.change_detection_result,
                association_confidence=decision.confidence,
                has_hypotheses=bool(track.hypotheses),
            )
        ).value

    return memory.process_decision(
        decision,
        observation,
        change_detection_result=change.change_detection_result,
    )
