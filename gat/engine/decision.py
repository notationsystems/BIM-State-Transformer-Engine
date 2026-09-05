"""Decision-focused belief assessment and evidence stopping.

GAT's product question is not "which sensor has the most information?" in
the abstract.  It is "is a stated architectural criterion resolved at the
required confidence, and if not, which available observation is worth its
burden?"  This module makes that contract explicit for a minimum scalar
criterion:

    P(target >= minimum | current evidence)

The criterion is SATISFIED or VIOLATED only when either conclusion reaches
the requested confidence; otherwise it is UNRESOLVED.  Evidence collection
stops for a resolved criterion.  For an unresolved criterion, candidate
observations are scored by the active-inference layer and selected only when
their decision-relevant information gain exceeds their declared cost.

This probabilistic assessment does not replace GAT's hard invariant
verification and does not authorize a physical intervention.  It is an
auditable decision-support boundary over the current immutable world state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math
from typing import Iterable

from gat.engine.active_inference import (
    MinimumPreference,
    ObservationCandidate,
    ObservationPlan,
    plan_observations,
)
from gat.engine.executor import World
from gat.engine.propagate import jacobian_rows
from gat.ids import VarId


class DecisionVerdict(StrEnum):
    """Posterior status of a probabilistic minimum criterion."""

    SATISFIED = "SATISFIED"
    VIOLATED = "VIOLATED"
    UNRESOLVED = "UNRESOLVED"


class EvidenceDisposition(StrEnum):
    """Why the planner selected, or declined to select, an observation."""

    OBSERVE = "OBSERVE"
    DECISION_RESOLVED = "DECISION_RESOLVED"
    NO_AVAILABLE_EVIDENCE = "NO_AVAILABLE_EVIDENCE"
    NO_WORTHWHILE_EVIDENCE = "NO_WORTHWHILE_EVIDENCE"


@dataclass(frozen=True)
class MinimumDecision:
    """A decision question of the form ``target >= minimum``.

    ``confidence`` applies symmetrically: SATISFIED requires
    ``P(target >= minimum) >= confidence`` and VIOLATED requires
    ``P(target < minimum) >= confidence``.
    """

    target: VarId
    minimum: float
    confidence: float = 0.95
    label: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.minimum):
            raise ValueError("decision minimum must be finite")
        if not math.isfinite(self.confidence) or not 0.5 < self.confidence < 1.0:
            raise ValueError("decision confidence must be finite and between 0.5 and 1")

    @property
    def name(self) -> str:
        return self.label or f"{self.target} >= {self.minimum}"

    def as_preference(self) -> MinimumPreference:
        return MinimumPreference(self.target, self.minimum)


@dataclass(frozen=True)
class DecisionAssessment:
    """A decision verdict bound to one exact canonical world digest."""

    decision: MinimumDecision
    target_mean: float
    target_sigma: float
    p_satisfies: float
    verdict: DecisionVerdict
    world_digest: str

    @property
    def p_violates(self) -> float:
        return 1.0 - self.p_satisfies

    @property
    def resolved(self) -> bool:
        return self.verdict is not DecisionVerdict.UNRESOLVED

    def render(self) -> str:
        return (
            f"{self.verdict}: {self.decision.name}; "
            f"target {self.target_mean:.6f} +- {self.target_sigma:.6f}; "
            f"P(satisfies)={self.p_satisfies:.6f}; "
            f"required confidence={self.decision.confidence:.6f}"
        )


@dataclass(frozen=True)
class DecisionEvidencePlan:
    """Decision assessment plus ranked, auditable next-evidence options."""

    assessment: DecisionAssessment
    options: tuple[ObservationPlan, ...]
    selected: ObservationPlan | None
    disposition: EvidenceDisposition

    @property
    def should_observe(self) -> bool:
        return self.selected is not None

    def render(self) -> str:
        lines = [self.assessment.render(), f"next={self.disposition}"]
        if self.selected is not None:
            lines.append(self.selected.describe())
        return "\n".join(lines)


def assess_decision(world: World, decision: MinimumDecision) -> DecisionAssessment:
    """Evaluate a minimum decision under the world's current joint belief."""
    H, predicted = jacobian_rows(world.binding, world.belief, (decision.target,))
    row = H[0]
    target_mean = float(predicted[0])
    target_variance = max(float(row @ world.belief.sigma @ row), 0.0)
    target_sigma = math.sqrt(target_variance)

    if target_sigma <= 1e-15:
        p_satisfies = 1.0 if target_mean >= decision.minimum else 0.0
    else:
        standardized_margin = (target_mean - decision.minimum) / target_sigma
        p_satisfies = 0.5 * (1.0 + math.erf(standardized_margin / math.sqrt(2.0)))

    if p_satisfies >= decision.confidence:
        verdict = DecisionVerdict.SATISFIED
    elif 1.0 - p_satisfies >= decision.confidence:
        verdict = DecisionVerdict.VIOLATED
    else:
        verdict = DecisionVerdict.UNRESOLVED

    return DecisionAssessment(
        decision=decision,
        target_mean=target_mean,
        target_sigma=target_sigma,
        p_satisfies=p_satisfies,
        verdict=verdict,
        world_digest=world.digest(),
    )


def plan_decision_evidence(
    world: World,
    decision: MinimumDecision,
    candidates: Iterable[ObservationCandidate],
) -> DecisionEvidencePlan:
    """Stop if resolved; otherwise select worthwhile decision evidence.

    The no-observation baseline has zero epistemic value and zero action
    cost.  Consequently an observation is worthwhile exactly when its
    target-relevant information gain exceeds its cost in nats.
    An unresolved decision with no candidates returns NO_AVAILABLE_EVIDENCE;
    it remains unresolved and no observation is selected.
    """
    assessment = assess_decision(world, decision)
    if assessment.resolved:
        return DecisionEvidencePlan(
            assessment=assessment,
            options=(),
            selected=None,
            disposition=EvidenceDisposition.DECISION_RESOLVED,
        )

    candidates = tuple(candidates)
    if not candidates:
        return DecisionEvidencePlan(
            assessment=assessment,
            options=(),
            selected=None,
            disposition=EvidenceDisposition.NO_AVAILABLE_EVIDENCE,
        )

    options = plan_observations(world, candidates, decision.as_preference())
    selected = options[0] if options[0].net_epistemic_value > 1e-15 else None
    return DecisionEvidencePlan(
        assessment=assessment,
        options=options,
        selected=selected,
        disposition=(
            EvidenceDisposition.OBSERVE
            if selected is not None
            else EvidenceDisposition.NO_WORTHWHILE_EVIDENCE
        ),
    )
