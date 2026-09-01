"""Decision-focused as-built clearance assurance.

This module turns probabilistic geometric clearance into an explicit
engineering decision and connects an accepted scan-evidence report to the
next inspection target.  It deliberately does *not* infer dimensions from
surface centroids or write scan-derived values into canonical BIM state.

For each existing solid element ``i`` and proposed route, the geometric
layer supplies a first-order Gaussian clearance ``C_i``.  A required gap
``c_min`` gives the modeled violation probability

    P_i = P(C_i < c_min).

The events can be correlated through shared BIM parameters, so the chance
of any violation is not computed with an independence assumption.  Instead
the assessment reports the valid bounds

    max_i P_i <= P(any violation) <= min(1, sum_i P_i).

The decision is SATISFIED only when the upper bound is below the permitted
risk and VIOLATED only when the lower bound reaches the requested confidence.
Everything between those conclusions is UNRESOLVED.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math

from gat.engine.decision import DecisionVerdict, EvidenceDisposition
from gat.errors import DecisionError
from gat.geometry.clash import score_proposed_box
from gat.geometry.gaussianize import OrientedBox
from gat.geometry.registration import ElementScanEvidence, ScanEvidenceReport
from gat.geometry.stateio import GeometryScene


@dataclass(frozen=True)
class ClearanceDecision:
    """A proposed box that must maintain a minimum geometric clearance."""

    proposed: OrientedBox
    required_clearance: float = 0.0
    confidence: float = 0.95
    position_sigma: float = 0.0
    label: str = "proposed MEP route"

    def __post_init__(self) -> None:
        values = (*self.proposed.origin, self.proposed.angle, *self.proposed.extents)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("proposed box coordinates and extents must be finite")
        if any(extent <= 0.0 for extent in self.proposed.extents):
            raise ValueError("proposed box extents must be positive")
        if not math.isfinite(self.required_clearance) or self.required_clearance < 0.0:
            raise ValueError("required_clearance must be finite and non-negative")
        if not math.isfinite(self.position_sigma) or self.position_sigma < 0.0:
            raise ValueError("position_sigma must be finite and non-negative")
        if not math.isfinite(self.confidence) or not 0.5 < self.confidence < 1.0:
            raise ValueError("confidence must be finite and between 0.5 and 1")


@dataclass(frozen=True)
class ClearanceRiskItem:
    """Modeled clearance risk between the proposal and one BIM element."""

    element_name: str
    clearance_mean: float
    clearance_sigma: float
    required_clearance: float
    p_violates: float
    direction: tuple[float, float, float]
    witness: tuple[float, float, float]

    def render(self) -> str:
        return (
            f"{self.element_name}: clearance {self.clearance_mean:+.4f} "
            f"+- {self.clearance_sigma:.4f} m; required "
            f"{self.required_clearance:.4f} m; P(violation)={self.p_violates:.6f}"
        )


@dataclass(frozen=True)
class ClearanceAssessment:
    """Dependence-safe decision bounds tied to one canonical scene version."""

    decision: ClearanceDecision
    risks: tuple[ClearanceRiskItem, ...]
    p_any_violation_lower: float
    p_any_violation_upper: float
    verdict: DecisionVerdict
    scene_version: str

    @property
    def resolved(self) -> bool:
        return self.verdict is not DecisionVerdict.UNRESOLVED

    def worst(self) -> ClearanceRiskItem | None:
        return self.risks[0] if self.risks else None

    def render(self) -> str:
        lines = [
            f"{self.verdict}: {self.decision.label}; "
            f"P(any violation) in [{self.p_any_violation_lower:.6f}, "
            f"{self.p_any_violation_upper:.6f}]; "
            f"required confidence={self.decision.confidence:.6f}"
        ]
        lines.extend(risk.render() for risk in self.risks)
        return "\n".join(lines)


class InspectionAction(StrEnum):
    """Next evidence operation; neither action mutates canonical state."""

    EXTRACT_SCAN_MEASUREMENT = "EXTRACT_SCAN_MEASUREMENT"
    RESCAN_ELEMENT = "RESCAN_ELEMENT"


@dataclass(frozen=True)
class InspectionThresholds:
    """Declared operational gates for reusing registered scan evidence."""

    min_effective_points: float = 25.0
    min_support_diversity: float = 0.50
    min_assignment_confidence: float = 0.80

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_effective_points) or self.min_effective_points <= 0:
            raise ValueError("min_effective_points must be finite and positive")
        for name, value in (
            ("min_support_diversity", self.min_support_diversity),
            ("min_assignment_confidence", self.min_assignment_confidence),
        ):
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be finite and in (0, 1]")


@dataclass(frozen=True)
class InspectionRecommendation:
    """An auditable target/action priority, not a calibrated information gain."""

    element_name: str
    action: InspectionAction
    p_violates: float
    decision_entropy_nats: float
    effective_points: float
    support_diversity: float
    assignment_confidence: float
    evidence_deficit: float
    priority_proxy: float
    direction: tuple[float, float, float]
    witness: tuple[float, float, float]

    def render(self) -> str:
        return (
            f"{self.action} {self.element_name}: priority={self.priority_proxy:.6f}; "
            f"decision_entropy={self.decision_entropy_nats:.6f} nat; "
            f"support={self.effective_points:.1f}; "
            f"diversity={self.support_diversity:.3f}; "
            f"assignment={self.assignment_confidence:.3f}"
        )


@dataclass(frozen=True)
class ClearanceEvidencePlan:
    """Clearance verdict plus ranked inspection recommendations."""

    assessment: ClearanceAssessment
    recommendations: tuple[InspectionRecommendation, ...]
    selected: InspectionRecommendation | None
    disposition: EvidenceDisposition
    scan_digest: str

    @property
    def should_observe(self) -> bool:
        return self.selected is not None

    def render(self) -> str:
        lines = [self.assessment.render(), f"next={self.disposition}"]
        if self.selected is not None:
            lines.append(self.selected.render())
        lines.append("canonical BIM state unchanged")
        return "\n".join(lines)


def assess_clearance(
    scene: GeometryScene, decision: ClearanceDecision
) -> ClearanceAssessment:
    """Assess a proposal against every solid element in the fresh scene."""
    scene.check_fresh(scene.world)
    report = score_proposed_box(
        scene,
        decision.proposed,
        position_sigma=decision.position_sigma,
        max_clearance=None,
    )

    risks: list[ClearanceRiskItem] = []
    for item in report.items:
        if item.sigma > 1e-15:
            standardized = (
                decision.required_clearance - item.clearance
            ) / item.sigma
            p_violates = 0.5 * (1.0 + math.erf(standardized / math.sqrt(2.0)))
        else:
            p_violates = (
                1.0 if item.clearance < decision.required_clearance else 0.0
            )
        risks.append(
            ClearanceRiskItem(
                element_name=item.element_a,
                clearance_mean=item.clearance,
                clearance_sigma=item.sigma,
                required_clearance=decision.required_clearance,
                p_violates=p_violates,
                direction=item.direction,
                witness=item.witness,
            )
        )

    risks.sort(key=lambda risk: (-risk.p_violates, risk.element_name))
    lower = max((risk.p_violates for risk in risks), default=0.0)
    upper = min(1.0, sum(risk.p_violates for risk in risks))
    permitted_risk = 1.0 - decision.confidence
    if upper <= permitted_risk:
        verdict = DecisionVerdict.SATISFIED
    elif lower >= decision.confidence:
        verdict = DecisionVerdict.VIOLATED
    else:
        verdict = DecisionVerdict.UNRESOLVED

    return ClearanceAssessment(
        decision=decision,
        risks=tuple(risks),
        p_any_violation_lower=lower,
        p_any_violation_upper=upper,
        verdict=verdict,
        scene_version=scene.version,
    )


def plan_clearance_evidence(
    assessment: ClearanceAssessment,
    evidence: ScanEvidenceReport,
    thresholds: InspectionThresholds = InspectionThresholds(),
) -> ClearanceEvidencePlan:
    """Use registered scan quality to route the next clearance inspection.

    The priority proxy is Bernoulli decision entropy multiplied by
    ``1 + evidence_deficit``.  It is an explicit triage heuristic, not an
    expected information gain: a calibrated geometry-observation likelihood
    is required before scan measurements may condition the BIM belief.
    """
    if evidence.scene_version != assessment.scene_version:
        raise DecisionError(
            "scan evidence and clearance assessment belong to different scenes"
        )
    if assessment.resolved:
        return ClearanceEvidencePlan(
            assessment=assessment,
            recommendations=(),
            selected=None,
            disposition=EvidenceDisposition.DECISION_RESOLVED,
            scan_digest=evidence.scan_digest,
        )

    evidence_by_name: dict[str, ElementScanEvidence] = {}
    for row in evidence.elements:
        if row.element_name in evidence_by_name:
            raise DecisionError(f"duplicate scan evidence for {row.element_name}")
        evidence_by_name[row.element_name] = row

    recommendations: list[InspectionRecommendation] = []
    for risk in assessment.risks:
        row = evidence_by_name.get(risk.element_name)
        effective_points = row.effective_points if row is not None else 0.0
        support_diversity = row.support_diversity if row is not None else 0.0
        assignment_confidence = (
            row.assignment_confidence if row is not None else 0.0
        )
        evidence_deficit = max(
            max(
                0.0,
                1.0 - effective_points / thresholds.min_effective_points,
            ),
            1.0 - support_diversity,
            1.0 - assignment_confidence,
        )
        reusable = (
            effective_points >= thresholds.min_effective_points
            and support_diversity >= thresholds.min_support_diversity
            and assignment_confidence >= thresholds.min_assignment_confidence
        )
        p = min(max(risk.p_violates, 0.0), 1.0)
        entropy = 0.0
        if 0.0 < p < 1.0:
            entropy = -p * math.log(p) - (1.0 - p) * math.log(1.0 - p)
        recommendations.append(
            InspectionRecommendation(
                element_name=risk.element_name,
                action=(
                    InspectionAction.EXTRACT_SCAN_MEASUREMENT
                    if reusable
                    else InspectionAction.RESCAN_ELEMENT
                ),
                p_violates=p,
                decision_entropy_nats=entropy,
                effective_points=effective_points,
                support_diversity=support_diversity,
                assignment_confidence=assignment_confidence,
                evidence_deficit=evidence_deficit,
                priority_proxy=entropy * (1.0 + evidence_deficit),
                direction=risk.direction,
                witness=risk.witness,
            )
        )

    recommendations.sort(
        key=lambda item: (-item.priority_proxy, -item.p_violates, item.element_name)
    )
    selected = (
        recommendations[0]
        if recommendations and recommendations[0].priority_proxy > 1e-15
        else None
    )
    return ClearanceEvidencePlan(
        assessment=assessment,
        recommendations=tuple(recommendations),
        selected=selected,
        disposition=(
            EvidenceDisposition.OBSERVE
            if selected is not None
            else EvidenceDisposition.NO_WORTHWHILE_EVIDENCE
        ),
        scan_digest=evidence.scan_digest,
    )
