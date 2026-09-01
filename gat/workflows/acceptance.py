"""Case-level acceptance decisions for construction evidence workflows.

The mathematical layers answer individual questions.  A workflow must also
answer the operational question: may this case be accepted, must it be
rejected, or is more evidence required?  This module provides that boundary
without granting authority to mutate BIM state or perform field work.

Acceptance is deliberately fail-closed:

* any confidently violated check rejects the case;
* any unresolved check requests evidence;
* satisfied checks are accepted only when calibrated, verified evidence is
  bound to the exact assessed world, unless a policy explicitly opts out.

Multiple checks can be aggregated for openings and prefabricated assemblies
(for example width, height, and route clearance) while preserving each
check's probability and evidence scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import hashlib
import json
import math
import re
from typing import Iterable

from gat.engine.decision import (
    DecisionAssessment,
    DecisionEvidencePlan,
    DecisionVerdict,
)
from gat.engine.executor import ExecutionResult, World
from gat.engine.propagate import jacobian_rows
from gat.errors import DecisionError
from gat.geometry.assurance import ClearanceAssessment, ClearanceEvidencePlan
from gat.geometry.scan_likelihood import ScanClearanceLikelihood
from gat.ids import VarId
from gat.ledger import LedgerEvent


_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")


class WorkflowKind(StrEnum):
    AS_BUILT_CLEARANCE = "AS_BUILT_CLEARANCE"
    PREFABRICATION_FIT = "PREFABRICATION_FIT"
    OPENING_VERIFICATION = "OPENING_VERIFICATION"


class AcceptanceDisposition(StrEnum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    REQUEST_EVIDENCE = "REQUEST_EVIDENCE"


class AcceptanceCheckKind(StrEnum):
    CLEARANCE = "CLEARANCE"
    MINIMUM = "MINIMUM"
    DIFFERENCE = "DIFFERENCE"


def _nonempty(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _digest(value: str, label: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _probability(value: float, label: str) -> float:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} must be finite and in [0, 1]")
    return value


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


@dataclass(frozen=True)
class DifferenceDecision:
    """A Gaussian fit criterion ``lhs - rhs >= minimum_margin``."""

    lhs: VarId
    rhs: VarId
    minimum_margin: float = 0.0
    confidence: float = 0.95
    label: str = ""

    def __post_init__(self) -> None:
        if self.lhs == self.rhs:
            raise ValueError("difference decision requires two distinct variables")
        if not math.isfinite(self.minimum_margin):
            raise ValueError("minimum_margin must be finite")
        if not math.isfinite(self.confidence) or not 0.5 < self.confidence < 1.0:
            raise ValueError("confidence must be finite and between 0.5 and 1")

    @property
    def name(self) -> str:
        return self.label or (
            f"{self.lhs} - {self.rhs} >= {self.minimum_margin}"
        )


@dataclass(frozen=True)
class DifferenceAssessment:
    decision: DifferenceDecision
    margin_mean: float
    margin_sigma: float
    p_satisfies: float
    verdict: DecisionVerdict
    world_digest: str

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.margin_mean)
            or not math.isfinite(self.margin_sigma)
            or self.margin_sigma < 0.0
        ):
            raise ValueError("difference assessment moments must be finite")
        _probability(self.p_satisfies, "p_satisfies")
        _digest(self.world_digest, "world_digest")


def assess_difference(world: World, decision: DifferenceDecision) -> DifferenceAssessment:
    """Assess an opening or fabrication fit margin under the joint belief."""
    rows, predicted = jacobian_rows(
        world.binding, world.belief, (decision.lhs, decision.rhs)
    )
    row = rows[0] - rows[1]
    mean = float(predicted[0] - predicted[1])
    variance = max(float(row @ world.belief.sigma @ row), 0.0)
    sigma = math.sqrt(variance)
    if sigma <= 1e-15:
        p_satisfies = 1.0 if mean >= decision.minimum_margin else 0.0
    else:
        p_satisfies = _normal_cdf((mean - decision.minimum_margin) / sigma)
    if p_satisfies >= decision.confidence:
        verdict = DecisionVerdict.SATISFIED
    elif 1.0 - p_satisfies >= decision.confidence:
        verdict = DecisionVerdict.VIOLATED
    else:
        verdict = DecisionVerdict.UNRESOLVED
    return DifferenceAssessment(
        decision,
        mean,
        sigma,
        p_satisfies,
        verdict,
        world.digest(),
    )


@dataclass(frozen=True)
class AcceptanceCheck:
    """Normalized probabilistic check used by a case-level policy."""

    check_id: str
    kind: AcceptanceCheckKind
    subject: str
    verdict: DecisionVerdict
    confidence: float
    p_satisfies_lower: float
    p_satisfies_upper: float
    world_digest: str
    details: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _nonempty(self.check_id, "check_id")
        _nonempty(self.subject, "subject")
        if not isinstance(self.kind, AcceptanceCheckKind):
            object.__setattr__(self, "kind", AcceptanceCheckKind(self.kind))
        if not isinstance(self.verdict, DecisionVerdict):
            object.__setattr__(self, "verdict", DecisionVerdict(self.verdict))
        if not math.isfinite(self.confidence) or not 0.5 < self.confidence < 1.0:
            raise ValueError("check confidence must be finite and between 0.5 and 1")
        lower = _probability(self.p_satisfies_lower, "p_satisfies_lower")
        upper = _probability(self.p_satisfies_upper, "p_satisfies_upper")
        if lower > upper:
            raise ValueError("p_satisfies_lower cannot exceed p_satisfies_upper")
        _digest(self.world_digest, "world_digest")
        canonical = json.loads(
            json.dumps(
                self.details,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        if not isinstance(canonical, dict):
            raise ValueError("check details must be a JSON object")
        object.__setattr__(self, "details", canonical)


def clearance_check(check_id: str, assessment: ClearanceAssessment) -> AcceptanceCheck:
    return AcceptanceCheck(
        check_id=check_id,
        kind=AcceptanceCheckKind.CLEARANCE,
        subject=assessment.decision.label,
        verdict=assessment.verdict,
        confidence=assessment.decision.confidence,
        p_satisfies_lower=1.0 - assessment.p_any_violation_upper,
        p_satisfies_upper=1.0 - assessment.p_any_violation_lower,
        world_digest=assessment.scene_version,
        details={
            "required_clearance": assessment.decision.required_clearance,
            "p_any_violation_lower": assessment.p_any_violation_lower,
            "p_any_violation_upper": assessment.p_any_violation_upper,
            "risks": [
                {
                    "element": risk.element_name,
                    "clearance_mean": risk.clearance_mean,
                    "clearance_sigma": risk.clearance_sigma,
                    "p_violates": risk.p_violates,
                }
                for risk in assessment.risks
            ],
        },
    )


def minimum_check(check_id: str, assessment: DecisionAssessment) -> AcceptanceCheck:
    return AcceptanceCheck(
        check_id=check_id,
        kind=AcceptanceCheckKind.MINIMUM,
        subject=assessment.decision.name,
        verdict=assessment.verdict,
        confidence=assessment.decision.confidence,
        p_satisfies_lower=assessment.p_satisfies,
        p_satisfies_upper=assessment.p_satisfies,
        world_digest=assessment.world_digest,
        details={
            "target_mean": assessment.target_mean,
            "target_sigma": assessment.target_sigma,
            "minimum": assessment.decision.minimum,
        },
    )


def difference_check(check_id: str, assessment: DifferenceAssessment) -> AcceptanceCheck:
    return AcceptanceCheck(
        check_id=check_id,
        kind=AcceptanceCheckKind.DIFFERENCE,
        subject=assessment.decision.name,
        verdict=assessment.verdict,
        confidence=assessment.decision.confidence,
        p_satisfies_lower=assessment.p_satisfies,
        p_satisfies_upper=assessment.p_satisfies,
        world_digest=assessment.world_digest,
        details={
            "lhs": str(assessment.decision.lhs),
            "rhs": str(assessment.decision.rhs),
            "minimum_margin": assessment.decision.minimum_margin,
            "margin_mean": assessment.margin_mean,
            "margin_sigma": assessment.margin_sigma,
        },
    )


@dataclass(frozen=True)
class EvidenceReceipt:
    """A verified evidence transition scoped to one or more checks."""

    receipt_id: str
    evidence_kind: str
    evidence_digest: str
    prior_world_digest: str
    result_world_digest: str
    calibration_id: str
    check_ids: tuple[str, ...]
    ledger_event_hash: str
    verification_passed: bool = True

    def __post_init__(self) -> None:
        _nonempty(self.receipt_id, "receipt_id")
        _nonempty(self.evidence_kind, "evidence_kind")
        _digest(self.evidence_digest, "evidence_digest")
        _digest(self.prior_world_digest, "prior_world_digest")
        _digest(self.result_world_digest, "result_world_digest")
        _nonempty(self.calibration_id, "calibration_id")
        _digest(self.ledger_event_hash, "ledger_event_hash")
        if not self.check_ids or len(set(self.check_ids)) != len(self.check_ids):
            raise ValueError("check_ids must be non-empty and unique")
        for check_id in self.check_ids:
            _nonempty(check_id, "check_id")
        if not isinstance(self.verification_passed, bool):
            raise ValueError("verification_passed must be boolean")

    @classmethod
    def from_scan_likelihood(
        cls,
        likelihood: ScanClearanceLikelihood,
        result: ExecutionResult,
        ledger_event: LedgerEvent,
        check_ids: Iterable[str],
        *,
        receipt_id: str | None = None,
    ) -> "EvidenceReceipt":
        """Bind an accepted scan likelihood to its verified posterior world."""
        check_ids = tuple(check_ids)
        if not result.committed or not result.report.passed:
            raise DecisionError("scan evidence transition was not committed and verified")
        if result.transformation is not likelihood.observation:
            raise DecisionError("execution result does not belong to this scan likelihood")
        if likelihood.scene_version != likelihood.observation.expected_world_digest:
            raise DecisionError("scan likelihood prior-world binding is inconsistent")
        if (
            ledger_event.kind != "transition"
            or ledger_event.prior_world_digest != likelihood.scene_version
            or ledger_event.result_world_digest != result.world.digest()
            or ledger_event.verification is None
            or ledger_event.verification.get("passed") is not True
            or ledger_event.operation.get("op") != "observe_linearized"
            or ledger_event.operation.get("evidence_digest")
            != likelihood.evidence_digest
            or ledger_event.provenance.get("evidence_kind")
            != "calibrated-scan-clearance-likelihood"
            or ledger_event.provenance.get("calibration_id")
            != likelihood.pose_source_id
            or ledger_event.provenance.get("scan_digest")
            != likelihood.scan_digest
            or ledger_event.provenance.get("check_ids") != list(check_ids)
        ):
            raise DecisionError("ledger event does not attest this scan transition")
        stable = receipt_id or f"scan:{likelihood.evidence_digest[:16]}"
        return cls(
            receipt_id=stable,
            evidence_kind="calibrated-scan-clearance-likelihood",
            evidence_digest=likelihood.evidence_digest,
            prior_world_digest=likelihood.scene_version,
            result_world_digest=result.world.digest(),
            calibration_id=likelihood.pose_source_id,
            check_ids=check_ids,
            ledger_event_hash=ledger_event.event_hash,
            verification_passed=True,
        )


@dataclass(frozen=True)
class EvidenceRequest:
    check_id: str
    action: str
    target: str
    reason: str
    priority: float = 0.0

    def __post_init__(self) -> None:
        for value, label in (
            (self.check_id, "check_id"),
            (self.action, "action"),
            (self.target, "target"),
            (self.reason, "reason"),
        ):
            _nonempty(value, label)
        if not math.isfinite(self.priority) or self.priority < 0.0:
            raise ValueError("evidence request priority must be finite and non-negative")


def clearance_evidence_request(
    check_id: str, plan: ClearanceEvidencePlan
) -> EvidenceRequest | None:
    if plan.selected is None:
        return None
    return EvidenceRequest(
        check_id,
        plan.selected.action.value,
        plan.selected.element_name,
        "clearance decision is unresolved",
        plan.selected.priority_proxy,
    )


def decision_evidence_request(
    check_id: str, plan: DecisionEvidencePlan
) -> EvidenceRequest | None:
    if plan.selected is None:
        return None
    return EvidenceRequest(
        check_id,
        "OBSERVE_QUANTITY",
        plan.selected.candidate.name,
        "minimum decision is unresolved",
        max(plan.selected.net_epistemic_value, 0.0),
    )


@dataclass(frozen=True)
class AcceptanceCase:
    case_id: str
    workflow: WorkflowKind
    subject: str
    checks: tuple[AcceptanceCheck, ...]

    def __post_init__(self) -> None:
        _nonempty(self.case_id, "case_id")
        _nonempty(self.subject, "subject")
        if not isinstance(self.workflow, WorkflowKind):
            object.__setattr__(self, "workflow", WorkflowKind(self.workflow))
        if not self.checks:
            raise ValueError("acceptance case requires at least one check")
        check_ids = [check.check_id for check in self.checks]
        if len(set(check_ids)) != len(check_ids):
            raise ValueError("acceptance check ids must be unique")
        digests = {check.world_digest for check in self.checks}
        if len(digests) != 1:
            raise DecisionError("all acceptance checks must assess the same world")

    @property
    def world_digest(self) -> str:
        return self.checks[0].world_digest

    @property
    def scope_digest(self) -> str:
        payload = {
            "case_id": self.case_id,
            "workflow": self.workflow.value,
            "subject": self.subject,
            "world_digest": self.world_digest,
            "checks": [acceptance_check_dict(check) for check in self.checks],
        }
        canonical = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AcceptancePolicy:
    policy_id: str = "gat-safe-acceptance-v1"
    require_verified_evidence_for_accept: bool = True
    accepted_evidence_kinds: frozenset[str] = frozenset(
        {"calibrated-scan-clearance-likelihood"}
    )

    def __post_init__(self) -> None:
        _nonempty(self.policy_id, "policy_id")
        if not isinstance(self.require_verified_evidence_for_accept, bool):
            raise ValueError("require_verified_evidence_for_accept must be boolean")
        if not self.accepted_evidence_kinds:
            raise ValueError("accepted_evidence_kinds must be non-empty")
        for kind in self.accepted_evidence_kinds:
            _nonempty(kind, "accepted evidence kind")


@dataclass(frozen=True)
class AcceptanceOutcome:
    case: AcceptanceCase
    policy_id: str
    disposition: AcceptanceDisposition
    reasons: tuple[str, ...]
    rejected_check_ids: tuple[str, ...]
    unresolved_check_ids: tuple[str, ...]
    uncovered_check_ids: tuple[str, ...]
    evidence_requests: tuple[EvidenceRequest, ...]
    evidence_receipt_ids: tuple[str, ...]

    @property
    def may_authorize(self) -> bool:
        """True only for an acceptance recommendation; still not an approval."""
        return self.disposition is AcceptanceDisposition.ACCEPT

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case.case_id,
            "case_digest": self.case.scope_digest,
            "workflow": self.case.workflow.value,
            "subject": self.case.subject,
            "world_digest": self.case.world_digest,
            "policy_id": self.policy_id,
            "disposition": self.disposition.value,
            "may_authorize": self.may_authorize,
            "reasons": list(self.reasons),
            "rejected_check_ids": list(self.rejected_check_ids),
            "unresolved_check_ids": list(self.unresolved_check_ids),
            "uncovered_check_ids": list(self.uncovered_check_ids),
            "evidence_receipt_ids": list(self.evidence_receipt_ids),
            "evidence_requests": [
                {
                    "check_id": request.check_id,
                    "action": request.action,
                    "target": request.target,
                    "reason": request.reason,
                    "priority": request.priority,
                }
                for request in self.evidence_requests
            ],
            "checks": [acceptance_check_dict(check) for check in self.case.checks],
        }


def evaluate_acceptance_case(
    case: AcceptanceCase,
    receipts: Iterable[EvidenceReceipt] = (),
    requests: Iterable[EvidenceRequest] = (),
    policy: AcceptancePolicy = AcceptancePolicy(),
) -> AcceptanceOutcome:
    """Apply a fail-closed case policy without recording an approval."""
    receipts = tuple(receipts)
    requests = tuple(requests)
    known = {check.check_id for check in case.checks}
    for request in requests:
        if request.check_id not in known:
            raise DecisionError(f"evidence request names unknown check {request.check_id}")

    covered: set[str] = set()
    valid_receipt_ids: list[str] = []
    for receipt in receipts:
        unknown = set(receipt.check_ids) - known
        if unknown:
            raise DecisionError(
                f"evidence receipt names unknown checks {sorted(unknown)}"
            )
        if receipt.result_world_digest != case.world_digest:
            raise DecisionError(
                f"evidence receipt {receipt.receipt_id} is stale for this case"
            )
        if (
            receipt.verification_passed
            and receipt.evidence_kind in policy.accepted_evidence_kinds
        ):
            covered.update(receipt.check_ids)
            valid_receipt_ids.append(receipt.receipt_id)

    rejected = tuple(
        check.check_id
        for check in case.checks
        if check.verdict is DecisionVerdict.VIOLATED
    )
    unresolved = tuple(
        check.check_id
        for check in case.checks
        if check.verdict is DecisionVerdict.UNRESOLVED
    )
    satisfied = tuple(
        check.check_id
        for check in case.checks
        if check.verdict is DecisionVerdict.SATISFIED
    )
    uncovered = (
        tuple(check_id for check_id in satisfied if check_id not in covered)
        if policy.require_verified_evidence_for_accept
        else ()
    )

    request_by_check = {request.check_id: request for request in requests}
    generated: list[EvidenceRequest] = []
    check_by_id = {check.check_id: check for check in case.checks}
    for check_id in (*unresolved, *uncovered):
        request = request_by_check.get(check_id)
        if request is None:
            check = check_by_id[check_id]
            request = EvidenceRequest(
                check_id,
                "ACQUIRE_CALIBRATED_EVIDENCE",
                check.subject,
                (
                    "decision is unresolved"
                    if check_id in unresolved
                    else "verified as-built evidence is required before acceptance"
                ),
            )
        generated.append(request)
    generated.sort(key=lambda item: (-item.priority, item.check_id, item.action))

    reasons: list[str] = []
    if rejected:
        disposition = AcceptanceDisposition.REJECT
        reasons.append("one or more checks are violated at the required confidence")
    elif unresolved:
        disposition = AcceptanceDisposition.REQUEST_EVIDENCE
        reasons.append("one or more checks remain probabilistically unresolved")
    elif uncovered:
        disposition = AcceptanceDisposition.REQUEST_EVIDENCE
        reasons.append("satisfied checks lack verified evidence for this exact world")
    else:
        disposition = AcceptanceDisposition.ACCEPT
        reasons.append("all checks are satisfied and required evidence is verified")

    return AcceptanceOutcome(
        case=case,
        policy_id=policy.policy_id,
        disposition=disposition,
        reasons=tuple(reasons),
        rejected_check_ids=rejected,
        unresolved_check_ids=unresolved,
        uncovered_check_ids=uncovered,
        evidence_requests=tuple(generated),
        evidence_receipt_ids=tuple(sorted(valid_receipt_ids)),
    )


def acceptance_check_dict(check: AcceptanceCheck) -> dict[str, object]:
    return {
        "check_id": check.check_id,
        "kind": check.kind.value,
        "subject": check.subject,
        "verdict": check.verdict.value,
        "confidence": check.confidence,
        "p_satisfies_lower": check.p_satisfies_lower,
        "p_satisfies_upper": check.p_satisfies_upper,
        "world_digest": check.world_digest,
        "details": check.details,
    }


__all__ = [
    "AcceptanceCase",
    "AcceptanceCheck",
    "AcceptanceCheckKind",
    "AcceptanceDisposition",
    "AcceptanceOutcome",
    "AcceptancePolicy",
    "DifferenceAssessment",
    "DifferenceDecision",
    "EvidenceReceipt",
    "EvidenceRequest",
    "WorkflowKind",
    "acceptance_check_dict",
    "assess_difference",
    "clearance_check",
    "clearance_evidence_request",
    "decision_evidence_request",
    "difference_check",
    "evaluate_acceptance_case",
    "minimum_check",
]
