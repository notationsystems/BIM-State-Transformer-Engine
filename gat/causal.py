"""Typed, non-mutating causal records for the GAT execution ledger.

Architectural belief changes are transformations. Assessing that belief,
selecting a policy, approving a proposal, and reporting an external action are
different event classes and must never masquerade as transformations. These
records are closed JSON contracts bound to one exact world digest; ledger
replay validates them while leaving the world bit-identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import hashlib
import json
import re
from typing import TYPE_CHECKING, Mapping

from gat.engine.decision import DecisionAssessment, DecisionEvidencePlan
from gat.geometry.assurance import ClearanceAssessment, ClearanceEvidencePlan
from gat.ids import VarId

if TYPE_CHECKING:  # pragma: no cover
    from gat.workflows.acceptance import AcceptanceOutcome
    from gat.workflows.change_impact import ChangeImpactReport


_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")


class ApprovalDecision(StrEnum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    DEFERRED = "DEFERRED"
    REVOKED = "REVOKED"


class ExternalActionStatus(StrEnum):
    PROPOSED = "PROPOSED"
    AUTHORIZED = "AUTHORIZED"
    STARTED = "STARTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


def _string(value: str, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _digest(value: str | None, label: str, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _details(value: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("details must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError("details keys must be strings")
    try:
        canonical = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"details must contain finite JSON values: {exc}") from exc
    decoded = json.loads(canonical)
    if not isinstance(decoded, dict):  # pragma: no cover - guarded above
        raise ValueError("details must be a JSON object")
    return decoded


@dataclass(frozen=True)
class AssessmentRecord:
    world_digest: str
    assessment_id: str
    assessment_type: str
    subject: str
    verdict: str
    method: str
    evidence_digest: str | None = None
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _digest(self.world_digest, "world_digest")
        for label in ("assessment_id", "assessment_type", "subject", "verdict", "method"):
            _string(getattr(self, label), label)
        _digest(self.evidence_digest, "evidence_digest", optional=True)
        object.__setattr__(self, "details", _details(self.details))


@dataclass(frozen=True)
class PolicyRecord:
    world_digest: str
    policy_id: str
    policy_type: str
    disposition: str
    selected_action: str | None = None
    evidence_digest: str | None = None
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _digest(self.world_digest, "world_digest")
        for label in ("policy_id", "policy_type", "disposition"):
            _string(getattr(self, label), label)
        if self.selected_action is not None:
            _string(self.selected_action, "selected_action")
        _digest(self.evidence_digest, "evidence_digest", optional=True)
        object.__setattr__(self, "details", _details(self.details))


@dataclass(frozen=True)
class ApprovalRecord:
    world_digest: str
    approval_id: str
    authority: str
    decision: ApprovalDecision
    scope_digest: str
    reason: str = ""
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _digest(self.world_digest, "world_digest")
        _string(self.approval_id, "approval_id")
        _string(self.authority, "authority")
        if not isinstance(self.decision, ApprovalDecision):
            object.__setattr__(self, "decision", ApprovalDecision(self.decision))
        _digest(self.scope_digest, "scope_digest")
        if not isinstance(self.reason, str):
            raise ValueError("reason must be a string")
        object.__setattr__(self, "details", _details(self.details))


@dataclass(frozen=True)
class ExternalActionRecord:
    world_digest: str
    action_id: str
    action_type: str
    status: ExternalActionStatus
    authorization_ref: str | None = None
    evidence_digest: str | None = None
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _digest(self.world_digest, "world_digest")
        _string(self.action_id, "action_id")
        _string(self.action_type, "action_type")
        if not isinstance(self.status, ExternalActionStatus):
            object.__setattr__(self, "status", ExternalActionStatus(self.status))
        if self.authorization_ref is not None:
            _string(self.authorization_ref, "authorization_ref")
        _digest(self.evidence_digest, "evidence_digest", optional=True)
        object.__setattr__(self, "details", _details(self.details))


CausalRecord = AssessmentRecord | PolicyRecord | ApprovalRecord | ExternalActionRecord
CAUSAL_EVENT_KINDS = frozenset({"assessment", "policy", "approval", "external_action"})


def encode_causal_record(record: CausalRecord) -> dict[str, object]:
    if isinstance(record, AssessmentRecord):
        return {
            "record_type": "assessment",
            "world_digest": record.world_digest,
            "assessment_id": record.assessment_id,
            "assessment_type": record.assessment_type,
            "subject": record.subject,
            "verdict": record.verdict,
            "method": record.method,
            "evidence_digest": record.evidence_digest,
            "details": dict(record.details),
        }
    if isinstance(record, PolicyRecord):
        return {
            "record_type": "policy",
            "world_digest": record.world_digest,
            "policy_id": record.policy_id,
            "policy_type": record.policy_type,
            "disposition": record.disposition,
            "selected_action": record.selected_action,
            "evidence_digest": record.evidence_digest,
            "details": dict(record.details),
        }
    if isinstance(record, ApprovalRecord):
        return {
            "record_type": "approval",
            "world_digest": record.world_digest,
            "approval_id": record.approval_id,
            "authority": record.authority,
            "decision": record.decision.value,
            "scope_digest": record.scope_digest,
            "reason": record.reason,
            "details": dict(record.details),
        }
    if isinstance(record, ExternalActionRecord):
        return {
            "record_type": "external_action",
            "world_digest": record.world_digest,
            "action_id": record.action_id,
            "action_type": record.action_type,
            "status": record.status.value,
            "authorization_ref": record.authorization_ref,
            "evidence_digest": record.evidence_digest,
            "details": dict(record.details),
        }
    raise TypeError(f"unsupported causal record {type(record).__name__}")


def decode_causal_record(value: object) -> CausalRecord:
    if not isinstance(value, dict):
        raise ValueError("causal record must be an object")
    record_type = value.get("record_type")
    fields: dict[str, set[str]] = {
        "assessment": {
            "record_type", "world_digest", "assessment_id", "assessment_type",
            "subject", "verdict", "method", "evidence_digest", "details",
        },
        "policy": {
            "record_type", "world_digest", "policy_id", "policy_type",
            "disposition", "selected_action", "evidence_digest", "details",
        },
        "approval": {
            "record_type", "world_digest", "approval_id", "authority",
            "decision", "scope_digest", "reason", "details",
        },
        "external_action": {
            "record_type", "world_digest", "action_id", "action_type", "status",
            "authorization_ref", "evidence_digest", "details",
        },
    }
    if record_type not in fields:
        raise ValueError(f"unknown causal record type {record_type!r}")
    if set(value) != fields[record_type]:
        raise ValueError(
            f"{record_type} fields differ; missing={sorted(fields[record_type] - set(value))}, "
            f"extra={sorted(set(value) - fields[record_type])}"
        )
    if record_type == "assessment":
        return AssessmentRecord(
            value["world_digest"], value["assessment_id"], value["assessment_type"],
            value["subject"], value["verdict"], value["method"],
            value["evidence_digest"], value["details"],
        )
    if record_type == "policy":
        return PolicyRecord(
            value["world_digest"], value["policy_id"], value["policy_type"],
            value["disposition"], value["selected_action"],
            value["evidence_digest"], value["details"],
        )
    if record_type == "approval":
        return ApprovalRecord(
            value["world_digest"], value["approval_id"], value["authority"],
            ApprovalDecision(value["decision"]), value["scope_digest"],
            value["reason"], value["details"],
        )
    return ExternalActionRecord(
        value["world_digest"], value["action_id"], value["action_type"],
        ExternalActionStatus(value["status"]), value["authorization_ref"],
        value["evidence_digest"], value["details"],
    )


def _var(var: VarId) -> dict[str, object]:
    return {
        "entity": {
            "ifc_class": var.entity.ifc_class,
            "global_id": var.entity.global_id,
        },
        "quantity": var.quantity,
    }


def _stable_id(prefix: str, payload: Mapping[str, object]) -> str:
    canonical = json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
    return f"{prefix}:{hashlib.sha256(canonical.encode()).hexdigest()[:16]}"


def decision_assessment_record(
    assessment: DecisionAssessment,
    assessment_id: str | None = None,
    *,
    evidence_digest: str | None = None,
) -> AssessmentRecord:
    details = {
        "criterion": {
            "kind": "minimum",
            "target": _var(assessment.decision.target),
            "minimum": assessment.decision.minimum,
            "confidence": assessment.decision.confidence,
        },
        "target_mean": assessment.target_mean,
        "target_sigma": assessment.target_sigma,
        "p_satisfies": assessment.p_satisfies,
        "p_violates": assessment.p_violates,
    }
    return AssessmentRecord(
        assessment.world_digest,
        assessment_id or _stable_id("assessment", details),
        "minimum-decision",
        assessment.decision.name,
        assessment.verdict.value,
        "minimum-gaussian-posterior-v1",
        evidence_digest,
        details,
    )


def decision_policy_record(
    plan: DecisionEvidencePlan,
    policy_id: str | None = None,
) -> PolicyRecord:
    details = {
        "assessment_verdict": plan.assessment.verdict.value,
        "options": [
            {
                "candidate": _var(option.candidate.var),
                "label": option.candidate.label,
                "noise_sigma": option.candidate.noise_sigma,
                "epistemic_value": option.epistemic_value,
                "action_cost": option.action_cost,
                "expected_free_energy": option.expected_free_energy,
            }
            for option in plan.options
        ],
    }
    selected = None if plan.selected is None else plan.selected.candidate.name
    return PolicyRecord(
        plan.assessment.world_digest,
        policy_id or _stable_id("policy", details),
        "one-step-decision-evidence",
        plan.disposition.value,
        selected,
        None,
        details,
    )


def clearance_assessment_record(
    assessment: ClearanceAssessment,
    assessment_id: str | None = None,
    *,
    evidence_digest: str | None = None,
) -> AssessmentRecord:
    details = {
        "required_clearance": assessment.decision.required_clearance,
        "confidence": assessment.decision.confidence,
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
    }
    return AssessmentRecord(
        assessment.scene_version,
        assessment_id or _stable_id("clearance", details),
        "clearance-assurance",
        assessment.decision.label,
        assessment.verdict.value,
        "dependence-safe-clearance-bounds-v1",
        evidence_digest,
        details,
    )


def clearance_policy_record(
    plan: ClearanceEvidencePlan,
    policy_id: str | None = None,
) -> PolicyRecord:
    details = {
        "assessment_verdict": plan.assessment.verdict.value,
        "recommendations": [
            {
                "element": item.element_name,
                "action": item.action.value,
                "priority_proxy": item.priority_proxy,
                "decision_entropy_nats": item.decision_entropy_nats,
            }
            for item in plan.recommendations
        ],
    }
    selected = None if plan.selected is None else f"{plan.selected.action.value}:{plan.selected.element_name}"
    return PolicyRecord(
        plan.assessment.scene_version,
        policy_id or _stable_id("clearance-policy", details),
        "clearance-inspection-routing",
        plan.disposition.value,
        selected,
        plan.scan_digest,
        details,
    )


def acceptance_assessment_record(
    outcome: "AcceptanceOutcome",
    assessment_id: str | None = None,
) -> AssessmentRecord:
    """Encode a case disposition without turning it into an approval."""
    details = {
        "case_digest": outcome.case.scope_digest,
        "workflow": outcome.case.workflow.value,
        "policy_id": outcome.policy_id,
        "checks": [
            {
                "check_id": check.check_id,
                "kind": check.kind.value,
                "verdict": check.verdict.value,
                "p_satisfies_lower": check.p_satisfies_lower,
                "p_satisfies_upper": check.p_satisfies_upper,
            }
            for check in outcome.case.checks
        ],
        "rejected_check_ids": list(outcome.rejected_check_ids),
        "unresolved_check_ids": list(outcome.unresolved_check_ids),
        "uncovered_check_ids": list(outcome.uncovered_check_ids),
        "evidence_receipt_ids": list(outcome.evidence_receipt_ids),
    }
    return AssessmentRecord(
        outcome.case.world_digest,
        assessment_id or _stable_id("acceptance", details),
        "construction-acceptance",
        outcome.case.subject,
        outcome.disposition.value,
        "case-acceptance-v1",
        None,
        details,
    )


def acceptance_policy_record(
    outcome: "AcceptanceOutcome",
    policy_event_id: str | None = None,
) -> PolicyRecord:
    """Encode the evidence request/stop policy selected for a case."""
    details = {
        "case_digest": outcome.case.scope_digest,
        "workflow": outcome.case.workflow.value,
        "acceptance_policy_id": outcome.policy_id,
        "requests": [
            {
                "check_id": request.check_id,
                "action": request.action,
                "target": request.target,
                "priority": request.priority,
            }
            for request in outcome.evidence_requests
        ],
    }
    selected = (
        None
        if not outcome.evidence_requests
        else f"{outcome.evidence_requests[0].action}:"
        f"{outcome.evidence_requests[0].target}"
    )
    return PolicyRecord(
        outcome.case.world_digest,
        policy_event_id or _stable_id("acceptance-policy", details),
        "construction-acceptance-routing",
        outcome.disposition.value,
        selected,
        None,
        details,
    )


def change_impact_assessment_record(
    report: "ChangeImpactReport",
    assessment_id: str | None = None,
) -> AssessmentRecord:
    """Encode a non-mutating RFI preview bound to its exact prior world."""
    details = {
        "scope_digest": report.scope_digest,
        "candidate_world_digest": report.candidate_world_digest,
        "transformation": report.transformation_payload,
        "targets": [str(var) for var in report.targets],
        "affected": [str(var) for var in report.affected],
        "impacted_entities": list(report.impacted_entities),
        "failure_ids": [item.invariant_id for item in report.failures],
        "warning_ids": [item.invariant_id for item in report.warnings],
    }
    return AssessmentRecord(
        report.prior_world_digest,
        assessment_id or _stable_id("change-impact", details),
        "design-change-impact",
        report.transformation.describe(),
        report.disposition.value,
        "execution-preview-v1",
        None,
        details,
    )


__all__ = [
    "acceptance_assessment_record",
    "acceptance_policy_record",
    "ApprovalDecision",
    "ApprovalRecord",
    "AssessmentRecord",
    "CAUSAL_EVENT_KINDS",
    "CausalRecord",
    "ExternalActionRecord",
    "ExternalActionStatus",
    "PolicyRecord",
    "clearance_assessment_record",
    "clearance_policy_record",
    "change_impact_assessment_record",
    "decision_assessment_record",
    "decision_policy_record",
    "decode_causal_record",
    "encode_causal_record",
]
