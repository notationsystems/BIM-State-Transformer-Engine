"""Strict JSON boundary for headless GAT workflow evaluation.

This is intentionally a command handler rather than a web server.  It keeps
transport, authentication, tenancy, and deployment policy outside the
mathematical core while giving Blender, CI, and future Kit extensions one
closed request/response contract.

Supported v1 operations are read-only:

* ``summary`` -- inspect a loaded authoritative state;
* ``acceptance`` -- aggregate clearance, minimum, and difference checks;
* ``beam_assurance`` -- condition a bounded ANSI/AISC beam check with one
  strict material certificate and report the reproducible verdict change;
* ``change_impact`` -- preview a design change through propagation and
  verification without committing it.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

from gat.engine.decision import MinimumDecision, assess_decision
from gat.engine.transform import ScaleParameter, SetParameter, ShiftParameter
from gat.engineering import (
    BeamBendingCheck,
    BeamBendingEvaluator,
    BeamCheckResult,
    beam_assessment_record,
    explain_beam_decision_change,
    read_material_certificate,
)
from gat.errors import GatError
from gat.geometry.assurance import ClearanceDecision, assess_clearance
from gat.geometry.gaussianize import OrientedBox
from gat.geometry.stateio import derive_scene
from gat.ids import VarId
from gat.session import GatSession
from gat.workflows.acceptance import (
    AcceptanceCase,
    AcceptancePolicy,
    DifferenceDecision,
    EvidenceReceipt,
    WorkflowKind,
    assess_difference,
    clearance_check,
    difference_check,
    evaluate_acceptance_case,
    minimum_check,
)
from gat.workflows.change_impact import preview_change


REQUEST_FORMAT = "gat-headless-request-v1"
RESPONSE_FORMAT = "gat-headless-response-v1"


def handle_request(
    value: object,
    *,
    trusted_public_keys: Mapping[str, bytes] | None = None,
) -> dict[str, object]:
    """Evaluate one closed, read-only headless request."""
    request = _object(value, "request")
    _fields(request, {"format", "request_id", "operation", "state", "payload"})
    if request["format"] != REQUEST_FORMAT:
        raise ValueError(f"unsupported request format {request['format']!r}")
    request_id = _string(request["request_id"], "request_id")
    operation = _string(request["operation"], "operation")
    session = _load_session(request["state"], trusted_public_keys)
    payload = _object(request["payload"], "payload")

    if operation == "summary":
        _fields(payload, set())
        result = _summary(session)
    elif operation == "acceptance":
        result = _acceptance(session, payload)
    elif operation == "beam_assurance":
        result = _beam_assurance(session, payload)
    elif operation == "change_impact":
        result = _change_impact(session, payload)
    else:
        raise ValueError(f"unsupported headless operation {operation!r}")

    return {
        "format": RESPONSE_FORMAT,
        "request_id": request_id,
        "operation": operation,
        "world_digest": session.world.digest(),
        "result": result,
    }


def _load_session(
    value: object,
    trusted_public_keys: Mapping[str, bytes] | None,
) -> GatSession:
    state = _object(value, "state")
    _fields(
        state,
        {"kind", "path"},
        {"require_signature"},
    )
    kind = _string(state["kind"], "state.kind")
    path = _string(state["path"], "state.path")
    if kind == "ifc":
        _reject_trust_options(state, kind)
        return GatSession.load_ifc(path)
    if kind == "snapshot":
        _reject_trust_options(state, kind)
        return GatSession.load_snapshot(path)
    if kind == "openusd":
        required = state.get("require_signature", False)
        if not isinstance(required, bool):
            raise ValueError("state.require_signature must be boolean")
        return GatSession.load_openusd(
            path,
            trusted_public_keys=trusted_public_keys,
            require_signature=required,
        )
    if kind == "usd":
        _reject_trust_options(state, kind)
        return GatSession.load_usd(path)
    raise ValueError(f"unsupported state kind {kind!r}")


def _summary(session: GatSession) -> dict[str, object]:
    report = session.verify()
    passed, warned, failed = report.counts()
    return {
        "source": session.world.module.meta.get("source", "<module>"),
        "entities": len(session.world.module.entities),
        "relationships": len(session.world.module.rels),
        "constraints": len(session.world.module.constraints),
        "raw_variables": session.world.binding.n_raw,
        "derived_variables": (
            session.world.binding.n_full - session.world.binding.n_raw
        ),
        "verification": {
            "passed": report.passed,
            "pass_count": passed,
            "warning_count": warned,
            "failure_count": failed,
        },
        "carrier_trust": {
            "signature_verified": session.carrier_signature_verified,
            "key_id": session.carrier_signing_key_id,
        },
    }


def _acceptance(session: GatSession, payload: Mapping[str, object]) -> dict[str, object]:
    _fields(
        payload,
        {"case_id", "workflow", "subject", "checks"},
        {"evidence_receipts", "policy"},
    )
    raw_checks = _array(payload["checks"], "checks")
    checks = tuple(_acceptance_check(session, item) for item in raw_checks)
    case = AcceptanceCase(
        case_id=_string(payload["case_id"], "case_id"),
        workflow=WorkflowKind(_string(payload["workflow"], "workflow")),
        subject=_string(payload["subject"], "subject"),
        checks=checks,
    )
    raw_receipts = _array(payload.get("evidence_receipts", []), "evidence_receipts")
    receipts = tuple(_evidence_receipt(item) for item in raw_receipts)
    for receipt in receipts:
        _validate_receipt_against_ledger(session, receipt)
    if receipts and not session.carrier_signature_verified:
        raise ValueError(
            "headless evidence acceptance requires a trusted signed OpenUSD carrier"
        )
    policy = _acceptance_policy(payload.get("policy"))
    return evaluate_acceptance_case(case, receipts, policy=policy).to_dict()


def _beam_assurance(
    session: GatSession,
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Run the validated beam chain in memory without modifying source files."""
    _fields(
        payload,
        {
            "case_id",
            "beam_name",
            "factored_demand_n_m",
            "confidence",
            "material_certificate_path",
        },
        {"label"},
    )
    case_id = _string(payload["case_id"], "case_id")
    beam_name = _string(payload["beam_name"], "beam_name")
    beam = session.entity_by_name(beam_name)
    if beam.ifc_class != "IfcBeam":
        raise ValueError("beam_assurance subject must resolve to an IfcBeam")
    check = BeamBendingCheck(
        beam,
        factored_demand_n_m=_number(
            payload["factored_demand_n_m"],
            "factored_demand_n_m",
        ),
        confidence=_number(payload["confidence"], "confidence"),
        label=_optional_string(payload.get("label")),
    )
    evaluator = BeamBendingEvaluator()
    prior_world = session.world
    prior = evaluator.evaluate(prior_world, check)
    session.record_assessment(
        beam_assessment_record(prior_world, prior),
        provenance={"phase": "headless-prior-beam-assessment"},
    )
    prior_assessment_event = session.ledger.events[-1]

    certificate_evidence = read_material_certificate(
        _string(payload["material_certificate_path"], "material_certificate_path")
    ).to_evidence(session.world)
    evidence = certificate_evidence.observation
    if evidence.subject.entity != beam:
        raise ValueError("material certificate subject differs from requested beam")
    transition = session.run(
        evidence.transformation(session.world),
        provenance=certificate_evidence.provenance(),
    )
    transition_event = session.ledger.events[-1]
    revised = evaluator.evaluate(
        session.world,
        check,
        changed_inputs=transition.targets,
        affected_variables=transition.affected,
    )
    change = explain_beam_decision_change(
        prior_world,
        session.world,
        evidence,
        transition,
        prior,
        revised,
    )
    session.record_assessment(
        beam_assessment_record(
            session.world,
            revised,
            evidence_digest=evidence.digest(),
            change=change,
        ),
        provenance={"phase": "headless-revised-beam-assessment"},
    )
    revised_assessment_event = session.ledger.events[-1]
    report = session.verify()
    passed, warned, failed = report.counts()
    certificate_record = certificate_evidence.certificate.provenance_record()
    return {
        "case_id": case_id,
        "subject": beam_name,
        "beam": {
            "ifc_class": beam.ifc_class,
            "global_id": beam.global_id,
            "name": beam_name,
        },
        "disposition": revised.verdict.value,
        "prior": _beam_result(prior),
        "revised": _beam_result(revised),
        "decision_change": {
            "verdict_changed": change.verdict_changed,
            "reason": change.reason,
            "changed_beliefs": list(change.changed_beliefs),
            "covariance_changes": list(change.covariance_changes),
        },
        "evidence": certificate_evidence.provenance(),
        "transition": {
            "prior_world_digest": prior_world.digest(),
            "result_world_digest": transition.world.digest(),
            "targets": [_variable_record(var) for var in transition.targets],
            "affected": [_variable_record(var) for var in transition.affected],
            "ledger_event_hash": transition_event.event_hash,
            "ledger_head_hash": session.ledger.head,
            "prior_assessment_event_hash": prior_assessment_event.event_hash,
            "revised_assessment_event_hash": revised_assessment_event.event_hash,
        },
        "verification": {
            "passed": report.passed,
            "pass_count": passed,
            "warning_count": warned,
            "failure_count": failed,
        },
        "assurance": {
            "design_code_profile_validated": True,
            "scope_assertions_independently_verified": False,
            "certificate_schema_validated": True,
            "certificate_signature_verified": certificate_record["assurance"][
                "signature_verified"
            ],
            "issuer_trust_verified": certificate_record["issuer"][
                "trust_verified"
            ],
            "may_authorize": False,
        },
    }


def _beam_result(result: BeamCheckResult) -> dict[str, object]:
    assessment = result.assessment
    return {
        "world_digest": assessment.world_digest,
        "target_mean_n_m": assessment.target_mean,
        "target_sigma_n_m": assessment.target_sigma,
        "p_satisfies": assessment.p_satisfies,
        "p_violates": assessment.p_violates,
        "verdict": result.verdict.value,
        "computation": result.computation_details(),
    }


def _variable_record(var: VarId) -> dict[str, object]:
    return {
        "entity": {
            "ifc_class": var.entity.ifc_class,
            "global_id": var.entity.global_id,
        },
        "quantity": var.quantity,
    }


def _acceptance_check(session: GatSession, value: object):
    check = _object(value, "check")
    kind = _string(check.get("kind"), "check.kind")
    if kind == "minimum":
        _fields(
            check,
            {"kind", "check_id", "target", "minimum", "confidence"},
            {"label"},
        )
        decision = MinimumDecision(
            target=_var(session, check["target"]),
            minimum=_number(check["minimum"], "minimum"),
            confidence=_number(check["confidence"], "confidence"),
            label=_optional_string(check.get("label")),
        )
        return minimum_check(
            _string(check["check_id"], "check_id"),
            assess_decision(session.world, decision),
        )
    if kind == "difference":
        _fields(
            check,
            {
                "kind",
                "check_id",
                "lhs",
                "rhs",
                "minimum_margin",
                "confidence",
            },
            {"label"},
        )
        decision = DifferenceDecision(
            lhs=_var(session, check["lhs"]),
            rhs=_var(session, check["rhs"]),
            minimum_margin=_number(check["minimum_margin"], "minimum_margin"),
            confidence=_number(check["confidence"], "confidence"),
            label=_optional_string(check.get("label")),
        )
        return difference_check(
            _string(check["check_id"], "check_id"),
            assess_difference(session.world, decision),
        )
    if kind == "clearance":
        _fields(
            check,
            {
                "kind",
                "check_id",
                "proposal",
                "required_clearance",
                "confidence",
                "position_sigma",
            },
            {"label"},
        )
        proposal = _object(check["proposal"], "proposal")
        _fields(proposal, {"origin", "angle", "extents"})
        origin = _vector(proposal["origin"], "proposal.origin", 3)
        extents = _vector(proposal["extents"], "proposal.extents", 3)
        decision = ClearanceDecision(
            proposed=OrientedBox(
                origin,
                _number(proposal["angle"], "proposal.angle"),
                extents,
            ),
            required_clearance=_number(
                check["required_clearance"], "required_clearance"
            ),
            confidence=_number(check["confidence"], "confidence"),
            position_sigma=_number(check["position_sigma"], "position_sigma"),
            label=_optional_string(check.get("label")) or "proposed route",
        )
        scene = derive_scene(session.world)
        return clearance_check(
            _string(check["check_id"], "check_id"),
            assess_clearance(scene, decision),
        )
    raise ValueError(f"unsupported acceptance check kind {kind!r}")


def _evidence_receipt(value: object) -> EvidenceReceipt:
    receipt = _object(value, "evidence_receipt")
    _fields(
        receipt,
        {
            "receipt_id",
            "evidence_kind",
            "evidence_digest",
            "prior_world_digest",
            "result_world_digest",
            "calibration_id",
            "check_ids",
            "ledger_event_hash",
            "verification_passed",
        },
    )
    check_ids = tuple(
        _string(item, "check_id")
        for item in _array(receipt["check_ids"], "check_ids")
    )
    verification = receipt["verification_passed"]
    if not isinstance(verification, bool):
        raise ValueError("verification_passed must be boolean")
    return EvidenceReceipt(
        receipt_id=_string(receipt["receipt_id"], "receipt_id"),
        evidence_kind=_string(receipt["evidence_kind"], "evidence_kind"),
        evidence_digest=_string(receipt["evidence_digest"], "evidence_digest"),
        prior_world_digest=_string(
            receipt["prior_world_digest"], "prior_world_digest"
        ),
        result_world_digest=_string(
            receipt["result_world_digest"], "result_world_digest"
        ),
        calibration_id=_string(receipt["calibration_id"], "calibration_id"),
        check_ids=check_ids,
        ledger_event_hash=_string(
            receipt["ledger_event_hash"], "ledger_event_hash"
        ),
        verification_passed=verification,
    )


def _validate_receipt_against_ledger(
    session: GatSession, receipt: EvidenceReceipt
) -> None:
    """Require the untrusted JSON receipt to resolve to a verified ledger event."""
    matches = [
        event
        for event in session.ledger.events
        if event.event_hash == receipt.ledger_event_hash
    ]
    if len(matches) != 1:
        raise ValueError(
            f"evidence receipt {receipt.receipt_id!r} is not in the state ledger"
        )
    event = matches[0]
    if (
        event.kind != "transition"
        or event.prior_world_digest != receipt.prior_world_digest
        or event.result_world_digest != receipt.result_world_digest
        or event.verification is None
        or event.verification.get("passed") is not True
        or event.operation.get("op") != "observe_linearized"
        or event.operation.get("evidence_digest") != receipt.evidence_digest
        or event.provenance.get("evidence_kind") != receipt.evidence_kind
        or event.provenance.get("calibration_id") != receipt.calibration_id
        or event.provenance.get("check_ids") != list(receipt.check_ids)
    ):
        raise ValueError(
            f"evidence receipt {receipt.receipt_id!r} does not match its ledger event"
        )


def _acceptance_policy(value: object | None) -> AcceptancePolicy:
    if value is None:
        return AcceptancePolicy()
    policy = _object(value, "policy")
    _fields(
        policy,
        {"policy_id", "require_verified_evidence_for_accept", "accepted_evidence_kinds"},
    )
    required = policy["require_verified_evidence_for_accept"]
    if not isinstance(required, bool):
        raise ValueError("require_verified_evidence_for_accept must be boolean")
    kinds = frozenset(
        _string(item, "accepted_evidence_kind")
        for item in _array(policy["accepted_evidence_kinds"], "accepted_evidence_kinds")
    )
    return AcceptancePolicy(
        _string(policy["policy_id"], "policy_id"),
        required,
        kinds,
    )


def _change_impact(session: GatSession, payload: Mapping[str, object]) -> dict[str, object]:
    _fields(payload, {"change"})
    change = _object(payload["change"], "change")
    op = _string(change.get("op"), "change.op")
    if op == "set_parameter":
        _fields(change, {"op", "target", "value", "design_sigma"})
        transformation = SetParameter(
            _var(session, change["target"]),
            _number(change["value"], "value"),
            _number(change["design_sigma"], "design_sigma"),
        )
    elif op == "shift_parameter":
        _fields(change, {"op", "target", "delta"})
        transformation = ShiftParameter(
            _var(session, change["target"]),
            _number(change["delta"], "delta"),
        )
    elif op == "scale_parameter":
        _fields(change, {"op", "target", "factor"})
        transformation = ScaleParameter(
            _var(session, change["target"]),
            _number(change["factor"], "factor"),
        )
    else:
        raise ValueError(f"unsupported design-change operation {op!r}")
    return preview_change(session.world, transformation).to_dict()


def _var(session: GatSession, value: object) -> VarId:
    ref = _object(value, "variable reference")
    _fields(ref, {"entity_name", "quantity"})
    return session.var(
        _string(ref["entity_name"], "entity_name"),
        _string(ref["quantity"], "quantity"),
    )


def _reject_trust_options(state: Mapping[str, object], kind: str) -> None:
    if "require_signature" in state:
        raise ValueError(f"signature trust options are unsupported for state kind {kind!r}")


def _load_trusted_key_args(values: Sequence[str]) -> dict[str, bytes]:
    """Load deployment trust roots from CLI arguments, never request JSON."""
    keys: dict[str, bytes] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("trusted key must use KEY_ID=PATH")
        key_id, path = value.split("=", 1)
        _string(key_id, "trusted key id")
        raw = Path(_string(path, "trusted key path")).read_bytes()
        public_key = raw
        if len(public_key) != 32:
            try:
                public_key = base64.b64decode(raw.strip(), validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ValueError(
                    f"trusted key {key_id!r} must be raw bytes or base64"
                ) from exc
        if len(public_key) != 32:
            raise ValueError(f"trusted key {key_id!r} must contain 32 bytes")
        if key_id in keys:
            raise ValueError(f"duplicate trusted key id {key_id!r}")
        keys[key_id] = public_key
    return keys


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a JSON object with string keys")
    return value


def _array(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return value


def _fields(
    value: Mapping[str, object],
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    allowed = required | (optional or set())
    missing = required - set(value)
    extra = set(value) - allowed
    if missing or extra:
        raise ValueError(
            f"JSON fields differ; missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _optional_string(value: object | None) -> str:
    if value is None:
        return ""
    return _string(value, "label")


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    number = float(value)
    if not (float("-inf") < number < float("inf")):
        raise ValueError(f"{label} must be finite")
    return number


def _vector(value: object, label: str, size: int) -> tuple[float, ...]:
    values = _array(value, label)
    if len(values) != size:
        raise ValueError(f"{label} must contain {size} numbers")
    return tuple(_number(item, label) for item in values)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a headless GAT request")
    parser.add_argument("request", nargs="?", default="-", help="JSON request path or -")
    parser.add_argument("-o", "--output", help="response path; stdout when omitted")
    parser.add_argument(
        "--trusted-key",
        action="append",
        default=[],
        metavar="KEY_ID=PATH",
        help="trusted Ed25519 public key configured by the deployment",
    )
    args = parser.parse_args(argv)

    try:
        if args.request == "-":
            value = json.load(sys.stdin)
        else:
            with open(args.request, "r", encoding="utf-8") as handle:
                value = json.load(handle)
        trusted_keys = _load_trusted_key_args(args.trusted_key)
        response = handle_request(value, trusted_public_keys=trusted_keys)
        status = 0
    except (OSError, ValueError, KeyError, GatError) as exc:
        response = {
            "format": RESPONSE_FORMAT,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
        status = 2

    rendered = json.dumps(
        response,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return status


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["REQUEST_FORMAT", "RESPONSE_FORMAT", "handle_request", "main"]
