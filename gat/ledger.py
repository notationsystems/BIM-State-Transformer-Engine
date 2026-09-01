"""Authoritative, hash-chained history for GAT state transitions.

The execution trace is intentionally human-oriented.  This module supplies
the complementary machine contract: every accepted or rejected operation is
encoded in a closed transformation algebra; assessments, policies, approvals,
and external actions use a separate closed causal vocabulary. Every event is
bound to its exact prior/result world digests and linked to the preceding event
by SHA-256. A ledger can be replayed from its genesis checkpoint and must
reproduce every verification result, rejection, causal binding, and final
covariance exactly.

There is deliberately no implicit wall-clock timestamp.  Sequence, causal
order, and state identity are authoritative; external time may be supplied as
hash-bound provenance when a trusted clock or sensor provides it.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping

import numpy as np

from gat.causal import (
    ApprovalDecision,
    ApprovalRecord,
    CAUSAL_EVENT_KINDS,
    CausalRecord,
    ExternalActionRecord,
    ExternalActionStatus,
    decode_causal_record,
    encode_causal_record,
)
from gat.engine.configuration import configuration_digest
from gat.engine.dynamics import EvolveLinearGaussian
from gat.engine.executor import ExecutionResult, World, execute
from gat.engine.transform import (
    CompositeTransformation,
    Measurement,
    ObserveLinearized,
    ObserveQuantity,
    ScaleParameter,
    SetParameter,
    ShiftParameter,
    Transformation,
)
from gat.engine.verify import VerificationReport, run_invariants
from gat.errors import GatError, LedgerError, VerificationError
from gat.ids import EntityId, VarId


LEDGER_FORMAT = "gat-execution-ledger"
LEDGER_SCHEMA_VERSION = 1
LEDGER_RUNTIME_CONTRACT = "gat-world-v1"
LEDGER_HASH_ALGORITHM = "sha256"
LEDGER_MAX_BYTES = 16 * 1024 * 1024
LEDGER_MAX_EVENTS = 100_000
ZERO_HASH = "0" * 64

_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_EVENT_FIELDS = {
    "seq",
    "kind",
    "operation",
    "provenance",
    "prior_world_digest",
    "result_world_digest",
    "verification",
    "verification_digest",
    "error_type",
    "error_message",
    "error_digest",
    "previous_hash",
    "event_hash",
}


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise LedgerError(f"value is not canonical JSON: {exc}") from exc


def _normalise_json(value: object, label: str) -> object:
    """Copy JSON data while rejecting non-string keys and non-finite numbers."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LedgerError(f"{label} contains a non-finite number")
        return value
    if isinstance(value, (list, tuple)):
        return [_normalise_json(item, f"{label}[]") for item in value]
    if isinstance(value, Mapping):
        out: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise LedgerError(f"{label} contains a non-string object key")
            out[key] = _normalise_json(item, f"{label}.{key}")
        return out
    raise LedgerError(f"{label} contains unsupported value {type(value).__name__}")


def _json_object(value: object, label: str) -> dict[str, object]:
    normalised = _normalise_json(value, label)
    if not isinstance(normalised, dict):
        raise LedgerError(f"{label} must be an object")
    return normalised


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise LedgerError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _optional_digest(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _require_digest(value, label)


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise LedgerError(f"{label} must be a non-empty string")
    return value


def _require_number(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LedgerError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise LedgerError(f"{label} must be finite")
    if positive and number <= 0.0:
        raise LedgerError(f"{label} must be positive")
    return number


def _expect_fields(record: Mapping[str, object], fields: set[str], label: str) -> None:
    actual = set(record)
    if actual != fields:
        missing = sorted(fields - actual)
        extra = sorted(actual - fields)
        raise LedgerError(f"{label} fields differ; missing={missing}, extra={extra}")


def _encode_var(var: VarId) -> dict[str, object]:
    return {
        "entity": {
            "ifc_class": var.entity.ifc_class,
            "global_id": var.entity.global_id,
        },
        "quantity": var.quantity,
    }


def _decode_var(value: object, label: str) -> VarId:
    record = _json_object(value, label)
    _expect_fields(record, {"entity", "quantity"}, label)
    entity = _json_object(record["entity"], f"{label}.entity")
    _expect_fields(entity, {"ifc_class", "global_id"}, f"{label}.entity")
    return VarId(
        EntityId(
            _require_string(entity["ifc_class"], f"{label}.entity.ifc_class"),
            _require_string(entity["global_id"], f"{label}.entity.global_id"),
        ),
        _require_string(record["quantity"], f"{label}.quantity"),
    )


def encode_transformation(t: Transformation) -> dict[str, object]:
    """Encode one operation in the closed ledger-v1 transformation algebra."""
    if isinstance(t, SetParameter):
        return {
            "op": "set_parameter",
            "var": _encode_var(t.var),
            "value": _require_number(t.value, "set_parameter.value"),
            "design_sigma": _require_number(
                t.design_sigma, "set_parameter.design_sigma", positive=True
            ),
        }
    if isinstance(t, ShiftParameter):
        return {
            "op": "shift_parameter",
            "var": _encode_var(t.var),
            "delta": _require_number(t.delta, "shift_parameter.delta"),
        }
    if isinstance(t, ScaleParameter):
        return {
            "op": "scale_parameter",
            "var": _encode_var(t.var),
            "factor": _require_number(t.factor, "scale_parameter.factor"),
        }
    if isinstance(t, ObserveQuantity):
        measurements = []
        for index, measurement in enumerate(t.measurements):
            measurements.append(
                {
                    "var": _encode_var(measurement.var),
                    "value": _require_number(
                        measurement.value, f"observe_quantity.measurements[{index}].value"
                    ),
                    "noise_sigma": _require_number(
                        measurement.noise_sigma,
                        f"observe_quantity.measurements[{index}].noise_sigma",
                        positive=True,
                    ),
                }
            )
        return {"op": "observe_quantity", "measurements": measurements}
    if isinstance(t, ObserveLinearized):
        return {
            "op": "observe_linearized",
            "row": [
                _require_number(value, f"observe_linearized.row[{index}]")
                for index, value in enumerate(t.row.tolist())
            ],
            "predicted": _require_number(t.predicted, "observe_linearized.predicted"),
            "observed": _require_number(t.observed, "observe_linearized.observed"),
            "noise_sigma": _require_number(
                t.noise_sigma, "observe_linearized.noise_sigma", positive=True
            ),
            "raw_targets": [_encode_var(var) for var in t.target_vars()],
            "expected_raw_order": [
                _encode_var(var) for var in t.expected_raw_order
            ],
            "expected_belief_digest": _require_digest(
                t.expected_belief_digest,
                "observe_linearized.expected_belief_digest",
            ),
            "expected_world_digest": _require_digest(
                t.expected_world_digest, "observe_linearized.expected_world_digest"
            ),
            "evidence_digest": _require_digest(
                t.evidence_digest, "observe_linearized.evidence_digest"
            ),
            "label": _require_string(t.label, "observe_linearized.label"),
        }
    if isinstance(t, EvolveLinearGaussian):
        return {
            "op": "evolve_linear_gaussian",
            "targets": [_encode_var(var) for var in t.targets],
            "transition": [
                [_require_number(value, "evolve_linear_gaussian.transition") for value in row]
                for row in t.transition.tolist()
            ],
            "offset": [
                _require_number(value, "evolve_linear_gaussian.offset")
                for value in t.offset.tolist()
            ],
            "process_covariance": [
                [
                    _require_number(value, "evolve_linear_gaussian.process_covariance")
                    for value in row
                ]
                for row in t.process_covariance.tolist()
            ],
            "elapsed_seconds": _require_number(
                t.elapsed_seconds, "evolve_linear_gaussian.elapsed_seconds", positive=True
            ),
            "model_id": _require_string(t.model_id, "evolve_linear_gaussian.model_id"),
            "calibration_digest": _require_digest(
                t.calibration_digest, "evolve_linear_gaussian.calibration_digest"
            ),
        }
    if isinstance(t, CompositeTransformation):
        return {"op": "composite", "steps": [encode_transformation(s) for s in t.steps]}
    raise LedgerError(
        f"transformation {type(t).__name__} is outside the ledger-v1 closed algebra"
    )


def decode_transformation(value: object) -> Transformation:
    """Decode and validate a ledger-v1 transformation record."""
    record = _json_object(value, "operation")
    op = _require_string(record.get("op"), "operation.op")
    try:
        if op == "set_parameter":
            _expect_fields(record, {"op", "var", "value", "design_sigma"}, op)
            return SetParameter(
                _decode_var(record["var"], f"{op}.var"),
                _require_number(record["value"], f"{op}.value"),
                _require_number(record["design_sigma"], f"{op}.design_sigma", positive=True),
            )
        if op == "shift_parameter":
            _expect_fields(record, {"op", "var", "delta"}, op)
            return ShiftParameter(
                _decode_var(record["var"], f"{op}.var"),
                _require_number(record["delta"], f"{op}.delta"),
            )
        if op == "scale_parameter":
            _expect_fields(record, {"op", "var", "factor"}, op)
            return ScaleParameter(
                _decode_var(record["var"], f"{op}.var"),
                _require_number(record["factor"], f"{op}.factor"),
            )
        if op == "observe_quantity":
            _expect_fields(record, {"op", "measurements"}, op)
            raw = record["measurements"]
            if not isinstance(raw, list) or not raw:
                raise LedgerError("observe_quantity.measurements must be a non-empty array")
            measurements = []
            for index, value in enumerate(raw):
                measurement = _json_object(value, f"{op}.measurements[{index}]")
                _expect_fields(
                    measurement, {"var", "value", "noise_sigma"},
                    f"{op}.measurements[{index}]",
                )
                measurements.append(
                    Measurement(
                        _decode_var(measurement["var"], f"{op}.measurements[{index}].var"),
                        _require_number(measurement["value"], f"{op}.measurements[{index}].value"),
                        _require_number(
                            measurement["noise_sigma"],
                            f"{op}.measurements[{index}].noise_sigma",
                            positive=True,
                        ),
                    )
                )
            return ObserveQuantity(tuple(measurements))
        if op == "observe_linearized":
            fields = {
                "op", "row", "predicted", "observed", "noise_sigma",
                "raw_targets", "expected_raw_order", "expected_belief_digest",
                "expected_world_digest", "evidence_digest", "label",
            }
            _expect_fields(record, fields, op)
            row = record["row"]
            raw_targets = record["raw_targets"]
            raw_order = record["expected_raw_order"]
            if not isinstance(row, list) or not row:
                raise LedgerError("observe_linearized.row must be a non-empty array")
            if not isinstance(raw_targets, list) or not raw_targets:
                raise LedgerError("observe_linearized.raw_targets must be a non-empty array")
            if not isinstance(raw_order, list) or not raw_order:
                raise LedgerError("observe_linearized.expected_raw_order must be a non-empty array")
            return ObserveLinearized(
                np.array(
                    [_require_number(item, f"{op}.row[{i}]") for i, item in enumerate(row)],
                    dtype=np.float64,
                ),
                _require_number(record["predicted"], f"{op}.predicted"),
                _require_number(record["observed"], f"{op}.observed"),
                _require_number(record["noise_sigma"], f"{op}.noise_sigma", positive=True),
                tuple(_decode_var(item, f"{op}.raw_targets[{i}]") for i, item in enumerate(raw_targets)),
                tuple(_decode_var(item, f"{op}.expected_raw_order[{i}]") for i, item in enumerate(raw_order)),
                _require_digest(record["expected_belief_digest"], f"{op}.expected_belief_digest"),
                _require_digest(record["expected_world_digest"], f"{op}.expected_world_digest"),
                _require_digest(record["evidence_digest"], f"{op}.evidence_digest"),
                _require_string(record["label"], f"{op}.label"),
            )
        if op == "evolve_linear_gaussian":
            fields = {
                "op", "targets", "transition", "offset", "process_covariance",
                "elapsed_seconds", "model_id", "calibration_digest",
            }
            _expect_fields(record, fields, op)
            targets = record["targets"]
            transition = record["transition"]
            offset = record["offset"]
            process_covariance = record["process_covariance"]
            if not isinstance(targets, list) or not targets:
                raise LedgerError("evolve_linear_gaussian.targets must be a non-empty array")
            if not isinstance(transition, list) or not isinstance(offset, list):
                raise LedgerError("evolve_linear_gaussian transition and offset must be arrays")
            if not isinstance(process_covariance, list):
                raise LedgerError("evolve_linear_gaussian.process_covariance must be an array")

            def matrix(values: list[object], label: str) -> np.ndarray:
                rows = []
                for i, value in enumerate(values):
                    if not isinstance(value, list):
                        raise LedgerError(f"{label}[{i}] must be an array")
                    rows.append(
                        [_require_number(item, f"{label}[{i}][{j}]") for j, item in enumerate(value)]
                    )
                try:
                    return np.array(rows, dtype=np.float64)
                except ValueError as exc:
                    raise LedgerError(f"{label} must be rectangular") from exc

            return EvolveLinearGaussian(
                tuple(_decode_var(item, f"{op}.targets[{i}]") for i, item in enumerate(targets)),
                matrix(transition, f"{op}.transition"),
                np.array(
                    [_require_number(item, f"{op}.offset[{i}]") for i, item in enumerate(offset)],
                    dtype=np.float64,
                ),
                matrix(process_covariance, f"{op}.process_covariance"),
                _require_number(record["elapsed_seconds"], f"{op}.elapsed_seconds", positive=True),
                _require_string(record["model_id"], f"{op}.model_id"),
                _require_digest(record["calibration_digest"], f"{op}.calibration_digest"),
            )
        if op == "composite":
            _expect_fields(record, {"op", "steps"}, op)
            steps = record["steps"]
            if not isinstance(steps, list) or not steps:
                raise LedgerError("composite.steps must be a non-empty array")
            return CompositeTransformation(tuple(decode_transformation(step) for step in steps))
    except (TypeError, ValueError) as exc:
        raise LedgerError(f"invalid {op} operation: {exc}") from exc
    raise LedgerError(f"unknown ledger-v1 transformation opcode {op!r}")


def verification_payload(report: VerificationReport) -> dict[str, object]:
    """Return the complete deterministic verification evidence."""
    return {
        "passed": report.passed,
        "results": [
            {
                "invariant_id": result.invariant_id,
                "status": result.status.value,
                "subject": result.subject,
                "residual": _require_number(result.residual, "verification.residual"),
                "detail": result.detail,
            }
            for result in report.results
        ],
    }


def verification_digest(report: VerificationReport) -> str:
    return _sha256_text(_canonical_json(verification_payload(report)))


def _validate_verification_record(value: object) -> dict[str, object]:
    record = _json_object(value, "verification")
    _expect_fields(record, {"passed", "results"}, "verification")
    if not isinstance(record["passed"], bool):
        raise LedgerError("verification.passed must be boolean")
    results = record["results"]
    if not isinstance(results, list) or not results:
        raise LedgerError("verification.results must be a non-empty array")
    saw_failure = False
    for index, value in enumerate(results):
        result = _json_object(value, f"verification.results[{index}]")
        _expect_fields(
            result,
            {"invariant_id", "status", "subject", "residual", "detail"},
            f"verification.results[{index}]",
        )
        _require_string(result["invariant_id"], f"verification.results[{index}].invariant_id")
        status = _require_string(result["status"], f"verification.results[{index}].status")
        if status not in {"PASS", "WARN", "FAIL"}:
            raise LedgerError(f"verification.results[{index}].status is unknown")
        saw_failure = saw_failure or status == "FAIL"
        if not isinstance(result["subject"], str) or not isinstance(result["detail"], str):
            raise LedgerError(f"verification.results[{index}] text fields must be strings")
        _require_number(result["residual"], f"verification.results[{index}].residual")
    if record["passed"] == saw_failure:
        raise LedgerError("verification.passed disagrees with invariant results")
    return record


def _error_digest_parts(error_type: str, message: str) -> str:
    return _sha256_text(f"{error_type}\0{message}")


def _error_digest(error: GatError) -> str:
    return _error_digest_parts(type(error).__name__, str(error))


@dataclass(frozen=True)
class LedgerEvent:
    """One immutable event. Nested JSON is held in canonical strings."""

    seq: int
    kind: str
    operation_json: str
    provenance_json: str
    prior_world_digest: str
    result_world_digest: str
    verification_json: str | None
    verification_digest: str | None
    error_type: str | None
    error_message: str | None
    error_digest: str | None
    previous_hash: str
    event_hash: str

    @property
    def operation(self) -> dict[str, object]:
        return json.loads(self.operation_json)

    @property
    def provenance(self) -> dict[str, object]:
        return json.loads(self.provenance_json)

    @property
    def verification(self) -> dict[str, object] | None:
        return None if self.verification_json is None else json.loads(self.verification_json)

    def _material(self) -> dict[str, object]:
        return {
            "seq": self.seq,
            "kind": self.kind,
            "operation": self.operation,
            "provenance": self.provenance,
            "prior_world_digest": self.prior_world_digest,
            "result_world_digest": self.result_world_digest,
            "verification": self.verification,
            "verification_digest": self.verification_digest,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "error_digest": self.error_digest,
            "previous_hash": self.previous_hash,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._material(), "event_hash": self.event_hash}

    @classmethod
    def create(
        cls,
        *,
        seq: int,
        kind: str,
        operation: Mapping[str, object],
        provenance: Mapping[str, object] | None,
        prior_world_digest: str,
        result_world_digest: str,
        report: VerificationReport | None,
        error_type: str | None,
        error_message: str | None,
        error_digest: str | None,
        previous_hash: str,
    ) -> "LedgerEvent":
        operation_obj = _json_object(operation, "operation")
        provenance_obj = _json_object(provenance or {}, "provenance")
        verification_obj = None if report is None else verification_payload(report)
        provisional = cls(
            seq,
            kind,
            _canonical_json(operation_obj),
            _canonical_json(provenance_obj),
            _require_digest(prior_world_digest, "prior_world_digest"),
            _require_digest(result_world_digest, "result_world_digest"),
            None if verification_obj is None else _canonical_json(verification_obj),
            None if verification_obj is None else _sha256_text(_canonical_json(verification_obj)),
            error_type,
            error_message,
            _optional_digest(error_digest, "error_digest"),
            _require_digest(previous_hash, "previous_hash"),
            "",
        )
        event_hash = _sha256_text(_canonical_json(provisional._material()))
        return cls(**{**provisional.__dict__, "event_hash": event_hash})

    @classmethod
    def from_dict(cls, value: object) -> "LedgerEvent":
        record = _json_object(value, "event")
        _expect_fields(record, _EVENT_FIELDS, "event")
        seq = record["seq"]
        if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
            raise LedgerError("event.seq must be a non-negative integer")
        kind = _require_string(record["kind"], "event.kind")
        error_type = record["error_type"]
        if error_type is not None:
            error_type = _require_string(error_type, "event.error_type")
        error_message = record["error_message"]
        if error_message is not None and not isinstance(error_message, str):
            raise LedgerError("event.error_message must be a string or null")
        verification = record["verification"]
        if verification is not None:
            verification = _json_object(verification, "event.verification")
        event = cls(
            seq,
            kind,
            _canonical_json(_json_object(record["operation"], "event.operation")),
            _canonical_json(_json_object(record["provenance"], "event.provenance")),
            _require_digest(record["prior_world_digest"], "event.prior_world_digest"),
            _require_digest(record["result_world_digest"], "event.result_world_digest"),
            None if verification is None else _canonical_json(verification),
            _optional_digest(record["verification_digest"], "event.verification_digest"),
            error_type,
            error_message,
            _optional_digest(record["error_digest"], "event.error_digest"),
            _require_digest(record["previous_hash"], "event.previous_hash"),
            _require_digest(record["event_hash"], "event.event_hash"),
        )
        expected = _sha256_text(_canonical_json(event._material()))
        if event.event_hash != expected:
            raise LedgerError(f"event {event.seq} hash mismatch")
        if event.verification is None:
            if event.verification_digest is not None:
                raise LedgerError(f"event {event.seq} has a digest but no verification record")
        elif _sha256_text(_canonical_json(event.verification)) != event.verification_digest:
            raise LedgerError(f"event {event.seq} verification digest mismatch")
        return event


class ExecutionLedger:
    """Append-only ledger rooted at an exact computational checkpoint."""

    def __init__(self, events: tuple[LedgerEvent, ...]):
        self._events = list(events)
        self.validate()

    @classmethod
    def genesis(
        cls,
        world: World,
        provenance: Mapping[str, object] | None = None,
    ) -> "ExecutionLedger":
        digest = world.digest()
        event = LedgerEvent.create(
            seq=0,
            kind="genesis",
            operation={
                "module_digest": world.module.digest(),
                "belief_digest": world.belief.digest(),
                "configuration_digest": configuration_digest(world),
                "runtime_contract": LEDGER_RUNTIME_CONTRACT,
            },
            provenance=provenance,
            prior_world_digest=digest,
            result_world_digest=digest,
            report=run_invariants(world),
            error_type=None,
            error_message=None,
            error_digest=None,
            previous_hash=ZERO_HASH,
        )
        return cls((event,))

    @property
    def events(self) -> tuple[LedgerEvent, ...]:
        return tuple(self._events)

    @property
    def head(self) -> str:
        return self._events[-1].event_hash

    def validate(self) -> None:
        if not self._events:
            raise LedgerError("ledger must contain a genesis event")
        if len(self._events) > LEDGER_MAX_EVENTS:
            raise LedgerError(f"ledger exceeds {LEDGER_MAX_EVENTS} events")
        previous = ZERO_HASH
        for index, event in enumerate(self._events):
            if event.seq != index:
                raise LedgerError(f"event sequence discontinuity at position {index}")
            if event.previous_hash != previous:
                raise LedgerError(f"event {index} does not link to its predecessor")
            expected = _sha256_text(_canonical_json(event._material()))
            if event.event_hash != expected:
                raise LedgerError(f"event {index} hash mismatch")
            if event.verification is None:
                if event.verification_digest is not None:
                    raise LedgerError(f"event {index} has a digest but no verification record")
            else:
                _validate_verification_record(event.verification)
                expected_verification = _sha256_text(_canonical_json(event.verification))
                if event.verification_digest != expected_verification:
                    raise LedgerError(f"event {index} verification digest mismatch")
            if index == 0:
                self._validate_genesis(event)
            else:
                if event.prior_world_digest != self._events[index - 1].result_world_digest:
                    raise LedgerError(f"event {index} prior world is not causally continuous")
                self._validate_operation_event(event)
            previous = event.event_hash
        self._validate_causal_sequence()

    def _validate_causal_sequence(self) -> None:
        approvals: dict[str, ApprovalDecision] = {}
        actions: dict[str, ExternalActionStatus] = {}
        for event in self._events:
            if event.kind not in CAUSAL_EVENT_KINDS:
                continue
            record = decode_causal_record(event.operation)
            if isinstance(record, ApprovalRecord):
                self._check_approval_transition(approvals.get(record.approval_id), record)
                approvals[record.approval_id] = record.decision
            elif isinstance(record, ExternalActionRecord):
                self._check_action_transition(actions.get(record.action_id), record)
                actions[record.action_id] = record.status

    @staticmethod
    def _check_approval_transition(
        previous: ApprovalDecision | None,
        record: ApprovalRecord,
    ) -> None:
        allowed = {
            None: {
                ApprovalDecision.APPROVED,
                ApprovalDecision.REJECTED,
                ApprovalDecision.DEFERRED,
            },
            ApprovalDecision.DEFERRED: {
                ApprovalDecision.APPROVED,
                ApprovalDecision.REJECTED,
                ApprovalDecision.DEFERRED,
            },
            ApprovalDecision.APPROVED: {ApprovalDecision.REVOKED},
            ApprovalDecision.REJECTED: set(),
            ApprovalDecision.REVOKED: set(),
        }
        if record.decision not in allowed[previous]:
            before = "NEW" if previous is None else previous.value
            raise LedgerError(
                f"invalid approval lifecycle {before} -> {record.decision.value} "
                f"for {record.approval_id!r}"
            )

    @staticmethod
    def _check_action_transition(
        previous: ExternalActionStatus | None,
        record: ExternalActionRecord,
    ) -> None:
        allowed = {
            None: {ExternalActionStatus.PROPOSED},
            ExternalActionStatus.PROPOSED: {
                ExternalActionStatus.AUTHORIZED,
                ExternalActionStatus.CANCELLED,
            },
            ExternalActionStatus.AUTHORIZED: {
                ExternalActionStatus.STARTED,
                ExternalActionStatus.CANCELLED,
            },
            ExternalActionStatus.STARTED: {
                ExternalActionStatus.COMPLETED,
                ExternalActionStatus.FAILED,
                ExternalActionStatus.CANCELLED,
            },
            ExternalActionStatus.COMPLETED: set(),
            ExternalActionStatus.FAILED: set(),
            ExternalActionStatus.CANCELLED: set(),
        }
        if record.status not in allowed[previous]:
            before = "NEW" if previous is None else previous.value
            raise LedgerError(
                f"invalid external-action lifecycle {before} -> {record.status.value} "
                f"for {record.action_id!r}"
            )
        if record.status in {
            ExternalActionStatus.AUTHORIZED,
            ExternalActionStatus.STARTED,
            ExternalActionStatus.COMPLETED,
            ExternalActionStatus.FAILED,
        } and record.authorization_ref is None:
            raise LedgerError(
                f"external action {record.action_id!r} status {record.status.value} "
                "requires an authorization reference"
            )

    @staticmethod
    def _validate_genesis(event: LedgerEvent) -> None:
        if event.kind != "genesis":
            raise LedgerError("first event must be genesis")
        if event.prior_world_digest != event.result_world_digest:
            raise LedgerError("genesis prior and result world digests must match")
        if (
            event.verification is None
            or event.verification_digest is None
            or event.error_type is not None
            or event.error_message is not None
            or event.error_digest is not None
        ):
            raise LedgerError("genesis verification/error fields are inconsistent")
        _validate_verification_record(event.verification)
        expected_fields = {
            "module_digest", "belief_digest", "configuration_digest", "runtime_contract"
        }
        _expect_fields(event.operation, expected_fields, "genesis.operation")
        for field in ("module_digest", "belief_digest", "configuration_digest"):
            _require_digest(event.operation[field], f"genesis.operation.{field}")
        if event.operation["runtime_contract"] != LEDGER_RUNTIME_CONTRACT:
            raise LedgerError("unsupported genesis runtime contract")

    @staticmethod
    def _validate_operation_event(event: LedgerEvent) -> None:
        if event.kind in CAUSAL_EVENT_KINDS:
            try:
                record = decode_causal_record(event.operation)
            except (TypeError, ValueError) as exc:
                raise LedgerError(f"invalid {event.kind} record: {exc}") from exc
            if event.operation.get("record_type") != event.kind:
                raise LedgerError("causal event kind differs from its record type")
            if (
                event.prior_world_digest != event.result_world_digest
                or event.prior_world_digest != record.world_digest
            ):
                raise LedgerError("causal event must preserve and bind the current world")
            if (
                event.verification is not None
                or event.verification_digest is not None
                or event.error_type is not None
                or event.error_message is not None
                or event.error_digest is not None
            ):
                raise LedgerError("causal event cannot carry transition verification or error fields")
            return
        if event.kind not in {"transition", "rejection"}:
            raise LedgerError(f"unsupported ledger event kind {event.kind!r}")
        decode_transformation(event.operation)
        if event.kind == "transition":
            if event.verification is None or event.verification_digest is None:
                raise LedgerError("transition must carry verification evidence")
            _validate_verification_record(event.verification)
            if event.verification["passed"] is not True:
                raise LedgerError("transition verification must pass")
            if (
                event.error_type is not None
                or event.error_message is not None
                or event.error_digest is not None
            ):
                raise LedgerError("transition cannot carry an error")
        else:
            if event.prior_world_digest != event.result_world_digest:
                raise LedgerError("rejection must preserve the world digest")
            if (
                event.error_type is None
                or event.error_message is None
                or event.error_digest is None
            ):
                raise LedgerError("rejection must carry an error type, message, and digest")
            if _error_digest_parts(event.error_type, event.error_message) != event.error_digest:
                raise LedgerError("rejection error digest does not match its type and message")
            if event.error_type == "VerificationError":
                if event.verification is None or event.verification_digest is None:
                    raise LedgerError("verification rejection must carry verification evidence")
                _validate_verification_record(event.verification)
                if event.verification["passed"] is not False:
                    raise LedgerError("verification rejection report must fail")
            elif event.verification is not None or event.verification_digest is not None:
                raise LedgerError("non-verification rejection cannot carry verification evidence")

    def _append(
        self,
        *,
        kind: str,
        operation: Mapping[str, object],
        provenance: Mapping[str, object] | None,
        prior_world_digest: str,
        result_world_digest: str,
        report: VerificationReport | None,
        error_type: str | None,
        error_message: str | None,
        error_digest: str | None,
    ) -> LedgerEvent:
        if len(self._events) >= LEDGER_MAX_EVENTS:
            raise LedgerError(f"ledger exceeds {LEDGER_MAX_EVENTS} events")
        event = LedgerEvent.create(
            seq=len(self._events),
            kind=kind,
            operation=operation,
            provenance=provenance,
            prior_world_digest=prior_world_digest,
            result_world_digest=result_world_digest,
            report=report,
            error_type=error_type,
            error_message=error_message,
            error_digest=error_digest,
            previous_hash=self.head,
        )
        self._validate_operation_event(event)
        self._events.append(event)
        return event

    def record_transition(
        self,
        before: World,
        result: ExecutionResult,
        provenance: Mapping[str, object] | None = None,
    ) -> LedgerEvent:
        if not result.committed or not result.report.passed:
            raise LedgerError("only a committed, verified result is a transition")
        if self._events[-1].result_world_digest != before.digest():
            raise LedgerError("ledger head does not describe the session's prior world")
        return self._append(
            kind="transition",
            operation=encode_transformation(result.transformation),
            provenance=provenance,
            prior_world_digest=before.digest(),
            result_world_digest=result.world.digest(),
            report=result.report,
            error_type=None,
            error_message=None,
            error_digest=None,
        )

    def record_rejection(
        self,
        before: World,
        transformation: Transformation,
        error: GatError,
        provenance: Mapping[str, object] | None = None,
    ) -> LedgerEvent:
        if self._events[-1].result_world_digest != before.digest():
            raise LedgerError("ledger head does not describe the session's prior world")
        report = error.report if isinstance(error, VerificationError) else None
        return self._append(
            kind="rejection",
            operation=encode_transformation(transformation),
            provenance=provenance,
            prior_world_digest=before.digest(),
            result_world_digest=before.digest(),
            report=report,
            error_type=type(error).__name__,
            error_message=str(error),
            error_digest=_error_digest(error),
        )

    def record_causal(
        self,
        world: World,
        record: CausalRecord,
        provenance: Mapping[str, object] | None = None,
    ) -> LedgerEvent:
        """Record a typed event that is about the world but does not mutate it."""
        digest = world.digest()
        if self._events[-1].result_world_digest != digest:
            raise LedgerError("ledger head does not describe the session's current world")
        if record.world_digest != digest:
            raise LedgerError("causal record is stale or belongs to a different world")
        if isinstance(record, ApprovalRecord):
            previous = None
            for event in reversed(self._events):
                if event.kind != "approval":
                    continue
                earlier = decode_causal_record(event.operation)
                if isinstance(earlier, ApprovalRecord) and earlier.approval_id == record.approval_id:
                    previous = earlier.decision
                    break
            self._check_approval_transition(previous, record)
        elif isinstance(record, ExternalActionRecord):
            previous = None
            for event in reversed(self._events):
                if event.kind != "external_action":
                    continue
                earlier = decode_causal_record(event.operation)
                if isinstance(earlier, ExternalActionRecord) and earlier.action_id == record.action_id:
                    previous = earlier.status
                    break
            self._check_action_transition(previous, record)
        operation = encode_causal_record(record)
        return self._append(
            kind=str(operation["record_type"]),
            operation=operation,
            provenance=provenance,
            prior_world_digest=digest,
            result_world_digest=digest,
            report=None,
            error_type=None,
            error_message=None,
            error_digest=None,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "format": LEDGER_FORMAT,
            "schema_version": LEDGER_SCHEMA_VERSION,
            "runtime_contract": LEDGER_RUNTIME_CONTRACT,
            "events": [event.to_dict() for event in self._events],
            "integrity": {"algorithm": LEDGER_HASH_ALGORITHM, "head": self.head},
        }

    @classmethod
    def from_dict(cls, value: object) -> "ExecutionLedger":
        record = _json_object(value, "ledger")
        _expect_fields(
            record,
            {"format", "schema_version", "runtime_contract", "events", "integrity"},
            "ledger",
        )
        if record["format"] != LEDGER_FORMAT:
            raise LedgerError("unsupported ledger format")
        if record["schema_version"] != LEDGER_SCHEMA_VERSION:
            raise LedgerError("unsupported ledger schema version")
        if record["runtime_contract"] != LEDGER_RUNTIME_CONTRACT:
            raise LedgerError("unsupported ledger runtime contract")
        raw_events = record["events"]
        if not isinstance(raw_events, list) or not raw_events:
            raise LedgerError("ledger.events must be a non-empty array")
        if len(raw_events) > LEDGER_MAX_EVENTS:
            raise LedgerError(f"ledger exceeds {LEDGER_MAX_EVENTS} events")
        ledger = cls(tuple(LedgerEvent.from_dict(event) for event in raw_events))
        integrity = _json_object(record["integrity"], "ledger.integrity")
        _expect_fields(integrity, {"algorithm", "head"}, "ledger.integrity")
        if integrity["algorithm"] != LEDGER_HASH_ALGORITHM:
            raise LedgerError("unsupported ledger integrity algorithm")
        if _require_digest(integrity["head"], "ledger.integrity.head") != ledger.head:
            raise LedgerError("ledger integrity head does not match the event chain")
        return ledger


@dataclass(frozen=True)
class LedgerReplayResult:
    world: World
    accepted: int
    rejected: int
    non_state: int
    events_replayed: int
    head: str


def replay_ledger(initial_world: World, ledger: ExecutionLedger) -> LedgerReplayResult:
    """Replay and verify every event from the supplied exact genesis world."""
    ledger.validate()
    genesis = ledger.events[0]
    operation = genesis.operation
    expected = {
        "world": initial_world.digest(),
        "module": initial_world.module.digest(),
        "belief": initial_world.belief.digest(),
        "configuration": configuration_digest(initial_world),
        "verification": verification_digest(run_invariants(initial_world)),
        "verification_record": verification_payload(run_invariants(initial_world)),
    }
    actual = {
        "world": genesis.result_world_digest,
        "module": operation["module_digest"],
        "belief": operation["belief_digest"],
        "configuration": operation["configuration_digest"],
        "verification": genesis.verification_digest,
        "verification_record": genesis.verification,
    }
    for label in expected:
        if actual[label] != expected[label]:
            raise LedgerError(f"genesis {label} digest does not match the checkpoint")

    current = initial_world
    accepted = 0
    rejected = 0
    non_state = 0
    for event in ledger.events[1:]:
        if current.digest() != event.prior_world_digest:
            raise LedgerError(f"event {event.seq} prior world digest does not match replay state")
        if event.kind in CAUSAL_EVENT_KINDS:
            try:
                record = decode_causal_record(event.operation)
            except (TypeError, ValueError) as exc:  # pragma: no cover - validate catches
                raise LedgerError(f"event {event.seq} causal record is invalid") from exc
            if record.world_digest != current.digest():
                raise LedgerError(f"event {event.seq} causal record is bound to another world")
            if event.result_world_digest != current.digest():
                raise LedgerError(f"event {event.seq} causal event changed world identity")
            non_state += 1
            continue
        transformation = decode_transformation(event.operation)
        try:
            result = execute(current, transformation, strict=True)
        except GatError as exc:
            if event.kind != "rejection":
                raise LedgerError(
                    f"event {event.seq} recorded a transition but replay rejected it as {type(exc).__name__}"
                ) from exc
            if (
                type(exc).__name__ != event.error_type
                or str(exc) != event.error_message
                or _error_digest(exc) != event.error_digest
            ):
                raise LedgerError(f"event {event.seq} rejection differs during replay") from exc
            report = exc.report if isinstance(exc, VerificationError) else None
            digest = None if report is None else verification_digest(report)
            record = None if report is None else verification_payload(report)
            if digest != event.verification_digest or record != event.verification:
                raise LedgerError(f"event {event.seq} rejection verification differs during replay") from exc
            rejected += 1
        else:
            if event.kind != "transition":
                raise LedgerError(f"event {event.seq} recorded rejection but replay accepted it")
            if not result.committed or result.world.digest() != event.result_world_digest:
                raise LedgerError(f"event {event.seq} result world differs during replay")
            if verification_digest(result.report) != event.verification_digest:
                raise LedgerError(f"event {event.seq} verification differs during replay")
            if verification_payload(result.report) != event.verification:
                raise LedgerError(f"event {event.seq} verification record differs during replay")
            current = result.world
            accepted += 1
        if current.digest() != event.result_world_digest:
            raise LedgerError(f"event {event.seq} result digest does not match replay state")
    return LedgerReplayResult(
        current,
        accepted,
        rejected,
        non_state,
        len(ledger.events) - 1,
        ledger.head,
    )


def write_ledger(ledger: ExecutionLedger, path: str | Path) -> str:
    """Write canonical ledger JSON and return its hash-chain head."""
    ledger.validate()
    text = json.dumps(ledger.to_dict(), allow_nan=False, indent=2, sort_keys=True) + "\n"
    if len(text.encode("utf-8")) > LEDGER_MAX_BYTES:
        raise LedgerError(f"ledger exceeds {LEDGER_MAX_BYTES} encoded bytes")
    Path(path).write_text(text, encoding="utf-8", newline="\n")
    return ledger.head


def read_ledger(path: str | Path) -> ExecutionLedger:
    """Read a bounded ledger JSON document and validate the complete chain."""
    source = Path(path)
    size = source.stat().st_size
    if size > LEDGER_MAX_BYTES:
        raise LedgerError(f"ledger exceeds {LEDGER_MAX_BYTES} encoded bytes")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LedgerError(f"could not read ledger: {exc}") from exc
    return ExecutionLedger.from_dict(value)
