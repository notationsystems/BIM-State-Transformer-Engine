"""Carrier-independent, restartable snapshots of a GAT computational world.

The existing JSON export is intentionally a downstream view.  A state
snapshot has a stronger contract: it must reconstruct the complete
authoritative :class:`~gat.engine.executor.World`, including the closed IR
expression algebra and the *indexed joint* raw covariance, so verified
computation can continue after a runtime boundary.

Only authoritative restart state is serialized:

* entity/variable identity, semantic attributes, placements and source refs,
* raw and derived quantity definitions, including closed expression ASTs,
* relationships and constraints,
* exact raw variable order, mean and dense covariance,
* execution-trace provenance supplied by the caller.

Derived means/covariances, dependency graphs, Jacobians, geometry views and
stability reports are recomputed by the receiving runtime.  The envelope is
versioned and SHA-256 bound; decoding compiles the IR, rebuilds the derived
view, runs the invariant registry, and requires the reconstructed module and
world digests to match the source.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from gat.engine.configuration import configuration_digest
from gat.engine.executor import World
from gat.engine.verify import run_invariants
from gat.errors import SnapshotError
from gat.gaussian.state import GaussianState
from gat.ids import EntityId, VarId
from gat.ir.core import (
    Entity,
    ExprEquals,
    LessEqual,
    Module,
    NonNegative,
    Placement,
    QtySlot,
    Rel,
    RelKind,
    Role,
    Unit,
)
from gat.ir.exprs import Add, Const, Expr, Mean, Mul, Neg, ScaledSum, Sub, VarRef
from gat.trace import TraceEvent


SNAPSHOT_FORMAT = "gat-state-snapshot"
SNAPSHOT_SCHEMA_VERSION = 1
RUNTIME_CONTRACT = "gat-world-v1"
INTEGRITY_ALGORITHM = "sha256"


@dataclass(frozen=True)
class SnapshotLoadResult:
    """A reconstructed world plus non-authoritative execution provenance."""

    world: World
    trace_events: tuple[TraceEvent, ...]
    snapshot_digest: str


@dataclass(frozen=True)
class EquivalenceCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class ComputationalEquivalenceReport:
    """Strict operational identity, distinct from configuration equivalence."""

    checks: tuple[EquivalenceCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failures(self) -> tuple[EquivalenceCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)

    def render(self) -> str:
        verdict = "EQUIVALENT" if self.passed else "NOT EQUIVALENT"
        lines = [f"computational state: {verdict}"]
        for check in self.checks:
            lines.append(
                f"{'PASS' if check.passed else 'FAIL':<4} "
                f"{check.name:<24} {check.detail}"
            )
        return "\n".join(lines)


def capture_snapshot(
    world: World,
    trace_events: Iterable[TraceEvent] = (),
) -> dict[str, object]:
    """Capture ``world`` as a pure JSON-compatible, integrity-bound value."""
    report = run_invariants(world)
    if not report.passed:
        raise SnapshotError("cannot snapshot a world that fails verification")

    raw_order = [_encode_var(var) for var in world.binding.raw_index.vars]
    payload: dict[str, object] = {
        "module": _encode_module(world.module),
        "belief": {
            "raw_variables": raw_order,
            "mean": [float(value) for value in world.belief.mu],
            "covariance": {
                "storage": "dense-row-major",
                "dimension": world.binding.n_raw,
                "values": [float(value) for value in world.belief.sigma.ravel()],
            },
        },
        "provenance": {
            "trace_events": [_encode_trace_event(event) for event in trace_events]
        },
        "source_module_digest": world.module.digest(),
        "source_world_digest": world.digest(),
        "source_configuration_digest": configuration_digest(world),
    }
    envelope: dict[str, object] = {
        "format": SNAPSHOT_FORMAT,
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "runtime_contract": RUNTIME_CONTRACT,
        "payload": payload,
    }
    envelope["integrity"] = {
        "algorithm": INTEGRITY_ALGORITHM,
        "digest": _content_digest(envelope),
    }
    # This is also the validation that every captured value is finite and
    # belongs to JSON's interoperable value model.
    _canonical_json(envelope)
    return envelope


def reconstruct_snapshot(document: Mapping[str, object]) -> SnapshotLoadResult:
    """Validate and reconstruct a complete verified world from ``document``."""
    root = _mapping(document, "snapshot")
    _require_keys(
        root,
        {"format", "schema_version", "runtime_contract", "payload", "integrity"},
        "snapshot",
    )
    if root["format"] != SNAPSHOT_FORMAT:
        raise SnapshotError(f"unsupported snapshot format {root['format']!r}")
    if root["schema_version"] != SNAPSHOT_SCHEMA_VERSION:
        raise SnapshotError(
            f"unsupported snapshot schema version {root['schema_version']!r}"
        )
    if root["runtime_contract"] != RUNTIME_CONTRACT:
        raise SnapshotError(
            f"incompatible runtime contract {root['runtime_contract']!r}"
        )

    integrity = _mapping(root["integrity"], "integrity")
    _require_keys(integrity, {"algorithm", "digest"}, "integrity")
    if integrity["algorithm"] != INTEGRITY_ALGORITHM:
        raise SnapshotError("unsupported snapshot integrity algorithm")
    expected_digest = _string(integrity["digest"], "integrity.digest")
    actual_digest = _content_digest(root)
    if not hmac.compare_digest(expected_digest, actual_digest):
        raise SnapshotError("snapshot integrity digest mismatch")

    payload = _mapping(root["payload"], "payload")
    _require_keys(
        payload,
        {
            "module",
            "belief",
            "provenance",
            "source_module_digest",
            "source_world_digest",
            "source_configuration_digest",
        },
        "payload",
    )
    module = _decode_module(_mapping(payload["module"], "payload.module"))
    source_module_digest = _string(
        payload["source_module_digest"], "source_module_digest"
    )
    if module.digest() != source_module_digest:
        raise SnapshotError("reconstructed module digest differs from source")

    try:
        compiled = World.compile(module)
    except Exception as exc:
        raise SnapshotError(f"snapshot IR could not be compiled: {exc}") from exc
    belief_record = _mapping(payload["belief"], "payload.belief")
    _require_keys(
        belief_record, {"raw_variables", "mean", "covariance"}, "payload.belief"
    )
    raw_values = _list(belief_record["raw_variables"], "belief.raw_variables")
    raw_order = tuple(
        _decode_var(_mapping(value, f"raw_variables[{index}]"))
        for index, value in enumerate(raw_values)
    )
    if raw_order != compiled.binding.raw_index.vars:
        raise SnapshotError("snapshot raw variable order differs from compiled IR")

    mean_values = _list(belief_record["mean"], "belief.mean")
    mean = np.asarray(
        [_finite_number(value, f"belief.mean[{index}]") for index, value in enumerate(mean_values)],
        dtype=np.float64,
    )
    covariance_record = _mapping(
        belief_record["covariance"], "belief.covariance"
    )
    _require_keys(
        covariance_record,
        {"storage", "dimension", "values"},
        "belief.covariance",
    )
    if covariance_record["storage"] != "dense-row-major":
        raise SnapshotError("unsupported covariance storage")
    dimension = _integer(covariance_record["dimension"], "covariance.dimension")
    if dimension != compiled.binding.n_raw or mean.shape != (dimension,):
        raise SnapshotError("belief dimension differs from compiled raw state")
    flat_values = _list(covariance_record["values"], "covariance.values")
    if len(flat_values) != dimension * dimension:
        raise SnapshotError("dense covariance value count is inconsistent")
    covariance = np.asarray(
        [
            _finite_number(value, f"covariance.values[{index}]")
            for index, value in enumerate(flat_values)
        ],
        dtype=np.float64,
    ).reshape(dimension, dimension)
    if not np.array_equal(covariance, covariance.T):
        raise SnapshotError("snapshot covariance must be exactly symmetric")

    try:
        belief = GaussianState(compiled.binding.raw_index, mean, covariance)
        world = compiled.with_belief(belief)
    except Exception as exc:
        raise SnapshotError(f"snapshot belief could not be reconstructed: {exc}") from exc
    verification = run_invariants(world)
    if not verification.passed:
        details = ", ".join(result.invariant_id for result in verification.failures)
        raise SnapshotError(f"reconstructed world failed verification: {details}")

    source_world_digest = _string(
        payload["source_world_digest"], "source_world_digest"
    )
    if world.digest() != source_world_digest:
        raise SnapshotError("reconstructed world digest differs from source")
    source_configuration_digest = _string(
        payload["source_configuration_digest"], "source_configuration_digest"
    )
    if configuration_digest(world) != source_configuration_digest:
        raise SnapshotError("reconstructed configuration digest differs from source")

    provenance = _mapping(payload["provenance"], "payload.provenance")
    _require_keys(provenance, {"trace_events"}, "payload.provenance")
    event_values = _list(provenance["trace_events"], "provenance.trace_events")
    events = tuple(
        _decode_trace_event(_mapping(value, f"trace_events[{index}]"))
        for index, value in enumerate(event_values)
    )
    if tuple(event.seq for event in events) != tuple(range(len(events))):
        raise SnapshotError("trace event sequence must be contiguous from zero")
    return SnapshotLoadResult(world, events, expected_digest)


def write_snapshot(
    world: World,
    path: str | Path,
    trace_events: Iterable[TraceEvent] = (),
) -> str:
    """Write a deterministic JSON carrier and return its integrity digest."""
    document = capture_snapshot(world, trace_events)
    text = json.dumps(
        document,
        indent=1,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    with open(path, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    integrity = _mapping(document["integrity"], "integrity")
    return _string(integrity["digest"], "integrity.digest")


def read_snapshot(path: str | Path) -> SnapshotLoadResult:
    """Read a JSON carrier and reconstruct its verified computational world."""
    def reject_constant(value: str) -> object:
        raise SnapshotError(f"non-finite JSON number {value!r}")

    try:
        with open(path, "r", encoding="utf-8") as stream:
            document = json.load(stream, parse_constant=reject_constant)
    except SnapshotError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SnapshotError(f"cannot read state snapshot: {exc}") from exc
    return reconstruct_snapshot(_mapping(document, "snapshot"))


def computational_equivalence(
    a: World,
    b: World,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> ComputationalEquivalenceReport:
    """Compare strict restart semantics rather than quotient configuration.

    The default is exact same-platform identity.  Nonzero tolerances support
    cross-platform carriers while identity, topology and expression semantics
    remain exact.
    """
    if not math.isfinite(atol) or not math.isfinite(rtol) or atol < 0 or rtol < 0:
        raise ValueError("equivalence tolerances must be finite and non-negative")

    module_equal = _encode_module(a.module) == _encode_module(b.module)
    raw_order_equal = a.binding.raw_index.vars == b.binding.raw_index.vars
    full_order_equal = a.binding.full_index.vars == b.binding.full_index.vars
    belief_mean_equal = _arrays_close(a.belief.mu, b.belief.mu, atol, rtol)
    belief_cov_equal = _arrays_close(a.belief.sigma, b.belief.sigma, atol, rtol)
    full_mean_equal = _arrays_close(a.full.mu, b.full.mu, atol, rtol)
    full_cov_equal = _arrays_close(a.full.sigma, b.full.sigma, atol, rtol)
    verify_a = run_invariants(a)
    verify_b = run_invariants(b)
    verification_equal = (
        verify_a.passed
        and verify_b.passed
        and verify_a.render() == verify_b.render()
    )
    checks = (
        EquivalenceCheck(
            "module semantics", module_equal,
            "closed IR, source identities and metadata",
        ),
        EquivalenceCheck("raw variable order", raw_order_equal, f"{a.binding.n_raw} variables"),
        EquivalenceCheck("full variable order", full_order_equal, f"{a.binding.n_full} variables"),
        EquivalenceCheck("raw belief mean", belief_mean_equal, f"atol={atol:g}, rtol={rtol:g}"),
        EquivalenceCheck("raw belief covariance", belief_cov_equal, f"atol={atol:g}, rtol={rtol:g}"),
        EquivalenceCheck("derived mean", full_mean_equal, "recomputed pushforward"),
        EquivalenceCheck("derived covariance", full_cov_equal, "recomputed pushforward"),
        EquivalenceCheck(
            "verification", verification_equal,
            "both invariant reports pass and agree",
        ),
        EquivalenceCheck(
            "configuration quotient",
            configuration_digest(a) == configuration_digest(b),
            "architectural equivalence is necessary but weaker",
        ),
    )
    return ComputationalEquivalenceReport(checks)


def _arrays_close(a: np.ndarray, b: np.ndarray, atol: float, rtol: float) -> bool:
    if a.shape != b.shape:
        return False
    if atol == 0.0 and rtol == 0.0:
        return bool(np.array_equal(a, b))
    return bool(np.allclose(a, b, atol=atol, rtol=rtol))


def _encode_module(module: Module) -> dict[str, object]:
    return {
        "meta": {str(key): str(module.meta[key]) for key in sorted(module.meta)},
        "entities": [_encode_entity(module.entities[eid]) for eid in module.entities],
        "relationships": [_encode_rel(rel) for rel in module.rels],
        "constraints": [_encode_constraint(constraint) for constraint in module.constraints],
    }


def _decode_module(record: Mapping[str, object]) -> Module:
    _require_keys(record, {"meta", "entities", "relationships", "constraints"}, "module")
    meta_record = _mapping(record["meta"], "module.meta")
    meta = {
        _text(key, "module.meta key"): _text(value, f"module.meta[{key!r}]")
        for key, value in meta_record.items()
    }
    entities_list = _list(record["entities"], "module.entities")
    entities: dict[EntityId, Entity] = {}
    for index, value in enumerate(entities_list):
        entity = _decode_entity(_mapping(value, f"entities[{index}]"))
        if entity.id in entities:
            raise SnapshotError(f"duplicate entity identity {entity.id}")
        entities[entity.id] = entity
    rels = tuple(
        _decode_rel(_mapping(value, f"relationships[{index}]"))
        for index, value in enumerate(_list(record["relationships"], "module.relationships"))
    )
    constraints = tuple(
        _decode_constraint(_mapping(value, f"constraints[{index}]"))
        for index, value in enumerate(_list(record["constraints"], "module.constraints"))
    )
    try:
        return Module(entities, rels, constraints, meta)
    except Exception as exc:
        raise SnapshotError(f"module could not be reconstructed: {exc}") from exc


def _encode_entity(entity: Entity) -> dict[str, object]:
    return {
        "id": _encode_entity_id(entity.id),
        "name": entity.name,
        "attrs": {key: entity.attrs[key] for key in sorted(entity.attrs)},
        "placement": (
            None
            if entity.placement is None
            else {
                "x": entity.placement.x,
                "y": entity.placement.y,
                "z": entity.placement.z,
                "angle": entity.placement.angle,
            }
        ),
        "source_ref": entity.source_ref,
        "slots": [_encode_slot(entity.slots[name]) for name in sorted(entity.slots)],
    }


def _decode_entity(record: Mapping[str, object]) -> Entity:
    _require_keys(record, {"id", "name", "attrs", "placement", "source_ref", "slots"}, "entity")
    eid = _decode_entity_id(_mapping(record["id"], "entity.id"))
    name = _text(record["name"], "entity.name")
    attrs_record = _mapping(record["attrs"], "entity.attrs")
    attrs: dict[str, str | int | float | bool] = {}
    for key, value in attrs_record.items():
        attr_key = _string(key, "entity attribute key")
        if not isinstance(value, (str, int, float, bool)):
            raise SnapshotError(f"entity attribute {attr_key!r} has unsupported value")
        if isinstance(value, float) and not math.isfinite(value):
            raise SnapshotError(f"entity attribute {attr_key!r} is non-finite")
        attrs[attr_key] = value
    placement_value = record["placement"]
    placement = None
    if placement_value is not None:
        placement_record = _mapping(placement_value, "entity.placement")
        _require_keys(placement_record, {"x", "y", "z", "angle"}, "entity.placement")
        placement = Placement(
            _finite_number(placement_record["x"], "placement.x"),
            _finite_number(placement_record["y"], "placement.y"),
            _finite_number(placement_record["z"], "placement.z"),
            _finite_number(placement_record["angle"], "placement.angle"),
        )
    source_ref = _optional_integer(record["source_ref"], "entity.source_ref")
    slots: dict[str, QtySlot] = {}
    for index, value in enumerate(_list(record["slots"], "entity.slots")):
        slot = _decode_slot(_mapping(value, f"entity.slots[{index}]"))
        if slot.var.entity != eid:
            raise SnapshotError("slot owner differs from enclosing entity")
        if slot.var.quantity in slots:
            raise SnapshotError(f"duplicate slot {slot.var.quantity!r}")
        slots[slot.var.quantity] = slot
    return Entity(eid, name, attrs, slots, placement, source_ref)


def _encode_slot(slot: QtySlot) -> dict[str, object]:
    return {
        "var": _encode_var(slot.var),
        "role": slot.role.value,
        "unit": slot.unit.value,
        "prior_mu": slot.prior_mu,
        "prior_sigma": slot.prior_sigma,
        "expr": None if slot.expr is None else _encode_expr(slot.expr),
        "source_ref": slot.source_ref,
    }


def _decode_slot(record: Mapping[str, object]) -> QtySlot:
    _require_keys(
        record,
        {"var", "role", "unit", "prior_mu", "prior_sigma", "expr", "source_ref"},
        "slot",
    )
    var = _decode_var(_mapping(record["var"], "slot.var"))
    try:
        role = Role(_string(record["role"], "slot.role"))
        unit = Unit(_string(record["unit"], "slot.unit"))
    except ValueError as exc:
        raise SnapshotError(f"unsupported slot role or unit: {exc}") from exc
    prior_mu = _finite_number(record["prior_mu"], "slot.prior_mu")
    prior_sigma = _finite_number(record["prior_sigma"], "slot.prior_sigma")
    expr_value = record["expr"]
    expr = None if expr_value is None else _decode_expr(_mapping(expr_value, "slot.expr"))
    source_ref = _optional_integer(record["source_ref"], "slot.source_ref")
    try:
        return QtySlot(var, role, unit, prior_mu, prior_sigma, expr, source_ref)
    except Exception as exc:
        raise SnapshotError(f"slot could not be reconstructed: {exc}") from exc


def _encode_expr(expr: Expr) -> dict[str, object]:
    if isinstance(expr, Const):
        return {"op": "const", "value": expr.value}
    if isinstance(expr, VarRef):
        return {"op": "var", "var": _encode_var(expr.var)}
    if isinstance(expr, (Add, Sub, Mul)):
        op = {Add: "add", Sub: "sub", Mul: "mul"}[type(expr)]
        return {"op": op, "left": _encode_expr(expr.left), "right": _encode_expr(expr.right)}
    if isinstance(expr, Neg):
        return {"op": "neg", "operand": _encode_expr(expr.operand)}
    if isinstance(expr, ScaledSum):
        return {
            "op": "scaled_sum",
            "const": expr.const,
            "terms": [
                {"coefficient": coefficient, "expr": _encode_expr(term)}
                for coefficient, term in expr.terms
            ],
        }
    if isinstance(expr, Mean):
        return {"op": "mean", "terms": [_encode_expr(term) for term in expr.terms]}
    raise SnapshotError(f"unsupported expression node {type(expr).__name__}")


def _decode_expr(record: Mapping[str, object]) -> Expr:
    op = _string(record.get("op"), "expression.op")
    if op == "const":
        _require_keys(record, {"op", "value"}, "const expression")
        return Const(_finite_number(record["value"], "const.value"))
    if op == "var":
        _require_keys(record, {"op", "var"}, "var expression")
        return VarRef(_decode_var(_mapping(record["var"], "var expression")))
    if op in {"add", "sub", "mul"}:
        _require_keys(record, {"op", "left", "right"}, f"{op} expression")
        left = _decode_expr(_mapping(record["left"], f"{op}.left"))
        right = _decode_expr(_mapping(record["right"], f"{op}.right"))
        return {"add": Add, "sub": Sub, "mul": Mul}[op](left, right)
    if op == "neg":
        _require_keys(record, {"op", "operand"}, "neg expression")
        return Neg(_decode_expr(_mapping(record["operand"], "neg.operand")))
    if op == "scaled_sum":
        _require_keys(record, {"op", "const", "terms"}, "scaled_sum expression")
        terms = []
        for index, value in enumerate(_list(record["terms"], "scaled_sum.terms")):
            term_record = _mapping(value, f"scaled_sum.terms[{index}]")
            _require_keys(term_record, {"coefficient", "expr"}, "scaled_sum term")
            terms.append(
                (
                    _finite_number(term_record["coefficient"], "term.coefficient"),
                    _decode_expr(_mapping(term_record["expr"], "term.expr")),
                )
            )
        return ScaledSum(
            tuple(terms), _finite_number(record["const"], "scaled_sum.const")
        )
    if op == "mean":
        _require_keys(record, {"op", "terms"}, "mean expression")
        terms = tuple(
            _decode_expr(_mapping(value, f"mean.terms[{index}]"))
            for index, value in enumerate(_list(record["terms"], "mean.terms"))
        )
        try:
            return Mean(terms)
        except ValueError as exc:
            raise SnapshotError(str(exc)) from exc
    raise SnapshotError(f"unsupported expression opcode {op!r}")


def _encode_constraint(constraint: object) -> dict[str, object]:
    if isinstance(constraint, NonNegative):
        return {"kind": "nonnegative", "var": _encode_var(constraint.var), "tol": constraint.tol}
    if isinstance(constraint, LessEqual):
        return {
            "kind": "less_equal", "lhs": _encode_var(constraint.lhs),
            "rhs": _encode_var(constraint.rhs), "tol": constraint.tol,
        }
    if isinstance(constraint, ExprEquals):
        return {
            "kind": "expr_equals", "var": _encode_var(constraint.var),
            "expr": _encode_expr(constraint.expr), "tol": constraint.tol,
        }
    raise SnapshotError(f"unsupported constraint {type(constraint).__name__}")


def _decode_constraint(record: Mapping[str, object]) -> object:
    kind = _string(record.get("kind"), "constraint.kind")
    if kind == "nonnegative":
        _require_keys(record, {"kind", "var", "tol"}, "nonnegative constraint")
        return NonNegative(
            _decode_var(_mapping(record["var"], "constraint.var")),
            _finite_number(record["tol"], "constraint.tol"),
        )
    if kind == "less_equal":
        _require_keys(record, {"kind", "lhs", "rhs", "tol"}, "less_equal constraint")
        return LessEqual(
            _decode_var(_mapping(record["lhs"], "constraint.lhs")),
            _decode_var(_mapping(record["rhs"], "constraint.rhs")),
            _finite_number(record["tol"], "constraint.tol"),
        )
    if kind == "expr_equals":
        _require_keys(record, {"kind", "var", "expr", "tol"}, "expr_equals constraint")
        return ExprEquals(
            _decode_var(_mapping(record["var"], "constraint.var")),
            _decode_expr(_mapping(record["expr"], "constraint.expr")),
            _finite_number(record["tol"], "constraint.tol"),
        )
    raise SnapshotError(f"unsupported constraint kind {kind!r}")


def _encode_rel(rel: Rel) -> dict[str, object]:
    return {
        "kind": rel.kind.value,
        "source": _encode_entity_id(rel.source),
        "target": _encode_entity_id(rel.target),
        "source_ref": rel.source_ref,
    }


def _decode_rel(record: Mapping[str, object]) -> Rel:
    _require_keys(record, {"kind", "source", "target", "source_ref"}, "relationship")
    try:
        kind = RelKind(_string(record["kind"], "relationship.kind"))
    except ValueError as exc:
        raise SnapshotError(f"unsupported relationship kind: {exc}") from exc
    return Rel(
        kind,
        _decode_entity_id(_mapping(record["source"], "relationship.source")),
        _decode_entity_id(_mapping(record["target"], "relationship.target")),
        _optional_integer(record["source_ref"], "relationship.source_ref"),
    )


def _encode_entity_id(eid: EntityId) -> dict[str, str]:
    return {"ifc_class": eid.ifc_class, "global_id": eid.global_id}


def _decode_entity_id(record: Mapping[str, object]) -> EntityId:
    _require_keys(record, {"ifc_class", "global_id"}, "entity id")
    return EntityId(
        _string(record["ifc_class"], "entity id.ifc_class"),
        _string(record["global_id"], "entity id.global_id"),
    )


def _encode_var(var: VarId) -> dict[str, object]:
    return {"entity": _encode_entity_id(var.entity), "quantity": var.quantity}


def _decode_var(record: Mapping[str, object]) -> VarId:
    _require_keys(record, {"entity", "quantity"}, "variable id")
    return VarId(
        _decode_entity_id(_mapping(record["entity"], "variable entity")),
        _string(record["quantity"], "variable quantity"),
    )


def _encode_trace_event(event: TraceEvent) -> dict[str, object]:
    return {
        "seq": event.seq,
        "stage": event.stage,
        "name": event.name,
        "detail": event.detail,
        "verify": event.verify,
        "digest": event.digest,
    }


def _decode_trace_event(record: Mapping[str, object]) -> TraceEvent:
    _require_keys(record, {"seq", "stage", "name", "detail", "verify", "digest"}, "trace event")
    return TraceEvent(
        _integer(record["seq"], "trace.seq"),
        _string(record["stage"], "trace.stage"),
        _text(record["name"], "trace.name"),
        _text(record["detail"], "trace.detail"),
        _string(record["verify"], "trace.verify"),
        _string(record["digest"], "trace.digest"),
    )


def _content_digest(envelope: Mapping[str, object]) -> str:
    material = {
        key: envelope[key]
        for key in ("format", "schema_version", "runtime_contract", "payload")
    }
    return hashlib.sha256(_canonical_json(material).encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise SnapshotError(f"snapshot contains unsupported JSON data: {exc}") from exc


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SnapshotError(f"{label} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise SnapshotError(f"{label} keys must be strings")
    return value  # type: ignore[return-value]


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise SnapshotError(f"{label} must be an array")
    return value


def _require_keys(record: Mapping[str, object], expected: set[str], label: str) -> None:
    actual = set(record)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise SnapshotError(f"{label} fields differ: missing={missing}, extra={extra}")


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise SnapshotError(f"{label} must be a non-empty string")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise SnapshotError(f"{label} must be a string")
    return value


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SnapshotError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise SnapshotError(f"{label} must be finite")
    return number


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SnapshotError(f"{label} must be an integer")
    return value


def _optional_integer(value: object, label: str) -> int | None:
    if value is None:
        return None
    return _integer(value, label)
