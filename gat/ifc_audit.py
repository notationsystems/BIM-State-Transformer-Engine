"""Non-mutating compatibility audit for real IFC inputs.

The ordinary IFC adapter is intentionally fail-closed: the first unsupported
unit, missing quantity, placement, or storey topology raises ``LoweringError``.
That is the right behavior for authoritative computation but a poor discovery
tool for unfamiliar models.  This module inventories the complete supported
product scope first, then attempts the real lower -> compile -> verify pipeline
without changing the source file or weakening the loader.

An audit never authorizes a construction decision.  A successful audit only
means that the current adapter can compile its declared product scope.  A
caller must still bind a decision to an explicit, fully covered scope.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

from gat.adapters.ifc.lower import REQUIRED_QUANTITIES, lower_ifc
from gat.adapters.ifc.parser import EnumVal, IfcFile, OMITTED, RawInstance, Ref, parse_ifc
from gat.adapters.ifc.reader import (
    attr,
    global_id,
    name_of,
    properties_of,
    pset_values,
    quantities_of,
    resolve_placement,
)
from gat.adapters.ifc.schema import (
    ANNOTATED_PRODUCT_CLASSES,
    PRODUCT_CLASSES,
    SUPPORTED_ENTITIES,
)
from gat.adapters.ifc.units import SI_PREFIX_SCALE, assigned_unit_ids
from gat.engine.executor import World
from gat.engine.verify import run_invariants
from gat.errors import GatError


AUDIT_FORMAT = "gat-ifc-audit-v1"


class AuditStatus(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    BLOCKED = "BLOCKED"
    NOT_RUN = "NOT_RUN"


class EntityStatus(str, Enum):
    READY = "READY"
    NEEDS_GEOMETRY_DERIVATION = "NEEDS_GEOMETRY_DERIVATION"
    MISSING_SOURCE_DATA = "MISSING_SOURCE_DATA"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class AuditIssue:
    code: str
    severity: str
    message: str
    step_id: int | None = None
    ifc_type: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "step_id": self.step_id,
            "ifc_type": self.ifc_type,
        }


@dataclass(frozen=True)
class LengthUnitAudit:
    step_id: int | None
    kind: str
    name: str
    prefix: str | None
    scale_to_metres: float | None
    accepted_by_current_adapter: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "step_id": self.step_id,
            "kind": self.kind,
            "name": self.name,
            "prefix": self.prefix,
            "scale_to_metres": self.scale_to_metres,
            "normalization_required": self.scale_to_metres not in (None, 1.0),
            "accepted_by_current_adapter": self.accepted_by_current_adapter,
        }


@dataclass(frozen=True)
class EntityAudit:
    step_id: int
    ifc_type: str
    canonical_class: str
    global_id: str | None
    name: str
    status: EntityStatus
    required_quantities: tuple[str, ...]
    available_quantities: tuple[str, ...]
    missing_quantities: tuple[str, ...]
    has_geometry_representation: bool
    issues: tuple[AuditIssue, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "step_id": self.step_id,
            "ifc_type": self.ifc_type,
            "canonical_class": self.canonical_class,
            "global_id": self.global_id,
            "name": self.name,
            "status": self.status.value,
            "required_quantities": list(self.required_quantities),
            "available_quantities": list(self.available_quantities),
            "missing_quantities": list(self.missing_quantities),
            "has_geometry_representation": self.has_geometry_representation,
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class StageAudit:
    status: AuditStatus
    error_type: str | None = None
    message: str | None = None
    details: tuple[tuple[str, object], ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "error_type": self.error_type,
            "message": self.message,
            "details": {key: value for key, value in self.details},
        }


@dataclass(frozen=True)
class IfcAuditReport:
    source: str
    source_sha256: str
    size_bytes: int
    parse: StageAudit
    schema: str | None
    type_counts: tuple[tuple[str, int], ...]
    length_units: tuple[LengthUnitAudit, ...]
    entities: tuple[EntityAudit, ...]
    model_issues: tuple[AuditIssue, ...]
    lowering: StageAudit
    compilation: StageAudit
    verification: StageAudit
    world_digest: str | None = None

    @property
    def pipeline_ready(self) -> bool:
        return (
            self.parse.status is AuditStatus.PASS
            and self.lowering.status is AuditStatus.PASS
            and self.compilation.status is AuditStatus.PASS
            and self.verification.status in (AuditStatus.PASS, AuditStatus.WARN)
        )

    @property
    def issue_counts(self) -> tuple[tuple[str, int], ...]:
        issues: Iterable[AuditIssue] = (
            *self.model_issues,
            *(issue for entity in self.entities for issue in entity.issues),
        )
        return tuple(sorted(Counter(issue.code for issue in issues).items()))

    def to_dict(self) -> dict[str, object]:
        status_counts = Counter(entity.status.value for entity in self.entities)
        supported_types = tuple(sorted(PRODUCT_CLASSES))
        opaque_counts = tuple(
            (name, count)
            for name, count in self.type_counts
            if name not in SUPPORTED_ENTITIES
        )
        return {
            "format": AUDIT_FORMAT,
            "source": {
                "path": self.source,
                "sha256": self.source_sha256,
                "size_bytes": self.size_bytes,
            },
            "parse": self.parse.to_dict(),
            "schema": self.schema,
            "units": [unit.to_dict() for unit in self.length_units],
            "inventory": {
                "instance_count": sum(count for _, count in self.type_counts),
                "type_counts": {name: count for name, count in self.type_counts},
                "opaque_type_counts": {name: count for name, count in opaque_counts},
                "opt_in_product_candidate_counts": {
                    name: dict(self.type_counts).get(name, 0)
                    for name in sorted(ANNOTATED_PRODUCT_CLASSES)
                },
                "supported_product_count": len(self.entities),
                "supported_product_status_counts": {
                    status: status_counts[status] for status in sorted(status_counts)
                },
            },
            "adapter_scope": {
                "supported_ifc_product_types": list(supported_types),
                "opt_in_ifc_product_types": {
                    name: marker
                    for name, (_, marker) in sorted(
                        ANNOTATED_PRODUCT_CLASSES.items()
                    )
                },
                "required_quantities": {
                    name: list(REQUIRED_QUANTITIES[canonical])
                    for name, canonical in sorted(PRODUCT_CLASSES.items())
                },
                "coverage_boundary": "supported-product-scope-only",
            },
            "entities": [entity.to_dict() for entity in self.entities],
            "model_issues": [issue.to_dict() for issue in self.model_issues],
            "issue_counts": {code: count for code, count in self.issue_counts},
            "pipeline": {
                "lowering": self.lowering.to_dict(),
                "compilation": self.compilation.to_dict(),
                "verification": self.verification.to_dict(),
                "world_digest": self.world_digest,
                "pipeline_ready": self.pipeline_ready,
            },
            "assurance": {
                "audit_authorizes_decisions": False,
                "requires_explicit_decision_scope": True,
                "partial_ingestion_may_authorize": False,
            },
        }

    def to_json(self, *, pretty: bool = True) -> str:
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
            sort_keys=True,
        ) + "\n"

    def render(self) -> str:
        counts = Counter(entity.status.value for entity in self.entities)
        lines = [
            f"IFC audit: {'PIPELINE READY' if self.pipeline_ready else 'BLOCKED'}",
            f"  schema: {self.schema or '<unparsed>'}",
            f"  source: {self.source_sha256[:12]} ({self.size_bytes} bytes)",
            f"  supported products: {len(self.entities)}",
        ]
        for status in sorted(counts):
            lines.append(f"    {status}: {counts[status]}")
        if self.issue_counts:
            lines.append("  issue counts:")
            for code, count in self.issue_counts:
                lines.append(f"    {code}: {count}")
        lines.append(
            "  assurance: audit only; an explicit fully covered decision scope is required"
        )
        return "\n".join(lines)


def _length_units(file: IfcFile) -> tuple[LengthUnitAudit, ...]:
    units: list[LengthUnitAudit] = []
    project_units = assigned_unit_ids(file)
    for inst in file.by_type("IFCCONVERSIONBASEDUNIT"):
        if project_units is not None and inst.step_id not in project_units:
            continue
        unit_type = inst.args[1] if len(inst.args) > 1 else None
        if isinstance(unit_type, EnumVal) and unit_type.name == "LENGTHUNIT":
            name = inst.args[2] if len(inst.args) > 2 and isinstance(inst.args[2], str) else ""
            units.append(
                LengthUnitAudit(inst.step_id, "CONVERSION_BASED", name, None, None, False)
            )
    for inst in file.by_type("IFCSIUNIT"):
        if project_units is not None and inst.step_id not in project_units:
            continue
        try:
            unit_type = attr(inst, "UnitType")
            if not isinstance(unit_type, EnumVal) or unit_type.name != "LENGTHUNIT":
                continue
            prefix_value = attr(inst, "Prefix")
            name_value = attr(inst, "Name")
        except GatError:
            units.append(LengthUnitAudit(inst.step_id, "MALFORMED_SI", "", None, None, False))
            continue
        prefix = prefix_value.name if isinstance(prefix_value, EnumVal) else None
        name = name_value.name if isinstance(name_value, EnumVal) else ""
        scale = (1.0 if prefix is None else SI_PREFIX_SCALE.get(prefix)) if name == "METRE" else None
        units.append(
            LengthUnitAudit(
                inst.step_id,
                "SI",
                name,
                prefix,
                scale,
                name == "METRE" and scale is not None,
            )
        )
    if not units and project_units is not None:
        raise GatError("project unit assignment has no supported length unit")
    if not units:
        units.append(LengthUnitAudit(None, "ASSUMED_SI", "METRE", None, 1.0, True))
    return tuple(sorted(units, key=lambda unit: (-1 if unit.step_id is None else unit.step_id)))


def _has_representation(inst: RawInstance) -> bool:
    # IfcProduct.Representation occupies position 6 in IFC2x3 and IFC4 for
    # every product class currently recognized by the adapter.
    return len(inst.args) > 6 and inst.args[6] is not None and inst.args[6] is not OMITTED


def _audit_entity(
    file: IfcFile,
    inst: RawInstance,
    canonical: str,
    definitions: list[RawInstance] | None,
    property_error: Exception | None,
) -> EntityAudit:
    issues: list[AuditIssue] = []
    gid: str | None = None
    name = name_of(inst)
    try:
        gid = global_id(inst)
    except GatError as exc:
        issues.append(
            AuditIssue("INVALID_GLOBAL_ID", "ERROR", str(exc), inst.step_id, inst.type_name)
        )

    quantities: dict[str, tuple[float, int]] = {}
    if property_error is not None:
        issues.append(
            AuditIssue(
                "PROPERTY_GRAPH_UNREADABLE",
                "ERROR",
                str(property_error),
                inst.step_id,
                inst.type_name,
            )
        )
    else:
        try:
            quantities = quantities_of(file, definitions or [])
        except GatError as exc:
            issues.append(
                AuditIssue(
                    "QUANTITY_SET_UNREADABLE",
                    "ERROR",
                    str(exc),
                    inst.step_id,
                    inst.type_name,
                )
            )

    required = tuple(REQUIRED_QUANTITIES.get(canonical, ()))
    available = tuple(sorted(quantities))
    missing = tuple(name for name in required if name not in quantities)
    has_geometry = _has_representation(inst)
    if missing:
        source = "geometry fallback is available to build" if has_geometry else "no geometry fallback exists"
        issues.append(
            AuditIssue(
                "MISSING_REQUIRED_QUANTITY",
                "ERROR",
                f"missing {', '.join(missing)}; {source}",
                inst.step_id,
                inst.type_name,
            )
        )

    if canonical == "IfcBeam" and property_error is None:
        try:
            structural = pset_values(file, definitions or [], "GAT_Structural")
            structural_required = {
                "YieldStrengthMPa",
                "YieldStrengthMPaSigma",
                "SectionModulusM3",
                "SectionModulusM3Sigma",
                "ResistanceFactor",
            }
            missing_structural = sorted(structural_required - set(structural))
            if missing_structural:
                issues.append(
                    AuditIssue(
                        "MISSING_STRUCTURAL_PROPERTY",
                        "ERROR",
                        "GAT_Structural lacks " + ", ".join(missing_structural),
                        inst.step_id,
                        inst.type_name,
                    )
                )
        except GatError as exc:
            issues.append(
                AuditIssue(
                    "STRUCTURAL_PROPERTY_SET_UNREADABLE",
                    "ERROR",
                    str(exc),
                    inst.step_id,
                    inst.type_name,
                )
            )

    try:
        placement = attr(inst, "ObjectPlacement")
        if isinstance(placement, Ref):
            resolve_placement(file, placement)
    except GatError as exc:
        issues.append(
            AuditIssue(
                "UNSUPPORTED_PLACEMENT",
                "ERROR",
                str(exc),
                inst.step_id,
                inst.type_name,
            )
        )

    non_quantity_error = any(issue.code != "MISSING_REQUIRED_QUANTITY" for issue in issues)
    if non_quantity_error:
        status = EntityStatus.BLOCKED
    elif missing and has_geometry:
        status = EntityStatus.NEEDS_GEOMETRY_DERIVATION
    elif missing:
        status = EntityStatus.MISSING_SOURCE_DATA
    else:
        status = EntityStatus.READY
    return EntityAudit(
        inst.step_id,
        inst.type_name,
        canonical,
        gid,
        name,
        status,
        required,
        available,
        missing,
        has_geometry,
        tuple(issues),
    )


def _blocked_stage(exc: Exception) -> StageAudit:
    return StageAudit(AuditStatus.BLOCKED, type(exc).__name__, str(exc))


def _not_run(message: str) -> StageAudit:
    return StageAudit(AuditStatus.NOT_RUN, None, message)


def _audit_parsed(
    file: IfcFile,
    *,
    source: str,
    source_sha256: str,
    size_bytes: int,
) -> IfcAuditReport:
    type_counts = tuple(sorted(Counter(inst.type_name for inst in file.instances.values()).items()))
    model_issues: list[AuditIssue] = []
    try:
        units = _length_units(file)
    except GatError as exc:
        units = ()
        model_issues.append(AuditIssue("UNSUPPORTED_LENGTH_UNIT", "ERROR", str(exc)))
    if units and any(not unit.accepted_by_current_adapter for unit in units):
        scalable = all(
            unit.scale_to_metres is not None and math.isfinite(unit.scale_to_metres)
            for unit in units
        )
        model_issues.append(
            AuditIssue(
                "LENGTH_UNIT_NORMALIZATION_REQUIRED" if scalable else "UNSUPPORTED_LENGTH_UNIT",
                "ERROR",
                "declared length units require normalization before authoritative lowering"
                if scalable
                else "declared length units cannot be normalized by the current adapter",
            )
        )
    elif any(unit.kind == "ASSUMED_SI" for unit in units):
        model_issues.append(
            AuditIssue(
                "ASSUMED_LENGTH_UNIT",
                "WARNING",
                "no explicit length unit was found; the current adapter assumes metres",
            )
        )

    products: list[tuple[RawInstance, str]] = []
    for type_name, canonical in PRODUCT_CLASSES.items():
        products.extend((inst, canonical) for inst in file.by_type(type_name))
    annotated_candidates: list[tuple[RawInstance, str, str]] = []
    for type_name, (canonical, marker) in ANNOTATED_PRODUCT_CLASSES.items():
        annotated_candidates.extend(
            (inst, canonical, marker) for inst in file.by_type(type_name)
        )
    product_ids = {
        inst.step_id for inst, _ in products
    } | {
        inst.step_id for inst, _, _ in annotated_candidates
    }
    property_map: dict[int, list[RawInstance]] = {}
    property_error: Exception | None = None
    try:
        property_map = properties_of(file, product_ids)
    except GatError as exc:
        property_error = exc
        model_issues.append(AuditIssue("PROPERTY_GRAPH_UNREADABLE", "ERROR", str(exc)))

    if property_error is None:
        for inst, canonical, marker in annotated_candidates:
            definitions = property_map.get(inst.step_id, [])
            if any(
                definition.type_name == "IFCPROPERTYSET"
                and attr(definition, "Name") == marker
                for definition in definitions
            ):
                products.append((inst, canonical))
    products.sort(key=lambda item: item[0].step_id)

    entities = tuple(
        _audit_entity(
            file,
            inst,
            canonical,
            property_map.get(inst.step_id),
            property_error,
        )
        for inst, canonical in products
    )
    storey_count = sum(1 for entity in entities if entity.canonical_class == "IfcBuildingStorey")
    if storey_count != 1:
        model_issues.append(
            AuditIssue(
                "UNSUPPORTED_STOREY_COUNT",
                "ERROR",
                f"current adapter requires exactly one storey; found {storey_count}",
            )
        )
    if not entities:
        model_issues.append(
            AuditIssue("NO_SUPPORTED_PRODUCTS", "ERROR", "model contains no supported IFC products")
        )

    lowering = _not_run("parse did not produce a module")
    compilation = _not_run("lowering did not produce a module")
    verification = _not_run("compilation did not produce a world")
    world_digest: str | None = None
    try:
        module = lower_ifc(file, source=source)
        lowering = StageAudit(
            AuditStatus.PASS,
            details=(("entity_count", len(module.entities)), ("relationship_count", len(module.rels))),
        )
    except Exception as exc:  # diagnostics must preserve unexpected adapter failures too
        lowering = _blocked_stage(exc)
    else:
        try:
            world = World.compile(module)
            compilation = StageAudit(
                AuditStatus.PASS,
                details=(("raw_variables", world.binding.n_raw), ("full_variables", world.binding.n_full)),
            )
        except Exception as exc:  # see lowering boundary above
            compilation = _blocked_stage(exc)
        else:
            try:
                report = run_invariants(world)
                passed, warnings, failures = report.counts()
                status = AuditStatus.BLOCKED if failures else (AuditStatus.WARN if warnings else AuditStatus.PASS)
                verification = StageAudit(
                    status,
                    None if not failures else "VerificationFailure",
                    None if not failures else report.render(),
                    (("passed", passed), ("warnings", warnings), ("failures", failures)),
                )
                world_digest = world.digest()
            except Exception as exc:  # preserve verifier boundary failures in the report
                verification = _blocked_stage(exc)

    return IfcAuditReport(
        source,
        source_sha256,
        size_bytes,
        StageAudit(AuditStatus.PASS),
        file.schema or None,
        type_counts,
        units,
        entities,
        tuple(model_issues),
        lowering,
        compilation,
        verification,
        world_digest,
    )


def audit_ifc_text(text: str, *, source: str = "<memory>") -> IfcAuditReport:
    raw = text.encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    try:
        file = parse_ifc(text)
    except Exception as exc:
        blocked = _blocked_stage(exc)
        return IfcAuditReport(
            source,
            digest,
            len(raw),
            blocked,
            None,
            (),
            (),
            (),
            (AuditIssue("PARSE_FAILED", "ERROR", str(exc)),),
            _not_run("parsing failed"),
            _not_run("parsing failed"),
            _not_run("parsing failed"),
        )
    return _audit_parsed(
        file,
        source=source,
        source_sha256=digest,
        size_bytes=len(raw),
    )


def audit_ifc_file(path: str | Path) -> IfcAuditReport:
    source = str(path)
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        text = raw.decode("utf-8")
        file = parse_ifc(text)
    except Exception as exc:
        blocked = _blocked_stage(exc)
        return IfcAuditReport(
            source,
            digest,
            len(raw),
            blocked,
            None,
            (),
            (),
            (),
            (AuditIssue("PARSE_FAILED", "ERROR", str(exc)),),
            _not_run("parsing failed"),
            _not_run("parsing failed"),
            _not_run("parsing failed"),
        )
    return _audit_parsed(
        file,
        source=source,
        source_sha256=digest,
        size_bytes=len(raw),
    )


__all__ = [
    "AUDIT_FORMAT",
    "AuditIssue",
    "AuditStatus",
    "EntityAudit",
    "EntityStatus",
    "IfcAuditReport",
    "LengthUnitAudit",
    "StageAudit",
    "audit_ifc_file",
    "audit_ifc_text",
]
