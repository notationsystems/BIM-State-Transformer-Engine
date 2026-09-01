"""Explicit IFC project-length normalization into GAT's metre state."""

from __future__ import annotations

from dataclasses import dataclass
import math

from gat.adapters.ifc.parser import EnumVal, IfcFile, Ref
from gat.errors import LoweringError


SI_PREFIX_SCALE: dict[str, float] = {
    "EXA": 1e18,
    "PETA": 1e15,
    "TERA": 1e12,
    "GIGA": 1e9,
    "MEGA": 1e6,
    "KILO": 1e3,
    "HECTO": 1e2,
    "DECA": 1e1,
    "DECI": 1e-1,
    "CENTI": 1e-2,
    "MILLI": 1e-3,
    "MICRO": 1e-6,
    "NANO": 1e-9,
    "PICO": 1e-12,
    "FEMTO": 1e-15,
    "ATTO": 1e-18,
}


@dataclass(frozen=True)
class LengthUnitContext:
    """One unambiguous source-length scale and its provenance."""

    scale_to_metres: float
    kind: str
    name: str
    prefix: str | None
    source_step_id: int | None
    assumed: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.scale_to_metres) or self.scale_to_metres <= 0.0:
            raise ValueError("length-unit scale must be finite and positive")

    @property
    def normalization_required(self) -> bool:
        return self.scale_to_metres != 1.0

    @property
    def label(self) -> str:
        if self.assumed:
            return "ASSUMED METRE"
        return f"{self.prefix + ' ' if self.prefix else ''}{self.name}"

    def to_metres(self, value: float) -> float:
        result = float(value) * self.scale_to_metres
        if not math.isfinite(result):
            raise LoweringError("length value is non-finite after metre normalization")
        return result

    def from_metres(self, value: float) -> float:
        result = float(value) / self.scale_to_metres
        if not math.isfinite(result):
            raise ValueError("length value is non-finite after source-unit conversion")
        return result


def length_unit_context(file: IfcFile) -> LengthUnitContext:
    """Resolve the file's supported project length unit.

    v1 accepts SI metres with any recognized IFC SI prefix. Conversion-based
    length units stay outside the assurance boundary until their complete
    ``IfcMeasureWithUnit`` chain is implemented and tested.
    """

    project_units = assigned_unit_ids(file)

    for unit in file.by_type("IFCCONVERSIONBASEDUNIT"):
        if project_units is not None and unit.step_id not in project_units:
            continue
        unit_type = unit.args[1] if len(unit.args) > 1 else None
        if isinstance(unit_type, EnumVal) and unit_type.name == "LENGTHUNIT":
            raise LoweringError(
                f"#{unit.step_id}: conversion-based length units are unsupported"
            )

    contexts: list[LengthUnitContext] = []
    for unit in file.by_type("IFCSIUNIT"):
        if project_units is not None and unit.step_id not in project_units:
            continue
        unit_type = unit.args[1] if len(unit.args) > 1 else None
        if not isinstance(unit_type, EnumVal) or unit_type.name != "LENGTHUNIT":
            continue
        prefix_value = unit.args[2] if len(unit.args) > 2 else None
        name_value = unit.args[3] if len(unit.args) > 3 else None
        if not isinstance(name_value, EnumVal) or name_value.name != "METRE":
            raise LoweringError(f"#{unit.step_id}: SI length unit must be METRE")
        prefix = prefix_value.name if isinstance(prefix_value, EnumVal) else None
        if prefix_value is not None and prefix is None:
            raise LoweringError(f"#{unit.step_id}: malformed SI length prefix")
        if prefix is None:
            scale = 1.0
        else:
            scale = SI_PREFIX_SCALE.get(prefix)
            if scale is None:
                raise LoweringError(
                    f"#{unit.step_id}: unsupported SI length prefix {prefix!r}"
                )
        contexts.append(
            LengthUnitContext(scale, "SI", "METRE", prefix, unit.step_id)
        )

    if not contexts and project_units is not None:
        raise LoweringError("project unit assignment has no supported length unit")
    if not contexts:
        return LengthUnitContext(1.0, "ASSUMED_SI", "METRE", None, None, True)
    first = contexts[0]
    if any(context.scale_to_metres != first.scale_to_metres for context in contexts[1:]):
        steps = ", ".join(f"#{context.source_step_id}" for context in contexts)
        raise LoweringError(f"ambiguous project length units at {steps}")
    return first


def assigned_unit_ids(file: IfcFile) -> set[int] | None:
    """Return units referenced by IfcProject.UnitsInContext when declared."""
    assignments: list[Ref] = []
    for project in file.by_type("IFCPROJECT"):
        value = project.args[8] if len(project.args) > 8 else None
        if isinstance(value, Ref):
            assignments.append(value)
        elif value is not None:
            raise LoweringError(
                f"#{project.step_id}: IfcProject.UnitsInContext must be a reference"
            )
    if not assignments:
        return None
    unit_ids: set[int] = set()
    for reference in assignments:
        assignment = file.deref(reference)
        if assignment.type_name != "IFCUNITASSIGNMENT":
            raise LoweringError(
                f"#{assignment.step_id}: project unit context is not IfcUnitAssignment"
            )
        values = assignment.args[0] if assignment.args else None
        if not isinstance(values, tuple):
            raise LoweringError(f"#{assignment.step_id}: unit assignment must contain a list")
        for value in values:
            if not isinstance(value, Ref):
                raise LoweringError(
                    f"#{assignment.step_id}: unit assignment contains a non-reference"
                )
            unit_ids.add(value.step_id)
    return unit_ids


__all__ = [
    "LengthUnitContext",
    "SI_PREFIX_SCALE",
    "assigned_unit_ids",
    "length_unit_context",
]
