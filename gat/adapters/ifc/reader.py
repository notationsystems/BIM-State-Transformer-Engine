"""Typed accessors over parsed IFC instances.

Bridges the schema-agnostic parser and the lowering pass: attribute lookup
by name, property-set and quantity extraction, and placement-chain
resolution.  Only this module and the lowering pass know IFC argument
positions.
"""

from __future__ import annotations

import math

from gat.adapters.ifc.parser import IfcFile, RawInstance, Ref, Typed
from gat.adapters.ifc.schema import SUPPORTED_ENTITIES
from gat.errors import LoweringError
from gat.ir.core import Placement


def attr(inst: RawInstance, name: str):
    """Attribute of a supported instance by schema position."""
    layout = SUPPORTED_ENTITIES.get(inst.type_name)
    if layout is None or name not in layout:
        raise LoweringError(f"no attribute map for {inst.type_name}.{name}")
    pos = layout[name]
    if pos >= len(inst.args):
        raise LoweringError(
            f"#{inst.step_id} {inst.type_name} has {len(inst.args)} args; "
            f"expected {name} at position {pos}"
        )
    return inst.args[pos]


def global_id(inst: RawInstance) -> str:
    gid = attr(inst, "GlobalId")
    if not isinstance(gid, str) or not gid:
        raise LoweringError(f"#{inst.step_id} {inst.type_name} has no GlobalId")
    return gid


def name_of(inst: RawInstance) -> str:
    layout = SUPPORTED_ENTITIES.get(inst.type_name, {})
    if "Name" not in layout:
        return ""
    value = inst.args[layout["Name"]] if layout["Name"] < len(inst.args) else None
    return value if isinstance(value, str) else ""


def numeric(value) -> float:
    """Unwrap a numeric argument that may be a bare number or a Typed value."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, Typed) and len(value.args) == 1:
        return numeric(value.args[0])
    raise LoweringError(f"expected a numeric value, got {value!r}")


def refs(value) -> tuple[Ref, ...]:
    if isinstance(value, tuple):
        out = []
        for item in value:
            if not isinstance(item, Ref):
                raise LoweringError(f"expected reference list, found {item!r}")
            out.append(item)
        return tuple(out)
    raise LoweringError(f"expected an aggregate of references, got {value!r}")


# -- property sets and quantities -----------------------------------------


def properties_of(file: IfcFile, target_step_ids: set[int]) -> dict[int, list[RawInstance]]:
    """Map product step-id -> property definitions attached via
    IfcRelDefinesByProperties (IfcPropertySet or IfcElementQuantity)."""
    out: dict[int, list[RawInstance]] = {sid: [] for sid in target_step_ids}
    for rel in file.by_type("IFCRELDEFINESBYPROPERTIES"):
        definition = attr(rel, "RelatingPropertyDefinition")
        if not isinstance(definition, Ref):
            continue
        def_inst = file.deref(definition)
        for obj in refs(attr(rel, "RelatedObjects")):
            if obj.step_id in out:
                out[obj.step_id].append(def_inst)
    return out


def quantities_of(file: IfcFile, defs: list[RawInstance]) -> dict[str, tuple[float, int]]:
    """Extract ``{quantity name: (value, quantity step id)}`` from the
    IfcElementQuantity definitions in ``defs``."""
    out: dict[str, tuple[float, int]] = {}
    for definition in defs:
        if definition.type_name != "IFCELEMENTQUANTITY":
            continue
        for qref in refs(attr(definition, "Quantities")):
            q = file.deref(qref)
            if q.type_name not in ("IFCQUANTITYLENGTH", "IFCQUANTITYAREA", "IFCQUANTITYVOLUME"):
                continue
            qname = attr(q, "Name")
            if not isinstance(qname, str):
                raise LoweringError(f"#{q.step_id} quantity has no name")
            out[qname] = (numeric(attr(q, "Value")), q.step_id)
    return out


def pset_value_refs(
    file: IfcFile, defs: list[RawInstance], pset_name: str
) -> dict[str, tuple[float, int]]:
    """``{property name: (value, property step id)}`` of the named pset."""
    out: dict[str, tuple[float, int]] = {}
    for definition in defs:
        if definition.type_name != "IFCPROPERTYSET":
            continue
        if attr(definition, "Name") != pset_name:
            continue
        for pref in refs(attr(definition, "HasProperties")):
            prop = file.deref(pref)
            if prop.type_name != "IFCPROPERTYSINGLEVALUE":
                continue
            pname = attr(prop, "Name")
            value = attr(prop, "NominalValue")
            if isinstance(pname, str) and value is not None:
                out[pname] = (numeric(value), prop.step_id)
    return out


def pset_values(file: IfcFile, defs: list[RawInstance], pset_name: str) -> dict[str, float]:
    """Numeric single values of the named property set, if present."""
    return {k: v for k, (v, _) in pset_value_refs(file, defs, pset_name).items()}


# -- placements ------------------------------------------------------------


def resolve_placement(
    file: IfcFile,
    placement_ref,
    *,
    length_scale_to_metres: float = 1.0,
) -> Placement:
    """Resolve an IfcLocalPlacement chain into an absolute Placement.

    v0 supports translation plus rotation about +Z (RefDirection in the XY
    plane).  Axis directions out of plane raise a LoweringError — the
    shipped models are storey-planar.
    """
    x = y = z = 0.0
    angle = 0.0
    if placement_ref is None:
        return Placement()
    if not isinstance(placement_ref, Ref):
        raise LoweringError(f"expected placement reference, got {placement_ref!r}")

    chain: list[RawInstance] = []
    seen: set[int] = set()
    cursor = placement_ref
    while cursor is not None:
        if cursor.step_id in seen:
            raise LoweringError(
                f"#{cursor.step_id}: cyclic IfcLocalPlacement chain"
            )
        seen.add(cursor.step_id)
        inst = file.deref(cursor)
        if inst.type_name != "IFCLOCALPLACEMENT":
            raise LoweringError(f"#{inst.step_id}: unsupported placement {inst.type_name}")
        chain.append(inst)
        parent = attr(inst, "PlacementRelTo")
        cursor = parent if isinstance(parent, Ref) else None

    # Compose from the root down.
    for inst in reversed(chain):
        rel = attr(inst, "RelativePlacement")
        if not isinstance(rel, Ref):
            continue
        axis2 = file.deref(rel)
        if axis2.type_name != "IFCAXIS2PLACEMENT3D":
            raise LoweringError(
                f"#{axis2.step_id}: unsupported placement type {axis2.type_name}"
            )
        loc = attr(axis2, "Location")
        lx = ly = lz = 0.0
        if isinstance(loc, Ref):
            point = file.deref(loc)
            coords = attr(point, "Coordinates")
            values = [numeric(v) for v in coords]
            while len(values) < 3:
                values.append(0.0)
            lx, ly, lz = (
                value * length_scale_to_metres for value in values[:3]
            )
        local_angle = 0.0
        ref_dir = attr(axis2, "RefDirection")
        if isinstance(ref_dir, Ref):
            direction = file.deref(ref_dir)
            ratios = [numeric(v) for v in attr(direction, "DirectionRatios")]
            while len(ratios) < 3:
                ratios.append(0.0)
            if abs(ratios[2]) > 1e-12:
                raise LoweringError(
                    f"#{direction.step_id}: out-of-plane RefDirection unsupported in v0"
                )
            local_angle = math.atan2(ratios[1], ratios[0])
        axis = attr(axis2, "Axis")
        if isinstance(axis, Ref):
            direction = file.deref(axis)
            ratios = [numeric(v) for v in attr(direction, "DirectionRatios")]
            while len(ratios) < 3:
                ratios.append(0.0)
            if not (abs(ratios[0]) < 1e-12 and abs(ratios[1]) < 1e-12 and ratios[2] > 0):
                raise LoweringError(
                    f"#{direction.step_id}: non-vertical Axis unsupported in v0"
                )
        # Compose: rotate the local offset by the accumulated angle.
        c, s = math.cos(angle), math.sin(angle)
        x += c * lx - s * ly
        y += s * lx + c * ly
        z += lz
        angle += local_angle
    return Placement(x, y, z, angle)


def unit_is_metres(file: IfcFile) -> bool:
    """Compatibility predicate for callers that require metre-native input."""
    from gat.adapters.ifc.units import length_unit_context

    try:
        return length_unit_context(file).scale_to_metres == 1.0
    except LoweringError:
        return False
