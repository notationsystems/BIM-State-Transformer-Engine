"""IFC export: emit the transformed state back into SPF text.

The writer re-serializes the parsed instance graph canonically:

* Every ``IfcQuantity*`` record whose quantity backs a state variable gets
  its value replaced by the current (posterior) mean.
* One ``GAT_Posterior`` property set per entity is appended, carrying the
  posterior sigma of each raw quantity — uncertainty round-trips *through*
  IFC (fresh step ids allocated deterministically from ``max_id + 1``).
* Everything else is re-emitted verbatim from the parsed value model.

Float serialization uses the shortest round-tripping representation in
STEP form, so exporting the same state twice yields byte-identical files.
"""

from __future__ import annotations

from gat.adapters.ifc.parser import (
    EnumVal,
    IfcFile,
    OMITTED,
    RawInstance,
    Ref,
    Typed,
)
from gat.adapters.ifc.units import length_unit_context
from gat.engine.executor import World
from gat.ir.core import Role, Unit


def format_real(value: float) -> str:
    """Shortest round-tripping STEP real literal."""
    text = repr(float(value))
    if "e" in text or "E" in text:
        mantissa, _, exponent = text.partition("e")
        if "." not in mantissa:
            mantissa += "."
        return f"{mantissa}E{int(exponent)}"
    if "." not in text:
        text += "."
    return text


def _serialize_value(value) -> str:
    if value is None:
        return "$"
    if value is OMITTED:
        return "*"
    if isinstance(value, bool):
        raise TypeError("boolean is not a STEP value in this model")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format_real(value)
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    if isinstance(value, EnumVal):
        return f".{value.name}."
    if isinstance(value, Ref):
        return f"#{value.step_id}"
    if isinstance(value, Typed):
        return f"{value.name}({','.join(_serialize_value(v) for v in value.args)})"
    if isinstance(value, tuple):
        return "(" + ",".join(_serialize_value(v) for v in value) + ")"
    raise TypeError(f"cannot serialize {value!r}")


def _serialize_instance(inst: RawInstance) -> str:
    args = ",".join(_serialize_value(v) for v in inst.args)
    return f"#{inst.step_id}={inst.type_name}({args});"


def export_ifc(file: IfcFile, world: World, path: str) -> tuple[int, int]:
    """Write the world state into a new SPF file at ``path``.

    Returns ``(n_patched, n_appended)`` — quantity records rewritten and
    new instances appended.
    """
    instances = dict(file.instances)
    length_units = length_unit_context(file)

    # 0. Strip any GAT_Posterior psets from a previous export, so repeated
    #    export/reload cycles replace rather than accumulate them.
    stale: set[int] = set()
    for inst in file.by_type("IFCPROPERTYSET"):
        if len(inst.args) > 2 and inst.args[2] == "GAT_Posterior":
            stale.add(inst.step_id)
            props = inst.args[4] if len(inst.args) > 4 else ()
            if isinstance(props, tuple):
                stale.update(p.step_id for p in props if isinstance(p, Ref))
    if stale:
        for inst in file.by_type("IFCRELDEFINESBYPROPERTIES"):
            definition = inst.args[5] if len(inst.args) > 5 else None
            if isinstance(definition, Ref) and definition.step_id in stale:
                stale.add(inst.step_id)
        for step_id in stale:
            instances.pop(step_id, None)

    # 1. Patch source-backed values with current means: IfcQuantity* records
    #    carry the value at position 3; pset properties (e.g. UnitCost in
    #    GAT_Material) carry a Typed value at position 2.
    quantity_to_slot = {}
    for entity in world.module.entities.values():
        for slot in entity.slots.values():
            if slot.source_ref is not None:
                quantity_to_slot[slot.source_ref] = slot
    n_patched = 0
    for step_id, slot in sorted(quantity_to_slot.items()):
        inst = instances.get(step_id)
        if inst is None:
            continue
        args = list(inst.args)
        mean = world.full.mean(slot.var)
        source_mean = (
            length_units.from_metres(mean) if slot.unit is Unit.M else mean
        )
        if inst.type_name == "IFCPROPERTYSINGLEVALUE":
            args[2] = Typed("IFCREAL", (source_mean,))
        else:
            args[3] = source_mean
        instances[step_id] = RawInstance(step_id, inst.type_name, tuple(args))
        n_patched += 1

    # 2. Append GAT_Posterior psets carrying raw-parameter sigmas.
    next_id = max(file.max_step_id(), max(instances, default=0)) + 1
    appended: list[RawInstance] = []
    owner_history = None
    for inst in file.by_type("IFCOWNERHISTORY"):
        owner_history = Ref(inst.step_id)
        break

    for eid in world.module.entities:
        entity = world.module.entities[eid]
        raw_slots = [
            entity.slots[q]
            for q in sorted(entity.slots)
            if entity.slots[q].role is Role.RAW
        ]
        if not raw_slots or entity.source_ref is None:
            continue
        prop_refs = []
        for slot in raw_slots:
            # Floor at the binding's minimum prior sigma so a file carrying
            # the posterior of an exact observation still re-binds.
            sigma = max(world.belief.std(slot.var), 1e-6)
            source_sigma = (
                length_units.from_metres(sigma) if slot.unit is Unit.M else sigma
            )
            prop = RawInstance(
                next_id,
                "IFCPROPERTYSINGLEVALUE",
                (
                    f"{slot.var.quantity}Sigma",
                    "posterior standard deviation",
                    Typed("IFCREAL", (source_sigma,)),
                    None,
                ),
            )
            appended.append(prop)
            prop_refs.append(Ref(next_id))
            next_id += 1
        pset = RawInstance(
            next_id,
            "IFCPROPERTYSET",
            (
                f"GATPOST{next_id:015d}",  # deterministic 22-char pseudo-GlobalId
                owner_history,
                "GAT_Posterior",
                None,
                tuple(prop_refs),
            ),
        )
        appended.append(pset)
        pset_id = next_id
        next_id += 1
        rel = RawInstance(
            next_id,
            "IFCRELDEFINESBYPROPERTIES",
            (
                f"GATRELP{next_id:015d}",
                owner_history,
                None,
                None,
                (Ref(entity.source_ref),),
                Ref(pset_id),
            ),
        )
        appended.append(rel)
        next_id += 1

    for inst in appended:
        instances[inst.step_id] = inst

    # 3. Canonical serialization.  Every parsed header entry re-emits in
    #    its original order — nothing the parser accepted is dropped.
    lines = ["ISO-10303-21;", "HEADER;"]
    for key, header_args in file.header.items():
        args = ",".join(_serialize_value(v) for v in header_args)
        lines.append(f"{key}({args});")
    lines.append("ENDSEC;")
    lines.append("DATA;")
    for step_id in sorted(instances):
        lines.append(_serialize_instance(instances[step_id]))
    lines.append("ENDSEC;")
    lines.append("END-ISO-10303-21;")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return n_patched, len(appended)
