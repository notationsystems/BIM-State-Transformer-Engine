"""Lowering: parsed IFC → Architectural IR.

``lower_ifc`` walks the supported entity subset, builds IR entities with
placements and RAW quantity slots (priors from a deterministic sigma
policy, overridable per-entity via a ``GAT_Uncertainty`` property set),
lowers the five relationship types to typed edges, and synthesizes the
DERIVED quantity layer plus the constraint set:

* ``wall.Height``          := storey.ClearHeight            (shared level)
* ``wall.GrossSideArea``   := Length * Height
* ``wall.NetSideArea``     := GrossSideArea - Σ opening.Area
* ``wall.GrossVolume``     := GrossSideArea * Width
* ``wall.NetVolume``       := NetSideArea * Width
* ``wall.Cost``            := NetVolume * UnitCost          (when priced)
* ``opening.Area``         := Width * Height
* ``door.Area``            := Width * Height
* ``space.FloorArea``      := Length * Width
* ``space.Volume``         := FloorArea * storey.ClearHeight
* storey rollups           := Σ member quantities

The shared ``ClearHeight`` raw variable is the coupling model: one design
change to the storey height cascades through every wall and space, and
room volumes become correlated through their shared parent — covariance
as the carrier of architectural dependency (README §16 Q3).

Geometry enters v0 through base quantities and placements, not through
solid-model parsing; that boundary is an explicit adapter decision
(README §12) — a geometric adapter can replace this one without touching
the engine.
"""

from __future__ import annotations

from gat.adapters.ifc.parser import IfcFile, RawInstance, Ref
from gat.adapters.ifc.reader import (
    attr,
    global_id,
    name_of,
    properties_of,
    pset_value_refs,
    pset_values,
    quantities_of,
    refs,
    resolve_placement,
)
from gat.adapters.ifc.schema import PRODUCT_CLASSES
from gat.adapters.ifc.units import LengthUnitContext, length_unit_context
from gat.errors import LoweringError
from gat.ids import EntityId, VarId
from gat.ir.core import (
    Entity,
    ExprEquals,
    LessEqual,
    Module,
    NonNegative,
    QtySlot,
    Rel,
    RelKind,
    Role,
    Unit,
)
from gat.ir.exprs import Mul, ScaledSum, Sub, VarRef

#: Default prior sigmas by (canonical class, quantity name).
DEFAULT_SIGMAS: dict[tuple[str, str], float] = {
    ("IfcBuildingStorey", "ClearHeight"): 0.01,
    ("IfcWall", "Length"): 0.005,
    ("IfcWall", "Width"): 0.002,
    ("IfcOpeningElement", "Width"): 0.005,
    ("IfcOpeningElement", "Height"): 0.005,
    ("IfcDoor", "Width"): 0.003,
    ("IfcDoor", "Height"): 0.003,
    ("IfcSpace", "Length"): 0.005,
    ("IfcSpace", "Width"): 0.005,
}

#: Relative default sigma for wall unit cost (8 % of the mean).
UNIT_COST_REL_SIGMA = 0.08

REQUIRED_QUANTITIES: dict[str, tuple[str, ...]] = {
    "IfcBuildingStorey": ("ClearHeight",),
    "IfcWall": ("Length", "Width"),
    "IfcOpeningElement": ("Width", "Height"),
    "IfcDoor": ("Width", "Height"),
    "IfcSpace": ("Length", "Width"),
}

QUANTITY_UNITS: dict[str, Unit] = {
    "ClearHeight": Unit.M,
    "Length": Unit.M,
    "Width": Unit.M,
    "Height": Unit.M,
}


def _sigma_for(
    canonical_class: str,
    quantity: str,
    mean: float,
    overrides: dict[str, float],
    length_units: LengthUnitContext,
) -> float:
    override = overrides.get(f"{quantity}Sigma")
    if override is not None:
        value = float(override)
        if QUANTITY_UNITS.get(quantity) is Unit.M:
            return length_units.to_metres(value)
        return value
    if quantity == "UnitCost":
        return max(abs(mean) * UNIT_COST_REL_SIGMA, 1e-6)
    default = DEFAULT_SIGMAS.get((canonical_class, quantity))
    if default is None:
        raise LoweringError(f"no sigma policy for {canonical_class}.{quantity}")
    return default


def lower_ifc(file: IfcFile, source: str = "<memory>") -> Module:
    length_units = length_unit_context(file)

    # -- products ----------------------------------------------------------
    products: dict[int, tuple[EntityId, RawInstance, str]] = {}
    for type_name, canonical in PRODUCT_CLASSES.items():
        for inst in file.by_type(type_name):
            eid = EntityId(canonical, global_id(inst))
            products[inst.step_id] = (eid, inst, canonical)

    step_to_eid = {sid: eid for sid, (eid, _, _) in products.items()}
    prop_map = properties_of(file, set(products))

    entities: dict[EntityId, Entity] = {}
    quantity_refs: dict[VarId, int] = {}
    used_source_refs: set[int] = set()
    storeys: list[EntityId] = []
    walls: list[EntityId] = []
    spaces: list[EntityId] = []
    openings: list[EntityId] = []
    doors: list[EntityId] = []
    priced_walls: set[EntityId] = set()

    for sid in sorted(products):
        eid, inst, canonical = products[sid]
        if eid in entities:
            raise LoweringError(f"duplicate GlobalId {eid.global_id}")
        defs = prop_map.get(sid, [])
        quantities = quantities_of(file, defs)
        overrides = pset_values(file, defs, "GAT_Uncertainty")
        # A previously exported file carries posterior sigmas; they take
        # precedence so uncertainty round-trips through IFC.
        overrides.update(pset_values(file, defs, "GAT_Posterior"))
        material = pset_value_refs(file, defs, "GAT_Material")

        slots: dict[str, QtySlot] = {}
        for qname in REQUIRED_QUANTITIES.get(canonical, ()):
            if qname not in quantities:
                raise LoweringError(
                    f"{canonical} {eid.global_id} ({name_of(inst)!r}) lacks "
                    f"required quantity {qname!r}"
                )
            source_value, qref = quantities[qname]
            value = length_units.to_metres(source_value)
            if qref in used_source_refs:
                raise LoweringError(
                    f"quantity #{qref} is shared by multiple products "
                    f"(second: {eid.global_id}); one quantity record per "
                    f"state variable is required in v0"
                )
            used_source_refs.add(qref)
            var = VarId(eid, qname)
            slots[qname] = QtySlot(
                var=var,
                role=Role.RAW,
                unit=QUANTITY_UNITS.get(qname, Unit.M),
                prior_mu=value,
                prior_sigma=_sigma_for(
                    canonical, qname, value, overrides, length_units
                ),
                source_ref=qref,
            )
            quantity_refs[var] = qref

        if canonical == "IfcWall" and "UnitCost" in material:
            mean, cost_ref = material["UnitCost"]
            if "UnitCostSigma" in material and "UnitCostSigma" not in overrides:
                overrides["UnitCostSigma"] = material["UnitCostSigma"][0]
            sigma = _sigma_for(
                canonical, "UnitCost", mean, overrides, length_units
            )
            var = VarId(eid, "UnitCost")
            slots["UnitCost"] = QtySlot(
                var=var,
                role=Role.RAW,
                unit=Unit.CURRENCY_PER_M3,
                prior_mu=mean,
                prior_sigma=float(sigma),
                source_ref=cost_ref,  # the writer patches the pset property
            )
            priced_walls.add(eid)

        placement = None
        layout_placement = attr(inst, "ObjectPlacement") if canonical != "IfcProject" else None
        if isinstance(layout_placement, Ref):
            placement = resolve_placement(
                file,
                layout_placement,
                length_scale_to_metres=length_units.scale_to_metres,
            )

        entities[eid] = Entity(
            id=eid,
            name=name_of(inst),
            attrs={},
            slots=slots,
            placement=placement,
            source_ref=sid,
        )

        {
            "IfcBuildingStorey": storeys,
            "IfcWall": walls,
            "IfcSpace": spaces,
            "IfcOpeningElement": openings,
            "IfcDoor": doors,
        }[canonical].append(eid)

    if len(storeys) != 1:
        raise LoweringError(f"v0 expects exactly one storey, found {len(storeys)}")
    storey = storeys[0]
    clear_height = VarId(storey, "ClearHeight")

    # -- relationships -----------------------------------------------------
    rels: list[Rel] = []

    def _edge(kind: RelKind, src_ref: Ref, dst_ref: Ref, source_ref: int) -> None:
        src = step_to_eid.get(src_ref.step_id)
        dst = step_to_eid.get(dst_ref.step_id)
        if src is not None and dst is not None:
            rels.append(Rel(kind, src, dst, source_ref))

    for rel in file.by_type("IFCRELAGGREGATES"):
        relating = attr(rel, "RelatingObject")
        if isinstance(relating, Ref):
            for related in refs(attr(rel, "RelatedObjects")):
                _edge(RelKind.AGGREGATES, relating, related, rel.step_id)
    for rel in file.by_type("IFCRELCONTAINEDINSPATIALSTRUCTURE"):
        structure = attr(rel, "RelatingStructure")
        if isinstance(structure, Ref):
            for element in refs(attr(rel, "RelatedElements")):
                _edge(RelKind.CONTAINS, structure, element, rel.step_id)
    for rel in file.by_type("IFCRELVOIDSELEMENT"):
        wall_ref = attr(rel, "RelatingBuildingElement")
        opening_ref = attr(rel, "RelatedOpeningElement")
        if isinstance(wall_ref, Ref) and isinstance(opening_ref, Ref):
            _edge(RelKind.VOIDS, opening_ref, wall_ref, rel.step_id)
    for rel in file.by_type("IFCRELFILLSELEMENT"):
        opening_ref = attr(rel, "RelatingOpeningElement")
        filler_ref = attr(rel, "RelatedBuildingElement")
        if isinstance(opening_ref, Ref) and isinstance(filler_ref, Ref):
            _edge(RelKind.FILLS, filler_ref, opening_ref, rel.step_id)
    external_elements: set[EntityId] = set()
    for rel in file.by_type("IFCRELSPACEBOUNDARY"):
        space_ref = attr(rel, "RelatingSpace")
        element_ref = attr(rel, "RelatedBuildingElement")
        if isinstance(space_ref, Ref) and isinstance(element_ref, Ref):
            _edge(RelKind.BOUNDS, element_ref, space_ref, rel.step_id)
            boundary_kind = attr(rel, "InternalOrExternalBoundary")
            eid = step_to_eid.get(element_ref.step_id)
            if (
                eid is not None
                and getattr(boundary_kind, "name", "") == "EXTERNAL"
            ):
                external_elements.add(eid)
    for eid in sorted(external_elements):
        entity = entities[eid]
        entities[eid] = Entity(
            id=entity.id,
            name=entity.name,
            attrs={**entity.attrs, "external": True},
            slots=entity.slots,
            placement=entity.placement,
            source_ref=entity.source_ref,
        )

    voids_of_wall: dict[EntityId, list[EntityId]] = {w: [] for w in walls}
    for rel in rels:
        if rel.kind is RelKind.VOIDS and rel.target in voids_of_wall:
            voids_of_wall[rel.target].append(rel.source)
    for wall in voids_of_wall:
        voids_of_wall[wall] = sorted(voids_of_wall[wall])

    fills: dict[EntityId, EntityId] = {}
    for rel in rels:
        if rel.kind is RelKind.FILLS:
            fills[rel.source] = rel.target  # door -> opening

    # -- derived synthesis -------------------------------------------------
    def add_derived(eid: EntityId, qname: str, unit: Unit, expr) -> VarId:
        entity = entities[eid]
        var = VarId(eid, qname)
        new_slots = dict(entity.slots)
        new_slots[qname] = QtySlot(var=var, role=Role.DERIVED, unit=unit, expr=expr)
        entities[eid] = Entity(
            id=entity.id,
            name=entity.name,
            attrs=entity.attrs,
            slots=new_slots,
            placement=entity.placement,
            source_ref=entity.source_ref,
        )
        return var

    for opening in openings:
        add_derived(
            opening,
            "Area",
            Unit.M2,
            Mul(VarRef(VarId(opening, "Width")), VarRef(VarId(opening, "Height"))),
        )
    for door in doors:
        add_derived(
            door,
            "Area",
            Unit.M2,
            Mul(VarRef(VarId(door, "Width")), VarRef(VarId(door, "Height"))),
        )

    for wall in walls:
        height = add_derived(wall, "Height", Unit.M, VarRef(clear_height))
        gross = add_derived(
            wall,
            "GrossSideArea",
            Unit.M2,
            Mul(VarRef(VarId(wall, "Length")), VarRef(height)),
        )
        net_terms: tuple[tuple[float, VarRef], ...] = ((1.0, VarRef(gross)),) + tuple(
            (-1.0, VarRef(VarId(o, "Area"))) for o in voids_of_wall[wall]
        )
        net = add_derived(wall, "NetSideArea", Unit.M2, ScaledSum(net_terms))
        add_derived(
            wall,
            "GrossVolume",
            Unit.M3,
            Mul(VarRef(gross), VarRef(VarId(wall, "Width"))),
        )
        net_volume = add_derived(
            wall,
            "NetVolume",
            Unit.M3,
            Mul(VarRef(net), VarRef(VarId(wall, "Width"))),
        )
        if wall in priced_walls:
            add_derived(
                wall,
                "Cost",
                Unit.CURRENCY,
                Mul(VarRef(net_volume), VarRef(VarId(wall, "UnitCost"))),
            )

    for space in spaces:
        floor = add_derived(
            space,
            "FloorArea",
            Unit.M2,
            Mul(VarRef(VarId(space, "Length")), VarRef(VarId(space, "Width"))),
        )
        add_derived(space, "Volume", Unit.M3, Mul(VarRef(floor), VarRef(clear_height)))

    # Rollup membership comes from the SAME relationship edges the QTY-01
    # invariant re-sums over (walls CONTAINed in the storey, spaces
    # AGGREGATED into it), so the DAG and the graph can never disagree
    # about who counts.
    contained_walls = sorted(
        rel.target
        for rel in rels
        if rel.kind is RelKind.CONTAINS
        and rel.source == storey
        and rel.target in set(walls)
    )
    aggregated_spaces = sorted(
        rel.target
        for rel in rels
        if rel.kind is RelKind.AGGREGATES
        and rel.source == storey
        and rel.target in set(spaces)
    )
    if contained_walls:
        add_derived(
            storey,
            "TotalWallNetVolume",
            Unit.M3,
            ScaledSum(
                tuple((1.0, VarRef(VarId(w, "NetVolume"))) for w in contained_walls)
            ),
        )
    contained_priced = [w for w in contained_walls if w in priced_walls]
    if contained_priced:
        add_derived(
            storey,
            "TotalWallCost",
            Unit.CURRENCY,
            ScaledSum(
                tuple((1.0, VarRef(VarId(w, "Cost"))) for w in contained_priced)
            ),
        )
    if aggregated_spaces:
        add_derived(
            storey,
            "TotalFloorArea",
            Unit.M2,
            ScaledSum(
                tuple((1.0, VarRef(VarId(s, "FloorArea"))) for s in aggregated_spaces)
            ),
        )

    # -- constraints -------------------------------------------------------
    constraints: list = []
    for eid in sorted(entities):
        for qname in sorted(entities[eid].slots):
            constraints.append(NonNegative(entities[eid].slots[qname].var))

    for wall in walls:
        for opening in voids_of_wall[wall]:
            constraints.append(LessEqual(VarId(opening, "Width"), VarId(wall, "Length")))
            constraints.append(LessEqual(VarId(opening, "Height"), VarId(wall, "Height")))
    for door, opening in sorted(fills.items()):
        constraints.append(LessEqual(VarId(door, "Width"), VarId(opening, "Width")))
        constraints.append(LessEqual(VarId(door, "Height"), VarId(opening, "Height")))

    for wall in walls:
        restatement = Sub(
            VarRef(VarId(wall, "GrossSideArea")),
            ScaledSum(tuple((1.0, VarRef(VarId(o, "Area"))) for o in voids_of_wall[wall])),
        )
        constraints.append(ExprEquals(VarId(wall, "NetSideArea"), restatement))
    for space in spaces:
        restatement = Mul(
            Mul(VarRef(VarId(space, "Length")), VarRef(VarId(space, "Width"))),
            VarRef(clear_height),
        )
        constraints.append(ExprEquals(VarId(space, "Volume"), restatement))

    module = Module(
        entities=entities,
        rels=tuple(rels),
        constraints=tuple(constraints),
        meta={
            "source": source,
            "schema": file.schema,
            "adapter": "gat.adapters.ifc v0",
            "ifc_length_scale_to_metres": repr(length_units.scale_to_metres),
            "ifc_length_unit": length_units.label,
        },
    )
    return module
