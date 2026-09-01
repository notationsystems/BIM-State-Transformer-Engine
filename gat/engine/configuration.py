"""Configuration identity: a working quotient of representation space.

Two BIM files can encode the *same architecture* while differing in every
representational artifact: STEP instance numbering, record order, GlobalId
strings, entity names, coordinate origin, and global orientation.  The
moduli-space stance distinguishes representation difference from
architectural difference; this module implements the working quotient

    configuration = state  /  (relabeling x rigid motion x re-encoding)

as a canonical digest:

* **Re-encoding** is quotiented by construction — the IR sorts everything
  and the digest never sees STEP ids or file order.
* **Rigid motion** (translation + rotation about gravity) is quotiented by
  using only rigid invariants of the placement geometry: center-to-center
  distances, height offsets, and relative bearings expressed in each
  element's own frame.
* **Relabeling** (GlobalIds, names) is quotiented by Weisfeiler-Leman-style
  color refinement: entities are canonically colored by their intrinsic
  content (class, quantity means/sigmas), then iteratively refined by
  their typed relationships and pairwise geometric signatures; the digest
  hashes the color multiset, never the labels.

Continuous values are rounded to ``QUANT`` (1e-6: one micrometre / one
micro-unit) — configuration identity is defined at representation
tolerance, which the docstring states rather than hides.  Two worlds with
``configuration_digest`` equal are the same architectural configuration in
this quotient; the demo proves the invariance by renumbering, relabeling,
translating, and rotating the model and hashing again.
"""

from __future__ import annotations

import hashlib
import json
import math

from gat.engine.executor import World
from gat.ids import EntityId
from gat.ir.core import Role

QUANT = 1e-6
WL_ROUNDS = 3


def _q(value: float) -> float:
    """Quantize a continuous value to the configuration tolerance."""
    return round(value / QUANT) * QUANT


def _entity_intrinsic(world: World, eid: EntityId) -> str:
    """Label-free intrinsic content of one entity."""
    entity = world.module.entities[eid]
    slots = []
    for qname in sorted(entity.slots):
        slot = entity.slots[qname]
        mean = _q(world.full.mean(slot.var))
        sigma = _q(world.full.std(slot.var))
        slots.append((qname, slot.role.value, slot.unit.value, mean, sigma))
    return json.dumps([eid.ifc_class, slots], sort_keys=True)


def _pair_signature(world: World, a: EntityId, b: EntityId) -> tuple | None:
    """Rigid-motion-invariant geometric signature of an ordered pair."""
    pa = world.module.entities[a].placement
    pb = world.module.entities[b].placement
    if pa is None or pb is None:
        return None
    dx, dy, dz = pb.x - pa.x, pb.y - pa.y, pb.z - pa.z
    dist = math.hypot(dx, dy)
    bearing = math.atan2(dy, dx) - pa.angle       # in a's frame: rotation-invariant
    rel_angle = pb.angle - pa.angle
    return (
        _q(dist),
        _q(dz),
        _q(math.cos(bearing) if dist > QUANT else 1.0),
        _q(math.sin(bearing) if dist > QUANT else 0.0),
        _q(math.cos(rel_angle)),
        _q(math.sin(rel_angle)),
    )


def _refine(world: World, colors: dict[EntityId, str]) -> dict[EntityId, str]:
    eids = sorted(world.module.entities)
    new_colors: dict[EntityId, str] = {}
    for eid in eids:
        rel_sig = []
        for rel in world.module.rels:
            if rel.source == eid:
                rel_sig.append(("out", rel.kind.value, colors[rel.target]))
            if rel.target == eid:
                rel_sig.append(("in", rel.kind.value, colors[rel.source]))
        rel_sig.sort()
        geo_sig = []
        for other in eids:
            if other == eid:
                continue
            sig = _pair_signature(world, eid, other)
            if sig is not None:
                geo_sig.append((sig, colors[other]))
        geo_sig.sort(key=lambda item: json.dumps(item, sort_keys=True))
        payload = json.dumps([colors[eid], rel_sig, geo_sig], sort_keys=True)
        new_colors[eid] = hashlib.sha256(payload.encode()).hexdigest()
    return new_colors


def configuration_digest(world: World) -> str:
    """The canonical configuration hash — invariant under re-encoding,
    relabeling, and global rigid motion."""
    colors = {
        eid: hashlib.sha256(_entity_intrinsic(world, eid).encode()).hexdigest()
        for eid in world.module.entities
    }
    for _ in range(WL_ROUNDS):
        colors = _refine(world, colors)

    color_multiset = sorted(colors.values())
    edge_multiset = sorted(
        (rel.kind.value, colors[rel.source], colors[rel.target])
        for rel in world.module.rels
    )
    payload = json.dumps([color_multiset, edge_multiset], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def same_configuration(a: World, b: World) -> bool:
    """Architectural equality in the quotient (at QUANT tolerance)."""
    return configuration_digest(a) == configuration_digest(b)
