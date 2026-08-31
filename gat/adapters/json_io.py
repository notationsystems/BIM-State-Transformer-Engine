"""JSON export of the canonical architectural state.

A machine-readable snapshot of the world — entities, placements,
relationships, means, sigmas, and pairwise correlations of interest — for
downstream toolchains that consume BIM state without speaking SPF (e.g.
render-prompt extractors, analysis pipelines, dashboards).
"""

from __future__ import annotations

import json

from gat.engine.executor import World
from gat.ir.core import Role


def world_to_dict(world: World) -> dict:
    entities = []
    for eid in world.module.entities:
        entity = world.module.entities[eid]
        quantities = {}
        for qname in sorted(entity.slots):
            slot = entity.slots[qname]
            quantities[qname] = {
                "role": slot.role.value,
                "unit": slot.unit.value,
                "mean": world.full.mean(slot.var),
                "sigma": world.full.std(slot.var),
            }
        record = {
            "ifc_class": eid.ifc_class,
            "global_id": eid.global_id,
            "name": entity.name,
            "quantities": quantities,
        }
        if entity.placement is not None:
            p = entity.placement
            record["placement"] = {"x": p.x, "y": p.y, "z": p.z, "angle": p.angle}
        entities.append(record)

    rels = [
        {
            "kind": rel.kind.value,
            "source": str(rel.source),
            "target": str(rel.target),
        }
        for rel in world.module.rels
    ]

    return {
        "format": "gat-state v0",
        "meta": dict(world.module.meta),
        "n_raw": world.binding.n_raw,
        "n_derived": world.binding.n_full - world.binding.n_raw,
        "digest": world.digest(),
        "entities": entities,
        "relationships": rels,
    }


def export_json(world: World, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(world_to_dict(world), fh, indent=1, sort_keys=True)
        fh.write("\n")
