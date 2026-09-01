"""JSON projection of the canonical architectural state.

A machine-readable downstream view of the world — entities, placements,
attributes, relationships, means, sigmas, and the strongest pairwise
correlations — for
downstream toolchains that consume BIM state without speaking SPF (e.g.
render-prompt extractors, analysis pipelines, dashboards).

This format is intentionally not a restart contract: it omits the closed IR,
the indexed full joint covariance, and constraints.  Use
``gat.state_snapshot`` when another runtime must continue computation.
"""

from __future__ import annotations

import json

from gat.engine.executor import World
from gat.ir.core import Role

#: Correlations below this magnitude are omitted from the export.
CORR_THRESHOLD = 0.2
#: Hard cap on exported correlation pairs (strongest first).
CORR_LIMIT = 200


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
            "attrs": {k: entity.attrs[k] for k in sorted(entity.attrs)},
            "quantities": quantities,
        }
        if entity.placement is not None:
            p = entity.placement
            record["placement"] = {"x": p.x, "y": p.y, "z": p.z, "angle": p.angle}
        entities.append(record)

    # Strongest pairwise couplings across the full joint (deterministic
    # order: descending |corr|, then variable names).
    correlations = []
    vars_all = world.full.index.vars
    for i in range(len(vars_all)):
        for j in range(i + 1, len(vars_all)):
            corr = world.full.corr(vars_all[i], vars_all[j])
            if abs(corr) >= CORR_THRESHOLD:
                correlations.append(
                    {"a": str(vars_all[i]), "b": str(vars_all[j]), "corr": corr}
                )
    correlations.sort(key=lambda r: (-abs(r["corr"]), r["a"], r["b"]))
    correlations = correlations[:CORR_LIMIT]

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
        "correlations": correlations,
    }


def export_json(world: World, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(world_to_dict(world), fh, indent=1, sort_keys=True)
        fh.write("\n")
