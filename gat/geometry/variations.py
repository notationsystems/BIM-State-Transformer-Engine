"""Belief-driven variation export for visualization pipelines.

Each sample of the architectural belief is a physically consistent
plausible as-built — imperfections correlated exactly as the model says.
This module renders ``n`` such realizations as standard 3D Gaussian
Splatting PLY files (loadable in stock splat viewers, Blender add-ons,
and USD/DCC pipelines via conversion), with a deterministic manifest
recording, per variation: the file, the sampled key dimensions, and
whether the realization passes the invariant registry.

Grounded procedural variation, in other words: instead of noise functions,
the spread artists see *is* the engineering uncertainty.
"""

from __future__ import annotations

import json
import os

from gat.engine.executor import World
from gat.engine.sampling import sample_worlds
from gat.engine.verify import Status, run_invariants
from gat.geometry.splat_io import export_splat_ply
from gat.geometry.stateio import derive_scene
from gat.ir.core import Role


def export_variations(
    world: World,
    out_dir: str,
    n: int = 25,
    seed: int = 0,
    spacing: float = 0.75,
    base_name: str = "variation",
) -> dict:
    """Write ``n`` sampled as-built splat PLYs plus ``manifest.json``.

    Returns the manifest dict.  Deterministic: same world + seed + n =>
    identical files and manifest.
    """
    if n < 1:
        raise ValueError("need at least one variation")
    os.makedirs(out_dir, exist_ok=True)

    raw_vars = world.binding.raw_index.vars
    samples = []
    for i, sampled in enumerate(sample_worlds(world, n, seed)):
        scene = derive_scene(sampled, spacing=spacing)
        filename = f"{base_name}_{i:03d}.ply"
        count = export_splat_ply(scene.cloud, os.path.join(out_dir, filename))
        report = run_invariants(sampled)
        failures = [
            f"{r.invariant_id} [{r.subject}]"
            for r in report.results
            if r.status is Status.FAIL
        ]
        samples.append(
            {
                "file": filename,
                "index": i,
                "primitives": count,
                "passed": not failures,
                "failures": failures,
                "raw_values": {
                    str(v): sampled.belief.mean(v) for v in raw_vars
                },
            }
        )

    manifest = {
        "format": "gat-variations v1",
        "seed": seed,
        "n": n,
        "spacing": spacing,
        "source_digest": world.digest(),
        "nominal": {
            "file": f"{base_name}_nominal.ply",
            "raw_values": {str(v): world.belief.mean(v) for v in raw_vars},
        },
        "samples": samples,
    }

    nominal_scene = derive_scene(world, spacing=spacing)
    export_splat_ply(
        nominal_scene.cloud, os.path.join(out_dir, f"{base_name}_nominal.ply")
    )

    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return manifest


def variation_spread(world: World) -> dict[str, float]:
    """Per-raw-parameter sigma — the spread a variation set will exhibit.

    A cheap pre-flight for artists: which dimensions actually vary, and by
    how much, before exporting a full set.
    """
    return {
        str(slot.var): world.belief.std(slot.var)
        for entity in world.module.entities.values()
        for slot in entity.slots.values()
        if slot.role is Role.RAW
    }
