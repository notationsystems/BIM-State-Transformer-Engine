"""Synthetic scale probe for dense and incremental GAT propagation.

This is a measurement harness, not a performance promise. It builds independent
storey-local nonlinear dependency chains, changes one raw variable, and compares
complete pushforward against the incremental path while reporting the dense
state footprint and actual rows invalidated.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from gat.engine.executor import World
from gat.engine.transform import ShiftParameter
from gat.engine.verify import run_invariants
from gat.ids import EntityId, VarId
from gat.ir.core import Entity, Module, QtySlot, Role, Unit
from gat.ir.exprs import Mul, VarRef


def synthetic_storey_module(storeys: int) -> Module:
    """Build ``storeys`` independent 1-raw/2-derived nonlinear chains."""
    if isinstance(storeys, bool) or not isinstance(storeys, int) or storeys <= 0:
        raise ValueError("storeys must be a positive integer")
    entities: dict[EntityId, Entity] = {}
    for index in range(storeys):
        entity_id = EntityId("IfcBuildingStorey", f"SYNTHETIC-STOREY-{index:08d}")
        height = VarId(entity_id, "Height")
        area = VarId(entity_id, "AreaProxy")
        volume = VarId(entity_id, "VolumeProxy")
        height_ref = VarRef(height)
        area_ref = VarRef(area)
        slots = {
            "Height": QtySlot(
                height,
                Role.RAW,
                Unit.M,
                prior_mu=3.0 + index * 1.0e-6,
                prior_sigma=0.01,
            ),
            "AreaProxy": QtySlot(
                area,
                Role.DERIVED,
                Unit.M2,
                expr=Mul(height_ref, height_ref),
            ),
            "VolumeProxy": QtySlot(
                volume,
                Role.DERIVED,
                Unit.M3,
                expr=Mul(area_ref, height_ref),
            ),
        }
        entities[entity_id] = Entity(entity_id, f"Synthetic Storey {index}", slots=slots)
    return Module(
        entities,
        (),
        (),
        {"source": "gat-synthetic-incremental-scale-v1"},
    )


def dense_state_bytes(raw_variables: int, full_variables: int) -> int:
    """Resident bytes for belief/full view, Jacobian, and ``J @ Sigma``."""
    return 8 * (
        raw_variables
        + raw_variables**2
        + full_variables
        + full_variables**2
        + 2 * full_variables * raw_variables
    )


def storeys_for_dense_budget(budget_bytes: int) -> int:
    """Largest synthetic storey count whose resident arrays fit ``budget``."""
    if isinstance(budget_bytes, bool) or not isinstance(budget_bytes, int) or budget_bytes <= 0:
        raise ValueError("budget_bytes must be a positive integer")
    low, high = 0, 1
    while dense_state_bytes(high, 3 * high) <= budget_bytes:
        high *= 2
    while low + 1 < high:
        middle = (low + high) // 2
        if dense_state_bytes(middle, 3 * middle) <= budget_bytes:
            low = middle
        else:
            high = middle
    return low


def _best_seconds(operation, repeats: int) -> tuple[float, object]:
    best = float("inf")
    result = None
    for _ in range(repeats):
        start = perf_counter()
        candidate = operation()
        elapsed = perf_counter() - start
        if elapsed < best:
            best = elapsed
            result = candidate
    return best, result


def measure_size(storeys: int, repeats: int = 3) -> dict[str, object]:
    """Measure one synthetic size without imposing a timing assertion."""
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    world = World.compile(synthetic_storey_module(storeys))
    target = world.binding.raw_index.vars[0]
    transformation = ShiftParameter(target, 0.01)
    belief = transformation.apply(world.binding, world.belief)

    full_seconds, full_world_obj = _best_seconds(
        lambda: world.with_belief(belief),
        repeats,
    )
    incremental_seconds, incremental_result = _best_seconds(
        lambda: world.with_belief_incremental(belief),
        repeats,
    )
    assert isinstance(full_world_obj, World)
    incremental_world, stats = incremental_result
    mean_error = float(np.max(np.abs(full_world_obj.full.mu - incremental_world.full.mu)))
    covariance_error = float(
        np.max(np.abs(full_world_obj.full.sigma - incremental_world.full.sigma))
    )
    if mean_error > 1e-12 or covariance_error > 1e-12:
        raise RuntimeError("incremental propagation differs from complete pushforward")
    verification_seconds, verification = _best_seconds(
        lambda: run_invariants(incremental_world),
        repeats,
    )
    if not verification.passed:
        raise RuntimeError("incrementally propagated synthetic world failed invariants")

    return {
        "storeys": storeys,
        "raw_variables": world.binding.n_raw,
        "derived_variables": len(world.binding.deps.derived_vars),
        "full_variables": world.binding.n_full,
        "resident_dense_state_bytes": dense_state_bytes(
            world.binding.n_raw,
            world.binding.n_full,
        ),
        "complete_pushforward_seconds": full_seconds,
        "incremental_pushforward_seconds": incremental_seconds,
        "verification_seconds": verification_seconds,
        "verified_incremental_pipeline_seconds": (
            incremental_seconds + verification_seconds
        ),
        "measured_speedup": (
            full_seconds / incremental_seconds if incremental_seconds else None
        ),
        "bitwise_mean_equal": np.array_equal(
            full_world_obj.full.mu,
            incremental_world.full.mu,
        ),
        "bitwise_covariance_equal": np.array_equal(
            full_world_obj.full.sigma,
            incremental_world.full.sigma,
        ),
        "max_abs_mean_error": mean_error,
        "max_abs_covariance_error": covariance_error,
        "incremental_work": {
            "mode": stats.mode,
            "raw_mean_rows_changed": stats.raw_mean_rows_changed,
            "raw_covariance_rows_changed": stats.raw_covariance_rows_changed,
            "derived_value_rows_recomputed": stats.derived_value_rows_recomputed,
            "derived_jacobian_rows_recomputed": (
                stats.derived_jacobian_rows_recomputed
            ),
            "covariance_left_rows_recomputed": (
                stats.covariance_left_rows_recomputed
            ),
            "full_covariance_rows_recomputed": (
                stats.full_covariance_rows_recomputed
            ),
        },
    }


def run_probe(
    sizes: tuple[int, ...] = (16, 32, 64, 128, 256),
    *,
    repeats: int = 3,
    time_cliff_seconds: float = 1.0,
    output_path: str | Path | None = None,
    quiet: bool = False,
) -> dict[str, object]:
    """Measure requested sizes and report observed and analytical cliffs."""
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("sizes must contain positive storey counts")
    if time_cliff_seconds <= 0.0:
        raise ValueError("time_cliff_seconds must be positive")
    measurements = [measure_size(size, repeats) for size in sizes]
    observed_cliff = next(
        (
            row["storeys"]
            for row in measurements
            if row["complete_pushforward_seconds"] >= time_cliff_seconds
        ),
        None,
    )
    verified_cliff = next(
        (
            row["storeys"]
            for row in measurements
            if row["verified_incremental_pipeline_seconds"] >= time_cliff_seconds
        ),
        None,
    )
    result: dict[str, object] = {
        "format": "gat-incremental-scale-probe-v1",
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "synthetic_model": {
            "raw_variables_per_storey": 1,
            "derived_variables_per_storey": 2,
            "dependency_scope_per_change": 2,
            "covariance_representation": "dense-float64",
        },
        "time_cliff_seconds": time_cliff_seconds,
        "first_measured_complete_pushforward_cliff_storeys": observed_cliff,
        "first_measured_verified_incremental_cliff_storeys": verified_cliff,
        "analytical_resident_memory_limits": {
            "one_gib_storeys": storeys_for_dense_budget(1024**3),
            "four_gib_storeys": storeys_for_dense_budget(4 * 1024**3),
        },
        "measurements": measurements,
        "conclusion": (
            "Incremental dependency/Jacobian and covariance-row recomputation is "
            "implemented, but the canonical raw and full covariances remain dense; "
            "resident memory and global observations therefore retain quadratic "
            "scaling. This probe determines when a sparse/factor representation is "
            "justified on the measured deployment hardware."
        ),
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(result, indent=1, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    if not quiet:
        print("GAT INCREMENTAL PROPAGATION SCALE PROBE")
        print(
            "storeys  full vars  complete(s)  incremental(s)  verify(s)  "
            "speedup  left/cov rows"
        )
        for row in measurements:
            work = row["incremental_work"]
            print(
                f"{row['storeys']:7d}  {row['full_variables']:9d}  "
                f"{row['complete_pushforward_seconds']:11.6f}  "
                f"{row['incremental_pushforward_seconds']:14.6f}  "
                f"{row['verification_seconds']:9.6f}  "
                f"{row['measured_speedup']:7.2f}x  "
                f"{work['covariance_left_rows_recomputed']}/"
                f"{work['full_covariance_rows_recomputed']}"
            )
        print(
            "dense resident-array limits: "
            f"1 GiB~{result['analytical_resident_memory_limits']['one_gib_storeys']} "
            "storeys; "
            f"4 GiB~{result['analytical_resident_memory_limits']['four_gib_storeys']} "
            "storeys"
        )
        print(f"measured complete-pushforward cliff: {observed_cliff}")
        print(f"measured verified-incremental cliff: {verified_cliff}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[16, 32, 64, 128, 256])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--time-cliff-seconds", type=float, default=1.0)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    run_probe(
        tuple(args.sizes),
        repeats=args.repeats,
        time_cliff_seconds=args.time_cliff_seconds,
        output_path=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
