"""Incremental pushforward equivalence, selectivity, and scale probe tests."""

from __future__ import annotations

import tempfile
import unittest

import numpy as np

from gat.demo.incremental_scale import (
    dense_state_bytes,
    measure_size,
    run_probe,
    storeys_for_dense_budget,
)
from gat.engine.transform import ScaleParameter, ShiftParameter
from gat.session import GatSession


MODEL = "gat/demo/model.ifc"


class IncrementalPropagationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session = GatSession.load_ifc(MODEL)

    def assert_matches_complete(self, transformation) -> None:
        world = self.session.world
        belief = transformation.apply(world.binding, world.belief)
        complete = world.with_belief(belief)
        incremental, stats = world.with_belief_incremental(belief)

        np.testing.assert_array_equal(incremental.full.mu, complete.full.mu)
        np.testing.assert_array_equal(incremental.full.sigma, complete.full.sigma)
        self.assertEqual(incremental.full.index.vars, complete.full.index.vars)
        self.assertEqual(stats.mode, "incremental")
        self.assertLess(
            stats.full_covariance_rows_recomputed,
            stats.full_variable_count,
        )

    def test_local_shift_recomputes_only_dependency_closed_rows(self) -> None:
        transformation = ShiftParameter(
            self.session.var("Wall-Party", "Width"),
            0.001,
        )
        result = self.session.run(transformation)
        stats = result.propagation

        self.assertIsNotNone(stats)
        self.assertEqual(stats.mode, "incremental")
        self.assertEqual(stats.raw_mean_rows_changed, 1)
        self.assertEqual(stats.raw_covariance_rows_changed, 0)
        self.assertEqual(
            stats.derived_value_rows_recomputed,
            len(result.affected),
        )
        self.assertEqual(
            stats.covariance_left_rows_recomputed,
            len(result.affected),
        )
        self.assertEqual(stats.full_covariance_rows_recomputed, len(result.affected))

    def test_covariance_change_reports_global_cached_product_refresh(self) -> None:
        transformation = ScaleParameter(
            self.session.var("Wall-Party", "Width"),
            1.01,
        )
        result = self.session.run(transformation)
        stats = result.propagation

        self.assertIsNotNone(stats)
        self.assertEqual(stats.mode, "incremental")
        self.assertEqual(stats.raw_mean_rows_changed, 1)
        self.assertEqual(stats.raw_covariance_rows_changed, 1)
        self.assertEqual(
            stats.covariance_left_rows_recomputed,
            stats.full_variable_count,
        )
        self.assertEqual(
            stats.full_covariance_rows_recomputed,
            len(result.affected) + 1,
        )

    def test_shift_and_scale_match_complete_pushforward(self) -> None:
        self.assert_matches_complete(
            ShiftParameter(self.session.var("Wall-Party", "Width"), 0.001)
        )
        self.assert_matches_complete(
            ScaleParameter(self.session.var("Wall-Party", "Width"), 1.01)
        )


class IncrementalScaleProbeTests(unittest.TestCase):
    def test_synthetic_probe_has_two_row_dependency_scope(self) -> None:
        row = measure_size(16, repeats=1)
        work = row["incremental_work"]

        self.assertEqual(row["raw_variables"], 16)
        self.assertEqual(row["derived_variables"], 32)
        self.assertEqual(row["full_variables"], 48)
        self.assertEqual(work["mode"], "incremental")
        self.assertEqual(work["derived_value_rows_recomputed"], 2)
        self.assertEqual(work["covariance_left_rows_recomputed"], 2)
        self.assertEqual(work["full_covariance_rows_recomputed"], 2)
        self.assertLessEqual(row["max_abs_covariance_error"], 1e-12)

    def test_dense_memory_limit_is_analytical_and_monotone(self) -> None:
        one_gib = storeys_for_dense_budget(1024**3)
        four_gib = storeys_for_dense_budget(4 * 1024**3)
        self.assertGreater(four_gib, one_gib)
        self.assertLessEqual(dense_state_bytes(one_gib, 3 * one_gib), 1024**3)
        self.assertGreater(
            dense_state_bytes(one_gib + 1, 3 * (one_gib + 1)),
            1024**3,
        )

    def test_probe_writes_machine_readable_result_without_timing_claim(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = f"{directory}/probe.json"
            result = run_probe(
                (4, 8),
                repeats=1,
                time_cliff_seconds=60.0,
                output_path=output,
                quiet=True,
            )
        self.assertEqual(result["format"], "gat-incremental-scale-probe-v1")
        self.assertIsNone(result["first_measured_complete_pushforward_cliff_storeys"])
        self.assertIsNone(result["first_measured_verified_incremental_cliff_storeys"])
        self.assertEqual(len(result["measurements"]), 2)


if __name__ == "__main__":
    unittest.main()
