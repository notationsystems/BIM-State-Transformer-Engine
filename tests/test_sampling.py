"""Belief sampling: determinism, moments, realization checks, calibration."""

from __future__ import annotations

import os
import unittest

import numpy as np

from gat.engine.sampling import (
    empirical_pair_clearance,
    sample_raw,
    sample_report,
    sample_worlds,
    sat_clearance,
)
from gat.geometry.clash import detect
from gat.geometry.stateio import derive_scene
from gat.session import GatSession

MODEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "gat", "demo", "model.ifc",
)


class SamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.world = GatSession.load_ifc(MODEL).world

    def test_seeded_determinism(self) -> None:
        a = sample_raw(self.world, 50, seed=7)
        b = sample_raw(self.world, 50, seed=7)
        self.assertTrue(np.array_equal(a, b))
        c = sample_raw(self.world, 50, seed=8)
        self.assertFalse(np.array_equal(a, c))

    def test_moments_converge_to_belief(self) -> None:
        n = 20000
        samples = sample_raw(self.world, n, seed=0)
        mu = self.world.belief.mu
        sigma = self.world.belief.sigma
        stds = np.sqrt(np.diag(sigma))
        # Per-variable scale-aware bound: 5 standard errors of the mean.
        err = np.abs(samples.mean(axis=0) - mu)
        self.assertTrue((err < 5.0 * stds / np.sqrt(n) + 1e-12).all())
        emp_cov = np.cov(samples.T)
        denom = np.outer(stds, stds) + 1e-12
        self.assertLess(np.abs((emp_cov - sigma) / denom).max(), 0.1)

    def test_sampled_worlds_carry_pushforward(self) -> None:
        (sampled,) = sample_worlds(self.world, 1, seed=3)
        # The realized derived layer must be the exact evaluation of the
        # realized raws (GAUSS-03 semantics survive sampling).
        env = sampled.belief.env()
        derived = self.world.binding.deps.evaluate(env)
        for var, value in derived.items():
            self.assertAlmostEqual(sampled.full.mean(var), value, places=12)

    def test_realizations_mostly_pass_on_demo(self) -> None:
        report = sample_report(self.world, 200, seed=0)
        # Demo margins are many sigma wide; realized hard failures are rare.
        self.assertGreaterEqual(report.pass_rate, 0.95)
        self.assertEqual(report.n, 200)

    def test_sat_clearance_matches_scene_definition_at_mean(self) -> None:
        scene = derive_scene(self.world)
        a = scene.element_by_name("Wall-South")
        b = scene.element_by_name("Wall-East")
        from gat.geometry.clash import score_pair

        item = score_pair(scene, a, b)
        direct = sat_clearance(a.box, b.box)
        self.assertAlmostEqual(direct, item.clearance, places=12)

    def test_monte_carlo_calibrates_analytic_clash_probability(self) -> None:
        # The delta-method P(clash) is a linearization; the sampler is the
        # ground truth of the same model.  They must agree within Monte-
        # Carlo error plus a small linearization allowance.
        emp = empirical_pair_clearance(
            self.world, "Wall-South", "Wall-East", n=4000, seed=1
        )
        scene = derive_scene(self.world)
        analytic = next(
            it
            for it in detect(scene).items
            if {it.element_a, it.element_b} == {"Wall-South", "Wall-East"}
        )
        self.assertLess(
            abs(emp.p_clash - analytic.p_clash),
            4.0 * emp.mc_standard_error + 0.01,
        )
        # Empirical clearance sigma should match the delta-method sigma.
        self.assertLess(abs(emp.std - analytic.sigma), 0.15 * analytic.sigma)

    def test_rejects_nonpositive_n(self) -> None:
        with self.assertRaises(ValueError):
            sample_raw(self.world, 0, seed=0)


if __name__ == "__main__":
    unittest.main()
