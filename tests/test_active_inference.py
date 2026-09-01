"""Tests for deterministic one-step active inference in GAT.

The tests use the shipped two-office model.  They establish that the planner
does not mutate the architectural state, chooses the most informative sensor
for a stated decision, and predicts the covariance reduction that the normal
GAT conditioning pipeline subsequently realizes.
"""

from __future__ import annotations

import math
import os
import unittest

from gat.engine.active_inference import (
    MinimumPreference,
    ObservationCandidate,
    plan_observations,
    select_observation,
)
from gat.session import GatSession


DEMO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "gat", "demo", "model.ifc",
)


class ActiveInferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session = GatSession.load_ifc(DEMO)
        self.target = self.session.var("Office-A", "Volume")
        self.clear_height = self.session.var("Level 1", "ClearHeight")
        self.unit_cost = self.session.var("Wall-South", "UnitCost")
        self.preference = MinimumPreference(self.target, minimum=59.0)

    def test_direct_target_sensor_has_closed_form_information_gain(self) -> None:
        # For a direct scalar observation y = t + eps, the information gain
        # is 1/2 log(1 + Var(t) / Var(eps)), and the posterior variance is
        # Var(t) Var(eps) / (Var(t) + Var(eps)).
        noise = 0.05
        plan = select_observation(
            self.session.world,
            [ObservationCandidate(self.target, noise)],
            self.preference,
        )
        prior_var = self.session.world.full.var_of(self.target)
        noise_var = noise**2
        expected_information = 0.5 * math.log1p(prior_var / noise_var)
        expected_posterior_var = prior_var * noise_var / (prior_var + noise_var)

        self.assertAlmostEqual(plan.epistemic_value, expected_information, delta=1e-12)
        self.assertIsNotNone(plan.posterior_target_sigma)
        assert plan.posterior_target_sigma is not None
        self.assertAlmostEqual(
            plan.posterior_target_sigma**2, expected_posterior_var, delta=1e-12
        )
        self.assertIsNotNone(plan.pragmatic_risk)
        assert plan.pragmatic_risk is not None
        self.assertAlmostEqual(
            plan.expected_free_energy,
            plan.pragmatic_risk - expected_information,
            delta=1e-12,
        )

    def test_policy_selects_relevant_observation_without_mutating_world(self) -> None:
        # Office-A.Volume depends on ClearHeight, but not Wall-South.UnitCost.
        # The direct volume sensor is the most informative of these choices.
        digest_before = self.session.world.digest()
        plans = plan_observations(
            self.session.world,
            [
                ObservationCandidate(self.unit_cost, 10.0, "cost survey"),
                ObservationCandidate(self.clear_height, 0.01, "laser height"),
                ObservationCandidate(self.target, 0.05, "laser volume"),
            ],
            self.preference,
        )
        self.assertEqual(self.session.world.digest(), digest_before)
        self.assertEqual(plans[0].candidate.var, self.target)
        self.assertEqual(plans[-1].candidate.var, self.unit_cost)
        self.assertEqual(plans[-1].epistemic_value, 0.0)
        self.assertGreater(plans[0].epistemic_value, plans[1].epistemic_value)

    def test_predicted_posterior_variance_matches_verified_conditioning(self) -> None:
        plan = select_observation(
            self.session.world,
            [ObservationCandidate(self.clear_height, 0.01, "laser height")],
            self.preference,
        )
        assert plan.posterior_target_sigma is not None
        # Covariance after Gaussian conditioning is independent of the actual
        # scalar reading.  Use the predicted reading so means remain unchanged
        # and the test isolates the planned uncertainty reduction.
        self.session.run(plan.candidate.observe(plan.predicted_measurement))
        self.assertAlmostEqual(
            self.session.world.full.std(self.target),
            plan.posterior_target_sigma,
            delta=1e-12,
        )
        self.assertTrue(self.session.verify().passed)

    def test_general_information_mode_ranks_raw_information(self) -> None:
        direct = ObservationCandidate(self.clear_height, 0.01)
        noisy = ObservationCandidate(self.clear_height, 0.1)
        plans = plan_observations(self.session.world, [noisy, direct])
        self.assertIsNone(plans[0].target)
        self.assertEqual(plans[0].candidate, direct)
        self.assertGreater(plans[0].epistemic_value, plans[1].epistemic_value)

    def test_exact_sensor_is_rejected_from_differential_entropy_policy(self) -> None:
        with self.assertRaises(ValueError):
            ObservationCandidate(self.target, 0.0)

    def test_non_finite_preference_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MinimumPreference(self.target, float("nan"))

    def test_action_cost_is_part_of_expected_free_energy(self) -> None:
        candidate = ObservationCandidate(
            self.target, 0.05, "burdensome volume survey", cost_nats=0.75
        )
        plan = select_observation(
            self.session.world, [candidate], self.preference
        )
        assert plan.pragmatic_risk is not None
        self.assertEqual(plan.action_cost, 0.75)
        self.assertAlmostEqual(
            plan.expected_free_energy,
            plan.pragmatic_risk + plan.action_cost - plan.epistemic_value,
            delta=1e-12,
        )
        self.assertAlmostEqual(
            plan.net_epistemic_value,
            plan.epistemic_value - plan.action_cost,
            delta=1e-12,
        )

    def test_invalid_action_cost_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ObservationCandidate(self.target, 0.05, cost_nats=-0.01)
        with self.assertRaises(ValueError):
            ObservationCandidate(self.target, 0.05, cost_nats=float("inf"))


if __name__ == "__main__":
    unittest.main()
