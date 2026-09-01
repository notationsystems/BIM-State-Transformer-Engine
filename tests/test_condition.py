"""Tests for gat/gaussian/condition.py — Gaussian conditioning updates.

Covers: an independent hand-computed 2-variable oracle, agreement between
the Joseph-form ``condition`` and the partitioned-formula
``condition_linear_exact`` oracle, Joseph symmetry/PSD stability over long
sequential runs, exact (R = 0) observations, redundant exact observations,
variance monotonicity, and input validation.
"""

from __future__ import annotations

import unittest

import numpy as np

from gat.errors import ConditioningError
from gat.gaussian.condition import condition, condition_linear_exact
from gat.gaussian.linalg import chol_psd, max_asymmetry
from gat.gaussian.state import GaussianState, VarIndex
from gat.ids import EntityId, VarId


def _state(mu, sigma, names=None):
    """Build a GaussianState over synthetic VarIds a, b, c, ..."""
    n = len(mu)
    if names is None:
        names = [f"q{i}" for i in range(n)]
    ent = EntityId("IfcWall", "TESTENTITY0000000000")
    vars_ = tuple(VarId(ent, name) for name in names)
    return GaussianState(VarIndex(vars_), np.asarray(mu), np.asarray(sigma))


def _random_spd(rng, n):
    """Random symmetric positive definite covariance."""
    A = rng.standard_normal((n, n))
    return A @ A.T + n * np.eye(n)


class TestTwoVariableHandOracle(unittest.TestCase):
    """Exact posterior for one linear scalar observation, by hand.

    Prior:  x = (a, b),  mu = (3, 2),
            Sigma = [[0.04, 0.01],
                     [0.01, 0.09]]
    Observe y = a + b  (H = [1, 1]),  y = 5.5,  R = 0.01.

    Partitioned-Gaussian formula, computed analytically:

        S       = H Sigma H^T + R
                = 0.04 + 0.01 + 0.01 + 0.09 + 0.01      = 0.16
        Sigma H^T = [0.04 + 0.01, 0.01 + 0.09]          = [0.05, 0.10]
        K       = Sigma H^T / S = [0.05, 0.10] / 0.16   = [0.3125, 0.625]
        innov   = y - H mu = 5.5 - (3 + 2)              = 0.5

        mu'     = mu + K * innov
                = [3 + 0.3125*0.5, 2 + 0.625*0.5]
                = [3.15625, 2.3125]

        Sigma'  = Sigma - (Sigma H^T)(Sigma H^T)^T / S
          Sigma'_aa = 0.04 - 0.05*0.05/0.16 = 0.04 - 0.015625 =  0.024375
          Sigma'_ab = 0.01 - 0.05*0.10/0.16 = 0.01 - 0.03125  = -0.02125
          Sigma'_bb = 0.09 - 0.10*0.10/0.16 = 0.09 - 0.0625   =  0.0275
    """

    MU = np.array([3.0, 2.0])
    SIGMA = np.array([[0.04, 0.01], [0.01, 0.09]])
    H = np.array([[1.0, 1.0]])
    Y = np.array([5.5])
    R = np.array([0.01])

    EXPECTED_MU = np.array([3.15625, 2.3125])
    EXPECTED_SIGMA = np.array([[0.024375, -0.02125], [-0.02125, 0.0275]])

    def test_condition_matches_hand_oracle(self):
        belief = _state(self.MU, self.SIGMA)
        post, record = condition(
            belief, self.H, self.H @ self.MU, self.Y, self.R
        )
        np.testing.assert_allclose(post.mu, self.EXPECTED_MU, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(
            post.sigma, self.EXPECTED_SIGMA, rtol=0.0, atol=1e-12
        )
        # Record carries the innovation 0.5 and S = 0.16 exactly as derived.
        self.assertAlmostEqual(record.innovations[0], 0.5, delta=1e-12)
        self.assertAlmostEqual(record.innovation_variances[0], 0.16, delta=1e-12)
        self.assertEqual(record.jitter, 0.0)

    def test_linear_exact_matches_hand_oracle(self):
        mu_post, sigma_post = condition_linear_exact(
            self.MU, self.SIGMA, self.H, self.Y, self.R
        )
        np.testing.assert_allclose(mu_post, self.EXPECTED_MU, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(
            sigma_post, self.EXPECTED_SIGMA, rtol=0.0, atol=1e-12
        )


class TestConditionAgreesWithLinearExact(unittest.TestCase):
    """condition() and condition_linear_exact() are independent code paths
    (Joseph + Cholesky vs partitioned formula + np.linalg.solve); on linear
    problems they must agree to 1e-12."""

    def test_five_seeded_random_linear_cases(self):
        rng = np.random.default_rng(20260831)
        for case in range(5):
            n = 3 + case % 4  # dims 3, 4, 5, 6, 3
            k = 1 + case % 3  # 1..3 simultaneous measurements
            with self.subTest(case=case, n=n, k=k):
                mu = rng.uniform(-5.0, 5.0, size=n)
                sigma = _random_spd(rng, n)
                H = rng.standard_normal((k, n))
                observed = rng.uniform(-5.0, 5.0, size=k)
                noise = rng.uniform(0.1, 1.0, size=k)

                belief = _state(mu, sigma)
                # Linear model: the exact prediction h(mu) is H @ mu.
                post, _ = condition(belief, H, H @ mu, observed, noise)
                mu_exact, sigma_exact = condition_linear_exact(
                    mu, sigma, H, observed, noise
                )
                np.testing.assert_allclose(
                    post.mu, mu_exact, rtol=0.0, atol=1e-12
                )
                np.testing.assert_allclose(
                    post.sigma, sigma_exact, rtol=0.0, atol=1e-12
                )


class TestJosephStability(unittest.TestCase):
    def test_hundred_sequential_scalar_conditionings_stay_symmetric_pd(self):
        """After every one of 100 sequential scalar updates the covariance
        has exactly zero asymmetry (condition() symmetrizes) and factors by
        Cholesky at jitter rung 0."""
        rng = np.random.default_rng(7)
        n = 6
        mu = rng.uniform(-2.0, 2.0, size=n)
        sigma = _random_spd(rng, n)
        belief = _state(mu, sigma)

        for step in range(100):
            H = rng.standard_normal((1, n))
            predicted = H @ belief.mu
            observed = predicted + rng.normal(0.0, 0.3, size=1)
            noise = np.array([rng.uniform(0.05, 0.5)])
            belief, record = condition(belief, H, predicted, observed, noise)

            self.assertEqual(
                max_asymmetry(belief.sigma),
                0.0,
                msg=f"asymmetry appeared at step {step}",
            )
            _, jitter = chol_psd(belief.sigma)
            self.assertEqual(
                jitter, 0.0, msg=f"jitter needed at step {step}"
            )


class TestExactObservations(unittest.TestCase):
    def _belief(self):
        return _state(
            np.array([1.0, 2.0, 3.0]),
            np.array(
                [
                    [0.04, 0.010, 0.005],
                    [0.010, 0.09, 0.020],
                    [0.005, 0.020, 0.16],
                ]
            ),
        )

    def test_r_zero_collapses_observed_marginal(self):
        """An exact (R = 0) direct observation of variable 0 drives its
        posterior marginal variance essentially to zero (< 1e-18): the
        Joseph form zeroes the observed row of (I - K H) up to roundoff."""
        belief = self._belief()
        H = np.array([[1.0, 0.0, 0.0]])
        post, record = condition(
            belief, H, H @ belief.mu, np.array([1.3]), np.array([0.0])
        )
        observed_var = post.var_of(belief.index.var(0))
        self.assertLess(abs(observed_var), 1e-18)
        # The observed mean is moved exactly onto the measurement.
        self.assertAlmostEqual(post.mean(belief.index.var(0)), 1.3, delta=1e-12)
        self.assertEqual(record.jitter, 0.0)

    # BUG: gat/gaussian/condition.py's module docstring promises that a
    # redundant set of exact observations makes S singular and raises
    # ConditioningError naming the offending measurement block.  condition()
    # detects a ladder engagement in the presence of an exact (R=0)
    # measurement and raises instead of silently regularizing the singular
    # innovation covariance.
    def test_redundant_exact_observations_raise(self):
        belief = self._belief()
        H = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])  # same row twice
        with self.assertRaises(ConditioningError):
            condition(
                belief,
                H,
                H @ belief.mu,
                np.array([1.3, 1.3]),
                np.array([0.0, 0.0]),  # both exact -> singular S
            )


class TestVarianceMonotonicityAndValidation(unittest.TestCase):
    def test_posterior_variances_never_exceed_priors(self):
        """Conditioning can only sharpen marginals: diag(Sigma') <=
        diag(Sigma) elementwise (tolerance 1e-12)."""
        rng = np.random.default_rng(99)
        for case in range(5):
            n = 3 + case
            k = 1 + case % 2
            with self.subTest(case=case):
                mu = rng.uniform(-3.0, 3.0, size=n)
                sigma = _random_spd(rng, n)
                belief = _state(mu, sigma)
                H = rng.standard_normal((k, n))
                observed = rng.uniform(-3.0, 3.0, size=k)
                noise = rng.uniform(0.05, 0.5, size=k)
                post, _ = condition(belief, H, H @ mu, observed, noise)
                prior_diag = np.diag(belief.sigma)
                post_diag = np.diag(post.sigma)
                self.assertTrue(
                    (post_diag <= prior_diag + 1e-12).all(),
                    msg=f"variance grew: {post_diag - prior_diag}",
                )

    def test_negative_noise_variance_raises(self):
        belief = _state(np.array([0.0, 0.0]), np.eye(2))
        H = np.array([[1.0, 0.0]])
        with self.assertRaises(ConditioningError):
            condition(belief, H, H @ belief.mu, np.array([0.5]), np.array([-0.01]))

    def test_shape_mismatch_raises(self):
        belief = _state(np.array([0.0, 0.0]), np.eye(2))
        H_bad = np.array([[1.0, 0.0, 0.0]])  # 3 columns vs n_raw = 2
        with self.assertRaises(ConditioningError):
            condition(belief, H_bad, np.array([0.0]), np.array([0.5]), np.array([0.1]))


if __name__ == "__main__":
    unittest.main()
