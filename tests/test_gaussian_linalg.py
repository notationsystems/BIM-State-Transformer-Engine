"""Tests for gat/gaussian/linalg.py and gat/gaussian/state.py.

Covers symmetrize exactness, the chol_psd jitter ladder (healthy PD case,
tiny-negative-eigenvalue repair with reported jitter, hard failure on an
indefinite matrix), chol_solve against np.linalg.solve, and the
GaussianState value-object contract (frozen arrays, symmetrization on
construction, marginals, corr guard, digest semantics, env round-trip).
"""

from __future__ import annotations

import unittest

import numpy as np

from gat.errors import NumericalError
from gat.gaussian.linalg import (
    JITTER_LADDER,
    chol_psd,
    chol_solve,
    max_asymmetry,
    symmetrize,
)
from gat.gaussian.state import GaussianState, VarIndex
from gat.ids import EntityId, VarId


def _vars(n: int) -> tuple[VarId, ...]:
    eid = EntityId("IfcWall", "TESTWALL00000000000000")
    return tuple(VarId(eid, f"Q{i}") for i in range(n))


class TestSymmetrize(unittest.TestCase):
    def test_symmetrize_is_exact_half_sum(self) -> None:
        rng = np.random.default_rng(42)
        a = rng.standard_normal((6, 6))
        s = symmetrize(a)
        # Bitwise identical to the definition, and exactly symmetric:
        # 0.5*(a_ij + a_ji) is computed by the same float ops for both
        # triangles, so S == S.T holds with no tolerance at all.
        np.testing.assert_array_equal(s, 0.5 * (a + a.T))
        np.testing.assert_array_equal(s, s.T)
        self.assertEqual(max_asymmetry(s), 0.0)

    def test_symmetrize_fixes_symmetric_input(self) -> None:
        rng = np.random.default_rng(3)
        a = rng.standard_normal((4, 4))
        sym = a + a.T
        np.testing.assert_array_equal(symmetrize(sym), sym)


class TestCholPsd(unittest.TestCase):
    def _pd_matrix(self, n: int = 5, seed: int = 11) -> np.ndarray:
        rng = np.random.default_rng(seed)
        r = rng.standard_normal((n, n))
        return r @ r.T + n * np.eye(n)

    def test_zero_jitter_on_pd_matrix_and_reconstruction(self) -> None:
        s = self._pd_matrix()
        L, jitter = chol_psd(s)
        self.assertEqual(jitter, 0.0)
        self.assertTrue(np.allclose(np.triu(L, k=1), 0.0), "L not lower-triangular")
        # L @ L.T must reconstruct S to 1e-12 (entries are O(10)).
        np.testing.assert_allclose(L @ L.T, s, rtol=0.0, atol=1e-12)

    def test_jitter_ladder_engages_and_reports(self) -> None:
        """A matrix with one tiny negative eigenvalue (-1e-11) fails plain
        Cholesky and the 1e-12 rung, then succeeds at 1e-10; the jitter
        actually used must be reported."""
        rng = np.random.default_rng(7)
        q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        s = symmetrize(q @ np.diag([1.0, 0.5, -1e-11]) @ q.T)
        # Sanity: plain Cholesky must reject this matrix.
        with self.assertRaises(np.linalg.LinAlgError):
            np.linalg.cholesky(s)
        L, jitter = chol_psd(s)
        # scale = max(trace/n, 1) = max(1.5/3, 1) = 1.0, so the reported
        # jitter is exactly a ladder rung.  -1e-11 + 1e-12 is still negative,
        # so the ladder must land on the 1e-10 rung.
        self.assertEqual(jitter, 1e-10)
        self.assertIn(jitter, JITTER_LADDER)
        np.testing.assert_allclose(
            L @ L.T, s + jitter * np.eye(3), rtol=0.0, atol=1e-12
        )

    def test_indefinite_matrix_raises_numerical_error(self) -> None:
        # diag(1, -1): min eigenvalue -1 is far beyond any ladder rung.
        s = np.diag([1.0, -1.0])
        with self.assertRaises(NumericalError):
            chol_psd(s)

    def test_chol_solve_matches_numpy_solve(self) -> None:
        rng = np.random.default_rng(19)
        n = 7
        r = rng.standard_normal((n, n))
        s = r @ r.T + n * np.eye(n)
        rhs = rng.standard_normal(n)
        L, jitter = chol_psd(s)
        self.assertEqual(jitter, 0.0)
        x = chol_solve(L, rhs)
        np.testing.assert_allclose(x, np.linalg.solve(s, rhs), rtol=1e-10, atol=1e-10)

    def test_chol_solve_matrix_rhs(self) -> None:
        rng = np.random.default_rng(23)
        n = 4
        r = rng.standard_normal((n, n))
        s = r @ r.T + n * np.eye(n)
        rhs = rng.standard_normal((n, 3))
        L, _ = chol_psd(s)
        np.testing.assert_allclose(
            chol_solve(L, rhs), np.linalg.solve(s, rhs), rtol=1e-10, atol=1e-10
        )


class TestGaussianState(unittest.TestCase):
    def setUp(self) -> None:
        self.vars = _vars(3)
        self.index = VarIndex(self.vars)
        self.mu = np.array([1.0, -2.0, 3.5])
        # PD covariance, hand-picked symmetric.
        self.sigma = np.array(
            [
                [2.0, 0.3, 0.1],
                [0.3, 1.5, -0.2],
                [0.1, -0.2, 1.0],
            ]
        )
        self.state = GaussianState(self.index, self.mu, self.sigma)

    def test_arrays_are_frozen(self) -> None:
        with self.assertRaises(ValueError):
            self.state.mu[0] = 99.0
        with self.assertRaises(ValueError):
            self.state.sigma[0, 0] = 99.0

    def test_construction_copies_input(self) -> None:
        # Mutating the arrays passed in must not affect the state.
        mu = self.mu.copy()
        sigma = self.sigma.copy()
        state = GaussianState(self.index, mu, sigma)
        mu[0] = 123.0
        sigma[0, 0] = 123.0
        self.assertEqual(state.mu[0], 1.0)
        self.assertEqual(state.sigma[0, 0], 2.0)

    def test_symmetrized_on_construction(self) -> None:
        asym = np.array(
            [
                [2.0, 1.0, 0.0],
                [0.0, 1.5, 0.4],
                [0.2, 0.0, 1.0],
            ]
        )
        state = GaussianState(self.index, self.mu, asym)
        np.testing.assert_array_equal(state.sigma, 0.5 * (asym + asym.T))
        np.testing.assert_array_equal(state.sigma, state.sigma.T)

    def test_marginal_extracts_right_blocks(self) -> None:
        v0, v1, v2 = self.vars
        marg = self.state.marginal((v2, v0))
        self.assertEqual(marg.index.vars, (v2, v0))
        np.testing.assert_array_equal(marg.mu, np.array([3.5, 1.0]))
        # Rows/cols permuted to (2, 0) of the original covariance.
        expected = np.array([[1.0, 0.1], [0.1, 2.0]])
        np.testing.assert_array_equal(marg.sigma, expected)
        self.assertEqual(marg.cov(v2, v0), self.state.cov(v2, v0))

    def test_corr_guards_zero_variance(self) -> None:
        v0, v1, _ = self.vars
        sigma = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.5, -0.2],
                [0.0, -0.2, 1.0],
            ]
        )
        state = GaussianState(self.index, self.mu, sigma)
        self.assertEqual(state.corr(v0, v1), 0.0)
        self.assertEqual(state.corr(v1, v0), 0.0)

    def test_corr_normalizes_covariance(self) -> None:
        v0, v1, _ = self.vars
        # corr = 0.3 / (sqrt(2.0) * sqrt(1.5)) = 0.3/sqrt(3) = 0.17320508...
        expected = 0.3 / np.sqrt(2.0 * 1.5)
        self.assertAlmostEqual(self.state.corr(v0, v1), expected, delta=1e-14)
        self.assertAlmostEqual(self.state.corr(v0, v0), 1.0, delta=1e-14)

    def test_digest_changes_iff_bytes_change(self) -> None:
        same = GaussianState(self.index, self.mu.copy(), self.sigma.copy())
        self.assertEqual(self.state.digest(), same.digest())
        # A change in any mu byte changes the digest ...
        mu2 = self.mu.copy()
        mu2[1] = np.nextafter(mu2[1], np.inf)
        self.assertNotEqual(
            self.state.digest(), GaussianState(self.index, mu2, self.sigma).digest()
        )
        # ... and so does a change in any Sigma byte.
        sigma2 = self.sigma.copy()
        sigma2[2, 2] = np.nextafter(sigma2[2, 2], np.inf)
        self.assertNotEqual(
            self.state.digest(),
            GaussianState(self.index, self.mu, sigma2).digest(),
        )
        # Calling digest twice on the same state is stable.
        self.assertEqual(self.state.digest(), self.state.digest())

    def test_env_round_trips_means(self) -> None:
        env = self.state.env()
        self.assertEqual(set(env), set(self.vars))
        for i, var in enumerate(self.vars):
            self.assertEqual(env[var], self.mu[i])
            self.assertEqual(env[var], self.state.mean(var))
        # env feeds back through replace: a state rebuilt from the same
        # numbers has identical means.
        rebuilt = self.state.replace(mu=np.array([env[v] for v in self.vars]))
        np.testing.assert_array_equal(rebuilt.mu, self.state.mu)


if __name__ == "__main__":
    unittest.main()
