"""Tests for gat.geometry.gaussianize, gat.geometry.overlap, gat.geometry.dual.

All oracles here are independent of the implementation under test:

* box moments are computed from the closed-form uniform-box formulas
  (center = origin + R E/2, covariance = R diag(E^2/12) R^T);
* Jacobians are checked against central finite differences;
* the chi-square(3) survival function is checked against Simpson
  integration of the chi2(3) density;
* dual-number derivatives are checked against central finite differences
  of the same composite function.

Everything is deterministic; the only randomness uses a fixed seed.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from gat.geometry.dual import (
    Dual,
    cos as dcos,
    directional_derivative,
    erf as derf,
    exp as dexp,
    log as dlog,
    sin as dsin,
    sqrt as dsqrt,
)
from gat.geometry.gaussianize import (
    OrientedBox,
    cov_jacobian_wrt_extents,
    gaussianize_box,
    mean_jacobian_wrt_extents,
    mean_jacobian_wrt_yaw,
    rot_z,
)
from gat.geometry.overlap import (
    bhattacharyya_coefficient,
    chi2_sf_3,
    mahalanobis2,
    normal_cdf,
    product_integral,
)
from gat.geometry.primitives import N_FEATURES, GaussianCloud


# Test boxes: axis-aligned, positively rotated, negatively rotated; the
# extents are chosen away from integer multiples of every spacing used so
# the grid resolution is stable under the +-1e-7 finite-difference
# perturbations below.
BOXES = (
    OrientedBox(origin=(0.0, 0.0, 0.0), angle=0.0, extents=(5.0, 4.0, 3.1)),
    OrientedBox(origin=(1.0, -2.0, 0.5), angle=0.3, extents=(4.0, 0.3, 2.9)),
    OrientedBox(origin=(-3.0, 2.0, 1.0), angle=-1.2, extents=(2.3, 0.31, 1.7)),
)
SPACINGS = (0.2, 0.5, 0.75, 1.1, 10.0)  # 10.0 => a single cell per box


def cloud_of(box: OrientedBox, spacing: float) -> GaussianCloud:
    means, covs, weights, _ = gaussianize_box(box, spacing)
    k = means.shape[0]
    return GaussianCloud(
        means,
        covs,
        weights,
        np.zeros((k, N_FEATURES)),
        np.zeros(k, dtype=np.intp),
    )


class MomentMatchTest(unittest.TestCase):
    def test_mixture_moments_equal_exact_box_moments(self):
        # Independent oracle: a uniform density on an oriented box with
        # full extents E has mean at the center origin + R (E/2) and
        # covariance R diag(E_i^2 / 12) R^T (Var(U[-E/2, E/2]) = E^2/12).
        # The tiling must reproduce these exactly (law of total covariance):
        # per axis, between-cell variance E^2 (n^2-1)/(12 n^2) plus
        # within-cell variance E^2/(12 n^2) sums to E^2/12.
        for box in BOXES:
            R = rot_z(box.angle)
            E = np.asarray(box.extents)
            exact_center = np.asarray(box.origin) + R @ (0.5 * E)
            exact_cov = R @ np.diag(E**2 / 12.0) @ R.T
            for spacing in SPACINGS:
                with self.subTest(box=box, spacing=spacing):
                    mean, cov = cloud_of(box, spacing).mixture_moments()
                    np.testing.assert_allclose(
                        mean, exact_center, rtol=0.0, atol=1e-12
                    )
                    np.testing.assert_allclose(
                        cov, exact_cov, rtol=0.0, atol=1e-12
                    )

    def test_total_weight_is_box_volume(self):
        for box in BOXES:
            for spacing in SPACINGS:
                with self.subTest(box=box, spacing=spacing):
                    self.assertAlmostEqual(
                        cloud_of(box, spacing).total_weight(),
                        box.volume,
                        delta=1e-12 * max(1.0, box.volume),
                    )


class JacobianTest(unittest.TestCase):
    H = 1e-7
    ATOL = 1e-6
    # 0.45 divides none of the test extents exactly, so the ceil-based grid
    # resolution is stable under the +-1e-7 perturbations.
    SPACING = 0.45

    @staticmethod
    def _with_extents(box: OrientedBox, extents) -> OrientedBox:
        return OrientedBox(origin=box.origin, angle=box.angle, extents=tuple(extents))

    def test_mean_jacobian_wrt_extents_vs_central_fd(self):
        for box in BOXES:
            means, _, _, fractions = gaussianize_box(box, self.SPACING)
            J = mean_jacobian_wrt_extents(box, fractions)  # (K, 3, 3)
            for i in range(3):
                e_p = np.asarray(box.extents, dtype=np.float64)
                e_m = e_p.copy()
                e_p = e_p.copy()
                e_p[i] += self.H
                e_m[i] -= self.H
                means_p = gaussianize_box(self._with_extents(box, e_p), self.SPACING)[0]
                means_m = gaussianize_box(self._with_extents(box, e_m), self.SPACING)[0]
                fd = (means_p - means_m) / (2.0 * self.H)  # (K, 3)
                with self.subTest(box=box, extent_axis=i):
                    np.testing.assert_allclose(
                        J[:, :, i], fd, rtol=0.0, atol=self.ATOL
                    )

    def test_cov_jacobian_wrt_extents_vs_central_fd(self):
        for box in BOXES:
            J = cov_jacobian_wrt_extents(box, self.SPACING)  # (3, 3, 3)
            for i in range(3):
                e_p = np.asarray(box.extents, dtype=np.float64)
                e_m = e_p.copy()
                e_p = e_p.copy()
                e_p[i] += self.H
                e_m[i] -= self.H
                cov_p = gaussianize_box(self._with_extents(box, e_p), self.SPACING)[1][0]
                cov_m = gaussianize_box(self._with_extents(box, e_m), self.SPACING)[1][0]
                fd = (cov_p - cov_m) / (2.0 * self.H)
                with self.subTest(box=box, extent_axis=i):
                    np.testing.assert_allclose(
                        J[:, :, i], fd, rtol=0.0, atol=self.ATOL
                    )

    def test_mean_jacobian_wrt_yaw_vs_central_fd(self):
        for box in BOXES:
            means, _, _, fractions = gaussianize_box(box, self.SPACING)
            J = mean_jacobian_wrt_yaw(box, fractions)  # (K, 3)
            box_p = OrientedBox(box.origin, box.angle + self.H, box.extents)
            box_m = OrientedBox(box.origin, box.angle - self.H, box.extents)
            means_p = gaussianize_box(box_p, self.SPACING)[0]
            means_m = gaussianize_box(box_m, self.SPACING)[0]
            fd = (means_p - means_m) / (2.0 * self.H)
            with self.subTest(box=box):
                np.testing.assert_allclose(J, fd, rtol=0.0, atol=self.ATOL)


def _scalar(x) -> float:
    """The overlap functions are batched: scalar inputs come back with a
    leading batch axis of length 1.  Extract the single value."""
    arr = np.asarray(x).reshape(-1)
    assert arr.size == 1
    return float(arr[0])


class OverlapTest(unittest.TestCase):
    def test_product_integral_of_gaussian_with_itself(self):
        # Hand check: for mu_i = mu_j = mu and S_i = S_j = Sigma,
        #   I = N(0; 0, 2 Sigma) = (2 pi)^{-3/2} det(2 Sigma)^{-1/2}
        #     = (2 pi)^{-3/2} 2^{-3/2} det(Sigma)^{-1/2}
        #     = 1 / ((4 pi)^{3/2} sqrt(det Sigma)).
        # For Sigma = diag(0.04, 0.09, 0.25): det = 0.0009, sqrt(det) = 0.03,
        # (4 pi)^{3/2} = 44.546623974653663..., so
        # I = 1 / (44.546623974653663 * 0.03) = 1 / 1.3363987192396098
        #   = 0.7482796755215274 (checked below against the closed form
        # written out literally, not against the implementation).
        mu = np.array([0.7, -1.3, 2.0])
        sigma = np.diag([0.04, 0.09, 0.25])
        expected = 1.0 / ((4.0 * math.pi) ** 1.5 * math.sqrt(0.0009))
        got = _scalar(product_integral(mu, sigma, mu, sigma))
        self.assertAlmostEqual(got, expected, delta=1e-12 * expected)

        # Rotated (non-diagonal) covariance: determinant is invariant under
        # rotation, so I = 1/((4 pi)^{3/2} sqrt(a1 a2 a3)) still.
        R = rot_z(0.6)
        a = np.array([0.05, 0.12, 0.30])
        sigma_rot = R @ np.diag(a) @ R.T
        expected_rot = 1.0 / ((4.0 * math.pi) ** 1.5 * math.sqrt(float(np.prod(a))))
        got_rot = _scalar(product_integral(mu, sigma_rot, mu, sigma_rot))
        self.assertAlmostEqual(got_rot, expected_rot, delta=1e-12 * expected_rot)

    def test_bhattacharyya_coefficient(self):
        rng = np.random.default_rng(42)
        A = rng.normal(size=(3, 3))
        S = A @ A.T + 0.5 * np.eye(3)  # SPD, non-diagonal
        mu = np.array([0.3, -1.0, 2.0])

        # Identical Gaussians: BC == 1.
        self.assertAlmostEqual(
            _scalar(bhattacharyya_coefficient(mu, S, mu, S)), 1.0, delta=1e-12
        )

        # Different means, same covariance: BC < 1.
        mu2 = mu + np.array([0.4, 0.0, -0.2])
        bc_mean = _scalar(bhattacharyya_coefficient(mu, S, mu2, S))
        self.assertLess(bc_mean, 1.0)
        self.assertGreater(bc_mean, 0.0)

        # Same mean, different covariance: BC < 1 (log-det term alone).
        S2 = 2.5 * S
        bc_cov = _scalar(bhattacharyya_coefficient(mu, S, mu, S2))
        self.assertLess(bc_cov, 1.0)
        self.assertGreater(bc_cov, 0.0)

        # Symmetry in the pair.
        self.assertAlmostEqual(
            _scalar(bhattacharyya_coefficient(mu, S, mu2, S2)),
            _scalar(bhattacharyya_coefficient(mu2, S2, mu, S)),
            delta=1e-12,
        )

    def test_mahalanobis2_hand_case(self):
        # Hand computation: d = mu_i - mu_j = (1, 2, -1),
        # S = S_i + S_j = diag(0.25, 0.25, 0.25) + diag(0.75, 0.25, 0.25)
        #   = diag(1.0, 0.5, 0.5), so
        # m2 = 1^2/1.0 + 2^2/0.5 + (-1)^2/0.5 = 1 + 8 + 2 = 11.
        mu_i = np.array([2.0, 1.0, -3.0])
        mu_j = np.array([1.0, -1.0, -2.0])
        S_i = np.diag([0.25, 0.25, 0.25])
        S_j = np.diag([0.75, 0.25, 0.25])
        got = _scalar(mahalanobis2(mu_i, S_i, mu_j, S_j))
        self.assertAlmostEqual(got, 11.0, delta=1e-12)

    def test_chi2_sf_3_vs_simpson_integration(self):
        # chi2(3) density: f(x) = sqrt(x) exp(-x/2) / sqrt(2 pi)
        # (x^{k/2-1} e^{-x/2} / (2^{k/2} Gamma(k/2)) with k=3,
        #  Gamma(3/2) = sqrt(pi)/2  =>  2^{3/2} Gamma(3/2) = sqrt(2 pi)).
        # SF(t) ~= integral_t^60 f(x) dx by composite Simpson; the tail
        # beyond 60 is O(1e-12), far below the 1e-6 tolerance.
        def pdf(x):
            return np.sqrt(x) * np.exp(-x / 2.0) / math.sqrt(2.0 * math.pi)

        def simpson(a, b, n):
            x = np.linspace(a, b, n + 1)
            y = pdf(x)
            h = (b - a) / n
            return h / 3.0 * (y[0] + y[-1] + 4.0 * y[1:-1:2].sum() + 2.0 * y[2:-1:2].sum())

        for t in (0.5, 2.0, 5.0, 10.0):
            with self.subTest(t=t):
                numeric = simpson(t, 60.0, 20000)
                self.assertAlmostEqual(float(chi2_sf_3(t)), numeric, delta=1e-6)

    def test_normal_cdf_reference_values(self):
        self.assertAlmostEqual(float(normal_cdf(0.0)), 0.5, delta=1e-15)
        # Phi(1) = 0.841344746068542948... (Abramowitz & Stegun 26.2.17 /
        # standard tables).
        self.assertAlmostEqual(float(normal_cdf(1.0)), 0.8413447460685429, delta=1e-12)
        self.assertAlmostEqual(float(normal_cdf(-1.0)), 1.0 - 0.8413447460685429, delta=1e-12)


def _composite(x):
    """A scalar composite exercising every Dual arithmetic rule plus the
    transcendental helpers.  ``x`` is a Dual over shape (3,); the return is
    a scalar Dual.  Evaluating it on ``Dual(v)`` (zero eps) gives the plain
    value, so the same definition drives the finite-difference oracle."""
    M = np.array([[0.5, -0.2, 0.1], [0.3, 0.8, -0.4], [-0.1, 0.2, 0.9]])
    y = Dual(M) @ x                               # __matmul__ (constant lhs)
    q = (y * y).sum()                             # __mul__, sum
    r = dsqrt(q + 1.0)                            # sqrt, __add__ w/ scalar
    s = dsin(x[0]) * dexp(-0.5 * x[1])            # sin, exp, __rmul__, __neg__
    s = s + dcos(x[2]) / (x[0] ** 2 + 2.0)        # cos, __truediv__, __pow__
    t = dlog(q + 3.0) + derf(x[1] - x[2] * 0.5)   # log, erf, __sub__
    u = (2.0 - x[1]) / (x[2] + 4.0)               # __rsub__, __truediv__
    w = 1.0 / (q + 5.0)                           # __rtruediv__
    return r + s + t + u + w - x[0] * 0.25


class DualTest(unittest.TestCase):
    X0 = np.array([0.4, -0.7, 1.2])

    @staticmethod
    def _value(v: np.ndarray) -> float:
        return float(np.asarray(_composite(Dual(v)).val).reshape(()))

    def _fd(self, direction: np.ndarray, h: float = 1e-6) -> float:
        return (
            self._value(self.X0 + h * direction) - self._value(self.X0 - h * direction)
        ) / (2.0 * h)

    def test_gradient_vs_central_fd_along_basis(self):
        for i in range(3):
            e = np.zeros(3)
            e[i] = 1.0
            dual = directional_derivative(_composite, self.X0, e)
            with self.subTest(axis=i):
                self.assertAlmostEqual(dual, self._fd(e), delta=1e-6)

    def test_directional_derivative_matches_fd_random_direction(self):
        rng = np.random.default_rng(7)
        for _ in range(4):
            direction = rng.normal(size=3)
            direction = direction / np.linalg.norm(direction)
            dual = directional_derivative(_composite, self.X0, direction)
            self.assertAlmostEqual(dual, self._fd(direction), delta=1e-6)

    def test_linearity_of_seed(self):
        # Forward-mode derivative is linear in the seed direction: the dual
        # derivative along (a u + b v) equals a f'(u) + b f'(v).
        u = np.array([1.0, 0.5, -0.25])
        v = np.array([-0.3, 1.1, 0.7])
        du = directional_derivative(_composite, self.X0, u)
        dv = directional_derivative(_composite, self.X0, v)
        dw = directional_derivative(_composite, self.X0, 2.0 * u - 3.0 * v)
        self.assertAlmostEqual(dw, 2.0 * du - 3.0 * dv, delta=1e-10)


if __name__ == "__main__":
    unittest.main()
