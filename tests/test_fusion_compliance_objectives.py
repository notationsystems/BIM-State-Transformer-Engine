"""Tests for gat/geometry/fusion.py, compliance.py, and objectives.py.

Covers: exact mixture moment matching against a hand-computed oracle,
kl_gauss identities, element-level LoD nodes reproducing exact box
moments, affine frame transport (moments transform affinely, weights
scale by |det A|), compliance PASS on the pristine demo and FAIL after
lowering the storey clear height below the 2.4 m rule, agreement between
the batched ray optical depth and the dual-friendly scalar path, the
dual-number gradient against central finite differences, and the layout
optimizer on the demo G6 configuration.
"""

from __future__ import annotations

import math
import os
import unittest

import numpy as np

import gat.demo
from gat.engine.executor import execute
from gat.engine.transform import SetParameter
from gat.geometry.compliance import check_compliance
from gat.geometry.dual import Dual
from gat.geometry.fusion import FrameTransform, element_level, kl_gauss, moment_match
from gat.geometry.gaussianize import rot_z
from gat.geometry.objectives import (
    LayoutObjective,
    optimize_layout,
    ray_optical_depth,
    scalar_ray_depth_dual,
)
from gat.geometry.stateio import derive_scene
from gat.ids import VarId
from gat.ir.core import LessEqual
from gat.session import GatSession

MODEL = os.path.join(os.path.dirname(gat.demo.__file__), "model.ifc")

_CACHE: dict = {}


def _fixture() -> dict:
    if not _CACHE:
        session = GatSession.load_ifc(MODEL)
        _CACHE["session"] = session
        _CACHE["scene"] = derive_scene(session.world)
    return _CACHE


class TestMomentMatch(unittest.TestCase):
    def test_hand_mixture_preserved_to_1e12(self):
        # Hand mixture: w = (1, 3); mu = (0,0,0), (4,0,0);
        # S1 = I, S2 = diag(2, 1, 1).
        # Total weight W = 4; fractions (1/4, 3/4).
        # mean = 1/4*(0,0,0) + 3/4*(4,0,0) = (3, 0, 0).
        # cov_xx = 1/4*1 + 3/4*2  (within-component)
        #        + 1/4*(0-3)^2 + 3/4*(4-3)^2  (between)
        #        = 0.25 + 1.5 + 2.25 + 0.75 = 4.75
        # cov_yy = cov_zz = 1; off-diagonals 0.
        weights = np.array([1.0, 3.0])
        means = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
        covs = np.array([np.eye(3), np.diag([2.0, 1.0, 1.0])])
        w, mu, S = moment_match(weights, means, covs)
        self.assertAlmostEqual(w, 4.0, delta=1e-12)
        self.assertLess(np.abs(mu - np.array([3.0, 0.0, 0.0])).max(), 1e-12)
        expected_cov = np.diag([4.75, 1.0, 1.0])
        self.assertLess(np.abs(S - expected_cov).max(), 1e-12)

    def test_random_mixture_second_moment_preserved(self):
        # Independent oracle: E[x x^T] of the mixture must be preserved:
        # sum_k frac_k (S_k + mu_k mu_k^T) == S + mu mu^T.
        rng = np.random.default_rng(11)
        weights = rng.random(5) + 0.1
        means = rng.normal(size=(5, 3))
        L = rng.normal(size=(5, 3, 3)) * 0.3
        covs = L @ np.swapaxes(L, 1, 2) + np.eye(3)[None] * 0.5
        w, mu, S = moment_match(weights, means, covs)
        frac = weights / weights.sum()
        second_in = np.einsum("k,kij->ij", frac, covs) + np.einsum(
            "k,ki,kj->ij", frac, means, means
        )
        second_out = S + np.outer(mu, mu)
        self.assertLess(np.abs(second_in - second_out).max(), 1e-12)
        self.assertAlmostEqual(w, float(weights.sum()), delta=1e-12)
        self.assertLess(np.abs(mu - frac @ means).max(), 1e-12)


class TestKlGauss(unittest.TestCase):
    def test_zero_for_identical_gaussians(self):
        mu = np.array([1.0, -2.0, 0.5])
        S = np.array([[0.5, 0.1, 0.0], [0.1, 0.4, 0.05], [0.0, 0.05, 0.3]])
        self.assertLess(abs(kl_gauss(mu, S, mu, S)), 1e-12)

    def test_mean_shift_oracle(self):
        # KL(N(d, I) || N(0, I)) = 0.5 * |d|^2 = 0.5 for |d| = 1.
        mu0 = np.array([1.0, 0.0, 0.0])
        mu1 = np.zeros(3)
        self.assertAlmostEqual(kl_gauss(mu0, np.eye(3), mu1, np.eye(3)), 0.5, delta=1e-12)

    def test_covariance_scale_oracle(self):
        # KL(N(0, 2I) || N(0, I)) = 0.5*(trace(2I) - 3 + ln det I - ln det 2I)
        #                        = 0.5*(6 - 3 - 3 ln 2) = 1.5 - 1.5 ln 2.
        kl = kl_gauss(np.zeros(3), 2.0 * np.eye(3), np.zeros(3), np.eye(3))
        self.assertAlmostEqual(kl, 1.5 - 1.5 * math.log(2.0), delta=1e-12)

    def test_positive_when_different(self):
        kl = kl_gauss(
            np.zeros(3), np.eye(3), np.array([0.2, 0.0, 0.0]), np.diag([1.1, 0.9, 1.0])
        )
        self.assertGreater(kl, 0.0)


class TestElementLevel(unittest.TestCase):
    def test_node_moments_are_exact_box_moments(self):
        # Gaussianization is moment-matched per sub-box and the sub-boxes
        # partition the box, so the per-element merged Gaussian must equal
        # the box's own uniform moments: mean = center,
        # cov = R diag(E^2 / 12) R^T, weight = volume.
        fx = _fixture()
        scene = fx["scene"]
        nodes = element_level(scene)
        self.assertEqual(len(nodes), len(scene.elements))
        for node, element in zip(nodes, scene.elements):
            self.assertEqual(node.label, element.name)
            R = rot_z(element.box.angle)
            E = np.asarray(element.box.extents)
            expected_cov = R @ np.diag(E**2 / 12.0) @ R.T
            self.assertLess(np.abs(node.cov - expected_cov).max(), 1e-12)
            self.assertLess(np.abs(node.mean - element.box.center()).max(), 1e-12)
            self.assertAlmostEqual(node.weight, element.box.volume, delta=1e-12)
            self.assertGreaterEqual(node.merge_error, 0.0)


class TestFrameTransform(unittest.TestCase):
    def test_moments_affine_and_weights_scale_by_det(self):
        fx = _fixture()
        cloud = fx["scene"].cloud
        A = np.array([[1.2, 0.3, 0.0], [0.0, 0.9, 0.1], [0.0, 0.0, 1.1]])
        b = np.array([100.0, -50.0, 7.0])
        det = 1.2 * 0.9 * 1.1  # = 1.188 (upper triangular)
        ft = FrameTransform(A=A, b=b)
        moved = ft.apply_cloud(cloud)

        m0, c0 = cloud.mixture_moments()
        m1, c1 = moved.mixture_moments()
        self.assertLess(np.abs(m1 - (A @ m0 + b)).max(), 1e-9)
        self.assertLess(np.abs(c1 - A @ c0 @ A.T).max(), 1e-9)
        # Weights scale by |det A| (measure transport, not a pdf).
        self.assertLess(np.abs(moved.weights - cloud.weights * det).max(), 1e-12)
        # Per-primitive means transform affinely too.
        self.assertLess(np.abs(moved.means - (cloud.means @ A.T + b)).max(), 1e-9)


class TestCompliance(unittest.TestCase):
    def test_demo_world_passes(self):
        fx = _fixture()
        report = check_compliance(fx["session"].world)
        self.assertTrue(report.passed)
        self.assertGreater(len(report.rows), 0)

    def test_fails_after_lowering_clear_height(self):
        fx = _fixture()
        session = fx["session"]
        world = session.world
        ch = session.var("Level 1", "ClearHeight")
        result = execute(world, SetParameter(ch, 2.3, 0.01), strict=False)
        self.assertTrue(result.committed)  # geometrically feasible change
        report = check_compliance(result.world)  # min_clear_height = 2.4
        self.assertFalse(report.passed)
        failing = [r for r in report.rows if r.status == "FAIL"]
        self.assertTrue(failing)
        self.assertTrue(all(r.rule == "min-clear-height" for r in failing))
        for r in failing:
            # margin mean = 2.3 - 2.4 = -0.1
            self.assertAlmostEqual(r.margin_mean, -0.1, delta=1e-9)


class TestRayDepth(unittest.TestCase):
    def test_batched_depth_matches_scalar_sum(self):
        fx = _fixture()
        scene = fx["scene"]
        prims = scene.cloud.of_element(scene.element_by_name("Wall-Party").row)
        origin = np.array([4.5, 2.0, 1.5])
        direction = np.array([1.0, 0.0, 0.0])  # unit
        total = ray_optical_depth(
            prims.means, prims.covs, prims.weights, origin, direction, 2.0, kappa=8.0
        )
        scalar_sum = 0.0
        for k in range(len(prims)):
            # cov is passed as a zero-eps Dual: the plain-ndarray cov branch
            # of scalar_ray_depth_dual is broken (see test_plain_cov_path).
            out = scalar_ray_depth_dual(
                prims.means[k],
                Dual(prims.covs[k]),
                float(prims.weights[k]),
                origin,
                direction,
                2.0,
                kappa=8.0,
            )
            scalar_sum += float(np.asarray(out.val).reshape(()))
        self.assertGreater(total, 0.0)
        self.assertLess(abs(total - scalar_sum), 1e-9)

    def test_plain_cov_path(self):
        # A plain ndarray cov is lifted to a zero-eps Dual inside
        # scalar_ray_depth_dual (and Dual carries __array_priority__ so
        # ndarray @ Dual defers correctly) — the value path works without
        # callers lifting anything themselves.
        out = scalar_ray_depth_dual(
            np.zeros(3),
            np.eye(3) * 0.1,
            1.0,
            np.array([-1.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            2.0,
        )
        self.assertGreater(float(np.asarray(out.val).reshape(())), 0.0)

    def test_dual_gradient_wrt_mu_y_matches_central_fd(self):
        mu = np.array([0.3, 0.1, -0.2])
        cov = np.array(
            [[0.05, 0.01, 0.0], [0.01, 0.04, 0.005], [0.0, 0.005, 0.06]]
        )
        weight = 2.0
        origin = np.array([-1.0, 0.35, 0.0])
        direction = np.array([1.0, 0.0, 0.0])
        length, kappa = 2.5, 3.0

        seed = Dual(mu, np.array([0.0, 1.0, 0.0]))
        # cov must be a Dual (zero eps) — the plain-ndarray cov branch is
        # broken; see test_plain_cov_path.
        out = scalar_ray_depth_dual(
            seed, Dual(cov), weight, origin, direction, length, kappa
        )
        dual_grad = float(np.asarray(out.eps).reshape(()))

        h = 1e-6
        def depth(m):
            return ray_optical_depth(
                m[None], cov[None], np.array([weight]), origin, direction, length, kappa
            )

        fd = (
            depth(mu + h * np.array([0.0, 1.0, 0.0]))
            - depth(mu - h * np.array([0.0, 1.0, 0.0]))
        ) / (2.0 * h)
        self.assertGreater(abs(dual_grad), 1e-4)  # informative direction
        self.assertLess(abs(dual_grad - fd), 1e-5 * max(1.0, abs(fd)))


class TestOptimizeLayout(unittest.TestCase):
    def test_demo_g6_configuration(self):
        # The G6 configuration from gat/demo/geometry.py, with max_iter=20.
        fx = _fixture()
        session = fx["session"]
        world = session.world
        opening = session.entity_by_name("Opening-1")
        door = session.entity_by_name("Door-1")
        storey = session.entity_by_name("Level 1")
        office_b = session.entity_by_name("Office-B")
        wall_party = session.entity_by_name("Wall-Party")
        objective = LayoutObjective(
            cost_var=VarId(storey, "TotalWallCost"),
            daylight_area_var=VarId(opening, "Area"),
            daylight_floor_var=VarId(office_b, "FloorArea"),
            energy_terms=(
                (VarId(session.entity_by_name("Wall-South"), "GrossSideArea"), 0.25),
                (VarId(session.entity_by_name("Wall-North"), "GrossSideArea"), 0.25),
                (VarId(session.entity_by_name("Wall-West"), "GrossSideArea"), 0.25),
                (VarId(session.entity_by_name("Wall-East"), "GrossSideArea"), 0.25),
            ),
            daylight_target=0.10,
            constraints=(
                LessEqual(VarId(door, "Width"), VarId(opening, "Width")),
                LessEqual(VarId(door, "Height"), VarId(opening, "Height")),
                LessEqual(VarId(opening, "Height"), VarId(wall_party, "Height")),
                LessEqual(VarId(opening, "Width"), VarId(wall_party, "Length")),
            ),
        )
        params = (VarId(opening, "Width"), VarId(opening, "Height"))
        result = optimize_layout(world, params, objective, max_iter=20)

        self.assertLess(result.objective_final, result.objective_initial)
        self.assertTrue(result.trajectory)  # at least one accepted step
        # Door-width chance margin held: opening W - door W > 0.02.
        door_w = world.full.mean(VarId(door, "Width"))
        self.assertGreater(result.optimized[0] - door_w, 0.02)


if __name__ == "__main__":
    unittest.main()
