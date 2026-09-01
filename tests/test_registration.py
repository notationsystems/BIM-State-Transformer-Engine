"""Tests for gat/geometry/registration.py — scan-to-BIM GMM registration.

Uses a small synthetic scan (700 points, 1% cm sensor noise, 2% outliers,
seed 3) with a withheld ground-truth pose of yaw 20 deg and translation
(0.2, -0.1, 0.03), and asserts: pose recovery within 0.5 deg / 60 mm,
monotone NLL traces in both annealing stages, bitwise determinism of
register(), a symmetric positive-definite information matrix,
deterministic scan synthesis, and rejection of degenerate scans.
"""

from __future__ import annotations

import math
import os
import unittest

import numpy as np

import gat.demo
from gat.errors import RegistrationError
from gat.geometry.registration import (
    RigidTransformZ,
    ScanRegistrar,
    synthesize_scan,
)
from gat.geometry.stateio import derive_scene
from gat.session import GatSession

MODEL = os.path.join(os.path.dirname(gat.demo.__file__), "model.ifc")

TRUTH = RigidTransformZ(theta=math.radians(20.0), t=(0.2, -0.1, 0.03))

_CACHE: dict = {}


def _fixture() -> dict:
    """Shared expensive fixture, computed once per test process."""
    if not _CACHE:
        session = GatSession.load_ifc(MODEL)
        scene = derive_scene(session.world)
        scan = synthesize_scan(
            scene,
            n_points=700,
            noise_sigma=0.01,
            outlier_frac=0.02,
            transform=TRUTH,
            seed=3,
        )
        registrar = ScanRegistrar(scene)
        result = registrar.register(scan)
        result_again = registrar.register(scan)
        _CACHE.update(
            scene=scene,
            scan=scan,
            registrar=registrar,
            result=result,
            result_again=result_again,
        )
    return _CACHE


class RegistrationTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fx = _fixture()
        cls.scene = fx["scene"]
        cls.scan = fx["scan"]
        cls.registrar = fx["registrar"]
        cls.result = fx["result"]
        cls.result_again = fx["result_again"]


class TestRecovery(RegistrationTestBase):
    def test_pose_recovered(self):
        yaw_err, trans_err = self.result.transform.compose_error(TRUTH)
        self.assertLess(yaw_err, math.radians(0.5))
        self.assertLess(trans_err, 0.060)  # 60 mm

    def test_fit_accepted(self):
        self.assertTrue(self.result.accepted)

    def test_traces_monotone(self):
        for stage in (self.result.coarse_trace, self.result.nll_trace):
            diffs = np.diff(np.asarray(stage))
            self.assertTrue(
                (diffs <= 1e-9).all(),
                f"non-monotone stage trace: max increase {diffs.max()!r}",
            )

    def test_deterministic_register(self):
        a, b = self.result, self.result_again
        self.assertEqual(a.transform.theta, b.transform.theta)
        self.assertEqual(a.transform.t, b.transform.t)
        self.assertEqual(a.nll, b.nll)
        self.assertEqual(a.nll_trace, b.nll_trace)
        self.assertEqual(a.coarse_trace, b.coarse_trace)
        self.assertEqual(a.start_nlls, b.start_nlls)
        self.assertTrue(np.array_equal(a.info_matrix, b.info_matrix))


class TestInformationMatrix(RegistrationTestBase):
    def test_symmetric(self):
        H = self.result.info_matrix
        self.assertEqual(H.shape, (4, 4))
        self.assertLess(np.abs(H - H.T).max(), 1e-9 * max(1.0, np.abs(H).max()))

    def test_positive_definite(self):
        eigvals = np.linalg.eigvalsh(self.result.info_matrix)
        self.assertGreater(eigvals.min(), 0.0)

    def test_pose_sigma_finite_positive(self):
        sig = self.result.pose_sigma()
        self.assertTrue(np.isfinite(sig).all())
        self.assertTrue((sig > 0).all())


class TestSynthesizeScan(RegistrationTestBase):
    def test_deterministic_for_fixed_seed(self):
        again = synthesize_scan(
            self.scene,
            n_points=700,
            noise_sigma=0.01,
            outlier_frac=0.02,
            transform=TRUTH,
            seed=3,
        )
        self.assertTrue(np.array_equal(self.scan, again))

    def test_different_for_different_seeds(self):
        other = synthesize_scan(
            self.scene,
            n_points=700,
            noise_sigma=0.01,
            outlier_frac=0.02,
            transform=TRUTH,
            seed=4,
        )
        self.assertFalse(np.array_equal(self.scan, other))

    def test_point_count(self):
        self.assertEqual(self.scan.shape, (700, 3))


class TestDegenerateInput(RegistrationTestBase):
    def test_rejects_five_point_scan(self):
        rng = np.random.default_rng(0)
        tiny = rng.random((5, 3))
        with self.assertRaises(RegistrationError):
            self.registrar.register(tiny)


if __name__ == "__main__":
    unittest.main()
