"""Tests for gat/geometry/registration.py — scan-to-BIM GMM registration.

Uses a small synthetic scan (700 points, 1% cm sensor noise, 2% outliers,
seed 3) with a withheld ground-truth pose of yaw 20 deg and translation
(0.2, -0.1, 0.03), and asserts: pose recovery within 0.5 deg / 60 mm,
monotone NLL traces in both annealing stages, bitwise determinism of
register(), a symmetric positive-definite information matrix,
deterministic scan synthesis, and rejection of degenerate scans.
"""

from __future__ import annotations

from dataclasses import replace
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

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
        self.assertEqual(a.scan_digest, b.scan_digest)
        self.assertEqual(a.scene_version, b.scene_version)
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

    def test_rejects_wrong_shape(self):
        with self.assertRaisesRegex(RegistrationError, "shape"):
            self.registrar.register(np.zeros((10, 2)))

    def test_rejects_non_finite_coordinates(self):
        bad = np.zeros((10, 3))
        bad[4, 1] = np.nan
        with self.assertRaisesRegex(RegistrationError, "non-finite"):
            self.registrar.register(bad)


class TestScanEvidence(RegistrationTestBase):
    def test_responsibility_mass_is_conserved_across_elements(self):
        report = self.registrar.evidence(self.scan, self.result)
        element_mass = sum(e.effective_points for e in report.elements)
        self.assertAlmostEqual(element_mass, report.inlier_effective_points, places=10)
        self.assertAlmostEqual(
            sum(e.responsibility_fraction for e in report.elements), 1.0, places=12
        )
        self.assertAlmostEqual(
            report.inlier_effective_points
            + report.outlier_fraction * report.point_count,
            report.point_count,
            places=10,
        )

    def test_metrics_are_bounded_and_finite(self):
        report = self.registrar.evidence(self.scan, self.result)
        solid_rows = {e.row for e in self.scene.elements if e.is_solid}
        self.assertEqual({e.element_row for e in report.elements}, solid_rows)
        self.assertGreater(report.inlier_effective_points, 0.0)
        self.assertGreaterEqual(report.outlier_fraction, 0.0)
        self.assertLessEqual(report.outlier_fraction, 1.0)
        for evidence in report.elements:
            self.assertGreater(evidence.primitive_count, 0)
            self.assertGreaterEqual(evidence.effective_points, 0.0)
            self.assertTrue(math.isfinite(evidence.mean_mahalanobis2))
            self.assertGreaterEqual(evidence.mean_mahalanobis2, 0.0)
            self.assertGreaterEqual(evidence.support_diversity, 0.0)
            self.assertLessEqual(evidence.support_diversity, 1.0 + 1e-12)
            self.assertGreaterEqual(evidence.assignment_confidence, 0.0)
            self.assertLessEqual(evidence.assignment_confidence, 1.0 + 1e-12)

    def test_report_is_deterministic_and_bound_to_provenance(self):
        first = self.registrar.evidence(self.scan, self.result)
        second = self.registrar.evidence(self.scan, self.result)
        self.assertEqual(first, second)
        self.assertEqual(first.scan_digest, self.result.scan_digest)
        self.assertEqual(first.scene_version, self.scene.version)

    def test_rejected_fit_produces_no_evidence(self):
        rejected = replace(self.result, accepted=False)
        with self.assertRaisesRegex(RegistrationError, "failed the fit gate"):
            self.registrar.evidence(self.scan, rejected)

    def test_different_scan_cannot_reuse_registration(self):
        different = self.scan.copy()
        different[0, 0] += 1e-12
        with self.assertRaisesRegex(RegistrationError, "differs"):
            self.registrar.evidence(different, self.result)

    def test_different_scene_version_cannot_reuse_registration(self):
        stale = replace(self.result, scene_version="not-this-scene")
        with self.assertRaisesRegex(RegistrationError, "different scene"):
            self.registrar.evidence(self.scan, stale)


class TestPlyHandoff(RegistrationTestBase):
    def test_register_ply_loads_vertices_then_delegates_to_same_gate(self):
        # The PLY adapter does not invent a second registration path.  It
        # decodes the producer artifact, then passes the same points and
        # caller-provided gates to ScanRegistrar.register().
        points = self.scan[:10]
        header = (
            "ply\nformat binary_little_endian 1.0\nelement vertex 10\n"
            "property float x\nproperty float y\nproperty float z\nend_header\n"
        ).encode("ascii")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "external_reconstruction.ply"
            path.write_bytes(header + np.asarray(points, dtype="<f4").tobytes())
            with patch.object(self.registrar, "register", return_value=self.result) as register:
                actual = self.registrar.register_ply(str(path), n_starts=4, accept_nll=2.5)

        self.assertIs(actual, self.result)
        register.assert_called_once()
        loaded, n_starts, accept_nll = register.call_args.args
        # The fixture deliberately serializes float32 PLY coordinates; the
        # loader promotes them to float64 without claiming lost source bits.
        np.testing.assert_allclose(loaded, points, rtol=0.0, atol=1e-6)
        self.assertEqual(n_starts, 4)
        self.assertEqual(accept_nll, 2.5)


if __name__ == "__main__":
    unittest.main()
