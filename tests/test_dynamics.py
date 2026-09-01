"""Exact temporal linear-Gaussian process dynamics and ledger replay."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import unittest

import numpy as np

import gat.demo
from gat.engine.dynamics import EvolveLinearGaussian, forecast_process
from gat.engine.executor import execute
from gat.engine.stability import analyze
from gat.engine.transform import ObserveQuantity
from gat.errors import BindingError, LedgerError, VerificationError
from gat.ledger import decode_transformation, encode_transformation, replay_ledger
from gat.session import GatSession


MODEL = os.path.join(os.path.dirname(gat.demo.__file__), "model.ifc")
CALIBRATION = hashlib.sha256(b"building-drift-model/calibration-2026-08").hexdigest()


class LinearGaussianDynamicsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session = GatSession.load_ifc(MODEL)
        self.initial = self.session.world
        self.height = self.session.var("Level 1", "ClearHeight")
        self.length = self.session.var("Office-A", "Length")
        self.width = self.session.var("Office-A", "Width")

    def process(
        self,
        *,
        targets=None,
        transition=None,
        offset=None,
        covariance=None,
        elapsed=3600.0,
    ) -> EvolveLinearGaussian:
        targets = tuple(targets or (self.height,))
        n = len(targets)
        return EvolveLinearGaussian(
            targets,
            np.eye(n) if transition is None else np.asarray(transition, dtype=float),
            np.zeros(n) if offset is None else np.asarray(offset, dtype=float),
            np.zeros((n, n)) if covariance is None else np.asarray(covariance, dtype=float),
            elapsed,
            "building-drift-v1",
            CALIBRATION,
        )

    def test_scalar_transition_has_exact_mean_variance_and_cross_covariance(self) -> None:
        # First create a real posterior correlation between height and room length.
        self.session.run(
            ObserveQuantity.single(
                self.session.var("Office-A", "Volume"), 59.4, 0.05
            )
        )
        before = self.session.world
        row_h = before.binding.raw_index.row(self.height)
        row_l = before.binding.raw_index.row(self.length)
        prior_mean = before.belief.mu[row_h]
        prior_var = before.belief.sigma[row_h, row_h]
        prior_cross = before.belief.sigma[row_h, row_l]
        self.assertNotEqual(prior_cross, 0.0)
        process = self.process(
            transition=[[0.9]], offset=[0.3], covariance=[[4.0e-6]]
        )

        result = execute(before, process)

        self.assertTrue(result.committed)
        self.assertAlmostEqual(
            result.world.belief.mu[row_h], 0.9 * prior_mean + 0.3, delta=1e-15
        )
        self.assertAlmostEqual(
            result.world.belief.sigma[row_h, row_h],
            0.9**2 * prior_var + 4.0e-6,
            delta=1e-18,
        )
        self.assertAlmostEqual(
            result.world.belief.sigma[row_h, row_l], 0.9 * prior_cross, delta=1e-18
        )
        self.assertTrue(result.report.passed)

    def test_multivariate_transition_matches_full_matrix_oracle(self) -> None:
        targets = (self.length, self.width)
        A = np.array([[0.98, 0.01], [-0.02, 1.01]])
        b = np.array([0.02, -0.01])
        Q = np.array([[4.0e-6, 1.0e-6], [1.0e-6, 9.0e-6]])
        process = self.process(
            targets=targets, transition=A, offset=b, covariance=Q, elapsed=86400.0
        )
        before = self.session.world
        rows = before.binding.raw_index.rows(targets)
        F = np.eye(before.binding.n_raw)
        F[np.ix_(rows, rows)] = A
        expected_mu = F @ before.belief.mu
        expected_mu[rows] += b
        expected_sigma = F @ before.belief.sigma @ F.T
        expected_sigma[np.ix_(rows, rows)] += Q
        expected_sigma = 0.5 * (expected_sigma + expected_sigma.T)

        result = execute(before, process)

        self.assertTrue(np.array_equal(result.world.belief.mu, expected_mu))
        self.assertTrue(np.array_equal(result.world.belief.sigma, expected_sigma))

    def test_forecast_is_non_mutating_and_matches_committed_steps(self) -> None:
        process = self.process(offset=[0.001], covariance=[[1.0e-6]], elapsed=60.0)
        digest = self.session.world.digest()
        rollout = forecast_process(self.session.world, process, steps=3)
        self.assertEqual(self.session.world.digest(), digest)
        self.assertEqual(rollout.elapsed_seconds, 180.0)
        self.assertEqual(len(rollout.steps), 3)
        for _ in range(3):
            self.session.run(process)
        self.assertEqual(rollout.final_world.digest(), self.session.world.digest())
        self.assertTrue(all(step.verification.passed for step in rollout.steps))

    def test_process_noise_increases_uncertainty_under_identity_dynamics(self) -> None:
        before = self.session.world.belief.var_of(self.height)
        result = execute(
            self.session.world, self.process(covariance=[[2.5e-5]])
        )
        self.assertAlmostEqual(
            result.world.belief.var_of(self.height), before + 2.5e-5, delta=1e-18
        )

    def test_process_is_in_closed_codec_and_replays_exactly(self) -> None:
        process = self.process(
            transition=[[0.99]], offset=[0.03], covariance=[[1.0e-6]], elapsed=30.0
        )
        encoded = encode_transformation(process)
        decoded = decode_transformation(encoded)
        self.assertEqual(encode_transformation(decoded), encoded)
        self.session.run(process, provenance={"clock": "building-controller-A"})
        replay = replay_ledger(self.initial, self.session.ledger)
        self.assertEqual(replay.accepted, 1)
        self.assertEqual(replay.world.digest(), self.session.world.digest())
        self.assertTrue(
            np.array_equal(replay.world.belief.sigma, self.session.world.belief.sigma)
        )

    def test_stability_uses_the_exact_process_transition(self) -> None:
        amplifying = self.process(transition=[[1.2]])
        report = analyze(self.session.world, [amplifying])
        self.assertEqual(report.verdict, "amplifying")
        self.assertAlmostEqual(report.sigma_max, 1.2, delta=1e-12)

    def test_invalid_process_contracts_fail_before_execution(self) -> None:
        with self.assertRaisesRegex(ValueError, "symmetric"):
            self.process(
                targets=(self.length, self.width),
                covariance=[[1.0, 1.0], [0.0, 1.0]],
            )
        with self.assertRaisesRegex(ValueError, "PSD"):
            self.process(covariance=[[-1.0]])
        with self.assertRaisesRegex(ValueError, "positive"):
            self.process(elapsed=0.0)
        with self.assertRaisesRegex(ValueError, "digest"):
            EvolveLinearGaussian(
                (self.height,), np.eye(1), np.zeros(1), np.zeros((1, 1)),
                1.0, "model", "not-a-digest",
            )

    def test_arrays_are_immutable_values(self) -> None:
        process = self.process(covariance=[[1.0e-6]])
        with self.assertRaises(ValueError):
            process.transition[0, 0] = 2.0
        with self.assertRaises(ValueError):
            process.process_covariance[0, 0] = 2.0

    def test_derived_target_is_rejected_and_replays_as_same_error(self) -> None:
        derived = self.session.var("Office-A", "Volume")
        process = self.process(targets=(derived,))
        with self.assertRaises(BindingError):
            self.session.run(process)
        replay = replay_ledger(self.initial, self.session.ledger)
        self.assertEqual(replay.rejected, 1)
        self.assertEqual(replay.world.digest(), self.initial.digest())

    def test_invalid_future_state_rolls_back_and_is_replayable(self) -> None:
        opening = self.session.var("Opening-1", "Height")
        process = self.process(targets=(opening,), offset=[1.2])
        with self.assertRaises(VerificationError):
            self.session.run(process)
        self.assertEqual(self.session.world.digest(), self.initial.digest())
        replay = replay_ledger(self.initial, self.session.ledger)
        self.assertEqual(replay.rejected, 1)

    def test_rollout_validates_step_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive integer"):
            forecast_process(self.session.world, self.process(), steps=0)

    def test_unknown_extra_process_field_fails_closed(self) -> None:
        encoded = encode_transformation(self.process())
        encoded["hidden_code"] = "none"
        with self.assertRaisesRegex(LedgerError, "fields differ"):
            decode_transformation(encoded)

    def test_predict_update_demo_runs_in_separate_process(self) -> None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = dict(os.environ)
        env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            [sys.executable, "-m", "gat.demo.temporal_process"],
            cwd=root,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("GAT TEMPORAL PREDICT-UPDATE", proc.stdout)
        self.assertIn("2 accepted transitions", proc.stdout)


if __name__ == "__main__":
    unittest.main()
