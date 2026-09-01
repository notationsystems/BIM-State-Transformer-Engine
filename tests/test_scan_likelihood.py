"""Tests for calibrated scan-to-clearance likelihood adaptation."""

from __future__ import annotations

from dataclasses import replace
import os
import unittest

import numpy as np

import gat.demo
from gat.engine.decision import DecisionVerdict
from gat.engine.executor import execute
from gat.engine.stability import analyze
from gat.engine.transform import ShiftParameter
from gat.errors import BindingError, LikelihoodCalibrationError
from gat.geometry import (
    ClearanceDecision,
    ClearanceLikelihoodCalibration,
    IndependentPoseCalibration,
    OrientedBox,
    RegistrationResult,
    RigidTransformZ,
    ScanRegistrar,
    adapt_clearance_likelihood,
    assess_clearance,
    derive_scene,
    plan_clearance_evidence,
)
from gat.geometry.registration import (
    ElementScanEvidence,
    ScanEvidenceReport,
    _scan_digest,
)
from gat.session import GatSession
from gat.workflows import (
    AcceptanceCase,
    AcceptanceDisposition,
    EvidenceReceipt,
    WorkflowKind,
    clearance_check,
    evaluate_acceptance_case,
)


MODEL = os.path.join(os.path.dirname(gat.demo.__file__), "model.ifc")


class ScanLikelihoodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.session = GatSession.load_ifc(MODEL)
        cls.scene = derive_scene(cls.session.world)
        cls.registrar = ScanRegistrar(cls.scene)

        rng = np.random.default_rng(4)
        count = 180
        # Independent survey points on the Wall-Party top face, whose true
        # support is 15 mm below the current BIM mean.
        cls.scan = np.column_stack(
            [
                np.full(count, 5.1) + rng.normal(0.0, 0.002, count),
                rng.uniform(0.35, 3.65, count),
                np.full(count, 2.985) + rng.normal(0.0, 0.003, count),
            ]
        )
        cls.scan_digest = _scan_digest(cls.scan)
        cls.registration = RegistrationResult(
            transform=RigidTransformZ(0.0, (0.0, 0.0, 0.0)),
            nll=0.0,
            nll_trace=(),
            coarse_trace=(),
            start_nlls=(),
            info_matrix=np.diag([1.0e4] * 4),
            accepted=True,
            scan_digest=cls.scan_digest,
            scene_version=cls.scene.version,
        )
        party = cls.scene.element_by_name("Wall-Party")
        cls.evidence = ScanEvidenceReport(
            scan_digest=cls.scan_digest,
            scene_version=cls.scene.version,
            point_count=count,
            inlier_effective_points=170.0,
            outlier_fraction=0.01,
            elements=(
                ElementScanEvidence(
                    element_row=party.row,
                    element_name=party.name,
                    primitive_count=20,
                    effective_points=170.0,
                    responsibility_fraction=1.0,
                    mean_mahalanobis2=2.0,
                    support_diversity=0.80,
                    assignment_confidence=0.98,
                ),
            ),
        )
        cls.assessment = assess_clearance(
            cls.scene,
            ClearanceDecision(
                proposed=OrientedBox(
                    (4.0, 1.8, 3.06), 0.0, (3.0, 0.4, 0.4)
                ),
                required_clearance=0.05,
                confidence=0.95,
                position_sigma=0.002,
                label="survey-conditioned route",
            ),
        )
        cls.plan = plan_clearance_evidence(cls.assessment, cls.evidence)
        cls.pose = IndependentPoseCalibration(
            transform=RigidTransformZ(0.0, (0.0, 0.0, 0.0)),
            covariance=np.diag([1.0e-8, 1.0e-6, 1.0e-6, 1.0e-6]),
            scan_digest=cls.scan_digest,
            source_id="survey-control-A",
        )

    def likelihood(self, **kwargs):
        return adapt_clearance_likelihood(
            self.scene,
            self.registrar,
            self.scan,
            self.registration,
            kwargs.pop("evidence", self.evidence),
            kwargs.pop("plan", self.plan),
            kwargs.pop("pose", self.pose),
            kwargs.pop("calibration", ClearanceLikelihoodCalibration()),
            **kwargs,
        )

    def test_extracts_controlling_support_face_not_centroid(self) -> None:
        likelihood = self.likelihood()
        self.assertEqual(likelihood.element_name, "Wall-Party")
        self.assertEqual(likelihood.direction, (0.0, 0.0, 1.0))
        self.assertAlmostEqual(likelihood.predicted_support, 3.0, delta=1e-12)
        self.assertAlmostEqual(likelihood.observed_support, 2.985, delta=0.002)
        self.assertGreater(likelihood.effective_face_points, 150.0)
        self.assertGreater(likelihood.face_assignment_confidence, 0.99)
        self.assertGreater(likelihood.tangent_rms, 0.5)

    def test_uncertainty_budget_includes_all_three_sources(self) -> None:
        likelihood = self.likelihood()
        self.assertGreater(likelihood.sampling_sigma, 0.0)
        self.assertGreater(likelihood.pose_sigma, 0.0)
        self.assertGreater(likelihood.calibration_sigma, 0.0)
        expected = np.sqrt(
            likelihood.sampling_sigma**2
            + likelihood.pose_sigma**2
            + likelihood.calibration_sigma**2
        )
        self.assertAlmostEqual(likelihood.noise_sigma, expected, delta=1e-15)

    def test_condition_propagate_verify_closes_clearance_decision(self) -> None:
        self.assertEqual(self.assessment.verdict, DecisionVerdict.UNRESOLVED)
        likelihood = self.likelihood()
        clear_height = likelihood.observation.target_vars()[0]
        prior_sigma = self.scene.world.belief.std(clear_height)

        result = execute(self.scene.world, likelihood.observation)

        self.assertTrue(result.committed)
        self.assertTrue(result.report.passed)
        self.assertLess(result.world.belief.std(clear_height), prior_sigma)
        updated_scene = derive_scene(result.world)
        resolved = assess_clearance(updated_scene, self.assessment.decision)
        self.assertEqual(resolved.verdict, DecisionVerdict.SATISFIED)
        self.assertLessEqual(resolved.p_any_violation_upper, 0.05)

    def test_linearized_observation_is_rejected_after_prior_changes(self) -> None:
        likelihood = self.likelihood()
        target = likelihood.observation.target_vars()[0]
        changed = execute(
            self.scene.world, ShiftParameter(target, 0.001)
        ).world
        with self.assertRaisesRegex(BindingError, "stale"):
            execute(changed, likelihood.observation)

    def test_observation_participates_in_stability_analysis(self) -> None:
        report = analyze(self.scene.world, [self.likelihood().observation])
        self.assertLess(report.energy_trace[-1], report.energy_trace[0])
        self.assertLess(report.sigma_min, 1.0)

    def test_pose_uncertainty_increases_likelihood_noise(self) -> None:
        baseline = self.likelihood()
        noisier_pose = replace(
            self.pose,
            covariance=np.diag([1.0e-8, 1.0e-6, 1.0e-6, 9.0e-6]),
            source_id="survey-control-B",
        )
        noisier = self.likelihood(pose=noisier_pose)
        self.assertGreater(noisier.pose_sigma, baseline.pose_sigma)
        self.assertGreater(noisier.noise_sigma, baseline.noise_sigma)

    def test_pose_disagreement_is_rejected(self) -> None:
        mismatched = replace(
            self.pose,
            transform=RigidTransformZ(0.0, (0.0, 0.0, 0.10)),
        )
        with self.assertRaisesRegex(LikelihoodCalibrationError, "disagrees"):
            self.likelihood(pose=mismatched)

    def test_scan_provenance_mismatch_is_rejected(self) -> None:
        mismatched = replace(self.pose, scan_digest="other-scan")
        with self.assertRaisesRegex(LikelihoodCalibrationError, "digests differ"):
            self.likelihood(pose=mismatched)

    def test_low_quality_element_evidence_is_rejected(self) -> None:
        weak_row = replace(self.evidence.elements[0], effective_points=4.0)
        weak = replace(self.evidence, elements=(weak_row,))
        with self.assertRaisesRegex(LikelihoodCalibrationError, "effective points"):
            self.likelihood(evidence=weak)

    def test_face_coverage_and_innovation_gates_are_enforced(self) -> None:
        with self.assertRaisesRegex(LikelihoodCalibrationError, "support face"):
            self.likelihood(
                calibration=replace(
                    ClearanceLikelihoodCalibration(),
                    min_face_effective_points=1000.0,
                )
            )
        with self.assertRaisesRegex(LikelihoodCalibrationError, "innovation gate"):
            self.likelihood(
                calibration=replace(
                    ClearanceLikelihoodCalibration(), max_innovation_sigma=1.0
                )
            )

    def test_adaptation_is_deterministic_and_does_not_mutate_world(self) -> None:
        digest_before = self.scene.world.digest()
        first = self.likelihood()
        second = self.likelihood()
        self.assertEqual(self.scene.world.digest(), digest_before)
        self.assertEqual(first.evidence_digest, second.evidence_digest)
        self.assertEqual(first.observed_support, second.observed_support)
        self.assertTrue(
            np.array_equal(first.observation.row, second.observation.row)
        )

    def test_verified_scan_transition_can_close_an_as_built_acceptance_case(self) -> None:
        likelihood = self.likelihood()
        session = GatSession(self.scene.world)
        result = session.run(
            likelihood.observation,
            provenance={
                "evidence_kind": "calibrated-scan-clearance-likelihood",
                "calibration_id": likelihood.pose_source_id,
                "scan_digest": likelihood.scan_digest,
                "check_ids": ["route-clearance"],
            },
        )
        receipt = EvidenceReceipt.from_scan_likelihood(
            likelihood,
            result,
            session.ledger.events[-1],
            ("route-clearance",),
        )
        posterior = assess_clearance(
            derive_scene(session.world), self.assessment.decision
        )
        case = AcceptanceCase(
            "as-built-route-1",
            WorkflowKind.AS_BUILT_CLEARANCE,
            "survey-conditioned route",
            (clearance_check("route-clearance", posterior),),
        )
        outcome = evaluate_acceptance_case(case, (receipt,))

        self.assertEqual(posterior.verdict, DecisionVerdict.SATISFIED)
        self.assertEqual(outcome.disposition, AcceptanceDisposition.ACCEPT)
        self.assertTrue(outcome.may_authorize)
        self.assertEqual(outcome.evidence_receipt_ids, (receipt.receipt_id,))


if __name__ == "__main__":
    unittest.main()
