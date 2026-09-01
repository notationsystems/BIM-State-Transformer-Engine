"""Tests for decision-focused MEP clearance and scan-evidence routing."""

from __future__ import annotations

import os
import unittest

import gat.demo
from gat.engine.decision import DecisionVerdict, EvidenceDisposition
from gat.errors import DecisionError
from gat.geometry import (
    ClearanceDecision,
    InspectionAction,
    OrientedBox,
    assess_clearance,
    derive_scene,
    plan_clearance_evidence,
)
from gat.geometry.registration import ElementScanEvidence, ScanEvidenceReport
from gat.session import GatSession


MODEL = os.path.join(os.path.dirname(gat.demo.__file__), "model.ifc")


class ClearanceAssuranceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.session = GatSession.load_ifc(MODEL)
        cls.scene = derive_scene(cls.session.world)
        cls.crossing = assess_clearance(
            cls.scene,
            ClearanceDecision(
                OrientedBox((4.0, 1.8, 2.6), 0.0, (3.0, 0.4, 0.4)),
                required_clearance=0.05,
                confidence=0.95,
                position_sigma=0.02,
                label="crossing duct",
            ),
        )
        cls.borderline = assess_clearance(
            cls.scene,
            ClearanceDecision(
                OrientedBox((4.0, 1.8, 3.05), 0.0, (3.0, 0.4, 0.4)),
                required_clearance=0.05,
                confidence=0.95,
                position_sigma=0.02,
                label="borderline duct",
            ),
        )
        cls.clear = assess_clearance(
            cls.scene,
            ClearanceDecision(
                OrientedBox((4.0, 1.8, 3.55), 0.0, (3.0, 0.4, 0.4)),
                required_clearance=0.05,
                confidence=0.95,
                position_sigma=0.02,
                label="clear duct",
            ),
        )

    def evidence(
        self,
        *,
        effective_points: float,
        support_diversity: float,
        assignment_confidence: float,
        scene_version: str | None = None,
    ) -> ScanEvidenceReport:
        party = self.scene.element_by_name("Wall-Party")
        row = ElementScanEvidence(
            element_row=party.row,
            element_name=party.name,
            primitive_count=32,
            effective_points=effective_points,
            responsibility_fraction=1.0,
            mean_mahalanobis2=2.0,
            support_diversity=support_diversity,
            assignment_confidence=assignment_confidence,
        )
        return ScanEvidenceReport(
            scan_digest="test-scan-digest",
            scene_version=scene_version or self.scene.version,
            point_count=100,
            inlier_effective_points=effective_points,
            outlier_fraction=0.0,
            elements=(row,),
        )

    def test_crossing_route_is_violated(self) -> None:
        self.assertEqual(self.crossing.verdict, DecisionVerdict.VIOLATED)
        self.assertGreaterEqual(self.crossing.p_any_violation_lower, 0.95)
        self.assertEqual(self.crossing.worst().element_name, "Wall-Party")

    def test_borderline_route_is_unresolved(self) -> None:
        self.assertEqual(self.borderline.verdict, DecisionVerdict.UNRESOLVED)
        self.assertAlmostEqual(
            self.borderline.p_any_violation_lower, 0.5, delta=1e-12
        )
        self.assertAlmostEqual(
            self.borderline.p_any_violation_upper, 0.5, delta=1e-12
        )
        self.assertEqual(len(self.borderline.risks), 6)

    def test_clear_route_is_satisfied(self) -> None:
        self.assertEqual(self.clear.verdict, DecisionVerdict.SATISFIED)
        self.assertLessEqual(self.clear.p_any_violation_upper, 0.05)

    def test_probability_bounds_do_not_assume_independence(self) -> None:
        for assessment in (self.crossing, self.borderline, self.clear):
            with self.subTest(label=assessment.decision.label):
                individual = [item.p_violates for item in assessment.risks]
                self.assertAlmostEqual(
                    assessment.p_any_violation_lower,
                    max(individual, default=0.0),
                    delta=1e-15,
                )
                self.assertAlmostEqual(
                    assessment.p_any_violation_upper,
                    min(1.0, sum(individual)),
                    delta=1e-15,
                )

    def test_reusable_scan_routes_to_measurement_extraction(self) -> None:
        digest_before = self.session.world.digest()
        plan = plan_clearance_evidence(
            self.borderline,
            self.evidence(
                effective_points=100.0,
                support_diversity=0.80,
                assignment_confidence=0.95,
            ),
        )
        self.assertEqual(self.session.world.digest(), digest_before)
        self.assertEqual(plan.disposition, EvidenceDisposition.OBSERVE)
        self.assertTrue(plan.should_observe)
        assert plan.selected is not None
        self.assertEqual(plan.selected.element_name, "Wall-Party")
        self.assertEqual(
            plan.selected.action, InspectionAction.EXTRACT_SCAN_MEASUREMENT
        )
        self.assertEqual(plan.scan_digest, "test-scan-digest")

    def test_weak_scan_routes_to_targeted_rescan(self) -> None:
        plan = plan_clearance_evidence(
            self.borderline,
            self.evidence(
                effective_points=5.0,
                support_diversity=0.20,
                assignment_confidence=0.40,
            ),
        )
        assert plan.selected is not None
        self.assertEqual(plan.selected.element_name, "Wall-Party")
        self.assertEqual(plan.selected.action, InspectionAction.RESCAN_ELEMENT)
        self.assertGreater(plan.selected.evidence_deficit, 0.0)

    def test_resolved_clearance_stops_inspection(self) -> None:
        plan = plan_clearance_evidence(
            self.clear,
            self.evidence(
                effective_points=5.0,
                support_diversity=0.20,
                assignment_confidence=0.40,
            ),
        )
        self.assertEqual(plan.disposition, EvidenceDisposition.DECISION_RESOLVED)
        self.assertFalse(plan.should_observe)
        self.assertEqual(plan.recommendations, ())

    def test_mismatched_scan_provenance_is_rejected(self) -> None:
        evidence = self.evidence(
            effective_points=100.0,
            support_diversity=0.80,
            assignment_confidence=0.95,
            scene_version="different-scene",
        )
        with self.assertRaisesRegex(DecisionError, "different scenes"):
            plan_clearance_evidence(self.borderline, evidence)

    def test_assessment_and_plan_are_deterministic(self) -> None:
        decision = self.borderline.decision
        again = assess_clearance(self.scene, decision)
        self.assertEqual(self.borderline, again)
        evidence = self.evidence(
            effective_points=100.0,
            support_diversity=0.80,
            assignment_confidence=0.95,
        )
        self.assertEqual(
            plan_clearance_evidence(self.borderline, evidence),
            plan_clearance_evidence(self.borderline, evidence),
        )


if __name__ == "__main__":
    unittest.main()
