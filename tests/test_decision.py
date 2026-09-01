"""Tests for decision confidence, stopping, and next-evidence planning."""

from __future__ import annotations

import os
import unittest

from gat import (
    DecisionVerdict,
    EvidenceDisposition,
    MinimumDecision,
    ObservationCandidate,
    assess_decision,
    plan_decision_evidence,
)
from gat.session import GatSession


DEMO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "gat", "demo", "model.ifc",
)


class DecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session = GatSession.load_ifc(DEMO)
        self.volume = self.session.var("Office-A", "Volume")
        self.clear_height = self.session.var("Level 1", "ClearHeight")
        self.unit_cost = self.session.var("Wall-South", "UnitCost")

    def test_current_threshold_is_unresolved(self) -> None:
        decision = MinimumDecision(
            self.volume, minimum=60.0, confidence=0.95, label="Office-A volume"
        )
        assessment = assess_decision(self.session.world, decision)
        self.assertEqual(assessment.verdict, DecisionVerdict.UNRESOLVED)
        self.assertAlmostEqual(assessment.p_satisfies, 0.5, delta=1e-12)
        self.assertFalse(assessment.resolved)
        self.assertEqual(assessment.world_digest, self.session.world.digest())

    def test_resolved_satisfied_decision_stops_without_candidates(self) -> None:
        decision = MinimumDecision(self.volume, minimum=59.0, confidence=0.95)
        plan = plan_decision_evidence(self.session.world, decision, [])
        self.assertEqual(plan.assessment.verdict, DecisionVerdict.SATISFIED)
        self.assertEqual(plan.disposition, EvidenceDisposition.DECISION_RESOLVED)
        self.assertFalse(plan.should_observe)
        self.assertEqual(plan.options, ())

    def test_resolved_violated_decision_stops_without_candidates(self) -> None:
        decision = MinimumDecision(self.volume, minimum=61.0, confidence=0.95)
        plan = plan_decision_evidence(self.session.world, decision, [])
        self.assertEqual(plan.assessment.verdict, DecisionVerdict.VIOLATED)
        self.assertEqual(plan.disposition, EvidenceDisposition.DECISION_RESOLVED)
        self.assertFalse(plan.should_observe)

    def test_unresolved_decision_selects_relevant_worthwhile_evidence(self) -> None:
        decision = MinimumDecision(self.volume, minimum=60.0, confidence=0.95)
        digest_before = self.session.world.digest()
        plan = plan_decision_evidence(
            self.session.world,
            decision,
            [
                ObservationCandidate(
                    self.unit_cost, 10.0, "cost survey", cost_nats=0.01
                ),
                ObservationCandidate(
                    self.clear_height, 0.01, "laser height", cost_nats=0.05
                ),
                ObservationCandidate(
                    self.volume, 0.05, "laser volume", cost_nats=0.10
                ),
            ],
        )
        self.assertEqual(self.session.world.digest(), digest_before)
        self.assertEqual(plan.disposition, EvidenceDisposition.OBSERVE)
        self.assertTrue(plan.should_observe)
        assert plan.selected is not None
        self.assertEqual(plan.selected.candidate.var, self.volume)
        self.assertGreater(plan.selected.net_epistemic_value, 0.0)
        self.assertEqual(plan.options[0], plan.selected)

    def test_observation_is_declined_when_burden_exceeds_information(self) -> None:
        decision = MinimumDecision(self.volume, minimum=60.0, confidence=0.95)
        plan = plan_decision_evidence(
            self.session.world,
            decision,
            [ObservationCandidate(self.volume, 0.05, cost_nats=100.0)],
        )
        self.assertEqual(
            plan.disposition, EvidenceDisposition.NO_WORTHWHILE_EVIDENCE
        )
        self.assertFalse(plan.should_observe)
        self.assertEqual(len(plan.options), 1)
        self.assertLess(plan.options[0].net_epistemic_value, 0.0)

    def test_real_evidence_changes_world_and_resolves_decision(self) -> None:
        decision = MinimumDecision(self.volume, minimum=60.0, confidence=0.95)
        before = assess_decision(self.session.world, decision)
        self.session.run(ObservationCandidate(self.volume, 0.05).observe(60.2))
        after = assess_decision(self.session.world, decision)

        self.assertEqual(before.verdict, DecisionVerdict.UNRESOLVED)
        self.assertEqual(after.verdict, DecisionVerdict.SATISFIED)
        self.assertGreater(after.p_satisfies, decision.confidence)
        self.assertNotEqual(after.world_digest, before.world_digest)
        self.assertTrue(self.session.verify().passed)

    def test_invalid_decision_confidence_is_rejected(self) -> None:
        for confidence in (0.5, 1.0, float("nan")):
            with self.subTest(confidence=confidence):
                with self.assertRaises(ValueError):
                    MinimumDecision(self.volume, 60.0, confidence)


if __name__ == "__main__":
    unittest.main()
