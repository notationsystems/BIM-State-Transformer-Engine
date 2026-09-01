"""Non-mutating design-change and RFI impact reports."""

from __future__ import annotations

import os
import unittest

import gat.demo
from gat.engine.executor import preview
from gat.engine.transform import SetParameter
from gat.session import GatSession
from gat.workflows import ChangeDisposition, preview_change


MODEL = os.path.join(os.path.dirname(gat.demo.__file__), "model.ifc")


class ChangeImpactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session = GatSession.load_ifc(MODEL)
        self.before = self.session.world.digest()

    def test_admissible_height_change_reports_every_propagated_impact(self) -> None:
        change = SetParameter(
            self.session.var("Level 1", "ClearHeight"), 3.4, 0.01
        )
        report = preview_change(self.session.world, change)

        self.assertEqual(report.disposition, ChangeDisposition.ADMISSIBLE)
        self.assertTrue(report.admissible)
        self.assertEqual(self.session.world.digest(), self.before)
        self.assertNotEqual(report.candidate_world_digest, self.before)
        self.assertEqual(len(report.affected), 34)
        self.assertIn("Level 1", report.impacted_entities)
        self.assertIn("Office-A", report.impacted_entities)
        self.assertTrue(any(item.target for item in report.impacts))
        self.assertTrue(any(item.affected for item in report.impacts))

    def test_infeasible_opening_change_exposes_failed_candidate_without_commit(self) -> None:
        change = SetParameter(
            self.session.var("Opening-1", "Height"), 3.6, 0.005
        )
        report = preview_change(self.session.world, change)

        self.assertEqual(report.disposition, ChangeDisposition.BLOCKED)
        self.assertFalse(report.admissible)
        self.assertTrue(report.failures)
        self.assertEqual(self.session.world.digest(), self.before)
        self.assertNotEqual(report.candidate_world_digest, self.before)
        self.assertTrue(
            any(item.invariant_id == "CONS-02" for item in report.failures)
        )

    def test_executor_preview_uses_the_same_verified_candidate_pipeline(self) -> None:
        change = SetParameter(
            self.session.var("Level 1", "ClearHeight"), 3.4, 0.01
        )
        raw = preview(self.session.world, change)
        report = preview_change(self.session.world, change)
        self.assertTrue(raw.admissible)
        self.assertEqual(raw.candidate.digest(), report.candidate_world_digest)
        self.assertEqual(raw.targets, report.targets)
        self.assertEqual(raw.affected, report.affected)

    def test_change_scope_and_rendering_are_deterministic(self) -> None:
        change = SetParameter(
            self.session.var("Level 1", "ClearHeight"), 3.4, 0.01
        )
        first = preview_change(self.session.world, change)
        second = preview_change(self.session.world, change)
        self.assertEqual(first.scope_digest, second.scope_digest)
        self.assertEqual(first.to_dict(), second.to_dict())


if __name__ == "__main__":
    unittest.main()
