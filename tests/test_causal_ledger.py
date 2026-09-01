"""Typed non-mutating assessment, policy, approval, and action history."""

from __future__ import annotations

import hashlib
import os
import unittest

import gat.demo
from gat.causal import (
    ApprovalDecision,
    ApprovalRecord,
    AssessmentRecord,
    ExternalActionRecord,
    ExternalActionStatus,
    decode_causal_record,
    decision_assessment_record,
    decision_policy_record,
    encode_causal_record,
)
from gat.engine.active_inference import ObservationCandidate
from gat.engine.decision import MinimumDecision, assess_decision, plan_decision_evidence
from gat.engine.transform import ShiftParameter
from gat.errors import LedgerError
from gat.ledger import replay_ledger
from gat.session import GatSession


MODEL = os.path.join(os.path.dirname(gat.demo.__file__), "model.ifc")


class CausalLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session = GatSession.load_ifc(MODEL)
        self.initial = self.session.world
        self.volume = self.session.var("Office-A", "Volume")
        self.clear_height = self.session.var("Level 1", "ClearHeight")

    def decision_plan(self):
        decision = MinimumDecision(
            self.volume, minimum=60.0, confidence=0.95, label="Office-A volume"
        )
        return plan_decision_evidence(
            self.session.world,
            decision,
            (
                ObservationCandidate(self.volume, 0.05, "laser volume", 0.02),
                ObservationCandidate(self.clear_height, 0.01, "laser height", 0.02),
            ),
        )

    def test_assessment_and_policy_are_state_bound_but_non_mutating(self) -> None:
        digest = self.session.world.digest()
        plan = self.decision_plan()
        assessment = decision_assessment_record(plan.assessment)
        policy = decision_policy_record(plan)

        self.session.record_assessment(
            assessment, provenance={"requested_by": "clearance-review"}
        )
        self.session.record_policy(policy)

        self.assertEqual(self.session.world.digest(), digest)
        self.assertEqual(
            [event.kind for event in self.session.ledger.events[-2:]],
            ["assessment", "policy"],
        )
        for event in self.session.ledger.events[-2:]:
            self.assertEqual(event.prior_world_digest, event.result_world_digest)
            self.assertIsNone(event.verification)
            self.assertIsNone(event.error_type)
        replay = replay_ledger(self.initial, self.session.ledger)
        self.assertEqual(replay.accepted, 0)
        self.assertEqual(replay.rejected, 0)
        self.assertEqual(replay.non_state, 2)
        self.assertEqual(replay.world.digest(), digest)

    def test_non_state_history_can_precede_a_real_evidence_transition(self) -> None:
        plan = self.decision_plan()
        self.session.record_assessment(decision_assessment_record(plan.assessment))
        self.session.record_policy(decision_policy_record(plan))
        assert plan.selected is not None
        self.session.run(plan.selected.candidate.observe(plan.selected.predicted_measurement))

        replay = replay_ledger(self.initial, self.session.ledger)
        self.assertEqual(replay.non_state, 2)
        self.assertEqual(replay.accepted, 1)
        self.assertEqual(replay.world.digest(), self.session.world.digest())

    def test_stale_assessment_is_refused_without_appending(self) -> None:
        assessment = assess_decision(
            self.session.world,
            MinimumDecision(self.volume, minimum=60.0),
        )
        record = decision_assessment_record(assessment)
        self.session.run(ShiftParameter(self.clear_height, 0.001))
        count = len(self.session.ledger.events)
        with self.assertRaisesRegex(LedgerError, "stale"):
            self.session.record_assessment(record)
        self.assertEqual(len(self.session.ledger.events), count)

    def test_approval_and_external_action_have_explicit_lifecycles(self) -> None:
        world_digest = self.session.world.digest()
        scope_digest = hashlib.sha256(b"route-proposal-17").hexdigest()
        approval = ApprovalRecord(
            world_digest,
            "approval-17",
            "licensed-engineer:CA-1234",
            ApprovalDecision.APPROVED,
            scope_digest,
            "route may proceed to field verification",
        )
        self.session.record_approval(approval)
        for status in (
            ExternalActionStatus.PROPOSED,
            ExternalActionStatus.AUTHORIZED,
            ExternalActionStatus.STARTED,
            ExternalActionStatus.COMPLETED,
        ):
            self.session.record_external_action(
                ExternalActionRecord(
                    world_digest,
                    "field-scan-17",
                    "targeted-clearance-scan",
                    status,
                    None if status is ExternalActionStatus.PROPOSED else approval.approval_id,
                    (
                        hashlib.sha256(b"field-scan-result-17").hexdigest()
                        if status is ExternalActionStatus.COMPLETED
                        else None
                    ),
                )
            )

        self.assertEqual(self.session.world.digest(), world_digest)
        replay = replay_ledger(self.initial, self.session.ledger)
        self.assertEqual(replay.non_state, 5)
        self.assertEqual(replay.world.digest(), world_digest)

    def test_invalid_lifecycle_transitions_fail_closed(self) -> None:
        digest = self.session.world.digest()
        with self.assertRaisesRegex(LedgerError, "NEW -> COMPLETED"):
            self.session.record_external_action(
                ExternalActionRecord(
                    digest,
                    "action-1",
                    "field-work",
                    ExternalActionStatus.COMPLETED,
                    "approval-1",
                )
            )
        with self.assertRaisesRegex(LedgerError, "NEW -> REVOKED"):
            self.session.record_approval(
                ApprovalRecord(
                    digest,
                    "approval-1",
                    "authority",
                    ApprovalDecision.REVOKED,
                    hashlib.sha256(b"scope").hexdigest(),
                )
            )
        self.assertEqual(len(self.session.ledger.events), 1)

    def test_authorized_action_requires_authorization_reference(self) -> None:
        digest = self.session.world.digest()
        self.session.record_external_action(
            ExternalActionRecord(
                digest,
                "action-1",
                "field-work",
                ExternalActionStatus.PROPOSED,
            )
        )
        with self.assertRaisesRegex(LedgerError, "authorization reference"):
            self.session.record_external_action(
                ExternalActionRecord(
                    digest,
                    "action-1",
                    "field-work",
                    ExternalActionStatus.AUTHORIZED,
                )
            )

    def test_causal_codec_is_closed_and_deterministic(self) -> None:
        record = AssessmentRecord(
            self.session.world.digest(),
            "assessment-1",
            "custom-check",
            "route",
            "UNRESOLVED",
            "method-v1",
            details={"score": 0.5},
        )
        encoded = encode_causal_record(record)
        self.assertEqual(encode_causal_record(decode_causal_record(encoded)), encoded)
        encoded["extra"] = "not allowed"
        with self.assertRaisesRegex(ValueError, "fields differ"):
            decode_causal_record(encoded)
        with self.assertRaisesRegex(ValueError, "unknown"):
            decode_causal_record({"record_type": "shell_command"})

    def test_non_finite_causal_details_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite JSON"):
            AssessmentRecord(
                self.session.world.digest(),
                "assessment-1",
                "custom",
                "subject",
                "UNRESOLVED",
                "method-v1",
                details={"score": float("nan")},
            )


if __name__ == "__main__":
    unittest.main()
