"""Tests for the authoritative, replayable GAT execution ledger."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import gat.demo
from gat.engine.transform import (
    CompositeTransformation,
    Measurement,
    ObserveLinearized,
    ObserveQuantity,
    ScaleParameter,
    SetParameter,
    ShiftParameter,
)
from gat.errors import BindingError, LedgerError, VerificationError
from gat.ledger import (
    ExecutionLedger,
    decode_transformation,
    encode_transformation,
    read_ledger,
    replay_ledger,
    write_ledger,
)
from gat.session import GatSession


MODEL = os.path.join(os.path.dirname(gat.demo.__file__), "model.ifc")


class LedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session = GatSession.load_ifc(MODEL)
        self.initial = self.session.world
        self.clear_height = self.session.var("Level 1", "ClearHeight")
        self.volume = self.session.var("Office-A", "Volume")
        self.opening_height = self.session.var("Opening-1", "Height")

    def linearized(self) -> ObserveLinearized:
        world = self.session.world
        row = np.zeros(world.binding.n_raw)
        row[world.binding.raw_index.row(self.clear_height)] = 1.0
        return ObserveLinearized(
            row=row,
            predicted=world.belief.mean(self.clear_height),
            observed=world.belief.mean(self.clear_height) - 0.01,
            noise_sigma=0.004,
            raw_targets=(self.clear_height,),
            expected_raw_order=world.binding.raw_index.vars,
            expected_belief_digest=world.belief.digest(),
            expected_world_digest=world.digest(),
            evidence_digest=hashlib.sha256(b"survey-control-A/scan-42").hexdigest(),
            label="scan support face Wall-Party",
        )

    def test_closed_codec_roundtrips_every_transformation(self) -> None:
        linearized = self.linearized()
        operations = (
            SetParameter(self.clear_height, 3.05, 0.01),
            ShiftParameter(self.clear_height, 0.01),
            ScaleParameter(self.clear_height, 1.01),
            ObserveQuantity(
                (
                    Measurement(self.clear_height, 3.0, 0.01),
                    Measurement(self.volume, 60.0, 0.05),
                )
            ),
            linearized,
            CompositeTransformation(
                (
                    ShiftParameter(self.clear_height, 0.01),
                    ScaleParameter(self.clear_height, 1.001),
                )
            ),
        )
        for operation in operations:
            with self.subTest(operation=operation.name):
                encoded = encode_transformation(operation)
                decoded = decode_transformation(encoded)
                self.assertEqual(encode_transformation(decoded), encoded)

    def test_unknown_opcode_and_extra_fields_fail_closed(self) -> None:
        with self.assertRaisesRegex(LedgerError, "unknown"):
            decode_transformation({"op": "execute_python", "source": "pass"})
        encoded = encode_transformation(ShiftParameter(self.clear_height, 0.1))
        encoded["unexpected"] = True
        with self.assertRaisesRegex(LedgerError, "fields differ"):
            decode_transformation(encoded)

    def test_accepted_history_replays_exact_mean_and_covariance(self) -> None:
        self.session.run(
            ObserveQuantity.single(self.volume, 59.4, 0.05),
            provenance={"sensor": "volume-laser-A", "calibration": "cal-2026-08"},
        )
        self.session.run(ShiftParameter(self.clear_height, 0.01))

        replay = replay_ledger(self.initial, self.session.ledger)

        self.assertEqual(replay.accepted, 2)
        self.assertEqual(replay.rejected, 0)
        self.assertEqual(replay.world.digest(), self.session.world.digest())
        self.assertTrue(np.array_equal(replay.world.belief.mu, self.session.world.belief.mu))
        self.assertTrue(
            np.array_equal(replay.world.belief.sigma, self.session.world.belief.sigma)
        )
        self.assertTrue(np.array_equal(replay.world.full.sigma, self.session.world.full.sigma))

    def test_verification_rejection_is_recorded_and_replayed(self) -> None:
        digest = self.session.world.digest()
        with self.assertRaises(VerificationError):
            self.session.run(SetParameter(self.opening_height, 3.2, 0.005))

        self.assertEqual(self.session.world.digest(), digest)
        event = self.session.ledger.events[-1]
        self.assertEqual(event.kind, "rejection")
        self.assertEqual(event.prior_world_digest, event.result_world_digest)
        self.assertIsNotNone(event.verification_digest)
        self.assertIsNotNone(event.verification)
        assert event.verification is not None
        self.assertFalse(event.verification["passed"])
        self.assertTrue(any(row["status"] == "FAIL" for row in event.verification["results"]))
        self.assertIn("verification failed", event.error_message or "")
        replay = replay_ledger(self.initial, self.session.ledger)
        self.assertEqual(replay.accepted, 0)
        self.assertEqual(replay.rejected, 1)
        self.assertEqual(replay.world.digest(), digest)

    def test_non_strict_rejection_is_equally_authoritative(self) -> None:
        result = self.session.run(
            SetParameter(self.opening_height, 3.2, 0.005), strict=False
        )
        self.assertFalse(result.committed)
        replay = replay_ledger(self.initial, self.session.ledger)
        self.assertEqual(replay.rejected, 1)
        self.assertEqual(replay.world.digest(), self.initial.digest())

    def test_stale_evidence_rejection_replays_as_same_binding_error(self) -> None:
        stale = self.linearized()
        self.session.run(ShiftParameter(self.clear_height, 0.001))
        with self.assertRaisesRegex(BindingError, "stale"):
            self.session.run(stale, provenance={"scan": "scan-42"})

        event = self.session.ledger.events[-1]
        self.assertEqual(event.error_type, "BindingError")
        self.assertIsNone(event.verification_digest)
        replay = replay_ledger(self.initial, self.session.ledger)
        self.assertEqual(replay.accepted, 1)
        self.assertEqual(replay.rejected, 1)
        self.assertEqual(replay.world.digest(), self.session.world.digest())

    def test_linearized_scan_evidence_replays_exactly(self) -> None:
        operation = self.linearized()
        self.session.run(
            operation,
            provenance={
                "scan_digest": operation.evidence_digest,
                "pose_source": "survey-control-A",
            },
        )
        replay = replay_ledger(self.initial, self.session.ledger)
        self.assertEqual(replay.world.digest(), self.session.world.digest())
        self.assertTrue(
            np.array_equal(replay.world.belief.sigma, self.session.world.belief.sigma)
        )

    def test_provenance_is_hash_bound_and_must_be_finite_json(self) -> None:
        self.session.run(
            ShiftParameter(self.clear_height, 0.001),
            provenance={"source": "survey", "external_time": "2026-09-01T04:00:00Z"},
        )
        self.assertEqual(self.session.ledger.events[-1].provenance["source"], "survey")
        tampered = self.session.ledger.to_dict()
        tampered["events"][-1]["provenance"]["source"] = "unknown"
        with self.assertRaisesRegex(LedgerError, "hash mismatch"):
            ExecutionLedger.from_dict(tampered)

        other = GatSession.load_ifc(MODEL)
        prior = other.world.digest()
        with self.assertRaisesRegex(LedgerError, "non-finite"):
            other.run(
                ShiftParameter(other.var("Level 1", "ClearHeight"), 0.001),
                provenance={"quality": float("nan")},
            )
        self.assertEqual(other.world.digest(), prior)

    def test_chain_detects_reordering_deletion_and_head_tampering(self) -> None:
        self.session.run(ShiftParameter(self.clear_height, 0.001))
        self.session.run(ShiftParameter(self.clear_height, 0.002))
        document = self.session.ledger.to_dict()

        reordered = json.loads(json.dumps(document))
        reordered["events"][1], reordered["events"][2] = (
            reordered["events"][2], reordered["events"][1]
        )
        with self.assertRaises(LedgerError):
            ExecutionLedger.from_dict(reordered)

        deleted = json.loads(json.dumps(document))
        del deleted["events"][1]
        with self.assertRaises(LedgerError):
            ExecutionLedger.from_dict(deleted)

        bad_head = json.loads(json.dumps(document))
        bad_head["integrity"]["head"] = "f" * 64
        with self.assertRaisesRegex(LedgerError, "head"):
            ExecutionLedger.from_dict(bad_head)

    def test_identical_execution_produces_identical_ledger(self) -> None:
        first = GatSession.load_ifc(MODEL)
        second = GatSession.load_ifc(MODEL)
        for session in (first, second):
            session.run(
                ObserveQuantity.single(session.var("Office-A", "Volume"), 59.4, 0.05),
                provenance={"sensor": "volume-laser-A"},
            )
        self.assertEqual(first.ledger.to_dict(), second.ledger.to_dict())
        self.assertEqual(first.ledger.head, second.ledger.head)

    def test_disk_roundtrip_and_session_export_are_deterministic(self) -> None:
        self.session.run(ShiftParameter(self.clear_height, 0.001))
        with tempfile.TemporaryDirectory() as tmp:
            direct = os.path.join(tmp, "direct.json")
            facade = os.path.join(tmp, "facade.json")
            self.assertEqual(write_ledger(self.session.ledger, direct), self.session.ledger.head)
            self.assertEqual(self.session.export_ledger(facade), self.session.ledger.head)
            with open(direct, "rb") as left, open(facade, "rb") as right:
                self.assertEqual(left.read(), right.read())
            loaded = read_ledger(direct)
            self.assertEqual(loaded.to_dict(), self.session.ledger.to_dict())
            self.assertEqual(replay_ledger(self.initial, loaded).world.digest(), self.session.world.digest())

    def test_separate_process_demo_persists_and_replays_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ledger.json")
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            env = dict(os.environ)
            env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
            proc = subprocess.run(
                [sys.executable, "-m", "gat.demo.ledger_replay", path],
                cwd=root,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("replayed exactly: 2 accepted, 1 rejected", proc.stdout)
            ledger = read_ledger(path)
            self.assertEqual(len(ledger.events), 4)


if __name__ == "__main__":
    unittest.main()
