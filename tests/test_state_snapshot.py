"""Lossless restart and operational-equivalence tests for state snapshots."""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
import os
import tempfile
import unittest

import numpy as np

import gat.demo
from gat.demo.portability import run_demo
from gat.engine.configuration import configuration_digest
from gat.engine.transform import ObserveQuantity, ShiftParameter
from gat.errors import SnapshotError
from gat.gaussian.state import GaussianState
from gat.session import GatSession
from gat.state_snapshot import (
    _content_digest,
    capture_snapshot,
    computational_equivalence,
    read_snapshot,
    reconstruct_snapshot,
)


MODEL = os.path.join(os.path.dirname(gat.demo.__file__), "model.ifc")


def _rehash(document: dict[str, object]) -> None:
    integrity = document["integrity"]
    assert isinstance(integrity, dict)
    integrity["digest"] = _content_digest(document)


class StateSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session = GatSession.load_ifc(MODEL)
        volume = self.session.var("Office-A", "Volume")
        self.session.run(ObserveQuantity.single(volume, 59.4, noise_sigma=0.05))

    def test_in_memory_roundtrip_is_exact_computational_state(self) -> None:
        document = capture_snapshot(self.session.world, self.session.trace.events)
        restored = reconstruct_snapshot(document)
        report = computational_equivalence(self.session.world, restored.world)
        self.assertTrue(report.passed, report.render())
        self.assertEqual(restored.trace_events, tuple(self.session.trace.events))
        self.assertEqual(restored.world.digest(), self.session.world.digest())

    def test_full_correlations_survive_not_only_marginal_sigmas(self) -> None:
        sigma = self.session.world.belief.sigma
        off_diagonal = sigma - np.diag(np.diag(sigma))
        self.assertGreater(np.abs(off_diagonal).max(), 0.0)
        restored = reconstruct_snapshot(capture_snapshot(self.session.world)).world
        self.assertTrue(np.array_equal(restored.belief.sigma, sigma))

    def test_capture_is_deterministic(self) -> None:
        first = capture_snapshot(self.session.world, self.session.trace.events)
        second = capture_snapshot(self.session.world, self.session.trace.events)
        self.assertEqual(first, second)
        self.assertEqual(
            json.dumps(first, sort_keys=True, separators=(",", ":")),
            json.dumps(second, sort_keys=True, separators=(",", ":")),
        )

    def test_session_disk_api_restores_trace_and_can_continue(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "checkpoint.gat.json")
            trace_at_capture = tuple(self.session.trace.events)
            digest = self.session.export_snapshot(path)
            restored = GatSession.load_snapshot(path)
        self.assertEqual(len(digest), 64)
        self.assertEqual(tuple(restored.trace.events[:-1]), trace_at_capture)
        self.assertEqual(restored.trace.events[-1].stage, "resume")
        report = computational_equivalence(self.session.world, restored.world)
        self.assertTrue(report.passed, report.render())

    def test_continuation_matches_uninterrupted_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "checkpoint.gat.json")
            self.session.export_snapshot(path)
            restored = GatSession.load_snapshot(path)
        target_a = self.session.var("Level 1", "ClearHeight")
        target_b = restored.var("Level 1", "ClearHeight")
        self.session.run(ShiftParameter(target_a, 0.10))
        restored.run(ShiftParameter(target_b, 0.10))
        report = computational_equivalence(self.session.world, restored.world)
        self.assertTrue(report.passed, report.render())

    def test_tampering_without_new_integrity_digest_is_rejected(self) -> None:
        document = capture_snapshot(self.session.world)
        payload = document["payload"]
        assert isinstance(payload, dict)
        belief = payload["belief"]
        assert isinstance(belief, dict)
        mean = belief["mean"]
        assert isinstance(mean, list)
        mean[0] += 0.1
        with self.assertRaisesRegex(SnapshotError, "integrity digest mismatch"):
            reconstruct_snapshot(document)

    def test_reordered_variable_index_is_rejected_even_with_valid_digest(self) -> None:
        document = capture_snapshot(self.session.world)
        payload = document["payload"]
        assert isinstance(payload, dict)
        belief = payload["belief"]
        assert isinstance(belief, dict)
        variables = belief["raw_variables"]
        assert isinstance(variables, list)
        variables[0], variables[1] = variables[1], variables[0]
        _rehash(document)
        with self.assertRaisesRegex(SnapshotError, "raw variable order"):
            reconstruct_snapshot(document)

    def test_invalid_covariance_is_rejected_even_with_valid_digest(self) -> None:
        document = capture_snapshot(self.session.world)
        payload = document["payload"]
        assert isinstance(payload, dict)
        belief = payload["belief"]
        assert isinstance(belief, dict)
        covariance = belief["covariance"]
        assert isinstance(covariance, dict)
        values = covariance["values"]
        assert isinstance(values, list)
        values[0] = -1.0
        _rehash(document)
        with self.assertRaisesRegex(SnapshotError, "failed verification"):
            reconstruct_snapshot(document)

    def test_unknown_expression_opcode_is_rejected(self) -> None:
        document = capture_snapshot(self.session.world)
        payload = document["payload"]
        assert isinstance(payload, dict)
        module = payload["module"]
        assert isinstance(module, dict)
        entities = module["entities"]
        assert isinstance(entities, list)
        changed = False
        for entity in entities:
            assert isinstance(entity, dict)
            slots = entity["slots"]
            assert isinstance(slots, list)
            for slot in slots:
                assert isinstance(slot, dict)
                if slot["expr"] is not None:
                    expr = slot["expr"]
                    assert isinstance(expr, dict)
                    expr["op"] = "execute_python"
                    changed = True
                    break
            if changed:
                break
        self.assertTrue(changed)
        _rehash(document)
        with self.assertRaisesRegex(SnapshotError, "unsupported expression opcode"):
            reconstruct_snapshot(document)

    def test_incompatible_runtime_contract_is_rejected(self) -> None:
        document = capture_snapshot(self.session.world)
        document["runtime_contract"] = "unknown-runtime"
        with self.assertRaisesRegex(SnapshotError, "incompatible runtime"):
            reconstruct_snapshot(document)

    def test_configuration_equivalence_is_weaker_than_computational(self) -> None:
        world = self.session.world
        covariance = world.belief.sigma.copy()
        i, j = 0, 1
        covariance[i, j] += 0.05 * np.sqrt(covariance[i, i] * covariance[j, j])
        covariance[j, i] = covariance[i, j]
        changed_belief = GaussianState(
            world.binding.raw_index, world.belief.mu, covariance
        )
        changed_world = world.with_belief(changed_belief)
        self.assertEqual(
            configuration_digest(world), configuration_digest(changed_world)
        )
        report = computational_equivalence(world, changed_world)
        self.assertFalse(report.passed)
        self.assertIn(
            "raw belief covariance", tuple(check.name for check in report.failures)
        )

    def test_separate_process_portability_demo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with redirect_stdout(StringIO()):
                digest = run_demo(directory)
            self.assertEqual(len(digest), 64)
            self.assertTrue(
                os.path.exists(os.path.join(directory, "state_t2_resumed.gat.json"))
            )


if __name__ == "__main__":
    unittest.main()
