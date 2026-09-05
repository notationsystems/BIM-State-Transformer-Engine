"""Relocation must preserve assessments without reinterpreting legacy states."""
import hashlib
import os
from pathlib import Path
import tempfile
import unittest

from gat.demo.workbench import clearance_request
from gat.engine.transform import ShiftParameter
from gat.errors import LedgerError
from gat.geometry.viewer import decision_overlay
from gat.headless import handle_request
from gat.ifc_audit import audit_ifc_file
from gat.ledger import read_ledger, replay_ledger
from gat.session import GatSession
from gat.source_identity import semantic_model_digest
from gat.state_snapshot import capture_snapshot, reconstruct_snapshot

MODEL = Path(__file__).parents[1] / "gat/demo/model.ifc"
FIXTURES = Path(__file__).parent / "fixtures/identity"
LEGACY_DIGEST = "f628952eaff3bac72edf1705da3d66e196bb6ee2736382535bd3f3c33a73a2ad"


class SourceIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.a = Path(self.temp.name) / "a.ifc"
        self.b = Path(self.temp.name) / "b.ifc"
        self.raw = MODEL.read_bytes()
        self.a.write_bytes(self.raw)
        self.b.write_bytes(self.raw)

    def test_relocated_bytes_preserve_world_and_keep_locators_in_trace(self):
        a, b = GatSession.load_ifc(str(self.a)), GatSession.load_ifc(str(self.b))
        self.assertEqual(a.world.digest(), b.world.digest())
        self.assertEqual(a.world.module.digest(), b.world.module.digest())
        self.assertEqual(a.world.module.meta["source_content_sha256"], hashlib.sha256(self.raw).hexdigest())
        self.assertEqual(a.source_locator, str(self.a))
        self.assertEqual(b.trace.events[0].name, str(self.b))
        self.assertNotEqual(a.trace.events[0].name, b.trace.events[0].name)

    def test_relative_absolute_and_text_import_agree_for_identical_bytes(self):
        absolute = GatSession.load_ifc(str(MODEL.resolve()))
        relative = GatSession.load_ifc(os.path.relpath(MODEL))
        text = GatSession.from_text(self.raw.decode("utf-8"), source="another-location")
        self.assertEqual(absolute.world.digest(), relative.world.digest())
        self.assertEqual(absolute.world.digest(), text.world.digest())

    def test_changed_bytes_at_same_path_change_identity_even_if_semantics_agree(self):
        before = GatSession.load_ifc(str(self.a))
        self.a.write_bytes(self.raw + b"\n")
        after = GatSession.load_ifc(str(self.a))
        self.assertNotEqual(before.world.digest(), after.world.digest())
        self.assertEqual(semantic_model_digest(before.world.module), semantic_model_digest(after.world.module))

    def test_audit_and_session_bind_the_same_exact_input(self):
        self.assertEqual(audit_ifc_file(self.a).world_digest, GatSession.load_ifc(str(self.a)).world.digest())

    def test_relocated_assessment_binds_but_changed_content_is_refused(self):
        request = clearance_request()
        request["state"]["path"] = str(self.a)
        response = handle_request(request)
        moved = GatSession.load_ifc(str(self.b))
        self.assertEqual(decision_overlay(moved.world, response, request)["disposition"], "REJECT")
        self.b.write_bytes(self.raw + b"\n")
        with self.assertRaisesRegex(ValueError, "different world"):
            decision_overlay(GatSession.load_ifc(str(self.b)).world, response, request)

    def test_snapshot_and_ledger_continue_across_relocation(self):
        original = GatSession.load_ifc(str(self.a))
        initial = GatSession.load_ifc(str(self.b)).world
        original.run(ShiftParameter(original.var("Level 1", "ClearHeight"), 0.01))
        replayed = replay_ledger(initial, original.ledger)
        self.assertEqual(replayed.world.digest(), original.world.digest())
        restored = reconstruct_snapshot(capture_snapshot(original.world)).world
        self.assertEqual(restored.digest(), original.world.digest())

    def test_checked_in_legacy_snapshot_and_ledger_keep_exact_identity(self):
        legacy = GatSession.load_snapshot(str(FIXTURES / "legacy-v1-snapshot.json"))
        self.assertEqual(legacy.world.digest(), LEGACY_DIGEST)
        ledger = read_ledger(FIXTURES / "legacy-v1-ledger.json")
        self.assertEqual(replay_ledger(legacy.world, ledger).world.digest(), LEGACY_DIGEST)
        explicit = GatSession.from_text(self.raw.decode("utf-8"), source="legacy-v1-model.ifc", identity_version=1)
        self.assertEqual(explicit.world.digest(), LEGACY_DIGEST)
        portable = GatSession.from_text(self.raw.decode("utf-8"), source="legacy-v1-model.ifc")
        self.assertNotEqual(portable.world.digest(), LEGACY_DIGEST)
        with self.assertRaises(LedgerError):
            replay_ledger(portable.world, ledger)

    def test_legacy_path_binding_is_explicit_and_still_location_dependent(self):
        a = GatSession.load_ifc(str(self.a), identity_version=1)
        b = GatSession.load_ifc(str(self.b), identity_version=1)
        self.assertNotEqual(a.world.digest(), b.world.digest())
        request = clearance_request()
        request["state"] = {"kind": "ifc", "path": str(self.a), "identity_version": 1}
        response = handle_request(request)
        self.assertEqual(response["world_digest"], a.world.digest())

    def test_unknown_identity_versions_are_rejected(self):
        for version in (0, 3, True, "2", None):
            with self.subTest(version=version):
                with self.assertRaises(ValueError):
                    GatSession.load_ifc(str(self.a), identity_version=version)
