"""Combined engine identity, audit, and Claude view/Workbench contracts."""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from gat.cli import main
from gat.demo.workbench import clearance_request
from gat.geometry.viewer import frame_record
from gat.headless import handle_request
from gat.ifc_audit import audit_ifc_file
from gat.session import GatSession

MODEL = Path(__file__).parents[1] / "gat/demo/model.ifc"


class EngineWorkbenchIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.original = self.root / "original.ifc"
        self.moved = self.root / "relocated.ifc"
        self.raw = MODEL.read_bytes()
        self.original.write_bytes(self.raw)
        self.moved.write_bytes(self.raw)

    def request(self, version):
        request = clearance_request()
        request["state"] = {
            "kind": "ifc", "path": str(self.original), "identity_version": version,
        }
        return request

    def render(self, command, model, request, response):
        request_path = self.root / "request.json"
        response_path = self.root / "response.json"
        output = self.root / f"{command}.html"
        request_path.write_text(json.dumps(request), encoding="utf-8")
        response_path.write_text(json.dumps(response), encoding="utf-8")
        args = [command, str(model), "-o", str(output), "--variations", "0",
                "--request", str(request_path), "--decision", str(response_path)]
        if command == "view":
            args.append("--audit")
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            code = main(args)
        return code, output

    def test_content_import_renders_after_original_locator_disappears(self):
        request = self.request(2)
        response = handle_request(request)
        self.original.unlink()
        for command in ("view", "workbench"):
            with self.subTest(command=command):
                code, output = self.render(command, self.moved, request, response)
                self.assertEqual(code, 0)
                html = output.read_text(encoding="utf-8")
                self.assertIn(response["world_digest"], html)
                self.assertIn("REJECT", html)
                self.assertIn("placements are exact metadata", html)
        self.assertEqual(self.moved.read_bytes(), self.raw)

    def test_explicit_legacy_import_renders_with_matching_audit(self):
        request = self.request(1)
        response = handle_request(request)
        self.assertEqual(
            audit_ifc_file(self.original, identity_version=1).world_digest,
            response["world_digest"],
        )
        self.assertNotEqual(audit_ifc_file(self.original).world_digest, response["world_digest"])
        for command in ("view", "workbench"):
            with self.subTest(command=command):
                code, output = self.render(command, self.original, request, response)
                self.assertEqual(code, 0)
                self.assertIn(response["world_digest"], output.read_text(encoding="utf-8"))

    def test_legacy_import_cannot_silently_rebind_to_a_copy(self):
        request = self.request(1)
        response = handle_request(request)
        for command in ("view", "workbench"):
            with self.subTest(command=command):
                code, output = self.render(command, self.moved, request, response)
                self.assertEqual(code, 2)
                self.assertFalse(output.exists())

    def test_changed_content_cannot_reuse_a_portable_assessment(self):
        request = self.request(2)
        response = handle_request(request)
        self.moved.write_bytes(self.raw + b"\n")
        for command in ("view", "workbench"):
            with self.subTest(command=command):
                code, output = self.render(command, self.moved, request, response)
                self.assertEqual(code, 2)
                self.assertFalse(output.exists())

    def test_frame_projection_remains_honest_and_nonmutating_for_both_imports(self):
        for version in (1, 2):
            session = GatSession.load_ifc(str(self.original), identity_version=version)
            before = session.world.digest()
            frame = frame_record(session.world)
            self.assertEqual(frame["units"], "m")
            self.assertIsNone(frame["transform"])
            self.assertIn("placements are exact metadata", frame["uncertainty"])
            self.assertEqual(session.world.digest(), before)

    def test_audit_rejects_unknown_import_versions(self):
        for version in (0, 3, True, "2"):
            with self.subTest(version=version):
                with self.assertRaises(ValueError):
                    audit_ifc_file(self.original, identity_version=version)
