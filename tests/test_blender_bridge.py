"""Blender extension response decoder tests without importing bpy."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import tomllib
import unittest

import gat.demo
from gat.headless import REQUEST_FORMAT, handle_request


ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "integrations" / "blender" / "gat_assurance" / "bridge.py"
MANIFEST = BRIDGE.with_name("blender_manifest.toml")
MODEL = os.path.join(os.path.dirname(gat.demo.__file__), "model.ifc")

spec = importlib.util.spec_from_file_location("gat_blender_bridge", BRIDGE)
assert spec is not None and spec.loader is not None
bridge = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bridge
spec.loader.exec_module(bridge)


class BlenderBridgeTests(unittest.TestCase):
    def response(self):
        return handle_request(
            {
                "format": REQUEST_FORMAT,
                "request_id": "blender-case",
                "operation": "acceptance",
                "state": {"kind": "ifc", "path": MODEL},
                "payload": {
                    "case_id": "route-1",
                    "workflow": "AS_BUILT_CLEARANCE",
                    "subject": "crossing duct",
                    "checks": [
                        {
                            "kind": "clearance",
                            "check_id": "route-clearance",
                            "proposal": {
                                "origin": [4.0, 1.8, 2.6],
                                "angle": 0.0,
                                "extents": [3.0, 0.4, 0.4],
                            },
                            "required_clearance": 0.05,
                            "confidence": 0.95,
                            "position_sigma": 0.02,
                            "label": "crossing duct",
                        }
                    ],
                },
            }
        )

    def test_response_decodes_to_read_only_view_model(self) -> None:
        view = bridge.parse_response(self.response())
        self.assertEqual(view.case_id, "route-1")
        self.assertEqual(view.disposition, "REJECT")
        self.assertIn("Wall-Party", view.overlay_subjects)
        self.assertEqual(view.color, (0.85, 0.08, 0.08, 1.0))

    def test_non_acceptance_response_is_rejected(self) -> None:
        response = self.response()
        response["operation"] = "change_impact"
        with self.assertRaisesRegex(ValueError, "requires an acceptance"):
            bridge.parse_response(response)

    def test_manifest_declares_only_local_file_access(self) -> None:
        manifest = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(manifest["id"], "gat_assurance")
        self.assertEqual(manifest["type"], "add-on")
        self.assertIn("files", manifest["permissions"])
        self.assertNotIn("network", manifest["permissions"])
        self.assertIn("SPDX:GPL-3.0-or-later", manifest["license"])


if __name__ == "__main__":
    unittest.main()
