"""The offline 3D viewer: deterministic payload, self-contained instrument."""

from __future__ import annotations

import math
import os
import tempfile
import unittest

import gat.demo
from gat.cli import main as cli_main
from gat.engine.verify import run_invariants
from gat.geometry.viewer import (
    VIEWER_SCENE_FORMAT,
    export_viewer_html,
    viewer_payload,
)
from gat.session import GatSession


MODEL = os.path.join(os.path.dirname(gat.demo.__file__), "model.ifc")


class ViewerPayloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.world = GatSession.load_ifc(MODEL).world
        cls.payload = viewer_payload(cls.world, n=3, seed=7, model_name="model.ifc")

    def test_payload_carries_nominal_plus_n_samples(self) -> None:
        self.assertEqual(self.payload["format"], VIEWER_SCENE_FORMAT)
        samples = self.payload["samples"]
        self.assertEqual(len(samples), 4)
        self.assertEqual(samples[0]["label"], "nominal")
        self.assertEqual(samples[0]["world_digest"], self.world.digest())
        self.assertEqual(samples[0]["passed"], run_invariants(self.world).passed)

    def test_arrays_are_consistent_and_finite(self) -> None:
        classes = self.payload["classes"]
        for element in self.payload["elements"]:
            self.assertLess(element["class"], len(classes))
        for sample in self.payload["samples"]:
            count = len(sample["element"])
            self.assertEqual(len(sample["centers"]), 3 * count)
            self.assertEqual(len(sample["axes"]), 9 * count)
            self.assertTrue(all(math.isfinite(v) for v in sample["centers"]))
            self.assertTrue(all(math.isfinite(v) for v in sample["axes"]))
            for index in sample["element"]:
                self.assertLess(index, len(self.payload["elements"]))

    def test_elements_carry_boxes_quantities_and_sigmas(self) -> None:
        by_name = {element["name"]: element for element in self.payload["elements"]}
        wall = by_name["Wall-Party"]
        self.assertEqual(wall["quantities"], ["Length", "Width", "Height"])
        self.assertEqual(len(wall["box"]["extents"]), 3)
        self.assertTrue(all(sigma > 0 for sigma in wall["extents_sigma"]))
        door = by_name["Door-1"]
        self.assertIsNone(door["quantities"][1])
        self.assertIsNone(door["extents_sigma"][1])
        for sample in self.payload["samples"]:
            self.assertEqual(len(sample["boxes"]), 7 * len(self.payload["elements"]))
        nominal = self.payload["samples"][0]["boxes"]
        index = self.payload["elements"].index(wall)
        self.assertEqual(nominal[index * 7 + 4 : index * 7 + 7], wall["box"]["extents"])

    def test_payload_is_deterministic(self) -> None:
        again = viewer_payload(self.world, n=3, seed=7, model_name="model.ifc")
        self.assertEqual(self.payload, again)

    def test_different_seed_changes_samples_not_nominal(self) -> None:
        other = viewer_payload(self.world, n=3, seed=8, model_name="model.ifc")
        self.assertEqual(
            self.payload["samples"][0]["centers"], other["samples"][0]["centers"]
        )
        self.assertNotEqual(
            self.payload["samples"][1]["centers"], other["samples"][1]["centers"]
        )


def clearance_request(request_id: str = "duct-route-1") -> dict:
    return {
        "format": "gat-headless-request-v1",
        "request_id": request_id,
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


class DecisionOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from gat.headless import handle_request

        cls.world = GatSession.load_ifc(MODEL).world
        cls.request = clearance_request()
        cls.response = handle_request(cls.request)

    def test_reject_paints_only_uncleared_elements_and_draws_proposal(self) -> None:
        from gat.geometry.viewer import decision_overlay

        overlay = decision_overlay(self.world, self.response, self.request)
        self.assertEqual(overlay["disposition"], "REJECT")
        self.assertEqual(overlay["color"], "#d91414")
        self.assertEqual(overlay["subjects"], ["Wall-Party"])
        self.assertGreater(len(overlay["risks"]), 1)
        self.assertEqual(len(overlay["proposals"]), 1)
        self.assertEqual(overlay["proposals"][0]["extents"], [3.0, 0.4, 0.4])
        self.assertIn("This report does not authorize any action.", overlay["footers"])

    def test_overlay_without_request_has_no_proposals(self) -> None:
        from gat.geometry.viewer import decision_overlay

        overlay = decision_overlay(self.world, self.response)
        self.assertEqual(overlay["proposals"], [])
        self.assertEqual(overlay["subjects"], ["Wall-Party"])

    def test_decision_from_another_world_is_refused(self) -> None:
        from gat.geometry.viewer import decision_overlay

        other = GatSession.load_ifc(
            os.path.join(os.path.dirname(gat.demo.__file__), "beam_model.ifc")
        ).world
        with self.assertRaisesRegex(ValueError, "different world"):
            decision_overlay(other, self.response, self.request)

    def test_mismatched_request_is_refused(self) -> None:
        from gat.geometry.viewer import decision_overlay

        with self.assertRaisesRegex(ValueError, "ids differ"):
            decision_overlay(self.world, self.response, clearance_request("other"))

    def test_non_decision_documents_are_refused(self) -> None:
        from gat.geometry.viewer import decision_overlay
        from gat.headless import handle_request

        summary = handle_request({**self.request, "operation": "summary", "payload": {}})
        with self.assertRaisesRegex(ValueError, "render decisions"):
            decision_overlay(self.world, summary)

    def test_beam_decision_binds_through_its_prior_world(self) -> None:
        from gat.geometry.viewer import decision_overlay
        from gat.headless import handle_request

        beam_model = os.path.join(os.path.dirname(gat.demo.__file__), "beam_model.ifc")
        response = handle_request(
            {
                "format": "gat-headless-request-v1",
                "request_id": "beam-view",
                "operation": "beam_assurance",
                "state": {"kind": "ifc", "path": beam_model},
                "payload": {
                    "case_id": "beam-b1-certificate",
                    "beam_name": "Beam-B1",
                    "factored_demand_n_m": 301_000.0,
                    "confidence": 0.95,
                    "material_certificate_path": os.path.join(
                        os.path.dirname(gat.demo.__file__), "material_certificate.json"
                    ),
                },
            }
        )
        overlay = decision_overlay(GatSession.load_ifc(beam_model).world, response)
        self.assertEqual(overlay["disposition"], "VIOLATED")
        self.assertEqual(overlay["subjects"], ["Beam-B1"])

    def test_cli_loads_model_through_the_request_path_form(self) -> None:
        import json

        from gat.headless import handle_request

        relative = os.path.relpath(MODEL)
        request = clearance_request("relative-form")
        request["state"]["path"] = relative
        response = handle_request(request)
        with tempfile.TemporaryDirectory() as tmp:
            request_path = os.path.join(tmp, "request.json")
            response_path = os.path.join(tmp, "response.json")
            with open(request_path, "w", encoding="utf-8") as handle:
                json.dump(request, handle)
            with open(response_path, "w", encoding="utf-8") as handle:
                json.dump(response, handle)
            out = os.path.join(tmp, "viewer.html")
            # absolute model path + relative request path: same file, so it binds
            self.assertEqual(
                cli_main(["view", MODEL, "-o", out, "--variations", "0",
                          "--decision", response_path, "--request", request_path]),
                0,
            )
            # without the request there is nothing to match against: refused with a hint
            self.assertEqual(
                cli_main(["view", MODEL, "-o", out, "--variations", "0",
                          "--decision", response_path]),
                2,
            )

    def test_cli_embeds_the_overlay(self) -> None:
        import json

        with tempfile.TemporaryDirectory() as tmp:
            request_path = os.path.join(tmp, "request.json")
            response_path = os.path.join(tmp, "response.json")
            with open(request_path, "w", encoding="utf-8") as handle:
                json.dump(self.request, handle)
            with open(response_path, "w", encoding="utf-8") as handle:
                json.dump(self.response, handle)
            out = os.path.join(tmp, "viewer.html")
            self.assertEqual(
                cli_main(
                    ["view", MODEL, "-o", out, "--variations", "1",
                     "--decision", response_path, "--request", request_path]
                ),
                0,
            )
            with open(out, encoding="utf-8") as handle:
                html = handle.read()
            self.assertIn('"disposition":"REJECT"', html)
            self.assertIn("Wall-Party", html)
            self.assertEqual(cli_main(["view", MODEL, "-o", out, "--request", request_path]), 2)


class ViewerHtmlTests(unittest.TestCase):
    def test_html_is_self_contained_and_offline(self) -> None:
        world = GatSession.load_ifc(MODEL).world
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "viewer.html")
            count = export_viewer_html(world, path, n=2, model_name="model.ifc")
            self.assertEqual(count, 3)
            with open(path, encoding="utf-8") as handle:
                html = handle.read()
        self.assertTrue(html.startswith("<!doctype html>"))
        self.assertNotIn("__GAT_SCENE_JSON__", html)
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)
        self.assertIn(VIEWER_SCENE_FORMAT, html)
        self.assertIn("Read-only: no BIM state was changed.", html)

    def test_cli_view_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "viewer.html")
            self.assertEqual(
                cli_main(["view", MODEL, "-o", path, "--variations", "2"]), 0
            )
            self.assertTrue(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
