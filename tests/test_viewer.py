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

    def test_elements_carry_their_entity_ids(self) -> None:
        by_name = {element["name"]: element for element in self.payload["elements"]}
        self.assertEqual(by_name["Wall-Party"]["entity"], "IfcWall:GATWAL0000000000000180")
        ids = [element["entity"] for element in self.payload["elements"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(":" in entity for entity in ids))

    def test_explode_offsets_follow_the_hierarchy(self) -> None:
        import numpy as np

        by_name = {element["name"]: element for element in self.payload["elements"]}
        centers = {}
        for element in self.payload["elements"]:
            box = element["box"]
            c, s = np.cos(box["angle"]), np.sin(box["angle"])
            hx, hy = box["extents"][0] / 2, box["extents"][1] / 2
            centers[element["name"]] = np.array(
                [box["origin"][0] + c * hx - s * hy, box["origin"][1] + s * hx + c * hy]
            )
        centroid = np.mean(list(centers.values()), axis=0)
        for element in self.payload["elements"]:
            self.assertEqual(len(element["explode"]), 3)
            self.assertTrue(all(math.isfinite(v) for v in element["explode"]))
        # perimeter walls move away from the plan centroid, in the plane
        for name in ("Wall-South", "Wall-North", "Wall-West", "Wall-East"):
            offset = np.array(by_name[name]["explode"][:2])
            self.assertGreater(float(offset @ (centers[name] - centroid)), 0.0, name)
            self.assertEqual(by_name[name]["explode"][2], 0.0)
        # the door travels with the wall its opening voids, further out and lifted
        wall = np.array(by_name["Wall-Party"]["explode"])
        door = np.array(by_name["Door-1"]["explode"])
        cosine = float(wall[:2] @ door[:2]) / (np.linalg.norm(wall[:2]) * np.linalg.norm(door[:2]))
        self.assertGreater(cosine, 0.999)
        self.assertGreater(np.linalg.norm(door[:2]), np.linalg.norm(wall[:2]))
        self.assertGreater(door[2], 0.0)
        # spaces lift straight up
        self.assertEqual(by_name["Office-A"]["explode"][:2], [0.0, 0.0])
        self.assertGreater(by_name["Office-A"]["explode"][2], 0.0)
        self.assertIsNone(self.payload["audit"])
        self.assertTrue(all(element["audit"] is None for element in self.payload["elements"]))

    def test_audit_statuses_bind_by_global_id_fail_closed(self) -> None:
        from gat.geometry.viewer import audit_statuses
        from gat.ifc_audit import audit_ifc_file

        document = audit_ifc_file(MODEL).to_dict()
        statuses = audit_statuses(document)
        self.assertEqual(set(statuses.values()), {"READY"})
        payload = viewer_payload(self.world, n=0, audit_statuses=statuses)
        self.assertEqual(payload["audit"]["matched"], len(payload["elements"]))
        wall = next(e for e in payload["elements"] if e["name"] == "Wall-Party")
        self.assertEqual(wall["audit"], {"status": "READY", "color": "#1ab233"})
        # a status the palette does not know is refused, never guessed
        with self.assertRaisesRegex(ValueError, "unsupported audit entity status"):
            viewer_payload(
                self.world, n=0, audit_statuses={"GATWAL0000000000000180": "FINE"}
            )
        with self.assertRaisesRegex(ValueError, "gat-ifc-audit-v1"):
            audit_statuses({"format": "something-else", "entities": []})
        tampered = dict(document)
        tampered["entities"] = [{**document["entities"][0], "status": "OK"}]
        with self.assertRaisesRegex(ValueError, "unsupported audit entity status"):
            audit_statuses(tampered)
        # an attention status paints an outline colour, and only that piece
        partial = viewer_payload(
            self.world,
            n=0,
            audit_statuses={"GATWAL0000000000000180": "NEEDS_GEOMETRY_DERIVATION"},
        )
        self.assertEqual(partial["audit"]["matched"], 1)
        wall = next(e for e in partial["elements"] if e["name"] == "Wall-Party")
        self.assertEqual(wall["audit"]["color"], "#f28c0d")

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

    def test_cli_binds_content_identity_across_request_path_forms(self) -> None:
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
            # Content identity also binds without a request-path workaround.
            self.assertEqual(
                cli_main(["view", MODEL, "-o", out, "--variations", "0",
                          "--decision", response_path]),
                0,
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


class FrameTests(unittest.TestCase):
    """Projections state their frame and behave consistently under a change of it."""

    @classmethod
    def setUpClass(cls) -> None:
        from gat.geometry.stateio import derive_scene

        cls.world = GatSession.load_ifc(MODEL).world
        cls.scene = derive_scene(cls.world, spacing=0.75)

    @staticmethod
    def transformed(scene, angle=0.0, shift=(0.0, 0.0, 0.0), scale=1.0):
        """The same scene expressed in another frame: rotated about +Z, then
        scaled (a unit change), then translated."""
        from types import SimpleNamespace

        import numpy as np

        from gat.geometry.gaussianize import OrientedBox, rot_z

        rotation = rot_z(angle)
        elements = []
        for element in scene.elements:
            origin = rotation @ np.asarray(element.box.origin, dtype=float) * scale + np.asarray(shift)
            box = OrientedBox(
                tuple(float(v) for v in origin),
                element.box.angle + angle,
                tuple(float(v) * scale for v in element.box.extents),
            )
            elements.append(SimpleNamespace(entity_id=element.entity_id, box=box))
        return SimpleNamespace(elements=tuple(elements))

    def test_frame_record_states_what_the_adapter_recorded(self) -> None:
        from gat.geometry.viewer import frame_record

        frame = frame_record(self.world)
        self.assertEqual(frame["id"], "model")
        self.assertIsNone(frame["parent"])
        self.assertIsNone(frame["crs"])
        self.assertEqual(frame["units"], "m")
        self.assertEqual(frame["source_unit"], "METRE")
        self.assertEqual(frame["scale_to_metres"], 1.0)
        self.assertEqual((frame["up"], frame["handedness"]), ("+Z", "right"))
        self.assertIn("dimensions only", frame["uncertainty"])
        payload = viewer_payload(self.world, n=0)
        self.assertEqual(payload["frame"], frame)

    def test_box_center_agrees_with_the_engine(self) -> None:
        import numpy as np

        from gat.geometry.viewer import _box_center

        for element in self.scene.elements:
            np.testing.assert_allclose(_box_center(element.box), element.box.center(), atol=1e-12)

    def test_explode_offsets_are_translation_invariant(self) -> None:
        from gat.geometry.viewer import explode_offsets

        base = explode_offsets(self.world, self.scene)
        moved = explode_offsets(self.world, self.transformed(self.scene, shift=(12.5, -3.0, 2.0)))
        self.assertEqual(base, moved)

    def test_explode_offsets_rotate_with_the_scene(self) -> None:
        import numpy as np

        from gat.geometry.gaussianize import rot_z
        from gat.geometry.viewer import explode_offsets

        angle = 0.7
        base = np.asarray(explode_offsets(self.world, self.scene))
        rotated = np.asarray(explode_offsets(self.world, self.transformed(self.scene, angle=angle)))
        expected = base @ rot_z(angle).T
        np.testing.assert_allclose(rotated, expected, atol=2e-3)

    def test_explode_offsets_scale_with_the_unit(self) -> None:
        import numpy as np

        from gat.geometry.viewer import explode_offsets

        base = np.asarray(explode_offsets(self.world, self.scene))
        in_mm = np.asarray(explode_offsets(self.world, self.transformed(self.scene, scale=1000.0)))
        np.testing.assert_allclose(in_mm / 1000.0, base, atol=2e-3)

    def test_display_frame_change_is_not_evidence(self) -> None:
        """Nothing frame-related touches the world: same digest, same belief."""
        from gat.geometry.viewer import explode_offsets, frame_record

        before = self.world.digest()
        explode_offsets(self.world, self.transformed(self.scene, angle=1.1, shift=(3, 4, 5), scale=0.001))
        frame_record(self.world)
        self.assertEqual(self.world.digest(), before)


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

    def test_viewer_speaks_the_workbench_message_contract(self) -> None:
        from gat.geometry.viewer import render_viewer_html

        world = GatSession.load_ifc(MODEL).world
        html = render_viewer_html(viewer_payload(world, n=0))
        self.assertIn('"gat-workbench-message-v1"', html)
        for phrase in ('kind: "ready"', 'kind: "selection"', 'message.kind !== "select"',
                       "message.world_digest !== SCENE.world_digest"):
            self.assertIn(phrase, html)

    def test_viewer_explodes_as_a_reading_offset(self) -> None:
        from gat.geometry.viewer import render_viewer_html

        world = GatSession.load_ifc(MODEL).world
        html = render_viewer_html(viewer_payload(world, n=0))
        self.assertIn('"frame":{"id":"model"', html)
        self.assertIn("SCENE.frame.uncertainty", html)
        for phrase in ("uExplode * aExplode", 'id="explode"', "a displacement is not a position",
                       "displacedBox(state.sample, index)", "for reading; not a position"):
            self.assertIn(phrase, html)

    def test_cli_view_audit_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "viewer.html")
            self.assertEqual(
                cli_main(["view", MODEL, "-o", path, "--variations", "0", "--audit"]), 0
            )
            with open(path, encoding="utf-8") as handle:
                html = handle.read()
            self.assertIn('"audit":{"format":"gat-ifc-audit-v1","matched":8,"elements":8}', html)
            self.assertIn('"audit":{"status":"READY","color":"#1ab233"}', html)

    def test_cli_view_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "viewer.html")
            self.assertEqual(
                cli_main(["view", MODEL, "-o", path, "--variations", "2"]), 0
            )
            self.assertTrue(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
