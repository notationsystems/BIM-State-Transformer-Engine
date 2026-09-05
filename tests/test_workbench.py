"""The Notation Workbench: eight modes, one identity, honest availability."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import gat.demo
from gat.cli import main as cli_main
from gat.geometry.viewer import decision_overlay, viewer_payload
from gat.report import (
    NON_AUTHORIZING_FOOTER,
    READ_ONLY_FOOTER,
    decode_ledger,
    decode_response,
    render_html,
    render_html_fragment,
)
from gat.session import GatSession
from gat.workbench import (
    AVAILABLE,
    EMPTY,
    MESSAGE_FORMAT,
    MODES,
    PROJECTION_SPEC_VERSION,
    UNAVAILABLE,
    WORKBENCH_FORMAT,
    export_workbench_html,
    graph_payload,
    projection_specs,
    render_workbench_html,
    state_payload,
    workbench_payload,
)


MODEL = os.path.join(os.path.dirname(gat.demo.__file__), "model.ifc")
WALL_PARTY = "IfcWall:GATWAL0000000000000180"


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


def build_ledger(tmp: str) -> str:
    from gat.causal import AssessmentRecord
    from gat.engine.transform import ShiftParameter

    session = GatSession.load_ifc(MODEL)
    session.run(
        ShiftParameter(session.var("Wall-Party", "Length"), 0.1),
        provenance={"phase": "test-shift"},
    )
    session.record_assessment(
        AssessmentRecord(
            world_digest=session.world.digest(),
            assessment_id="fit-1",
            assessment_type="test-assessment",
            subject="Wall-Party",
            verdict="VIOLATED",
            method="test-method-v1",
        ),
        provenance={"phase": "test-assessment"},
    )
    path = os.path.join(tmp, "ledger.json")
    session.export_ledger(path)
    return path


class ProjectionSpecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.world = GatSession.load_ifc(MODEL).world

    def specs(self, **bound):
        flags = {"decision_bound": False, "ledger_bound": False, "audit_bound": False}
        flags.update(bound)
        return projection_specs(self.world, **flags)

    def test_modes_follow_the_synthesis_order(self) -> None:
        self.assertEqual(
            MODES,
            ("MAP", "GLOBE", "STRUCTURE", "GRAPH", "STATE", "TIME", "EVIDENCE", "COMPLEXITY"),
        )
        self.assertEqual(tuple(spec.mode for spec in self.specs()), MODES)

    def test_geodetic_modes_are_unavailable_with_their_reason(self) -> None:
        by_mode = {spec.mode: spec for spec in self.specs()}
        for mode in ("MAP", "GLOBE"):
            self.assertEqual(by_mode[mode].availability, UNAVAILABLE)
            self.assertIn("IfcSite", by_mode[mode].reason)
            self.assertIn("coordinate", by_mode[mode].reason)
        self.assertTrue(by_mode["GLOBE"].surface_class.startswith("connected instrument"))
        self.assertIn("network", by_mode["GLOBE"].reason)

    def test_unbound_modes_are_empty_and_say_what_fills_them(self) -> None:
        by_mode = {spec.mode: spec for spec in self.specs()}
        for mode in ("STRUCTURE", "GRAPH", "STATE"):
            self.assertEqual(by_mode[mode].availability, AVAILABLE)
        self.assertEqual(by_mode["TIME"].availability, EMPTY)
        self.assertIn("--ledger", by_mode["TIME"].reason)
        self.assertEqual(by_mode["EVIDENCE"].availability, EMPTY)
        self.assertIn("--decision", by_mode["EVIDENCE"].reason)
        self.assertEqual(by_mode["COMPLEXITY"].availability, EMPTY)
        bound = {
            spec.mode: spec
            for spec in self.specs(decision_bound=True, ledger_bound=True, audit_bound=True)
        }
        for mode in ("TIME", "EVIDENCE", "COMPLEXITY"):
            self.assertEqual(bound[mode].availability, AVAILABLE)
            self.assertEqual(bound[mode].reason, "")

    def test_every_mode_declares_its_loss_frame_and_time(self) -> None:
        for spec in self.specs(decision_bound=True, ledger_bound=True, audit_bound=True):
            for field in ("source", "transformation", "meaning", "loss", "identity",
                          "frame", "time"):
                self.assertTrue(getattr(spec, field), f"{spec.mode}.{field}")
        by_mode = {spec.mode: spec for spec in self.specs()}
        self.assertIn("marginals only", by_mode["STATE"].loss)
        self.assertIn("carry no information", by_mode["GRAPH"].loss)
        self.assertIn("no geodetic frame", by_mode["STRUCTURE"].frame)
        # frame text is derived from the stated frame record, not typed by hand
        self.assertIn("corner-origin box with yaw about +Z", by_mode["STRUCTURE"].frame)
        self.assertIn("dimensions only", by_mode["STRUCTURE"].frame)
        self.assertIn("METRE x 1.0 -> m", by_mode["STATE"].frame)

    def test_spec_dict_declares_version_and_no_mutation(self) -> None:
        record = self.specs()[0].to_dict()
        self.assertEqual(record["version"], PROJECTION_SPEC_VERSION)
        self.assertIs(record["mutates_source"], False)
        self.assertEqual(
            set(record),
            {"version", "mode", "seat", "question", "surface_class", "source",
             "transformation", "meaning", "loss", "identity", "frame", "time",
             "availability", "reason", "mutates_source"},
        )


class GraphPayloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.world = GatSession.load_ifc(MODEL).world
        cls.graph = graph_payload(cls.world)

    def test_layout_is_deterministic_and_bounded(self) -> None:
        self.assertEqual(self.graph, graph_payload(self.world))
        self.assertGreaterEqual(self.graph["rows"], 3)
        self.assertEqual(len(self.graph["row_classes"]), self.graph["rows"])
        for node in self.graph["nodes"]:
            self.assertLess(node["row"], self.graph["rows"])
            self.assertLess(node["column"], node["columns"])
            self.assertIn(node["class"], self.graph["row_classes"][node["row"]])

    def test_edges_are_typed_and_reference_known_nodes(self) -> None:
        ids = {node["entity"] for node in self.graph["nodes"]}
        self.assertEqual(len(self.graph["edges"]), len(self.world.module.rels))
        for edge in self.graph["edges"]:
            self.assertIn(edge["source"], ids)
            self.assertIn(edge["target"], ids)
            self.assertIn("source_ref", edge)
        self.assertEqual(
            sum(entry["count"] for entry in self.graph["kinds"]), len(self.graph["edges"])
        )
        self.assertEqual(
            [entry["kind"] for entry in self.graph["kinds"]],
            ["aggregates", "contains", "bounds", "voids", "fills"],
        )

    def test_containers_read_above_what_they_contain(self) -> None:
        row = {node["entity"]: node["row"] for node in self.graph["nodes"]}
        for edge in self.graph["edges"]:
            if edge["kind"] in ("aggregates", "contains"):  # container above content
                self.assertLess(row[edge["source"]], row[edge["target"]], edge)
            if edge["kind"] == "bounds":  # wall bounds space: the wall reads below
                self.assertGreater(row[edge["source"]], row[edge["target"]], edge)
            if edge["kind"] == "voids":  # opening voids wall: below the wall
                self.assertGreater(row[edge["source"]], row[edge["target"]], edge)
            if edge["kind"] == "fills":  # door fills opening: below the opening
                self.assertGreater(row[edge["source"]], row[edge["target"]], edge)


class StatePayloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.world = GatSession.load_ifc(MODEL).world
        cls.state = state_payload(cls.world)

    def test_identity_survives_representation(self) -> None:
        state_ids = {entity["entity"] for entity in self.state["entities"]}
        graph_ids = {node["entity"] for node in graph_payload(self.world)["nodes"]}
        viewer_ids = {
            element["entity"] for element in viewer_payload(self.world, n=0)["elements"]
        }
        self.assertEqual(state_ids, graph_ids)
        self.assertTrue(viewer_ids <= state_ids)
        self.assertEqual(self.state["frame"]["id"], "model")
        self.assertEqual(self.state["frame"], viewer_payload(self.world, n=0)["frame"])
        self.assertIn(WALL_PARTY, viewer_ids)
        self.assertEqual(self.state["world_digest"], self.world.digest())

    def test_quantities_carry_mean_sigma_role_and_unit(self) -> None:
        wall = next(e for e in self.state["entities"] if e["entity"] == WALL_PARTY)
        self.assertEqual(wall["name"], "Wall-Party")
        by_name = {q["name"]: q for q in wall["quantities"]}
        length = by_name["Length"]
        self.assertEqual(length["role"], "raw")
        self.assertEqual(length["unit"], "m")
        self.assertAlmostEqual(length["mean"], 4.0, places=2)
        self.assertGreater(length["sigma"], 0)
        volume = by_name["NetVolume"]
        self.assertEqual(volume["role"], "derived")
        self.assertEqual(volume["unit"], "m3")
        self.assertGreater(self.state["raw"], 0)
        self.assertGreater(self.state["derived"], 0)


class WorkbenchDocumentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from gat.headless import handle_request
        from gat.ifc_audit import audit_ifc_file

        cls.world = GatSession.load_ifc(MODEL).world
        cls.request = clearance_request()
        cls.response = handle_request(cls.request)
        cls.decision = decision_overlay(cls.world, cls.response, cls.request)
        cls.decision_report = decode_response(cls.response)
        cls.audit = decode_response(audit_ifc_file(MODEL).to_dict())
        cls.tmp = tempfile.TemporaryDirectory()
        cls.ledger = decode_ledger(build_ledger(cls.tmp.name))
        cls.payload = workbench_payload(
            cls.world,
            model_name="model.ifc",
            n=2,
            decision=cls.decision,
            decision_report=cls.decision_report,
            ledger=cls.ledger,
            audit=cls.audit,
        )
        cls.html = render_workbench_html(
            cls.payload,
            decision_report=cls.decision_report,
            ledger=cls.ledger,
            audit=cls.audit,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_payload_is_deterministic_and_complete(self) -> None:
        again = workbench_payload(
            self.world,
            model_name="model.ifc",
            n=2,
            decision=self.decision,
            decision_report=self.decision_report,
            ledger=self.ledger,
            audit=self.audit,
        )
        self.assertEqual(self.payload, again)
        self.assertEqual(self.payload["format"], WORKBENCH_FORMAT)
        self.assertEqual(self.payload["message_format"], MESSAGE_FORMAT)
        self.assertEqual([m["mode"] for m in self.payload["modes"]], list(MODES))
        self.assertTrue(all(m["availability"] == AVAILABLE
                            for m in self.payload["modes"] if m["mode"] not in ("MAP", "GLOBE")))
        self.assertEqual(self.payload["decision"]["disposition"], "REJECT")
        self.assertEqual(self.payload["decision"]["subjects"], ["Wall-Party"])
        self.assertEqual(self.payload["structure"]["world_digest"], self.world.digest())

    def test_document_is_one_offline_file(self) -> None:
        self.assertTrue(self.html.startswith("<!doctype html>"))
        self.assertNotIn("http://", self.html)
        self.assertNotIn("https://", self.html)
        self.assertIn('sandbox="allow-scripts"', self.html)
        self.assertIn("srcdoc=", self.html)
        self.assertIn(WORKBENCH_FORMAT, self.html)
        self.assertIn(MESSAGE_FORMAT, self.html)
        self.assertIn(READ_ONLY_FOOTER, self.html)
        self.assertIn(NON_AUTHORIZING_FOOTER, self.html)
        self.assertIn("frame model (m, +Z up, no CRS)", self.html)
        for rule in self.payload["rules"]:
            self.assertIn(rule, self.html)

    def test_every_mode_has_a_tab_and_a_panel(self) -> None:
        for mode in MODES:
            self.assertIn(f'role="tab" data-mode="{mode}"', self.html)
            self.assertIn(f'class="panel" data-mode="{mode}" role="tabpanel"', self.html)
        self.assertIn('data-mode="MAP" class="unavailable"', self.html)
        self.assertIn('data-mode="STRUCTURE" class="available"', self.html)

    def test_reports_compose_byte_identically(self) -> None:
        for report in (self.decision_report, self.ledger, self.audit):
            fragment = render_html_fragment(report)
            self.assertIn(fragment, self.html)
            self.assertIn(fragment, render_html(report))
        self.assertIn("hash chain verified", self.html)
        self.assertIn("REJECT", self.html)

    def test_decision_and_report_bind_together_or_not_at_all(self) -> None:
        with self.assertRaisesRegex(ValueError, "bound together"):
            workbench_payload(self.world, n=0, decision=self.decision)
        with self.assertRaisesRegex(ValueError, "bound together"):
            workbench_payload(self.world, n=0, decision_report=self.decision_report)
        with self.assertRaisesRegex(ValueError, "disagree"):
            workbench_payload(
                self.world, n=0, decision=self.decision, decision_report=self.ledger
            )

    def test_unbound_modes_state_their_reason_in_the_page(self) -> None:
        payload = workbench_payload(self.world, n=0, audit_reason="The IFC audit was skipped.")
        html = render_workbench_html(payload)
        self.assertIn("No decision is bound", html)
        self.assertIn("No execution ledger is bound", html)
        self.assertIn("The IFC audit was skipped.", html)
        self.assertIn('data-mode="TIME" class="empty"', html)
        self.assertIn("no decision bound", html)

    def test_structure_scene_carries_audit_statuses_only_with_the_audit(self) -> None:
        from gat.geometry.viewer import audit_statuses
        from gat.ifc_audit import audit_ifc_file

        statuses = audit_statuses(audit_ifc_file(MODEL).to_dict())
        payload = workbench_payload(self.world, n=0, audit=self.audit, audit_statuses=statuses)
        self.assertEqual(payload["structure"]["audit"]["matched"], 8)
        self.assertIn("EXPLODE", payload["modes"][2]["transformation"])
        with self.assertRaisesRegex(ValueError, "bind both or neither"):
            workbench_payload(self.world, n=0, audit_statuses=statuses)

    def test_untrusted_names_are_escaped(self) -> None:
        payload = json.loads(json.dumps(workbench_payload(self.world, n=0)))
        payload["model"] = "<img src=x onerror=alert(1)>"
        html = render_workbench_html(payload)
        self.assertNotIn("<img", html)
        self.assertIn("&lt;img", html)


class WorkbenchCliTests(unittest.TestCase):
    def test_cli_writes_the_instrument_with_every_binding(self) -> None:
        from gat.headless import handle_request

        request = clearance_request("cli-route")
        response = handle_request(request)
        with tempfile.TemporaryDirectory() as tmp:
            request_path = os.path.join(tmp, "request.json")
            response_path = os.path.join(tmp, "response.json")
            with open(request_path, "w", encoding="utf-8") as handle:
                json.dump(request, handle)
            with open(response_path, "w", encoding="utf-8") as handle:
                json.dump(response, handle)
            ledger_path = build_ledger(tmp)
            out = os.path.join(tmp, "workbench.html")
            self.assertEqual(
                cli_main(["workbench", MODEL, "-o", out, "--variations", "1",
                          "--decision", response_path, "--request", request_path,
                          "--ledger", ledger_path]),
                0,
            )
            with open(out, encoding="utf-8") as handle:
                html = handle.read()
            self.assertIn('"disposition":"REJECT"', html)
            self.assertIn("hash chain verified", html)
            self.assertIn("gat-ifc-audit-v1", html)
            self.assertIn("&quot;audit&quot;:{&quot;status&quot;:&quot;READY&quot;", html)
            self.assertNotIn("No decision is bound", html)
            # --request without --decision has nothing to bind: refused
            self.assertEqual(
                cli_main(["workbench", MODEL, "-o", out, "--request", request_path]), 2
            )

    def test_cli_no_audit_leaves_complexity_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "workbench.html")
            self.assertEqual(
                cli_main(["workbench", MODEL, "-o", out, "--variations", "0", "--no-audit"]), 0
            )
            with open(out, encoding="utf-8") as handle:
                html = handle.read()
            self.assertIn("--no-audit", html)
            self.assertIn('data-mode="COMPLEXITY" class="empty"', html)
            self.assertIn("No decision is bound", html)

    def test_cli_refuses_a_tampered_ledger_and_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = build_ledger(tmp)
            with open(ledger_path, encoding="utf-8") as handle:
                document = json.load(handle)
            document["events"][1]["operation"]["delta"] = 0.5
            with open(ledger_path, "w", encoding="utf-8") as handle:
                json.dump(document, handle)
            out = os.path.join(tmp, "workbench.html")
            self.assertEqual(
                cli_main(["workbench", MODEL, "-o", out, "--variations", "0",
                          "--ledger", ledger_path]),
                2,
            )
            self.assertFalse(os.path.exists(out))

    def test_export_returns_availability_per_mode(self) -> None:
        world = GatSession.load_ifc(MODEL).world
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "workbench.html")
            availability = export_workbench_html(world, out, n=0)
            self.assertTrue(os.path.exists(out))
        self.assertEqual(list(availability), list(MODES))
        self.assertEqual(availability["MAP"], UNAVAILABLE)
        self.assertEqual(availability["STRUCTURE"], AVAILABLE)
        self.assertEqual(availability["TIME"], EMPTY)


class WalkthroughDemoTests(unittest.TestCase):
    def test_clearance_walkthrough_is_self_asserting(self) -> None:
        from gat.demo.workbench import run

        with tempfile.TemporaryDirectory() as tmp:
            result = run(tmp)
            self.assertEqual(result["disposition"], "REJECT")
            for name in ("request.json", "response.json", "ledger.json", "workbench.html"):
                self.assertTrue(os.path.exists(os.path.join(tmp, name)), name)
            with open(result["page"], encoding="utf-8") as handle:
                html = handle.read()
            self.assertIn("hash chain verified", html)
            self.assertIn('data-mode="EVIDENCE" class="available"', html)


if __name__ == "__main__":
    unittest.main()
