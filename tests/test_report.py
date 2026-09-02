"""The shared design language: one palette, fail-closed human rendering."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import gat.demo
from gat import report
from gat.cli import main as cli_main
from gat.headless import REQUEST_FORMAT, handle_request


ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "integrations" / "blender" / "gat_assurance" / "bridge.py"
MODEL = os.path.join(os.path.dirname(gat.demo.__file__), "model.ifc")
BEAM_MODEL = os.path.join(os.path.dirname(gat.demo.__file__), "beam_model.ifc")
MATERIAL_CERTIFICATE = os.path.join(
    os.path.dirname(gat.demo.__file__),
    "material_certificate.json",
)

spec = importlib.util.spec_from_file_location("gat_report_bridge", BRIDGE)
assert spec is not None and spec.loader is not None
bridge = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bridge
spec.loader.exec_module(bridge)


def request(operation: str, payload: dict, model: str = MODEL) -> dict:
    return {
        "format": REQUEST_FORMAT,
        "request_id": f"report-{operation}",
        "operation": operation,
        "state": {"kind": "ifc", "path": model},
        "payload": payload,
    }


def acceptance_response() -> dict:
    return handle_request(
        request(
            "acceptance",
            {
                "case_id": "opening-fit-1",
                "workflow": "OPENING_VERIFICATION",
                "subject": "Door-1 into Opening-1",
                "checks": [
                    {
                        "kind": "difference",
                        "check_id": "width",
                        "lhs": {"entity_name": "Opening-1", "quantity": "Width"},
                        "rhs": {"entity_name": "Door-1", "quantity": "Width"},
                        "minimum_margin": 0.05,
                        "confidence": 0.95,
                        "label": "opening width fit",
                    }
                ],
            },
        )
    )


def beam_response() -> dict:
    return handle_request(
        request(
            "beam_assurance",
            {
                "case_id": "beam-b1-certificate",
                "beam_name": "Beam-B1",
                "factored_demand_n_m": 301_000.0,
                "confidence": 0.95,
                "material_certificate_path": MATERIAL_CERTIFICATE,
                "label": "Beam-B1 factored bending",
            },
            model=BEAM_MODEL,
        )
    )


def change_response() -> dict:
    return handle_request(
        request(
            "change_impact",
            {
                "change": {
                    "op": "set_parameter",
                    "target": {"entity_name": "Wall-Party", "quantity": "Length"},
                    "value": 5.2,
                    "design_sigma": 0.01,
                }
            },
        )
    )


class PaletteLockstepTests(unittest.TestCase):
    def test_shared_terms_match_the_blender_panel_bit_for_bit(self) -> None:
        for term in (
            "ACCEPT",
            "REJECT",
            "REQUEST_EVIDENCE",
            "SATISFIED",
            "VIOLATED",
            "UNRESOLVED",
        ):
            self.assertEqual(
                report.disposition_color(term),
                bridge.disposition_color(term),
                term,
            )

    def test_unknown_vocabulary_is_refused_not_guessed(self) -> None:
        with self.assertRaises(ValueError):
            report.disposition_color("MAYBE")
        with self.assertRaises(ValueError):
            report.disposition_hex("MAYBE")

    def test_hex_values_match_the_documented_palette(self) -> None:
        self.assertEqual(report.disposition_hex("ACCEPT"), "#1ab233")
        self.assertEqual(report.disposition_hex("VIOLATED"), "#d91414")
        self.assertEqual(report.disposition_hex("UNRESOLVED"), "#f28c0d")
        self.assertEqual(report.disposition_hex("ERROR"), "#595959")


class SummaryRenderingTests(unittest.TestCase):
    def test_summary_renders_state_and_footers(self) -> None:
        decoded = report.decode_response(handle_request(request("summary", {})))
        self.assertEqual(decoded.disposition, "PASS")
        text = report.render_text(decoded)
        self.assertIn("PASS: ", text)
        self.assertIn("12 pass / 0 warn / 0 fail", text)
        self.assertIn("24 raw + 39 derived variables", text)
        self.assertIn(report.NON_AUTHORIZING_FOOTER, text)
        self.assertIn(report.READ_ONLY_FOOTER, text)

    def test_summary_refuses_verified_claim_with_failures(self) -> None:
        response = handle_request(request("summary", {}))
        tampered = copy.deepcopy(response)
        tampered["result"]["verification"]["failure_count"] = 1
        with self.assertRaises(ValueError):
            report.decode_response(tampered)


class AcceptanceRenderingTests(unittest.TestCase):
    def test_request_evidence_case_shows_checks_and_next_evidence(self) -> None:
        decoded = report.decode_response(acceptance_response())
        self.assertEqual(decoded.disposition, "REQUEST_EVIDENCE")
        self.assertEqual(decoded.headline, "REQUEST_EVIDENCE: Door-1 into Opening-1")
        text = report.render_text(decoded)
        self.assertIn("OPENING_VERIFICATION case opening-fit-1", text)
        self.assertIn("SATISFIED", text)
        self.assertIn("next evidence", text)
        self.assertIn("gat-safe-acceptance-v1", text)
        self.assertIn(report.NON_AUTHORIZING_FOOTER, text)

    def test_inconsistent_authorization_claim_is_refused(self) -> None:
        tampered = copy.deepcopy(acceptance_response())
        tampered["result"]["may_authorize"] = True
        with self.assertRaises(ValueError):
            report.decode_response(tampered)

    def test_unknown_disposition_is_refused(self) -> None:
        tampered = copy.deepcopy(acceptance_response())
        tampered["result"]["disposition"] = "PROBABLY_FINE"
        with self.assertRaises(ValueError):
            report.decode_response(tampered)


class BeamRenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.response = beam_response()

    def test_verdict_change_capacity_and_honest_assurance_flags(self) -> None:
        decoded = report.decode_response(self.response)
        self.assertEqual(decoded.disposition, "VIOLATED")
        text = report.render_text(decoded)
        self.assertIn("VIOLATED: Beam-B1", text)
        self.assertIn("prior SATISFIED -> revised VIOLATED", text)
        self.assertIn("315.0 +- 7.9 kN*m -> 293.8 +- 3.4 kN*m", text)
        self.assertIn("0.96258 -> 0.01788", text)
        self.assertIn("ansi-aisc-360-22-f2-1-lrfd-v1", text)
        self.assertIn("issuer_trust_verified                    no", text)
        self.assertIn(report.NON_AUTHORIZING_FOOTER, text)

    def test_unverified_beam_response_is_refused(self) -> None:
        tampered = copy.deepcopy(self.response)
        tampered["result"]["verification"]["passed"] = False
        with self.assertRaises(ValueError):
            report.decode_response(tampered)

    def test_authorizing_beam_response_is_refused(self) -> None:
        tampered = copy.deepcopy(self.response)
        tampered["result"]["assurance"]["may_authorize"] = True
        with self.assertRaises(ValueError):
            report.decode_response(tampered)

    def test_world_identity_mismatch_is_refused(self) -> None:
        tampered = copy.deepcopy(self.response)
        tampered["result"]["prior"]["world_digest"] = "0" * 64
        with self.assertRaises(ValueError):
            report.decode_response(tampered)

    def test_verdict_change_claim_must_be_consistent(self) -> None:
        tampered = copy.deepcopy(self.response)
        tampered["result"]["decision_change"]["verdict_changed"] = False
        with self.assertRaises(ValueError):
            report.decode_response(tampered)


class ChangeRenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.response = change_response()

    def test_preview_renders_impacts_and_preview_footer(self) -> None:
        decoded = report.decode_response(self.response)
        self.assertEqual(decoded.disposition, "ADMISSIBLE")
        self.assertEqual(decoded.subject, "set_parameter IfcWall.Length = 5.2")
        text = report.render_text(decoded)
        self.assertIn("design-change preview", text)
        self.assertIn("target", text)
        self.assertIn("affected", text)
        self.assertIn(report.PREVIEW_FOOTER, text)
        self.assertIn(report.READ_ONLY_FOOTER, text)

    def test_admissibility_claim_must_match_failures(self) -> None:
        tampered = copy.deepcopy(self.response)
        tampered["result"]["disposition"] = "BLOCKED"
        tampered["result"]["admissible"] = False
        with self.assertRaises(ValueError):
            report.decode_response(tampered)


class HtmlRenderingTests(unittest.TestCase):
    def test_html_is_self_contained_and_script_free(self) -> None:
        decoded = report.decode_response(beam_response())
        html = report.render_html(decoded)
        self.assertTrue(html.startswith("<!doctype html>"))
        self.assertNotIn("<script", html)
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)
        self.assertIn(report.disposition_hex("VIOLATED"), html)
        self.assertIn("<details", html)
        self.assertIn(decoded.world_digest, html)

    def test_html_escapes_untrusted_response_text(self) -> None:
        tampered = copy.deepcopy(acceptance_response())
        tampered["result"]["reasons"] = ["<img src=x onerror=alert(1)>"]
        html = report.render_html(report.decode_response(tampered))
        self.assertNotIn("<img", html)
        self.assertIn("&lt;img", html)


class ErrorRenderingTests(unittest.TestCase):
    def test_error_response_renders_grey_and_undecided(self) -> None:
        decoded = report.decode_response(
            {
                "format": report.RESPONSE_FORMAT,
                "error": {"type": "ValueError", "message": "boom"},
            }
        )
        self.assertEqual(decoded.disposition, "ERROR")
        self.assertEqual(decoded.operation, "error")
        text = report.render_text(decoded)
        self.assertIn("ERROR: ValueError", text)
        self.assertIn("boom", text)
        self.assertIn(report.READ_ONLY_FOOTER, text)


class LedgerTimelineTests(unittest.TestCase):
    def build_ledger(self, tmp: str) -> str:
        from gat.causal import AssessmentRecord
        from gat.engine.transform import ShiftParameter
        from gat.session import GatSession

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

    def test_timeline_renders_chain_events_and_accents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            decoded = report.decode_ledger(self.build_ledger(tmp))
            self.assertEqual(decoded.operation, "ledger")
            self.assertEqual(decoded.disposition, "PASS")
            text = report.render_text(decoded)
            self.assertIn("0 - genesis", text)
            self.assertIn("1 - transition: shift_parameter", text)
            self.assertIn("2 - assessment", text)
            self.assertIn("hash chain verified", text)
            self.assertIn(report.READ_ONLY_FOOTER, text)
            html = report.render_html(decoded)
            self.assertNotIn("<script", html)
            self.assertIn('class="stop"', html)
            self.assertIn("VIOLATED", html)

    def test_tampered_chain_is_refused_not_drawn(self) -> None:
        from gat.errors import LedgerError

        with tempfile.TemporaryDirectory() as tmp:
            path = self.build_ledger(tmp)
            with open(path, encoding="utf-8") as handle:
                document = json.load(handle)
            document["events"][1]["operation"]["delta"] = 0.5
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(document, handle)
            with self.assertRaises(LedgerError):
                report.decode_ledger(path)
            self.assertEqual(cli_main(["ledger", path]), 2)

    def test_cli_renders_timeline_and_exits_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self.build_ledger(tmp)
            self.assertEqual(cli_main(["ledger", path, "-o", os.devnull]), 0)
            out = os.path.join(tmp, "timeline.html")
            self.assertEqual(cli_main(["ledger", path, "--html", "-o", out]), 0)
            with open(out, encoding="utf-8") as handle:
                self.assertTrue(handle.read().startswith("<!doctype html>"))


class CliReportTests(unittest.TestCase):
    def run_cli(self, *argv: str) -> int:
        return cli_main(list(argv))

    def test_report_command_renders_and_exits_zero(self) -> None:
        response = handle_request(request("summary", {}))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "response.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(response, handle)
            self.assertEqual(self.run_cli("report", path, "-o", os.devnull), 0)
            out = os.path.join(tmp, "report.html")
            self.assertEqual(self.run_cli("report", path, "--html", "-o", out), 0)
            with open(out, encoding="utf-8") as handle:
                self.assertTrue(handle.read().startswith("<!doctype html>"))

    def test_error_response_exits_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "error.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "format": report.RESPONSE_FORMAT,
                        "error": {"type": "ValueError", "message": "boom"},
                    },
                    handle,
                )
            self.assertEqual(self.run_cli("report", path, "-o", os.devnull), 1)

    def test_invalid_input_exits_two(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.json")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write('{"format": "not-a-response"}')
            self.assertEqual(self.run_cli("report", path), 2)


if __name__ == "__main__":
    unittest.main()
