"""Tests for the fail-closed, non-mutating IFC compatibility audit."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import gat.demo
from gat.cli import main as cli_main
from gat.ifc_audit import AuditStatus, EntityStatus, audit_ifc_file, audit_ifc_text


MODEL = os.path.join(os.path.dirname(gat.demo.__file__), "model.ifc")
BEAM_MODEL = os.path.join(os.path.dirname(gat.demo.__file__), "beam_model.ifc")


class IfcAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with open(MODEL, "r", encoding="utf-8") as stream:
            cls.demo_text = stream.read()

    def test_demo_is_pipeline_ready_but_audit_never_authorizes(self) -> None:
        report = audit_ifc_file(MODEL)
        document = report.to_dict()
        self.assertTrue(report.pipeline_ready)
        self.assertEqual(report.schema, "IFC4")
        self.assertEqual(len(report.entities), 10)
        self.assertTrue(all(entity.status is EntityStatus.READY for entity in report.entities))
        self.assertFalse(document["assurance"]["audit_authorizes_decisions"])
        self.assertTrue(document["assurance"]["requires_explicit_decision_scope"])

    def test_audit_is_byte_deterministic(self) -> None:
        first = audit_ifc_text(self.demo_text, source="same.ifc")
        second = audit_ifc_text(self.demo_text, source="same.ifc")
        self.assertEqual(first.to_json(), second.to_json())
        self.assertEqual(first.world_digest, second.world_digest)

    def test_opt_in_structural_beam_is_audited_as_authoritative_scope(self) -> None:
        report = audit_ifc_file(BEAM_MODEL)
        beam = next(
            entity for entity in report.entities
            if entity.canonical_class == "IfcBeam"
        )
        document = report.to_dict()

        self.assertTrue(report.pipeline_ready)
        self.assertEqual(beam.status, EntityStatus.READY)
        self.assertEqual(
            document["adapter_scope"]["opt_in_ifc_product_types"]["IFCBEAM"],
            "GAT_Structural",
        )

    def test_prefixed_units_are_normalized_without_hiding_inventory(self) -> None:
        millimetres = self.demo_text.replace(
            "IFCSIUNIT(*,.LENGTHUNIT.,$,.METRE.)",
            "IFCSIUNIT(*,.LENGTHUNIT.,.MILLI.,.METRE.)",
        )
        report = audit_ifc_text(millimetres)
        self.assertTrue(report.pipeline_ready)
        self.assertEqual(report.lowering.status, AuditStatus.PASS)
        self.assertNotIn("LENGTH_UNIT_NORMALIZATION_REQUIRED", dict(report.issue_counts))
        self.assertEqual(len(report.entities), 10)
        self.assertTrue(report.length_units[0].scale_to_metres == 0.001)
        self.assertTrue(report.length_units[0].accepted_by_current_adapter)

    def test_missing_quantity_is_classified_by_geometry_availability(self) -> None:
        missing = self.demo_text.replace(
            "#111=IFCQUANTITYLENGTH('Width',$,$,0.3);",
            "#111=IFCQUANTITYLENGTH('Thickness',$,$,0.3);",
        )
        report = audit_ifc_text(missing)
        wall = next(entity for entity in report.entities if entity.step_id == 100)
        self.assertEqual(wall.status, EntityStatus.MISSING_SOURCE_DATA)
        self.assertEqual(wall.missing_quantities, ("Width",))
        self.assertFalse(wall.has_geometry_representation)
        self.assertFalse(report.pipeline_ready)

    def test_multi_storey_limitation_is_reported_before_lowering_stops(self) -> None:
        second_storey = (
            "#999=IFCBUILDINGSTOREY('GATSTY0000000000000999',$,'Level 2',$,$,$,$,$,$,$);\n"
        )
        text = self.demo_text.replace("ENDSEC;\nEND-ISO", second_storey + "ENDSEC;\nEND-ISO")
        report = audit_ifc_text(text)
        self.assertEqual(dict(report.issue_counts)["UNSUPPORTED_STOREY_COUNT"], 1)
        self.assertEqual(
            sum(entity.canonical_class == "IfcBuildingStorey" for entity in report.entities),
            2,
        )
        self.assertFalse(report.pipeline_ready)

    def test_parse_failure_is_a_structured_report(self) -> None:
        report = audit_ifc_text("not an IFC file")
        self.assertEqual(report.parse.status, AuditStatus.BLOCKED)
        self.assertEqual(dict(report.issue_counts), {"PARSE_FAILED": 1})
        self.assertEqual(report.lowering.status, AuditStatus.NOT_RUN)

    def test_malformed_project_unit_context_is_a_structured_finding(self) -> None:
        project = (
            "#1=IFCPROJECT('GATPRJ0000000000000001',$,'GAT Demo Project',"
            "$,$,$,$,$,#997);"
        )
        text = self.demo_text.replace(
            "#1=IFCPROJECT('GATPRJ0000000000000001',$,'GAT Demo Project',$,$,$,$,$,$);",
            project,
        ).replace(
            "ENDSEC;\nEND-ISO",
            "#997=IFCUNITASSIGNMENT((42));\nENDSEC;\nEND-ISO",
        )
        report = audit_ifc_text(text)
        self.assertEqual(dict(report.issue_counts)["UNSUPPORTED_LENGTH_UNIT"], 1)
        self.assertEqual(report.length_units, ())
        self.assertEqual(report.lowering.status, AuditStatus.BLOCKED)

    def test_cli_writes_machine_readable_report_and_meaningful_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "audit.json")
            status = cli_main(["audit", MODEL, "--compact", "--output", output])
            self.assertEqual(status, 0)
            with open(output, "r", encoding="utf-8") as stream:
                document = json.load(stream)
        self.assertEqual(document["format"], "gat-ifc-audit-v1")
        self.assertTrue(document["pipeline"]["pipeline_ready"])


if __name__ == "__main__":
    unittest.main()
