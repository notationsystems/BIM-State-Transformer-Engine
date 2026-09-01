"""Pinned public-model compatibility baselines, enabled by CI corpus fetch."""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import unittest

from gat.ifc_audit import audit_ifc_file


ROOT = os.environ.get("GAT_IFC_VALIDATION_ROOT")
MANIFEST = Path(__file__).parents[1] / "validation" / "ifc-corpus-v1.json"


class PublicIfcCorpusManifestTests(unittest.TestCase):
    def test_ci_corpus_includes_a_measured_multistorey_beam_model(self) -> None:
        document = json.loads(MANIFEST.read_text(encoding="utf-8"))
        candidates = [
            model
            for model in document["models"]
            if model.get("ci")
            and model.get("use_case") == "multi-storey-structural-beam"
        ]
        self.assertEqual(len(candidates), 1)
        expected = candidates[0]["expected"]
        self.assertGreaterEqual(expected["type_counts"]["IFCBUILDINGSTOREY"], 2)
        self.assertGreater(expected["type_counts"]["IFCBEAM"], 0)
        self.assertEqual(
            expected["opt_in_product_candidate_counts"]["IFCBEAM"],
            expected["type_counts"]["IFCBEAM"],
        )
        self.assertEqual(
            candidates[0]["validation_status"],
            "MEASURED_REAL_STRUCTURAL_BASELINE",
        )


@unittest.skipUnless(ROOT, "public IFC corpus not fetched")
class PublicIfcCorpusTests(unittest.TestCase):
    def test_commit_pinned_public_models_match_measured_baselines(self) -> None:
        document = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for model in document["models"]:
            path = Path(ROOT) / model["destination"]
            if not model["ci"] and not path.exists():
                continue
            with self.subTest(model=model["id"]):
                report = audit_ifc_file(path)
                expected = model["expected"]
                serialized = report.to_dict()
                actual_statuses = Counter(entity.status.value for entity in report.entities)
                self.assertEqual(report.source_sha256, model["sha256"])
                self.assertEqual(report.size_bytes, model["size_bytes"])
                self.assertEqual(report.schema, expected["schema"])
                self.assertEqual(report.pipeline_ready, expected["pipeline_ready"])
                self.assertEqual(len(report.entities), expected["supported_product_count"])
                self.assertEqual(dict(actual_statuses), expected["status_counts"])
                self.assertEqual(dict(report.issue_counts), expected["issue_counts"])
                self.assertEqual(len(report.length_units), 1)
                self.assertEqual(
                    report.length_units[0].scale_to_metres,
                    expected["length_scale_to_metres"],
                )
                self.assertEqual(
                    report.length_units[0].to_dict()["normalization_required"],
                    expected["length_unit_normalization_required"],
                )
                self.assertTrue(
                    report.length_units[0].accepted_by_current_adapter
                )
                self.assertFalse(
                    serialized["assurance"]["audit_authorizes_decisions"]
                )
                if "instance_count" in expected:
                    self.assertEqual(
                        serialized["inventory"]["instance_count"],
                        expected["instance_count"],
                    )
                for ifc_type, count in expected.get("type_counts", {}).items():
                    self.assertEqual(
                        serialized["inventory"]["type_counts"].get(ifc_type),
                        count,
                    )
                if "opt_in_product_candidate_counts" in expected:
                    self.assertEqual(
                        serialized["inventory"]["opt_in_product_candidate_counts"],
                        expected["opt_in_product_candidate_counts"],
                    )
                if "pipeline_lowering_status" in expected:
                    self.assertEqual(
                        serialized["pipeline"]["lowering"]["status"],
                        expected["pipeline_lowering_status"],
                    )
                    self.assertEqual(
                        serialized["pipeline"]["lowering"]["error_type"],
                        expected["pipeline_lowering_error_type"],
                    )
                if "beam_geometry_status_counts" in expected:
                    beam_geometry = serialized["inventory"]["beam_geometry"]
                    self.assertIsNotNone(beam_geometry)
                    self.assertEqual(
                        beam_geometry["status_counts"],
                        expected["beam_geometry_status_counts"],
                    )
                    self.assertEqual(
                        beam_geometry["derived_quantity_counts"],
                        expected["beam_geometry_quantity_counts"],
                    )
                    self.assertFalse(
                        beam_geometry["authorizes_structural_decisions"]
                    )


if __name__ == "__main__":
    unittest.main()
