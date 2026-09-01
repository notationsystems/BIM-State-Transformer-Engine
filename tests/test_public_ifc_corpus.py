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
                    report.to_dict()["assurance"]["audit_authorizes_decisions"]
                )


if __name__ == "__main__":
    unittest.main()
