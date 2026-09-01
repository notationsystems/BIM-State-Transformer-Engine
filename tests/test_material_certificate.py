"""Closed material-certificate ingestion and provenance preservation."""

from __future__ import annotations

import json
import os
import unittest

import gat.demo
from gat.engineering import parse_material_certificate, read_material_certificate
from gat.errors import CertificateIngestionError
from gat.session import GatSession


MODEL = os.path.join(os.path.dirname(gat.demo.__file__), "beam_model.ifc")
CERTIFICATE = os.path.join(
    os.path.dirname(gat.demo.__file__),
    "material_certificate.json",
)


class MaterialCertificateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session = GatSession.load_ifc(MODEL)
        with open(CERTIFICATE, "rb") as stream:
            self.source = stream.read()

    def test_ingestion_binds_artifact_identity_and_full_certificate_context(self) -> None:
        certificate = read_material_certificate(CERTIFICATE)
        evidence = certificate.to_evidence(self.session.world)
        provenance = evidence.provenance()

        self.assertEqual(evidence.observation.subject.entity.global_id, "GATBEAMELEMENT00000100")
        self.assertEqual(evidence.observation.observed_value, 325.0)
        self.assertEqual(evidence.observation.noise_sigma, 2.0)
        record = provenance["material_certificate"]
        self.assertEqual(record["issuer"]["organization_id"], "LAB-STRUCTURAL-001")
        self.assertEqual(record["material"]["batch_id"], "HEAT-B1-2026-08")
        self.assertEqual(record["material"]["specimen_id"], "COUPON-B1-17")
        self.assertEqual(record["calibration"]["calibration_id"], "CAL-UTM-2026-08")
        self.assertEqual(record["unit"], "MPa")
        self.assertFalse(record["assurance"]["signature_verified"])
        self.assertFalse(record["assurance"]["authorizes_engineering_decision"])

    def test_unknown_or_duplicate_fields_fail_closed(self) -> None:
        document = json.loads(self.source)
        document["unreviewed"] = True
        with self.assertRaisesRegex(CertificateIngestionError, "extra"):
            parse_material_certificate(json.dumps(document).encode("utf-8"))
        duplicate = self.source.replace(
            b'"certificate_id": "MAT-CERT-B1-325",',
            b'"certificate_id": "MAT-CERT-B1-325",\n  "certificate_id": "forged",',
        )
        with self.assertRaisesRegex(CertificateIngestionError, "duplicate"):
            parse_material_certificate(duplicate)

    def test_certificate_for_another_beam_cannot_condition_this_world(self) -> None:
        document = json.loads(self.source)
        document["subject"]["ifc_global_id"] = "ANOTHER-BEAM"
        certificate = parse_material_certificate(json.dumps(document).encode("utf-8"))
        with self.assertRaisesRegex(CertificateIngestionError, "absent"):
            certificate.to_evidence(self.session.world)


if __name__ == "__main__":
    unittest.main()
