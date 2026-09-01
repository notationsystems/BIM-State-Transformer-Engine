"""Tests for proof-carrying computation claims.

These tests deliberately use opaque fake proof bytes.  They exercise GAT's
binding contract and prove that an absent backend verifier can never be
reported as cryptographic verification.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import gat.demo
from gat.engine.transform import ShiftParameter
from gat.errors import ProofManifestError, VerificationError
from gat.proof_manifest import (
    ComputationProofManifest,
    NumericContract,
    ProofCheckStatus,
    create_computation_proof_manifest,
    read_computation_proof_manifest,
    verify_computation_proof_manifest,
    write_computation_proof_manifest,
)
from gat.session import GatSession


MODEL = os.path.join(os.path.dirname(gat.demo.__file__), "model.ifc")


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class ComputationProofManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session = GatSession.load_ifc(MODEL)
        self.session.run(
            ShiftParameter(self.session.var("Level 1", "ClearHeight"), 0.01),
            provenance={"request": "raise-level"},
        )
        self.proof_bytes = b"opaque-sp1-proof-fixture-v1"
        self.numeric = NumericContract(
            "clearance-micrometre-v1",
            digest("numeric-profile"),
            "signed-fixed-point",
            "nearest-ties-to-even",
            "checked",
        )

    def manifest(self, computation_result_digest=None):
        return create_computation_proof_manifest(
            self.session.ledger,
            1,
            numeric_contract=self.numeric,
            model_contract_digest=digest("engineering-model-contract"),
            validation_profile_digest=digest("validation-profile"),
            computation_result_digest=computation_result_digest,
            evidence_commitments=(digest("scan-42"), digest("survey-control-A")),
            proof_system="sp1",
            proof_type="groth16",
            program_digest=digest("guest-elf"),
            verifying_key_digest=digest("sp1-vkey"),
            proof_artifact=self.proof_bytes,
            media_type="application/vnd.succinct.sp1-proof",
            locator="proofs/transition-1.bin",
        )

    def test_roundtrip_is_deterministic_and_binds_public_values(self) -> None:
        manifest = self.manifest()
        restored = ComputationProofManifest.from_dict(manifest.to_dict())

        self.assertEqual(restored, manifest)
        self.assertEqual(
            manifest.proof.public_values_digest,
            digest_json(manifest.public_values()),
        )
        self.assertEqual(
            manifest.evidence_commitments,
            tuple(sorted((digest("scan-42"), digest("survey-control-A")))),
        )

    def test_file_roundtrip_returns_manifest_integrity_digest(self) -> None:
        manifest = self.manifest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transition.proof.json"
            written = write_computation_proof_manifest(manifest, path)
            restored = read_computation_proof_manifest(path)

        self.assertEqual(written, manifest.manifest_digest)
        self.assertEqual(restored, manifest)

    def test_binding_is_not_reported_as_cryptographic_verification(self) -> None:
        report = verify_computation_proof_manifest(
            self.manifest(), self.session.ledger, self.proof_bytes
        )

        self.assertTrue(report.bound)
        self.assertFalse(report.proof_verified)
        self.assertEqual(report.checks[-1].status, ProofCheckStatus.NOT_CHECKED)
        self.assertIn("NOT PROOF VERIFIED", report.render())

    def test_explicit_backend_verifier_can_close_the_proof_check(self) -> None:
        calls = []

        def verifier(manifest, artifact):
            calls.append((manifest.proof.public_values_digest, artifact))
            return artifact == self.proof_bytes

        report = verify_computation_proof_manifest(
            self.manifest(), self.session.ledger, self.proof_bytes, verifier
        )

        self.assertTrue(report.bound)
        self.assertTrue(report.proof_verified)
        self.assertEqual(calls, [(self.manifest().proof.public_values_digest, self.proof_bytes)])

    def test_wrong_proof_bytes_fail_before_backend_verification(self) -> None:
        called = False

        def verifier(_manifest, _artifact):
            nonlocal called
            called = True
            return True

        report = verify_computation_proof_manifest(
            self.manifest(), self.session.ledger, b"wrong-proof", verifier
        )

        self.assertFalse(report.bound)
        self.assertFalse(report.proof_verified)
        self.assertFalse(called)
        self.assertEqual(report.checks[-1].status, ProofCheckStatus.NOT_CHECKED)

    def test_manifest_for_another_ledger_does_not_bind(self) -> None:
        other = GatSession.load_ifc(MODEL)
        other.run(ShiftParameter(other.var("Level 1", "ClearHeight"), 0.02))

        report = verify_computation_proof_manifest(
            self.manifest(), other.ledger, self.proof_bytes, lambda *_: True
        )

        self.assertFalse(report.bound)
        self.assertFalse(report.proof_verified)

    def test_later_ledger_head_requires_a_new_manifest(self) -> None:
        manifest = self.manifest()
        self.session.run(
            ShiftParameter(self.session.var("Level 1", "ClearHeight"), 0.001)
        )

        report = verify_computation_proof_manifest(
            manifest, self.session.ledger, self.proof_bytes, lambda *_: True
        )

        self.assertFalse(report.bound)
        self.assertEqual(
            next(check for check in report.checks if check.name == "ledger head").status,
            ProofCheckStatus.FAIL,
        )

    def test_declared_computation_result_requires_later_ledger_assessment(self) -> None:
        manifest = self.manifest(digest("unrecorded-computation-result"))
        report = verify_computation_proof_manifest(
            manifest,
            self.session.ledger,
            self.proof_bytes,
        )

        self.assertFalse(report.bound)
        self.assertEqual(
            next(
                check
                for check in report.checks
                if check.name == "computation result"
            ).status,
            ProofCheckStatus.FAIL,
        )

    def test_tampered_document_and_in_memory_digest_fail_closed(self) -> None:
        manifest = self.manifest()
        document = manifest.to_dict()
        document["statement"]["result_world_digest"] = digest("forged-world")
        with self.assertRaisesRegex(ProofManifestError, "integrity digest mismatch"):
            ComputationProofManifest.from_dict(document)

        forged = replace(manifest, manifest_digest=digest("forged-manifest"))
        report = verify_computation_proof_manifest(
            forged, self.session.ledger, self.proof_bytes, lambda *_: True
        )
        self.assertFalse(report.bound)

    def test_rehashed_statement_still_must_match_proof_public_values(self) -> None:
        document = self.manifest().to_dict()
        document["engineering_context"]["model_contract_digest"] = digest("other-model")
        unsigned = dict(document)
        del unsigned["integrity"]
        document["integrity"]["digest"] = digest_json(unsigned)

        with self.assertRaisesRegex(ProofManifestError, "public-values commitment"):
            ComputationProofManifest.from_dict(document)

    def test_rejected_transition_cannot_receive_computation_proof(self) -> None:
        opening = self.session.var("Opening-1", "Height")
        with self.assertRaises(VerificationError):
            self.session.run(
                # SetParameter would work too; this large shift violates the wall bound.
                ShiftParameter(opening, 10.0)
            )
        with self.assertRaisesRegex(ProofManifestError, "accepted transitions"):
            create_computation_proof_manifest(
                self.session.ledger,
                2,
                numeric_contract=self.numeric,
                model_contract_digest=digest("engineering-model-contract"),
                validation_profile_digest=digest("validation-profile"),
                proof_system="sp1",
                proof_type="groth16",
                program_digest=digest("guest-elf"),
                verifying_key_digest=digest("sp1-vkey"),
                proof_artifact=self.proof_bytes,
            )

    def test_numeric_contract_requires_explicit_safe_overflow_semantics(self) -> None:
        with self.assertRaisesRegex(ProofManifestError, "checked overflow"):
            NumericContract(
                "unsafe-fixed",
                digest("unsafe-fixed"),
                "signed-fixed-point",
                "truncate",
                "wrap",
            )


def digest_json(value: object) -> str:
    text = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    unittest.main()
