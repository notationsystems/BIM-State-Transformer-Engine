from __future__ import annotations

from dataclasses import replace
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from gat.errors import ProofManifestError
from gat.demo.beam_assurance import run_beam_assurance
from gat.ledger import read_ledger
from gat.proof_manifest import computation_proof_public_values_digest
from gat.sp1_beam import (
    SP1_BEAM_NUMERIC_PROFILE_DIGEST,
    SP1_BEAM_PROOF_TYPE,
    SP1_BEAM_RECEIPT_FORMAT,
    SP1_BEAM_VERSION,
    Sp1BeamClaim,
    Sp1BeamClaimInput,
    Sp1BeamProofReceipt,
    Sp1BeamPublicValues,
    Sp1BeamRequest,
    read_sp1_beam_request,
    sp1_beam_numeric_contract,
    write_sp1_beam_request,
)


KNOWN_COMPUTATION_DIGEST = (
    "1443b90bc95f146a0a4c1e8e4beeb7db9c7cd59e9431f05e91958bb6c97e54e6"
)
KNOWN_PUBLIC_VALUES = (
    "6761742d7370312d6265616d2d7075626c69632d763100"
    + "55" * 32
    + KNOWN_COMPUTATION_DIGEST
    + "00000000000000000000004bab827200"
    + "0000000000000000000000441a5bcd00"
    + "00000000000000000000004614ff8200"
    + "00"
)


def known_input() -> Sp1BeamClaimInput:
    return Sp1BeamClaimInput(
        325_000,
        1_000_000,
        301_000_000_000,
        900_000,
        SP1_BEAM_NUMERIC_PROFILE_DIGEST,
        "11" * 32,
        "22" * 32,
        "33" * 32,
        "44" * 32,
    )


class Sp1BeamArithmeticTests(unittest.TestCase):
    def test_python_matches_the_rust_known_vector(self) -> None:
        claim = Sp1BeamClaim.evaluate(known_input())
        self.assertEqual(claim.nominal_milli_n_mm, 325_000_000_000)
        self.assertEqual(claim.available_milli_n_mm, 292_500_000_000)
        self.assertEqual(claim.verdict, "FAIL")
        self.assertEqual(claim.computation_digest, KNOWN_COMPUTATION_DIGEST)
        public = Sp1BeamPublicValues(
            "55" * 32,
            claim.computation_digest,
            claim.nominal_milli_n_mm,
            claim.available_milli_n_mm,
            claim.input.factored_demand_milli_n_mm,
            claim.verdict,
        )
        self.assertEqual(public.to_bytes().hex(), KNOWN_PUBLIC_VALUES)
        self.assertEqual(Sp1BeamPublicValues.from_bytes(public.to_bytes()), public)

    def test_claim_rejects_profile_phi_output_and_overflow_drift(self) -> None:
        with self.assertRaises(ProofManifestError):
            replace(known_input(), resistance_factor_ppm=899_999)
        with self.assertRaises(ProofManifestError):
            replace(known_input(), numeric_profile_digest="00" * 32)
        claim = Sp1BeamClaim.evaluate(known_input())
        with self.assertRaises(ProofManifestError):
            replace(claim, available_milli_n_mm=claim.available_milli_n_mm + 1)
        overflowing = replace(
            known_input(),
            yield_strength_milli_mpa=(1 << 64) - 1,
            plastic_section_modulus_mm3=(1 << 64) - 1,
        )
        with self.assertRaisesRegex(ProofManifestError, "numerator overflows"):
            Sp1BeamClaim.evaluate(overflowing)

    def test_request_is_strict_and_roundtrips(self) -> None:
        request = Sp1BeamRequest(
            2,
            "55" * 32,
            sp1_beam_numeric_contract(),
            Sp1BeamClaim.evaluate(known_input()),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            written = write_sp1_beam_request(request, path)
            self.assertEqual(read_sp1_beam_request(path), request)
            self.assertEqual(written, hashlib.sha256(path.read_bytes()).hexdigest())
            value = json.loads(path.read_text(encoding="utf-8"))
            value["claim"]["output"]["verdict"] = "PASS"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(ProofManifestError):
                read_sp1_beam_request(path)

    def test_receipt_requires_exact_types_and_consistent_public_values(self) -> None:
        receipt = {
            "format": SP1_BEAM_RECEIPT_FORMAT,
            "schema_version": 1,
            "sp1_version": SP1_BEAM_VERSION,
            "proof_type": SP1_BEAM_PROOF_TYPE,
            "program_digest": "66" * 32,
            "verifying_key_digest": "77" * 32,
            "proof_artifact_digest": "88" * 32,
            "public_values_hex": KNOWN_PUBLIC_VALUES,
            "public_statement_digest": "55" * 32,
            "computation_result_digest": KNOWN_COMPUTATION_DIGEST,
            "proof_verified": True,
            "cycles": None,
        }
        parsed = Sp1BeamProofReceipt.from_dict(receipt)
        self.assertTrue(parsed.proof_verified)
        wrong_type = copy.deepcopy(receipt)
        wrong_type["sp1_version"] = 640
        with self.assertRaises(ProofManifestError):
            Sp1BeamProofReceipt.from_dict(wrong_type)
        inconsistent = copy.deepcopy(receipt)
        inconsistent["public_statement_digest"] = "99" * 32
        with self.assertRaises(ProofManifestError):
            Sp1BeamProofReceipt.from_dict(inconsistent)

    def test_reference_chain_emits_the_exact_ledger_bound_guest_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_beam_assurance(directory, quiet=True)
            request = read_sp1_beam_request(Path(directory) / "beam_sp1_request.json")
            ledger = read_ledger(Path(directory) / "beam_ledger.json")
            expected = computation_proof_public_values_digest(
                ledger,
                request.transition_event_seq,
                numeric_contract=request.numeric_contract,
                model_contract_digest=request.claim.input.model_contract_digest,
                validation_profile_digest=request.claim.input.validation_profile_digest,
                computation_result_digest=request.claim.computation_digest,
                evidence_commitments=(
                    request.claim.input.evidence_digest,
                    request.claim.input.evidence_source_digest,
                ),
            )
            self.assertEqual(request.public_statement_digest, expected)
            fixed_assessment = ledger.events[-1].operation
            self.assertEqual(
                fixed_assessment["details"]["computation"]["computation_digest"],
                request.claim.computation_digest,
            )


if __name__ == "__main__":
    unittest.main()
