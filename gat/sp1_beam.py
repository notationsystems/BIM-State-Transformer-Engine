"""Checked fixed-point beam claim and optional SP1 subprocess adapter.

The guest proves only the bounded deterministic ANSI/AISC 360-22 F2-1
calculation over an explicitly quantized posterior beam slice.  It does not
prove evidence truth, the Gaussian update, design-code applicability, or a
professional acceptance decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Callable, Mapping

from gat.causal import AssessmentRecord
from gat.engine.executor import World
from gat.engineering.aisc360_22 import AISC360_22_PHI_B
from gat.engineering.beam import BeamCheckResult
from gat.evidence import CalibratedObservation
from gat.errors import ProofManifestError
from gat.ids import VarId
from gat.ledger import ExecutionLedger
from gat.proof_manifest import (
    ComputationProofManifest,
    NumericContract,
    computation_proof_public_values_digest,
)


SP1_BEAM_REQUEST_FORMAT = "gat-sp1-beam-request-v1"
SP1_BEAM_RECEIPT_FORMAT = "gat-sp1-beam-proof-receipt-v1"
SP1_BEAM_SCHEMA_VERSION = 1
SP1_BEAM_METHOD = "ansi-aisc-360-22-f2-1-lrfd-checked-fixed-v1"
SP1_BEAM_NUMERIC_PROFILE_ID = "beam-milli-mpa-mm3-milli-nmm-v1"
SP1_BEAM_PROOF_SYSTEM = "sp1"
SP1_BEAM_CIRCUIT_VERSION = "v6.1.0"
SP1_BEAM_PROOF_TYPE = "core-v6.1.0"
SP1_BEAM_VERSION = "6.5.0"
SP1_BEAM_MEDIA_TYPE = "application/vnd.succinct.sp1-proof"

_CLAIM_DOMAIN = b"gat-sp1-beam-claim-v1\x00"
_PUBLIC_DOMAIN = b"gat-sp1-beam-public-v1\x00"
_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_U64_MAX = (1 << 64) - 1
_U128_MAX = (1 << 128) - 1
_ONE_MILLION = 1_000_000


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


SP1_BEAM_NUMERIC_PROFILE: Mapping[str, object] = {
    "profile_id": SP1_BEAM_NUMERIC_PROFILE_ID,
    "arithmetic": "checked-integer",
    "rounding": "nearest-ties-to-even",
    "overflow": "checked",
    "input_scales": {
        "yield_strength": "0.001 MPa",
        "plastic_section_modulus": "1 mm3",
        "factored_demand": "0.001 N*mm",
        "resistance_factor": "1 ppm",
    },
    "calculation": {
        "nominal_milli_n_mm": "fy_milli_mpa * zx_mm3",
        "available_milli_n_mm": (
            "round_ties_even(nominal_milli_n_mm * phi_ppm / 1000000)"
        ),
        "verdict": "PASS iff available_milli_n_mm >= demand_milli_n_mm",
    },
    "scope": "deterministic mean-value AISC F2-1 yielding check only",
}
SP1_BEAM_NUMERIC_PROFILE_DIGEST = _canonical_digest(SP1_BEAM_NUMERIC_PROFILE)


def sp1_beam_numeric_contract() -> NumericContract:
    return NumericContract(
        SP1_BEAM_NUMERIC_PROFILE_ID,
        SP1_BEAM_NUMERIC_PROFILE_DIGEST,
        "checked-integer",
        "nearest-ties-to-even",
        "checked",
    )


def _digest(value: str, label: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ProofManifestError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _integer(value: object, label: str, *, maximum: int, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProofManifestError(f"{label} must be an integer")
    minimum = 1 if positive else 0
    if value < minimum or value > maximum:
        raise ProofManifestError(f"{label} is outside [{minimum}, {maximum}]")
    return value


def _quantize(value: float, scale: int, label: str, *, positive: bool) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProofManifestError(f"{label} must be numeric")
    try:
        scaled = Decimal(str(value)) * Decimal(scale)
        if not scaled.is_finite():
            raise ValueError("value is not finite")
        rounded = scaled.to_integral_value(rounding=ROUND_HALF_EVEN)
        result = int(rounded)
    except Exception as exc:
        raise ProofManifestError(f"could not quantize {label}: {exc}") from exc
    return _integer(result, label, maximum=_U64_MAX, positive=positive)


def _round_div_ties_even(numerator: int, denominator: int) -> int:
    quotient, remainder = divmod(numerator, denominator)
    twice = remainder * 2
    if twice > denominator or (twice == denominator and quotient % 2 == 1):
        quotient += 1
    return quotient


@dataclass(frozen=True)
class Sp1BeamClaimInput:
    yield_strength_milli_mpa: int
    plastic_section_modulus_mm3: int
    factored_demand_milli_n_mm: int
    resistance_factor_ppm: int
    numeric_profile_digest: str
    model_contract_digest: str
    validation_profile_digest: str
    evidence_digest: str
    evidence_source_digest: str

    def __post_init__(self) -> None:
        _integer(
            self.yield_strength_milli_mpa,
            "yield_strength_milli_mpa",
            maximum=_U64_MAX,
            positive=True,
        )
        _integer(
            self.plastic_section_modulus_mm3,
            "plastic_section_modulus_mm3",
            maximum=_U64_MAX,
            positive=True,
        )
        _integer(
            self.factored_demand_milli_n_mm,
            "factored_demand_milli_n_mm",
            maximum=_U64_MAX,
        )
        _integer(
            self.resistance_factor_ppm,
            "resistance_factor_ppm",
            maximum=_U64_MAX,
            positive=True,
        )
        if self.resistance_factor_ppm != 900_000:
            raise ProofManifestError("the v1 beam guest requires phi_b = 900000 ppm")
        for name in (
            "numeric_profile_digest",
            "model_contract_digest",
            "validation_profile_digest",
            "evidence_digest",
            "evidence_source_digest",
        ):
            _digest(getattr(self, name), name)
        if self.numeric_profile_digest != SP1_BEAM_NUMERIC_PROFILE_DIGEST:
            raise ProofManifestError("beam claim numeric profile digest is not v1")

    def canonical_bytes(self) -> bytes:
        parts = [
            _CLAIM_DOMAIN,
            self.yield_strength_milli_mpa.to_bytes(8, "big"),
            self.plastic_section_modulus_mm3.to_bytes(8, "big"),
            self.factored_demand_milli_n_mm.to_bytes(8, "big"),
            self.resistance_factor_ppm.to_bytes(8, "big"),
        ]
        parts.extend(
            bytes.fromhex(value)
            for value in (
                self.numeric_profile_digest,
                self.model_contract_digest,
                self.validation_profile_digest,
                self.evidence_digest,
                self.evidence_source_digest,
            )
        )
        return b"".join(parts)

    def to_dict(self) -> dict[str, object]:
        return {
            "yield_strength_milli_mpa": self.yield_strength_milli_mpa,
            "plastic_section_modulus_mm3": self.plastic_section_modulus_mm3,
            "factored_demand_milli_n_mm": self.factored_demand_milli_n_mm,
            "resistance_factor_ppm": self.resistance_factor_ppm,
            "numeric_profile_digest": self.numeric_profile_digest,
            "model_contract_digest": self.model_contract_digest,
            "validation_profile_digest": self.validation_profile_digest,
            "evidence_digest": self.evidence_digest,
            "evidence_source_digest": self.evidence_source_digest,
        }

    @classmethod
    def from_dict(cls, value: object) -> "Sp1BeamClaimInput":
        record = _record(value, "claim.input")
        expected = {
            "yield_strength_milli_mpa",
            "plastic_section_modulus_mm3",
            "factored_demand_milli_n_mm",
            "resistance_factor_ppm",
            "numeric_profile_digest",
            "model_contract_digest",
            "validation_profile_digest",
            "evidence_digest",
            "evidence_source_digest",
        }
        _fields(record, expected, "claim.input")
        return cls(**record)  # type: ignore[arg-type]


@dataclass(frozen=True)
class Sp1BeamClaim:
    input: Sp1BeamClaimInput
    nominal_milli_n_mm: int
    available_milli_n_mm: int
    verdict: str
    computation_digest: str

    def __post_init__(self) -> None:
        _integer(self.nominal_milli_n_mm, "nominal_milli_n_mm", maximum=_U128_MAX)
        _integer(
            self.available_milli_n_mm,
            "available_milli_n_mm",
            maximum=_U128_MAX,
        )
        if self.verdict not in {"PASS", "FAIL"}:
            raise ProofManifestError("beam fixed-point verdict must be PASS or FAIL")
        _digest(self.computation_digest, "computation_digest")
        expected = self.evaluate(self.input)
        if (
            self.nominal_milli_n_mm != expected.nominal_milli_n_mm
            or self.available_milli_n_mm != expected.available_milli_n_mm
            or self.verdict != expected.verdict
            or self.computation_digest != expected.computation_digest
        ):
            raise ProofManifestError("beam claim output differs from checked evaluation")

    @classmethod
    def evaluate(cls, value: Sp1BeamClaimInput) -> "Sp1BeamClaim":
        nominal = (
            value.yield_strength_milli_mpa
            * value.plastic_section_modulus_mm3
        )
        if nominal > _U128_MAX:
            raise ProofManifestError("nominal beam capacity overflows u128")
        numerator = nominal * value.resistance_factor_ppm
        if numerator > _U128_MAX:
            raise ProofManifestError("available beam capacity numerator overflows u128")
        available = _round_div_ties_even(numerator, _ONE_MILLION)
        verdict = (
            "PASS"
            if available >= value.factored_demand_milli_n_mm
            else "FAIL"
        )
        output = (
            nominal.to_bytes(16, "big")
            + available.to_bytes(16, "big")
            + (b"\x01" if verdict == "PASS" else b"\x00")
        )
        digest = hashlib.sha256(value.canonical_bytes() + output).hexdigest()
        instance = object.__new__(cls)
        object.__setattr__(instance, "input", value)
        object.__setattr__(instance, "nominal_milli_n_mm", nominal)
        object.__setattr__(instance, "available_milli_n_mm", available)
        object.__setattr__(instance, "verdict", verdict)
        object.__setattr__(instance, "computation_digest", digest)
        return instance

    def to_dict(self) -> dict[str, object]:
        return {
            "input": self.input.to_dict(),
            "output": {
                "nominal_milli_n_mm": self.nominal_milli_n_mm,
                "available_milli_n_mm": self.available_milli_n_mm,
                "verdict": self.verdict,
                "computation_digest": self.computation_digest,
            },
        }

    @classmethod
    def from_dict(cls, value: object) -> "Sp1BeamClaim":
        record = _record(value, "claim")
        _fields(record, {"input", "output"}, "claim")
        output = _record(record["output"], "claim.output")
        _fields(
            output,
            {
                "nominal_milli_n_mm",
                "available_milli_n_mm",
                "verdict",
                "computation_digest",
            },
            "claim.output",
        )
        return cls(Sp1BeamClaimInput.from_dict(record["input"]), **output)  # type: ignore[arg-type]


@dataclass(frozen=True)
class Sp1BeamPublicValues:
    public_statement_digest: str
    computation_result_digest: str
    nominal_milli_n_mm: int
    available_milli_n_mm: int
    factored_demand_milli_n_mm: int
    verdict: str

    def to_bytes(self) -> bytes:
        for name in ("public_statement_digest", "computation_result_digest"):
            _digest(getattr(self, name), name)
        for name in (
            "nominal_milli_n_mm",
            "available_milli_n_mm",
            "factored_demand_milli_n_mm",
        ):
            _integer(getattr(self, name), name, maximum=_U128_MAX)
        if self.verdict not in {"PASS", "FAIL"}:
            raise ProofManifestError("public beam verdict must be PASS or FAIL")
        return b"".join(
            (
                _PUBLIC_DOMAIN,
                bytes.fromhex(self.public_statement_digest),
                bytes.fromhex(self.computation_result_digest),
                self.nominal_milli_n_mm.to_bytes(16, "big"),
                self.available_milli_n_mm.to_bytes(16, "big"),
                self.factored_demand_milli_n_mm.to_bytes(16, "big"),
                b"\x01" if self.verdict == "PASS" else b"\x00",
            )
        )

    @classmethod
    def from_bytes(cls, value: bytes) -> "Sp1BeamPublicValues":
        expected = len(_PUBLIC_DOMAIN) + 32 + 32 + 16 + 16 + 16 + 1
        if not isinstance(value, bytes) or len(value) != expected:
            raise ProofManifestError("SP1 beam public values have the wrong length")
        if not value.startswith(_PUBLIC_DOMAIN):
            raise ProofManifestError("SP1 beam public values have the wrong domain")
        offset = len(_PUBLIC_DOMAIN)
        statement = value[offset : offset + 32].hex()
        offset += 32
        computation = value[offset : offset + 32].hex()
        offset += 32
        numbers = []
        for _ in range(3):
            numbers.append(int.from_bytes(value[offset : offset + 16], "big"))
            offset += 16
        verdict_byte = value[offset]
        if verdict_byte not in {0, 1}:
            raise ProofManifestError("SP1 beam public verdict byte is invalid")
        return cls(
            statement,
            computation,
            numbers[0],
            numbers[1],
            numbers[2],
            "PASS" if verdict_byte == 1 else "FAIL",
        )


@dataclass(frozen=True)
class Sp1BeamRequest:
    transition_event_seq: int
    public_statement_digest: str
    numeric_contract: NumericContract
    claim: Sp1BeamClaim

    def __post_init__(self) -> None:
        _integer(
            self.transition_event_seq,
            "transition_event_seq",
            maximum=_U64_MAX,
            positive=True,
        )
        _digest(self.public_statement_digest, "public_statement_digest")
        if self.numeric_contract != sp1_beam_numeric_contract():
            raise ProofManifestError("SP1 beam request has the wrong numeric contract")
        if self.claim.input.numeric_profile_digest != self.numeric_contract.profile_digest:
            raise ProofManifestError("claim and numeric-contract profiles differ")

    @property
    def public_values(self) -> Sp1BeamPublicValues:
        return Sp1BeamPublicValues(
            self.public_statement_digest,
            self.claim.computation_digest,
            self.claim.nominal_milli_n_mm,
            self.claim.available_milli_n_mm,
            self.claim.input.factored_demand_milli_n_mm,
            self.claim.verdict,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "format": SP1_BEAM_REQUEST_FORMAT,
            "schema_version": SP1_BEAM_SCHEMA_VERSION,
            "sp1_version": SP1_BEAM_VERSION,
            "sp1_circuit_version": SP1_BEAM_CIRCUIT_VERSION,
            "transition_event_seq": self.transition_event_seq,
            "public_statement_digest": self.public_statement_digest,
            "numeric_contract": self.numeric_contract.to_dict(),
            "claim": self.claim.to_dict(),
            "expected_public_values_hex": self.public_values.to_bytes().hex(),
        }

    @classmethod
    def from_dict(cls, value: object) -> "Sp1BeamRequest":
        record = _record(value, "SP1 beam request")
        expected = {
            "format",
            "schema_version",
            "sp1_version",
            "sp1_circuit_version",
            "transition_event_seq",
            "public_statement_digest",
            "numeric_contract",
            "claim",
            "expected_public_values_hex",
        }
        _fields(record, expected, "SP1 beam request")
        if record["format"] != SP1_BEAM_REQUEST_FORMAT:
            raise ProofManifestError("unsupported SP1 beam request format")
        if record["schema_version"] != SP1_BEAM_SCHEMA_VERSION:
            raise ProofManifestError("unsupported SP1 beam request schema")
        if record["sp1_version"] != SP1_BEAM_VERSION:
            raise ProofManifestError("SP1 beam request version is not pinned")
        if record["sp1_circuit_version"] != SP1_BEAM_CIRCUIT_VERSION:
            raise ProofManifestError("SP1 beam request circuit version is not pinned")
        request = cls(
            _integer(
                record["transition_event_seq"],
                "transition_event_seq",
                maximum=_U64_MAX,
                positive=True,
            ),
            _digest(record["public_statement_digest"], "public_statement_digest"),
            NumericContract.from_dict(record["numeric_contract"]),
            Sp1BeamClaim.from_dict(record["claim"]),
        )
        if record["expected_public_values_hex"] != request.public_values.to_bytes().hex():
            raise ProofManifestError("SP1 beam expected public values are inconsistent")
        return request


def build_sp1_beam_claim(
    world: World,
    result: BeamCheckResult,
    evidence: CalibratedObservation,
) -> Sp1BeamClaim:
    if result.assessment.world_digest != world.digest():
        raise ProofManifestError("beam result is not bound to the supplied world")
    if evidence.subject.entity != result.check.beam:
        raise ProofManifestError("beam evidence and result subjects differ")
    fy = VarId(result.check.beam, "YieldStrengthMPa")
    zx = VarId(result.check.beam, "PlasticSectionModulusMajorM3")
    input_value = Sp1BeamClaimInput(
        _quantize(world.belief.mean(fy), 1_000, "yield strength", positive=True),
        _quantize(world.belief.mean(zx), 1_000_000_000, "plastic modulus", positive=True),
        _quantize(result.check.factored_demand_n_m, 1_000_000, "factored demand", positive=False),
        _quantize(AISC360_22_PHI_B, 1_000_000, "resistance factor", positive=True),
        SP1_BEAM_NUMERIC_PROFILE_DIGEST,
        result.model_contract_digest,
        result.validation_profile_digest,
        evidence.digest(),
        evidence.source_digest,
    )
    return Sp1BeamClaim.evaluate(input_value)


def sp1_beam_assessment_record(
    world: World,
    result: BeamCheckResult,
    evidence: CalibratedObservation,
    claim: Sp1BeamClaim,
) -> AssessmentRecord:
    expected = build_sp1_beam_claim(world, result, evidence)
    if claim != expected:
        raise ProofManifestError("SP1 beam claim differs from the bound world slice")
    details = {
        "state_identity": world.digest(),
        "criterion": {
            "kind": "deterministic-fixed-point-mean-capacity",
            "factored_demand_milli_n_mm": claim.input.factored_demand_milli_n_mm,
        },
        "result": {
            "nominal_milli_n_mm": claim.nominal_milli_n_mm,
            "available_milli_n_mm": claim.available_milli_n_mm,
            "verdict": claim.verdict,
        },
        "computation": {
            "method": SP1_BEAM_METHOD,
            "computation_digest": claim.computation_digest,
            "numeric_profile_digest": claim.input.numeric_profile_digest,
            "model_contract_digest": claim.input.model_contract_digest,
            "validation_profile_digest": claim.input.validation_profile_digest,
            "input": claim.input.to_dict(),
        },
        "claim_limits": {
            "proves_gaussian_update": False,
            "proves_evidence_truth": False,
            "proves_design_code_applicability": False,
            "authorizes_physical_action": False,
        },
    }
    return AssessmentRecord(
        world.digest(),
        f"sp1-beam-input:{claim.computation_digest[:16]}",
        "sp1-beam-fixed-computation",
        result.check.beam.global_id,
        claim.verdict,
        SP1_BEAM_METHOD,
        evidence.digest(),
        details,
    )


def build_sp1_beam_request(
    ledger: ExecutionLedger,
    transition_event_seq: int,
    claim: Sp1BeamClaim,
) -> Sp1BeamRequest:
    numeric = sp1_beam_numeric_contract()
    evidence = (
        claim.input.evidence_digest,
        claim.input.evidence_source_digest,
    )
    statement_digest = computation_proof_public_values_digest(
        ledger,
        transition_event_seq,
        numeric_contract=numeric,
        model_contract_digest=claim.input.model_contract_digest,
        validation_profile_digest=claim.input.validation_profile_digest,
        computation_result_digest=claim.computation_digest,
        evidence_commitments=evidence,
    )
    return Sp1BeamRequest(transition_event_seq, statement_digest, numeric, claim)


def write_sp1_beam_request(request: Sp1BeamRequest, path: str | Path) -> str:
    text = json.dumps(
        request.to_dict(),
        indent=1,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    Path(path).write_text(text, encoding="utf-8", newline="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_sp1_beam_request(path: str | Path) -> Sp1BeamRequest:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProofManifestError(f"could not read SP1 beam request: {exc}") from exc
    return Sp1BeamRequest.from_dict(value)


@dataclass(frozen=True)
class Sp1BeamProofReceipt:
    sp1_version: str
    sp1_circuit_version: str
    proof_type: str
    program_digest: str
    verifying_key_digest: str
    proof_artifact_digest: str
    public_values_hex: str
    public_statement_digest: str
    computation_result_digest: str
    proof_verified: bool
    cycles: int | None

    @classmethod
    def from_dict(cls, value: object) -> "Sp1BeamProofReceipt":
        record = _record(value, "SP1 beam proof receipt")
        expected = {
            "format",
            "schema_version",
            "sp1_version",
            "sp1_circuit_version",
            "proof_type",
            "program_digest",
            "verifying_key_digest",
            "proof_artifact_digest",
            "public_values_hex",
            "public_statement_digest",
            "computation_result_digest",
            "proof_verified",
            "cycles",
        }
        _fields(record, expected, "SP1 beam proof receipt")
        if record["format"] != SP1_BEAM_RECEIPT_FORMAT:
            raise ProofManifestError("unsupported SP1 beam receipt format")
        if record["schema_version"] != SP1_BEAM_SCHEMA_VERSION:
            raise ProofManifestError("unsupported SP1 beam receipt schema")
        if not isinstance(record["proof_verified"], bool):
            raise ProofManifestError("proof_verified must be boolean")
        cycles = record["cycles"]
        if cycles is not None:
            cycles = _integer(cycles, "cycles", maximum=_U64_MAX)
        text_fields = (
            "sp1_version",
            "sp1_circuit_version",
            "proof_type",
            "public_values_hex",
        )
        for name in text_fields:
            if not isinstance(record[name], str) or not record[name]:
                raise ProofManifestError(f"{name} must be non-empty text")
        if record["sp1_version"] != SP1_BEAM_VERSION:
            raise ProofManifestError("SP1 beam receipt version is not pinned")
        if record["sp1_circuit_version"] != SP1_BEAM_CIRCUIT_VERSION:
            raise ProofManifestError("SP1 beam receipt circuit version is not pinned")
        if record["proof_type"] != SP1_BEAM_PROOF_TYPE:
            raise ProofManifestError("SP1 beam receipt proof type is not supported")
        receipt = cls(
            record["sp1_version"],
            record["sp1_circuit_version"],
            record["proof_type"],
            _digest(record["program_digest"], "program_digest"),
            _digest(record["verifying_key_digest"], "verifying_key_digest"),
            _digest(record["proof_artifact_digest"], "proof_artifact_digest"),
            record["public_values_hex"],
            _digest(record["public_statement_digest"], "public_statement_digest"),
            _digest(record["computation_result_digest"], "computation_result_digest"),
            record["proof_verified"],
            cycles,
        )
        try:
            public = Sp1BeamPublicValues.from_bytes(bytes.fromhex(receipt.public_values_hex))
        except ValueError as exc:
            raise ProofManifestError("receipt public values are not hexadecimal") from exc
        if (
            public.public_statement_digest != receipt.public_statement_digest
            or public.computation_result_digest != receipt.computation_result_digest
        ):
            raise ProofManifestError("receipt summaries differ from its public values")
        return receipt


def read_sp1_beam_receipt(path: str | Path) -> Sp1BeamProofReceipt:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProofManifestError(f"could not read SP1 beam receipt: {exc}") from exc
    return Sp1BeamProofReceipt.from_dict(value)


def sp1_beam_subprocess_verifier(
    executable: str | Path,
    request_path: str | Path,
    *,
    timeout_seconds: float = 120.0,
) -> Callable[[ComputationProofManifest, bytes], bool]:
    """Return a fail-closed verifier backed by the explicit SP1 script binary."""
    command = str(Path(executable).resolve())
    request_file = str(Path(request_path).resolve())

    def verify(manifest: ComputationProofManifest, artifact: bytes) -> bool:
        request = read_sp1_beam_request(request_file)
        if (
            manifest.proof.proof_system != SP1_BEAM_PROOF_SYSTEM
            or manifest.proof.proof_type != SP1_BEAM_PROOF_TYPE
            or manifest.proof.media_type != SP1_BEAM_MEDIA_TYPE
            or manifest.proof.public_values_digest != request.public_statement_digest
            or manifest.computation_result_digest != request.claim.computation_digest
        ):
            return False
        with tempfile.TemporaryDirectory(prefix="gat-sp1-verify-") as directory:
            proof_path = Path(directory) / "proof.bin"
            receipt_path = Path(directory) / "receipt.json"
            proof_path.write_bytes(artifact)
            try:
                completed = subprocess.run(
                    [
                        command,
                        "verify",
                        "--request",
                        request_file,
                        "--proof",
                        str(proof_path),
                        "--receipt",
                        str(receipt_path),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                )
            except (OSError, subprocess.SubprocessError):
                return False
            if completed.returncode != 0 or not receipt_path.is_file():
                return False
            try:
                receipt = read_sp1_beam_receipt(receipt_path)
            except ProofManifestError:
                return False
        return bool(
            receipt.proof_verified
            and receipt.sp1_version == SP1_BEAM_VERSION
            and receipt.sp1_circuit_version == SP1_BEAM_CIRCUIT_VERSION
            and receipt.proof_type == SP1_BEAM_PROOF_TYPE
            and receipt.program_digest == manifest.proof.program_digest
            and receipt.verifying_key_digest == manifest.proof.verifying_key_digest
            and receipt.proof_artifact_digest == manifest.proof.artifact_digest
            and receipt.public_statement_digest == manifest.proof.public_values_digest
            and receipt.computation_result_digest == manifest.computation_result_digest
            and receipt.public_values_hex == request.public_values.to_bytes().hex()
        )

    return verify


def _record(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ProofManifestError(f"{label} must be an object with string keys")
    return dict(value)


def _fields(record: Mapping[str, object], expected: set[str], label: str) -> None:
    actual = set(record)
    if actual != expected:
        raise ProofManifestError(
            f"{label} fields differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


__all__ = [
    "SP1_BEAM_CIRCUIT_VERSION",
    "SP1_BEAM_MEDIA_TYPE",
    "SP1_BEAM_METHOD",
    "SP1_BEAM_NUMERIC_PROFILE",
    "SP1_BEAM_NUMERIC_PROFILE_DIGEST",
    "SP1_BEAM_PROOF_SYSTEM",
    "SP1_BEAM_PROOF_TYPE",
    "SP1_BEAM_RECEIPT_FORMAT",
    "SP1_BEAM_REQUEST_FORMAT",
    "SP1_BEAM_SCHEMA_VERSION",
    "SP1_BEAM_VERSION",
    "Sp1BeamClaim",
    "Sp1BeamClaimInput",
    "Sp1BeamProofReceipt",
    "Sp1BeamPublicValues",
    "Sp1BeamRequest",
    "build_sp1_beam_claim",
    "build_sp1_beam_request",
    "read_sp1_beam_receipt",
    "read_sp1_beam_request",
    "sp1_beam_assessment_record",
    "sp1_beam_numeric_contract",
    "sp1_beam_subprocess_verifier",
    "write_sp1_beam_request",
]
