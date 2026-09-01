"""Proof-carrying computation claims bound to GAT ledger transitions.

The execution ledger proves replayability and hash-chain integrity.  This
module defines the smaller portable statement that an external verifiable-
computation system may prove: one accepted ledger transition maps an exact
prior world to an exact result world under a committed program and numerical
contract.

The manifest deliberately does *not* claim that the engineering model,
evidence, calibration, or decision policy is valid.  Those remain separate
GAT responsibilities.  It also does not bundle a zkVM runtime.  A caller may
provide a backend verifier (for example an SP1 verifier) while the core keeps
its NumPy-only dependency boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import hmac
import json
import math
from pathlib import Path
import re
from typing import Callable, Mapping

from gat.errors import ProofManifestError
from gat.ledger import ExecutionLedger, LEDGER_RUNTIME_CONTRACT


PROOF_MANIFEST_FORMAT = "gat-computation-proof-manifest"
PROOF_MANIFEST_SCHEMA_VERSION = 1
PROOF_CLAIM_SCOPE = "computational-integrity-only"
PROOF_MANIFEST_HASH_ALGORITHM = "sha256"
PROOF_MANIFEST_MAX_BYTES = 1024 * 1024

_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_NUMERIC_ARITHMETIC = {
    "checked-integer",
    "signed-fixed-point",
    "ieee754-binary64",
}


class ProofCheckStatus(Enum):
    """Outcome of one manifest verification check."""

    PASS = "PASS"
    FAIL = "FAIL"
    NOT_CHECKED = "NOT_CHECKED"


@dataclass(frozen=True)
class NumericContract:
    """Digest-bound numerical semantics used by the proof program.

    ``profile_digest`` identifies the complete external numerical profile,
    including per-field units/scales where applicable.  The summary fields
    remain inspectable and prevent a proof from silently changing arithmetic,
    rounding, or overflow behaviour.
    """

    profile_id: str
    profile_digest: str
    arithmetic: str
    rounding: str
    overflow: str

    def __post_init__(self) -> None:
        _require_text(self.profile_id, "numeric_contract.profile_id")
        _require_digest(self.profile_digest, "numeric_contract.profile_digest")
        if self.arithmetic not in _NUMERIC_ARITHMETIC:
            raise ProofManifestError(
                f"unsupported numeric arithmetic {self.arithmetic!r}"
            )
        _require_text(self.rounding, "numeric_contract.rounding")
        _require_text(self.overflow, "numeric_contract.overflow")
        if self.arithmetic != "ieee754-binary64" and self.overflow != "checked":
            raise ProofManifestError(
                "integer and fixed-point proof arithmetic must use checked overflow"
            )
        if self.arithmetic == "ieee754-binary64" and self.overflow != "reject-nonfinite":
            raise ProofManifestError(
                "binary64 proof arithmetic must reject non-finite results"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "profile_digest": self.profile_digest,
            "arithmetic": self.arithmetic,
            "rounding": self.rounding,
            "overflow": self.overflow,
        }

    @classmethod
    def from_dict(cls, value: object) -> "NumericContract":
        record = _object(value, "numeric_contract")
        _expect_fields(
            record,
            {"profile_id", "profile_digest", "arithmetic", "rounding", "overflow"},
            "numeric_contract",
        )
        return cls(
            _require_text(record["profile_id"], "numeric_contract.profile_id"),
            _require_digest(record["profile_digest"], "numeric_contract.profile_digest"),
            _require_text(record["arithmetic"], "numeric_contract.arithmetic"),
            _require_text(record["rounding"], "numeric_contract.rounding"),
            _require_text(record["overflow"], "numeric_contract.overflow"),
        )


@dataclass(frozen=True)
class ProofArtifactCommitment:
    """Content-addressed proof and verifier commitments.

    ``locator`` is descriptive only.  Readers never fetch it implicitly; the
    exact proof bytes must be supplied to verification by the host.
    """

    proof_system: str
    proof_type: str
    program_digest: str
    verifying_key_digest: str
    public_values_digest: str
    artifact_digest: str
    media_type: str = "application/octet-stream"
    locator: str | None = None

    def __post_init__(self) -> None:
        for name in ("proof_system", "proof_type", "media_type"):
            _require_text(getattr(self, name), f"proof.{name}")
        for name in (
            "program_digest",
            "verifying_key_digest",
            "public_values_digest",
            "artifact_digest",
        ):
            _require_digest(getattr(self, name), f"proof.{name}")
        if self.locator is not None:
            _require_text(self.locator, "proof.locator")
            if len(self.locator) > 4096:
                raise ProofManifestError("proof.locator exceeds 4096 characters")

    def to_dict(self) -> dict[str, object]:
        return {
            "proof_system": self.proof_system,
            "proof_type": self.proof_type,
            "program_digest": self.program_digest,
            "verifying_key_digest": self.verifying_key_digest,
            "public_values_digest": self.public_values_digest,
            "artifact_digest": self.artifact_digest,
            "media_type": self.media_type,
            "locator": self.locator,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ProofArtifactCommitment":
        record = _object(value, "proof")
        fields = {
            "proof_system",
            "proof_type",
            "program_digest",
            "verifying_key_digest",
            "public_values_digest",
            "artifact_digest",
            "media_type",
            "locator",
        }
        _expect_fields(record, fields, "proof")
        locator = record["locator"]
        if locator is not None and not isinstance(locator, str):
            raise ProofManifestError("proof.locator must be a string or null")
        return cls(
            _require_text(record["proof_system"], "proof.proof_system"),
            _require_text(record["proof_type"], "proof.proof_type"),
            _require_digest(record["program_digest"], "proof.program_digest"),
            _require_digest(
                record["verifying_key_digest"], "proof.verifying_key_digest"
            ),
            _require_digest(
                record["public_values_digest"], "proof.public_values_digest"
            ),
            _require_digest(record["artifact_digest"], "proof.artifact_digest"),
            _require_text(record["media_type"], "proof.media_type"),
            locator,
        )


@dataclass(frozen=True)
class ComputationProofManifest:
    """One integrity-bound, proof-carrying accepted state transition."""

    ledger_head: str
    event_seq: int
    event_hash: str
    prior_world_digest: str
    result_world_digest: str
    operation_digest: str
    verification_digest: str
    computation_result_digest: str | None
    numeric_contract: NumericContract
    model_contract_digest: str
    validation_profile_digest: str
    evidence_commitments: tuple[str, ...]
    proof: ProofArtifactCommitment
    manifest_digest: str

    def __post_init__(self) -> None:
        for name in (
            "ledger_head",
            "event_hash",
            "prior_world_digest",
            "result_world_digest",
            "operation_digest",
            "verification_digest",
            "model_contract_digest",
            "validation_profile_digest",
            "manifest_digest",
        ):
            _require_digest(getattr(self, name), name)
        if self.computation_result_digest is not None:
            _require_digest(
                self.computation_result_digest, "computation_result_digest"
            )
        if isinstance(self.event_seq, bool) or not isinstance(self.event_seq, int):
            raise ProofManifestError("event_seq must be a non-negative integer")
        if self.event_seq < 1:
            raise ProofManifestError("a computation proof must bind a non-genesis event")
        if len(set(self.evidence_commitments)) != len(self.evidence_commitments):
            raise ProofManifestError("evidence commitments must be unique")
        if tuple(sorted(self.evidence_commitments)) != self.evidence_commitments:
            raise ProofManifestError("evidence commitments must be sorted")
        for index, digest in enumerate(self.evidence_commitments):
            _require_digest(digest, f"evidence_commitments[{index}]")

    def public_values(self) -> dict[str, object]:
        """Canonical values the proof program must expose to its verifier."""
        return {
            "claim_scope": PROOF_CLAIM_SCOPE,
            "runtime_contract": LEDGER_RUNTIME_CONTRACT,
            "ledger_head": self.ledger_head,
            "event_seq": self.event_seq,
            "event_hash": self.event_hash,
            "prior_world_digest": self.prior_world_digest,
            "result_world_digest": self.result_world_digest,
            "operation_digest": self.operation_digest,
            "verification_digest": self.verification_digest,
            "computation_result_digest": self.computation_result_digest,
            "numeric_contract_digest": _digest_json(self.numeric_contract.to_dict()),
            "model_contract_digest": self.model_contract_digest,
            "validation_profile_digest": self.validation_profile_digest,
            "evidence_commitments": list(self.evidence_commitments),
        }

    def to_dict(self) -> dict[str, object]:
        document = _manifest_document(self)
        document["integrity"] = {
            "algorithm": PROOF_MANIFEST_HASH_ALGORITHM,
            "digest": self.manifest_digest,
        }
        return document

    @classmethod
    def from_dict(cls, value: object) -> "ComputationProofManifest":
        root = _object(value, "proof manifest")
        fields = {
            "format",
            "schema_version",
            "runtime_contract",
            "claim_scope",
            "statement",
            "numeric_contract",
            "engineering_context",
            "proof",
            "integrity",
        }
        _expect_fields(root, fields, "proof manifest")
        if root["format"] != PROOF_MANIFEST_FORMAT:
            raise ProofManifestError(f"unsupported proof manifest format {root['format']!r}")
        if (
            isinstance(root["schema_version"], bool)
            or not isinstance(root["schema_version"], int)
            or root["schema_version"] != PROOF_MANIFEST_SCHEMA_VERSION
        ):
            raise ProofManifestError(
                f"unsupported proof manifest schema version {root['schema_version']!r}"
            )
        if root["runtime_contract"] != LEDGER_RUNTIME_CONTRACT:
            raise ProofManifestError("proof manifest runtime contract differs from GAT")
        if root["claim_scope"] != PROOF_CLAIM_SCOPE:
            raise ProofManifestError("proof manifest overstates its permitted claim scope")

        statement = _object(root["statement"], "statement")
        _expect_fields(
            statement,
            {
                "ledger_head",
                "event_seq",
                "event_hash",
                "prior_world_digest",
                "result_world_digest",
                "operation_digest",
                "verification_digest",
                "computation_result_digest",
            },
            "statement",
        )
        context = _object(root["engineering_context"], "engineering_context")
        _expect_fields(
            context,
            {
                "model_contract_digest",
                "validation_profile_digest",
                "evidence_commitments",
            },
            "engineering_context",
        )
        raw_evidence = context["evidence_commitments"]
        if not isinstance(raw_evidence, list):
            raise ProofManifestError("evidence_commitments must be an array")
        integrity = _object(root["integrity"], "integrity")
        _expect_fields(integrity, {"algorithm", "digest"}, "integrity")
        if integrity["algorithm"] != PROOF_MANIFEST_HASH_ALGORITHM:
            raise ProofManifestError("unsupported proof manifest integrity algorithm")
        expected_digest = _require_digest(integrity["digest"], "integrity.digest")
        unsigned = dict(root)
        del unsigned["integrity"]
        actual_digest = _digest_json(unsigned)
        if not hmac.compare_digest(expected_digest, actual_digest):
            raise ProofManifestError("proof manifest integrity digest mismatch")

        manifest = cls(
            _require_digest(statement["ledger_head"], "statement.ledger_head"),
            _require_int(statement["event_seq"], "statement.event_seq"),
            _require_digest(statement["event_hash"], "statement.event_hash"),
            _require_digest(
                statement["prior_world_digest"], "statement.prior_world_digest"
            ),
            _require_digest(
                statement["result_world_digest"], "statement.result_world_digest"
            ),
            _require_digest(statement["operation_digest"], "statement.operation_digest"),
            _require_digest(
                statement["verification_digest"], "statement.verification_digest"
            ),
            (
                None
                if statement["computation_result_digest"] is None
                else _require_digest(
                    statement["computation_result_digest"],
                    "statement.computation_result_digest",
                )
            ),
            NumericContract.from_dict(root["numeric_contract"]),
            _require_digest(
                context["model_contract_digest"], "model_contract_digest"
            ),
            _require_digest(
                context["validation_profile_digest"], "validation_profile_digest"
            ),
            tuple(
                _require_digest(item, f"evidence_commitments[{index}]")
                for index, item in enumerate(raw_evidence)
            ),
            ProofArtifactCommitment.from_dict(root["proof"]),
            expected_digest,
        )
        expected_public_values = _digest_json(manifest.public_values())
        if not hmac.compare_digest(
            expected_public_values, manifest.proof.public_values_digest
        ):
            raise ProofManifestError("proof public-values commitment is inconsistent")
        return manifest


@dataclass(frozen=True)
class ProofCheck:
    name: str
    status: ProofCheckStatus
    detail: str


@dataclass(frozen=True)
class ComputationProofVerification:
    """Separates ledger/artifact binding from cryptographic verification."""

    checks: tuple[ProofCheck, ...]

    @property
    def bound(self) -> bool:
        required = self.checks[:-1]
        return bool(required) and all(check.status is ProofCheckStatus.PASS for check in required)

    @property
    def proof_verified(self) -> bool:
        return self.bound and self.checks[-1].status is ProofCheckStatus.PASS

    def render(self) -> str:
        lines = [
            "computation claim: "
            + ("PROOF VERIFIED" if self.proof_verified else "NOT PROOF VERIFIED")
        ]
        for check in self.checks:
            lines.append(f"{check.status.value:<11} {check.name:<24} {check.detail}")
        return "\n".join(lines)


ProofVerifier = Callable[[ComputationProofManifest, bytes], bool]


def computation_proof_public_values(
    ledger: ExecutionLedger,
    event_seq: int,
    *,
    numeric_contract: NumericContract,
    model_contract_digest: str,
    validation_profile_digest: str,
    computation_result_digest: str | None = None,
    evidence_commitments: tuple[str, ...] = (),
) -> dict[str, object]:
    """Prepare the exact public statement before an external proof exists.

    Proof generation needs this statement digest as a guest public value, but
    :func:`create_computation_proof_manifest` also needs the resulting proof
    bytes.  This helper exposes the non-circular portion of the manifest and
    deliberately applies the same validation and ordering rules as creation.
    """
    ledger.validate()
    if isinstance(event_seq, bool) or not isinstance(event_seq, int):
        raise ProofManifestError("event_seq must be an integer")
    if event_seq < 1 or event_seq >= len(ledger.events):
        raise ProofManifestError("event_seq does not identify a non-genesis ledger event")
    event = ledger.events[event_seq]
    if event.kind != "transition":
        raise ProofManifestError("computation proofs may bind only accepted transitions")
    if event.verification_digest is None:
        raise ProofManifestError("accepted transition has no verification digest")
    if not isinstance(numeric_contract, NumericContract):
        raise ProofManifestError("numeric_contract must be a NumericContract")
    result_digest = (
        None
        if computation_result_digest is None
        else _require_digest(
            computation_result_digest,
            "computation_result_digest",
        )
    )
    evidence = tuple(sorted(evidence_commitments))
    if len(set(evidence)) != len(evidence):
        raise ProofManifestError("evidence commitments must be unique")
    for index, digest in enumerate(evidence):
        _require_digest(digest, f"evidence_commitments[{index}]")
    return {
        "claim_scope": PROOF_CLAIM_SCOPE,
        "runtime_contract": LEDGER_RUNTIME_CONTRACT,
        "ledger_head": ledger.head,
        "event_seq": event.seq,
        "event_hash": event.event_hash,
        "prior_world_digest": event.prior_world_digest,
        "result_world_digest": event.result_world_digest,
        "operation_digest": _digest_json(event.operation),
        "verification_digest": event.verification_digest,
        "computation_result_digest": result_digest,
        "numeric_contract_digest": _digest_json(numeric_contract.to_dict()),
        "model_contract_digest": _require_digest(
            model_contract_digest,
            "model_contract_digest",
        ),
        "validation_profile_digest": _require_digest(
            validation_profile_digest,
            "validation_profile_digest",
        ),
        "evidence_commitments": list(evidence),
    }


def computation_proof_public_values_digest(
    ledger: ExecutionLedger,
    event_seq: int,
    *,
    numeric_contract: NumericContract,
    model_contract_digest: str,
    validation_profile_digest: str,
    computation_result_digest: str | None = None,
    evidence_commitments: tuple[str, ...] = (),
) -> str:
    """Return the SHA-256 commitment an external guest must publish."""
    return _digest_json(
        computation_proof_public_values(
            ledger,
            event_seq,
            numeric_contract=numeric_contract,
            model_contract_digest=model_contract_digest,
            validation_profile_digest=validation_profile_digest,
            computation_result_digest=computation_result_digest,
            evidence_commitments=evidence_commitments,
        )
    )


def create_computation_proof_manifest(
    ledger: ExecutionLedger,
    event_seq: int,
    *,
    numeric_contract: NumericContract,
    model_contract_digest: str,
    validation_profile_digest: str,
    computation_result_digest: str | None = None,
    evidence_commitments: tuple[str, ...] = (),
    proof_system: str,
    proof_type: str,
    program_digest: str,
    verifying_key_digest: str,
    proof_artifact: bytes,
    media_type: str = "application/octet-stream",
    locator: str | None = None,
) -> ComputationProofManifest:
    """Create a manifest for one accepted transition in ``ledger``.

    The caller must already have generated ``proof_artifact``.  GAT binds it
    to the ledger statement but does not infer that its proof system is valid.
    """
    artifact = _bytes(proof_artifact, "proof_artifact")
    evidence = tuple(sorted(evidence_commitments))
    public_values = computation_proof_public_values(
        ledger,
        event_seq,
        numeric_contract=numeric_contract,
        model_contract_digest=model_contract_digest,
        validation_profile_digest=validation_profile_digest,
        computation_result_digest=computation_result_digest,
        evidence_commitments=evidence,
    )
    event = ledger.events[event_seq]

    provisional = ComputationProofManifest(
        ledger.head,
        event.seq,
        event.event_hash,
        event.prior_world_digest,
        event.result_world_digest,
        _digest_json(event.operation),
        event.verification_digest,
        (
            None
            if computation_result_digest is None
            else _require_digest(
                computation_result_digest, "computation_result_digest"
            )
        ),
        numeric_contract,
        _require_digest(model_contract_digest, "model_contract_digest"),
        _require_digest(validation_profile_digest, "validation_profile_digest"),
        evidence,
        ProofArtifactCommitment(
            proof_system,
            proof_type,
            program_digest,
            verifying_key_digest,
            "0" * 64,
            hashlib.sha256(artifact).hexdigest(),
            media_type,
            locator,
        ),
        "0" * 64,
    )
    public_values_digest = _digest_json(public_values)
    proof = ProofArtifactCommitment(
        proof_system,
        proof_type,
        program_digest,
        verifying_key_digest,
        public_values_digest,
        hashlib.sha256(artifact).hexdigest(),
        media_type,
        locator,
    )
    with_proof = ComputationProofManifest(
        **{
            **provisional.__dict__,
            "proof": proof,
        }
    )
    digest = _digest_json(_manifest_document(with_proof))
    return ComputationProofManifest(**{**with_proof.__dict__, "manifest_digest": digest})


def verify_computation_proof_manifest(
    manifest: ComputationProofManifest,
    ledger: ExecutionLedger,
    proof_artifact: bytes,
    verifier: ProofVerifier | None = None,
) -> ComputationProofVerification:
    """Verify manifest binding, then optionally invoke a cryptographic verifier."""
    checks: list[ProofCheck] = []
    manifest_ok = hmac.compare_digest(
        manifest.manifest_digest, _digest_json(_manifest_document(manifest))
    )
    checks.append(
        _check("manifest integrity", manifest_ok, manifest.manifest_digest)
    )
    try:
        ledger.validate()
        ledger_ok = True
        ledger_detail = f"{len(ledger.events)} hash-chained events"
    except Exception as exc:  # the report is a boundary; do not leak a partial pass
        ledger_ok = False
        ledger_detail = str(exc)
    checks.append(_check("ledger integrity", ledger_ok, ledger_detail))

    event = None
    if ledger_ok and manifest.event_seq < len(ledger.events):
        candidate = ledger.events[manifest.event_seq]
        event = candidate if candidate.kind == "transition" else None
    event_ok = event is not None
    checks.append(_check("accepted event", event_ok, f"event {manifest.event_seq}"))

    head_ok = ledger_ok and hmac.compare_digest(manifest.ledger_head, ledger.head)
    checks.append(_check("ledger head", head_ok, manifest.ledger_head))

    statement_ok = bool(
        event is not None
        and event.verification_digest is not None
        and hmac.compare_digest(manifest.event_hash, event.event_hash)
        and hmac.compare_digest(manifest.prior_world_digest, event.prior_world_digest)
        and hmac.compare_digest(manifest.result_world_digest, event.result_world_digest)
        and hmac.compare_digest(manifest.operation_digest, _digest_json(event.operation))
        and hmac.compare_digest(
            manifest.verification_digest, event.verification_digest
        )
    )
    checks.append(_check("transition statement", statement_ok, "event and world commitments"))

    if manifest.computation_result_digest is None:
        computation_ok = True
        computation_detail = "no optional result commitment declared"
    else:
        computation_ok = ledger_ok and _ledger_commits_computation_result(
            manifest, ledger
        )
        computation_detail = manifest.computation_result_digest
    checks.append(
        _check("computation result", computation_ok, computation_detail)
    )

    public_ok = hmac.compare_digest(
        manifest.proof.public_values_digest, _digest_json(manifest.public_values())
    )
    checks.append(_check("public values", public_ok, manifest.proof.public_values_digest))

    artifact = _bytes(proof_artifact, "proof_artifact")
    artifact_ok = hmac.compare_digest(
        manifest.proof.artifact_digest, hashlib.sha256(artifact).hexdigest()
    )
    checks.append(_check("proof artifact", artifact_ok, manifest.proof.artifact_digest))

    prerequisites = all(check.status is ProofCheckStatus.PASS for check in checks)
    if verifier is None:
        checks.append(
            ProofCheck(
                "cryptographic proof",
                ProofCheckStatus.NOT_CHECKED,
                "no backend verifier supplied",
            )
        )
    elif not prerequisites:
        checks.append(
            ProofCheck(
                "cryptographic proof",
                ProofCheckStatus.NOT_CHECKED,
                "binding checks failed",
            )
        )
    else:
        try:
            verified = verifier(manifest, artifact)
            if not isinstance(verified, bool):
                raise TypeError("verifier must return bool")
            checks.append(
                _check(
                    "cryptographic proof",
                    verified,
                    f"{manifest.proof.proof_system}/{manifest.proof.proof_type}",
                )
            )
        except Exception as exc:
            checks.append(
                ProofCheck(
                    "cryptographic proof",
                    ProofCheckStatus.FAIL,
                    f"verifier error: {exc}",
                )
            )
    return ComputationProofVerification(tuple(checks))


def write_computation_proof_manifest(
    manifest: ComputationProofManifest, path: str | Path
) -> str:
    """Write a deterministic manifest and return its integrity digest."""
    # Reparse the in-memory representation before persistence so constructed
    # objects cannot carry an inconsistent stored digest.
    validated = ComputationProofManifest.from_dict(manifest.to_dict())
    text = json.dumps(
        validated.to_dict(),
        indent=1,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    if len(text.encode("utf-8")) > PROOF_MANIFEST_MAX_BYTES:
        raise ProofManifestError("proof manifest exceeds the encoded-size limit")
    try:
        with open(path, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
    except OSError as exc:
        raise ProofManifestError(f"could not write proof manifest: {exc}") from exc
    return validated.manifest_digest


def read_computation_proof_manifest(path: str | Path) -> ComputationProofManifest:
    """Read and integrity-check a proof manifest without fetching its proof."""
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise ProofManifestError(f"could not read proof manifest: {exc}") from exc
    if len(raw) > PROOF_MANIFEST_MAX_BYTES:
        raise ProofManifestError("proof manifest exceeds the encoded-size limit")

    def reject_constant(token: str) -> object:
        raise ProofManifestError(f"proof manifest contains non-finite number {token!r}")

    try:
        document = json.loads(raw.decode("utf-8"), parse_constant=reject_constant)
    except ProofManifestError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ProofManifestError(f"invalid proof manifest JSON: {exc}") from exc
    return ComputationProofManifest.from_dict(document)


def _ledger_commits_computation_result(
    manifest: ComputationProofManifest,
    ledger: ExecutionLedger,
) -> bool:
    """Require an exact, later assessment to bind the optional result digest."""
    expected = manifest.computation_result_digest
    if expected is None:
        return True
    for event in ledger.events[manifest.event_seq + 1 :]:
        if (
            event.kind != "assessment"
            or event.result_world_digest != manifest.result_world_digest
        ):
            continue
        details = event.operation.get("details")
        if not isinstance(details, dict):
            continue
        computation = details.get("computation")
        if not isinstance(computation, dict):
            continue
        if computation.get("computation_digest") == expected:
            return True
    return False


def _manifest_document(manifest: ComputationProofManifest) -> dict[str, object]:
    return {
        "format": PROOF_MANIFEST_FORMAT,
        "schema_version": PROOF_MANIFEST_SCHEMA_VERSION,
        "runtime_contract": LEDGER_RUNTIME_CONTRACT,
        "claim_scope": PROOF_CLAIM_SCOPE,
        "statement": {
            "ledger_head": manifest.ledger_head,
            "event_seq": manifest.event_seq,
            "event_hash": manifest.event_hash,
            "prior_world_digest": manifest.prior_world_digest,
            "result_world_digest": manifest.result_world_digest,
            "operation_digest": manifest.operation_digest,
            "verification_digest": manifest.verification_digest,
            "computation_result_digest": manifest.computation_result_digest,
        },
        "numeric_contract": manifest.numeric_contract.to_dict(),
        "engineering_context": {
            "model_contract_digest": manifest.model_contract_digest,
            "validation_profile_digest": manifest.validation_profile_digest,
            "evidence_commitments": list(manifest.evidence_commitments),
        },
        "proof": manifest.proof.to_dict(),
    }


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ProofManifestError(f"value is not canonical JSON: {exc}") from exc


def _digest_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ProofManifestError(f"{label} must be an object")
    out: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ProofManifestError(f"{label} contains a non-string key")
        out[key] = _normalise_json(item, f"{label}.{key}")
    return out


def _normalise_json(value: object, label: str) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProofManifestError(f"{label} contains a non-finite number")
        return value
    if isinstance(value, list):
        return [_normalise_json(item, f"{label}[]") for item in value]
    if isinstance(value, Mapping):
        return _object(value, label)
    raise ProofManifestError(f"{label} contains unsupported value {type(value).__name__}")


def _expect_fields(record: Mapping[str, object], fields: set[str], label: str) -> None:
    actual = set(record)
    if actual != fields:
        raise ProofManifestError(
            f"{label} fields differ; missing={sorted(fields - actual)}, "
            f"extra={sorted(actual - fields)}"
        )


def _require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProofManifestError(f"{label} must be non-empty text")
    return value


def _require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ProofManifestError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProofManifestError(f"{label} must be a non-negative integer")
    return value


def _bytes(value: object, label: str) -> bytes:
    if not isinstance(value, bytes):
        raise ProofManifestError(f"{label} must be bytes")
    return value


def _check(name: str, passed: bool, detail: str) -> ProofCheck:
    return ProofCheck(
        name,
        ProofCheckStatus.PASS if passed else ProofCheckStatus.FAIL,
        detail,
    )


__all__ = [
    "ComputationProofManifest",
    "ComputationProofVerification",
    "NumericContract",
    "PROOF_CLAIM_SCOPE",
    "PROOF_MANIFEST_FORMAT",
    "PROOF_MANIFEST_HASH_ALGORITHM",
    "PROOF_MANIFEST_SCHEMA_VERSION",
    "ProofArtifactCommitment",
    "ProofCheck",
    "ProofCheckStatus",
    "ProofVerifier",
    "computation_proof_public_values",
    "computation_proof_public_values_digest",
    "create_computation_proof_manifest",
    "read_computation_proof_manifest",
    "verify_computation_proof_manifest",
    "write_computation_proof_manifest",
]
