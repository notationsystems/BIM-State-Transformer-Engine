"""Closed material-certificate ingestion into calibrated GAT evidence.

Certificate bytes remain distinct from belief state.  This adapter validates a
small, versioned JSON contract, binds it to one ``IfcBeam`` raw variable, and
preserves issuer, batch/specimen, calibration, units, and the exact artifact
digest in ledger provenance.  It validates structure and identity only; it
does not claim issuer trust, signature validation, revocation status, or
professional approval.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping

from gat.engine.executor import World
from gat.evidence import CalibratedObservation, EvidenceKind
from gat.errors import CertificateIngestionError
from gat.ids import EntityId, VarId


MATERIAL_CERTIFICATE_FORMAT = "gat-material-certificate-observation-v1"
MATERIAL_CERTIFICATE_SCHEMA_VERSION = 1
_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")


def _reject_constant(value: str) -> None:
    raise CertificateIngestionError(f"non-finite JSON number {value!r} is forbidden")


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CertificateIngestionError(f"duplicate certificate field {key!r}")
        result[key] = value
    return result


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CertificateIngestionError(f"{label} must be a JSON object")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise CertificateIngestionError(
            f"{label} fields differ; missing={missing}, extra={extra}"
        )


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CertificateIngestionError(f"{label} must be non-empty text")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CertificateIngestionError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CertificateIngestionError(f"{label} must be finite")
    return result


@dataclass(frozen=True)
class MaterialCertificate:
    certificate_id: str
    issued_at: str
    issuer_organization_id: str
    issuer_name: str
    subject_ifc_class: str
    subject_ifc_global_id: str
    material_batch_id: str
    specimen_id: str
    property_name: str
    reported_value: float
    standard_uncertainty_sigma: float
    unit: str
    method: str
    calibration_id: str
    calibration_digest: str
    source_digest: str

    def __post_init__(self) -> None:
        for label, value in (
            ("certificate_id", self.certificate_id),
            ("issued_at", self.issued_at),
            ("issuer_organization_id", self.issuer_organization_id),
            ("issuer_name", self.issuer_name),
            ("subject_ifc_global_id", self.subject_ifc_global_id),
            ("material_batch_id", self.material_batch_id),
            ("specimen_id", self.specimen_id),
            ("method", self.method),
            ("calibration_id", self.calibration_id),
        ):
            _text(value, label)
        if self.subject_ifc_class != "IfcBeam":
            raise CertificateIngestionError("certificate subject must be IfcBeam")
        if self.property_name != "YieldStrengthMPa" or self.unit != "MPa":
            raise CertificateIngestionError(
                "v1 certificates support only YieldStrengthMPa in MPa"
            )
        if not math.isfinite(self.reported_value):
            raise CertificateIngestionError("reported value must be finite")
        if (
            not math.isfinite(self.standard_uncertainty_sigma)
            or self.standard_uncertainty_sigma <= 0.0
        ):
            raise CertificateIngestionError(
                "standard uncertainty sigma must be finite and positive"
            )
        for label, digest in (
            ("calibration_digest", self.calibration_digest),
            ("source_digest", self.source_digest),
        ):
            if _DIGEST_RE.fullmatch(digest) is None:
                raise CertificateIngestionError(
                    f"{label} must be a lowercase SHA-256 digest"
                )

    @property
    def subject(self) -> VarId:
        return VarId(
            EntityId(self.subject_ifc_class, self.subject_ifc_global_id),
            self.property_name,
        )

    def provenance_record(self) -> dict[str, object]:
        return {
            "format": MATERIAL_CERTIFICATE_FORMAT,
            "schema_version": MATERIAL_CERTIFICATE_SCHEMA_VERSION,
            "certificate_id": self.certificate_id,
            "issued_at": self.issued_at,
            "issuer": {
                "organization_id": self.issuer_organization_id,
                "name": self.issuer_name,
                "trust_verified": False,
            },
            "subject": {
                "ifc_class": self.subject_ifc_class,
                "ifc_global_id": self.subject_ifc_global_id,
                "property": self.property_name,
            },
            "material": {
                "batch_id": self.material_batch_id,
                "specimen_id": self.specimen_id,
            },
            "calibration": {
                "calibration_id": self.calibration_id,
                "calibration_digest": self.calibration_digest,
            },
            "unit": self.unit,
            "source_digest": self.source_digest,
            "assurance": {
                "schema_validated": True,
                "signature_verified": False,
                "revocation_checked": False,
                "authorizes_engineering_decision": False,
            },
        }

    def to_evidence(self, world: World) -> "MaterialCertificateEvidence":
        entity = world.module.entities.get(self.subject.entity)
        if entity is None:
            raise CertificateIngestionError(
                f"certificate subject {self.subject.entity.global_id} is absent"
            )
        slot = entity.slots.get(self.property_name)
        if slot is None or not world.binding.is_raw(self.subject):
            raise CertificateIngestionError(
                f"certificate subject {self.subject} is not a raw state variable"
            )
        if slot.unit.value != self.unit:
            raise CertificateIngestionError(
                f"certificate unit {self.unit!r} differs from {slot.unit.value!r}"
            )
        observation = CalibratedObservation(
            self.certificate_id,
            self.subject,
            EvidenceKind.MEASURED,
            self.reported_value,
            self.standard_uncertainty_sigma,
            self.unit,
            self.source_digest,
            self.method,
            self.calibration_digest,
        )
        return MaterialCertificateEvidence(self, observation)


@dataclass(frozen=True)
class MaterialCertificateEvidence:
    certificate: MaterialCertificate
    observation: CalibratedObservation

    def __post_init__(self) -> None:
        if self.observation.subject != self.certificate.subject:
            raise ValueError("certificate and observation subjects differ")
        if self.observation.source_digest != self.certificate.source_digest:
            raise ValueError("certificate and observation source digests differ")

    def provenance(self) -> dict[str, object]:
        return {
            **self.observation.provenance(),
            "material_certificate": self.certificate.provenance_record(),
        }


def parse_material_certificate(source_bytes: bytes) -> MaterialCertificate:
    if not isinstance(source_bytes, bytes):
        raise TypeError("source_bytes must be bytes")
    try:
        document = json.loads(
            source_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except CertificateIngestionError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CertificateIngestionError(f"invalid certificate JSON: {exc}") from exc
    root = _mapping(document, "certificate")
    _exact_keys(
        root,
        {
            "format",
            "schema_version",
            "certificate_id",
            "issued_at",
            "issuer",
            "subject",
            "material",
            "observation",
            "calibration",
        },
        "certificate",
    )
    if root["format"] != MATERIAL_CERTIFICATE_FORMAT:
        raise CertificateIngestionError("unsupported material certificate format")
    if root["schema_version"] != MATERIAL_CERTIFICATE_SCHEMA_VERSION:
        raise CertificateIngestionError("unsupported material certificate schema version")
    issuer = _mapping(root["issuer"], "issuer")
    subject = _mapping(root["subject"], "subject")
    material = _mapping(root["material"], "material")
    observation = _mapping(root["observation"], "observation")
    calibration = _mapping(root["calibration"], "calibration")
    _exact_keys(issuer, {"organization_id", "name"}, "issuer")
    _exact_keys(subject, {"ifc_class", "ifc_global_id"}, "subject")
    _exact_keys(material, {"batch_id", "specimen_id"}, "material")
    _exact_keys(
        observation,
        {
            "property",
            "reported_value",
            "standard_uncertainty_sigma",
            "unit",
            "method",
        },
        "observation",
    )
    _exact_keys(
        calibration,
        {"calibration_id", "calibration_digest"},
        "calibration",
    )
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    return MaterialCertificate(
        _text(root["certificate_id"], "certificate_id"),
        _text(root["issued_at"], "issued_at"),
        _text(issuer["organization_id"], "issuer.organization_id"),
        _text(issuer["name"], "issuer.name"),
        _text(subject["ifc_class"], "subject.ifc_class"),
        _text(subject["ifc_global_id"], "subject.ifc_global_id"),
        _text(material["batch_id"], "material.batch_id"),
        _text(material["specimen_id"], "material.specimen_id"),
        _text(observation["property"], "observation.property"),
        _finite(observation["reported_value"], "observation.reported_value"),
        _finite(
            observation["standard_uncertainty_sigma"],
            "observation.standard_uncertainty_sigma",
        ),
        _text(observation["unit"], "observation.unit"),
        _text(observation["method"], "observation.method"),
        _text(calibration["calibration_id"], "calibration.calibration_id"),
        _text(calibration["calibration_digest"], "calibration.calibration_digest"),
        source_digest,
    )


def read_material_certificate(path: str | Path) -> MaterialCertificate:
    try:
        source = Path(path).read_bytes()
    except OSError as exc:
        raise CertificateIngestionError(f"cannot read material certificate: {exc}") from exc
    return parse_material_certificate(source)


__all__ = [
    "MATERIAL_CERTIFICATE_FORMAT",
    "MATERIAL_CERTIFICATE_SCHEMA_VERSION",
    "MaterialCertificate",
    "MaterialCertificateEvidence",
    "parse_material_certificate",
    "read_material_certificate",
]
