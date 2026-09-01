"""Typed, calibrated observations at GAT's epistemic boundary.

An evidence item is not a state value.  It is an immutable claim about one
identified state variable, its uncertainty, provenance, and source class.
Only its explicit observation transformation may condition the canonical
belief.  This prevents an inferred or assumed value from silently becoming
equivalent to a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
import math
import re

from gat.engine.executor import World
from gat.engine.transform import ObserveQuantity
from gat.ids import VarId


_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")


class EvidenceKind(StrEnum):
    """Epistemic origin of a claim entering the belief state."""

    MEASURED = "MEASURED"
    ESTIMATED = "ESTIMATED"
    INFERRED = "INFERRED"
    ASSUMED = "ASSUMED"
    SIMULATED = "SIMULATED"
    DERIVED = "DERIVED"


def _var_record(var: VarId) -> dict[str, object]:
    return {
        "entity": {
            "ifc_class": var.entity.ifc_class,
            "global_id": var.entity.global_id,
        },
        "quantity": var.quantity,
    }


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CalibratedObservation:
    """One source-bound scalar likelihood for a canonical raw variable."""

    evidence_id: str
    subject: VarId
    kind: EvidenceKind
    observed_value: float
    noise_sigma: float
    unit: str
    source_digest: str
    method: str
    calibration_digest: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.evidence_id, str) or not self.evidence_id:
            raise ValueError("evidence_id must be non-empty")
        if not isinstance(self.subject, VarId):
            raise TypeError("subject must be a VarId")
        if not isinstance(self.kind, EvidenceKind):
            object.__setattr__(self, "kind", EvidenceKind(self.kind))
        if not math.isfinite(self.observed_value):
            raise ValueError("observed_value must be finite")
        if not math.isfinite(self.noise_sigma) or self.noise_sigma <= 0.0:
            raise ValueError("noise_sigma must be finite and positive")
        if not isinstance(self.unit, str) or not self.unit:
            raise ValueError("unit must be non-empty")
        if (
            not isinstance(self.source_digest, str)
            or _DIGEST_RE.fullmatch(self.source_digest) is None
        ):
            raise ValueError("source_digest must be a lowercase SHA-256 digest")
        if not isinstance(self.method, str) or not self.method:
            raise ValueError("method must be non-empty")
        if (
            self.calibration_digest is not None
            and (
                not isinstance(self.calibration_digest, str)
                or _DIGEST_RE.fullmatch(self.calibration_digest) is None
            )
        ):
            raise ValueError(
                "calibration_digest must be a lowercase SHA-256 digest or None"
            )

    @classmethod
    def from_source_bytes(
        cls,
        evidence_id: str,
        subject: VarId,
        kind: EvidenceKind,
        observed_value: float,
        noise_sigma: float,
        unit: str,
        source_bytes: bytes,
        method: str,
        calibration_digest: str | None = None,
    ) -> "CalibratedObservation":
        if not isinstance(source_bytes, bytes):
            raise TypeError("source_bytes must be bytes")
        return cls(
            evidence_id,
            subject,
            kind,
            observed_value,
            noise_sigma,
            unit,
            hashlib.sha256(source_bytes).hexdigest(),
            method,
            calibration_digest,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "evidence_id": self.evidence_id,
            "subject": _var_record(self.subject),
            "kind": self.kind.value,
            "observed_value": self.observed_value,
            "noise_sigma": self.noise_sigma,
            "unit": self.unit,
            "source_digest": self.source_digest,
            "method": self.method,
            "calibration_digest": self.calibration_digest,
        }

    def digest(self) -> str:
        """Content identity used by the ledger and engineering record."""
        return _canonical_digest(self.to_dict())

    def transformation(self, world: World) -> ObserveQuantity:
        """Validate the subject/unit boundary and return explicit conditioning."""
        if not world.binding.is_raw(self.subject):
            raise ValueError(f"evidence subject {self.subject} is not a raw variable")
        slot = world.module.entities[self.subject.entity].slots[self.subject.quantity]
        if slot.unit.value != self.unit:
            raise ValueError(
                f"evidence unit {self.unit!r} differs from canonical unit "
                f"{slot.unit.value!r} for {self.subject}"
            )
        return ObserveQuantity.single(
            self.subject,
            self.observed_value,
            self.noise_sigma,
        )

    def provenance(self) -> dict[str, object]:
        """Closed JSON provenance attached to the accepted ledger transition."""
        return {
            "evidence_digest": self.digest(),
            "evidence": self.to_dict(),
        }


__all__ = ["CalibratedObservation", "EvidenceKind"]
