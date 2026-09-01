"""Narrow ANSI/AISC 360-22 F2 yielding check with a pinned oracle profile.

This module implements only LRFD major-axis yielding for a continuously braced,
compact, doubly symmetric W-shape.  It intentionally requires the plastic
section modulus ``Zx``; an elastic section modulus derived by the IFC geometry
adapter is not interchangeable with that design-code input.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
import math
from types import MappingProxyType
from typing import Mapping


AISC360_22_F2_LRFD_METHOD = "ansi-aisc-360-22-f2-1-lrfd-v1"
AISC360_22_F2_LRFD_ORACLE_ID = "aisc-v16-example-f1-1b-lrfd-v1"
AISC360_22_F2_LRFD_ORACLE_RECORD_DIGEST = (
    "3a6b0e5a4b1d3020d6e4b28f7941f251b77bc6ec896c6f1d35f055f6378f9a77"
)
AISC360_22_PHI_B = 0.90


def _canonical_digest(value: object) -> str:
    def json_value(item: object) -> object:
        if isinstance(item, Mapping):
            return {str(key): json_value(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [json_value(child) for child in item]
        return item

    encoded = json.dumps(
        json_value(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


AISC360_22_F2_LRFD_VALIDATION_PROFILE: Mapping[str, object] = MappingProxyType({
    "method": AISC360_22_F2_LRFD_METHOD,
    "standard": "ANSI/AISC 360-22",
    "chapter": "F",
    "section": "F2",
    "equation": "F2-1",
    "design_basis": "LRFD",
    "resistance_factor_phi_b": AISC360_22_PHI_B,
    "implemented_limit_state": "major-axis flexural yielding",
    "required_scope": MappingProxyType({
        "shape_family": "doubly-symmetric-w-shape",
        "section_classification": "compact",
        "bracing": "continuously-braced",
        "bending_axis": "major",
        "section_property": "plastic-section-modulus-zx",
    }),
    "excluded_limit_states": (
        "lateral-torsional-buckling",
        "flange-local-buckling",
        "web-local-buckling",
        "shear",
        "deflection",
        "combined-forces",
    ),
    "oracle_id": AISC360_22_F2_LRFD_ORACLE_ID,
    "oracle_record_digest": AISC360_22_F2_LRFD_ORACLE_RECORD_DIGEST,
})
AISC360_22_F2_LRFD_VALIDATION_PROFILE_DIGEST = _canonical_digest(
    AISC360_22_F2_LRFD_VALIDATION_PROFILE
)


class Aisc36022Verdict(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass(frozen=True)
class Aisc36022F2YieldingCheck:
    """SI inputs admitted by the bounded F2-1 LRFD calculation."""

    yield_strength_mpa: float
    plastic_section_modulus_major_m3: float
    factored_demand_n_m: float
    shape_family: str = "doubly-symmetric-w-shape"
    section_classification: str = "compact"
    bracing: str = "continuously-braced"
    bending_axis: str = "major"
    section_property_kind: str = "plastic-section-modulus-zx"

    def __post_init__(self) -> None:
        for label, value in (
            ("yield_strength_mpa", self.yield_strength_mpa),
            (
                "plastic_section_modulus_major_m3",
                self.plastic_section_modulus_major_m3,
            ),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{label} must be finite and positive")
        if (
            not math.isfinite(self.factored_demand_n_m)
            or self.factored_demand_n_m < 0.0
        ):
            raise ValueError("factored_demand_n_m must be finite and non-negative")
        required = AISC360_22_F2_LRFD_VALIDATION_PROFILE["required_scope"]
        actual = {
            "shape_family": self.shape_family,
            "section_classification": self.section_classification,
            "bracing": self.bracing,
            "bending_axis": self.bending_axis,
            "section_property": self.section_property_kind,
        }
        if actual != required:
            differences = {
                key: {"required": required[key], "actual": actual[key]}
                for key in required
                if actual[key] != required[key]
            }
            raise ValueError(
                f"check is outside the implemented AISC 360-22 F2 scope: {differences}"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "yield_strength_mpa": self.yield_strength_mpa,
            "plastic_section_modulus_major_m3": (
                self.plastic_section_modulus_major_m3
            ),
            "factored_demand_n_m": self.factored_demand_n_m,
            "scope": {
                "shape_family": self.shape_family,
                "section_classification": self.section_classification,
                "bracing": self.bracing,
                "bending_axis": self.bending_axis,
                "section_property": self.section_property_kind,
            },
        }


@dataclass(frozen=True)
class Aisc36022F2YieldingResult:
    check: Aisc36022F2YieldingCheck
    nominal_moment_n_m: float
    available_moment_n_m: float
    utilization: float
    verdict: Aisc36022Verdict

    def computation_record(self) -> dict[str, object]:
        return {
            "method": AISC360_22_F2_LRFD_METHOD,
            "validation_profile_digest": (
                AISC360_22_F2_LRFD_VALIDATION_PROFILE_DIGEST
            ),
            "input": self.check.to_dict(),
            "calculation": {
                "nominal_expression": "1e6 * Fy_MPa * Zx_m3",
                "available_expression": "0.90 * nominal_moment",
                "nominal_moment_n_m": self.nominal_moment_n_m,
                "available_moment_n_m": self.available_moment_n_m,
                "utilization": self.utilization,
                "verdict": self.verdict.value,
            },
        }

    @property
    def computation_digest(self) -> str:
        return _canonical_digest(self.computation_record())

    def to_dict(self) -> dict[str, object]:
        return {
            **self.computation_record(),
            "computation_digest": self.computation_digest,
        }


def evaluate_aisc36022_f2_yielding(
    check: Aisc36022F2YieldingCheck,
) -> Aisc36022F2YieldingResult:
    """Evaluate ANSI/AISC 360-22 Eq. F2-1 within the declared narrow scope."""
    nominal = (
        1_000_000.0
        * check.yield_strength_mpa
        * check.plastic_section_modulus_major_m3
    )
    available = AISC360_22_PHI_B * nominal
    utilization = check.factored_demand_n_m / available
    verdict = (
        Aisc36022Verdict.PASS
        if check.factored_demand_n_m <= available
        else Aisc36022Verdict.FAIL
    )
    return Aisc36022F2YieldingResult(
        check,
        nominal,
        available,
        utilization,
        verdict,
    )


__all__ = [
    "AISC360_22_F2_LRFD_METHOD",
    "AISC360_22_F2_LRFD_ORACLE_ID",
    "AISC360_22_F2_LRFD_ORACLE_RECORD_DIGEST",
    "AISC360_22_F2_LRFD_VALIDATION_PROFILE",
    "AISC360_22_F2_LRFD_VALIDATION_PROFILE_DIGEST",
    "AISC360_22_PHI_B",
    "Aisc36022F2YieldingCheck",
    "Aisc36022F2YieldingResult",
    "Aisc36022Verdict",
    "evaluate_aisc36022_f2_yielding",
]
