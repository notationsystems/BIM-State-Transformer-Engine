"""Non-mutating design-change and RFI impact analysis.

The execution engine already knows how a proposed transformation propagates
through the architectural dependency graph and whether its candidate state
passes verification.  This module turns that result into a stable workflow
artifact suitable for an RFI, design review, or CI gate.  It never commits
the candidate and never records an approval.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
import math

import numpy as np

from gat.engine.executor import World, preview
from gat.engine.transform import Transformation
from gat.engine.verify import InvariantResult, Status
from gat.ids import VarId
from gat.ir.core import Role
from gat.ledger import encode_transformation


class ChangeDisposition(StrEnum):
    ADMISSIBLE = "ADMISSIBLE"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class VariableImpact:
    var: VarId
    role: Role
    unit: str
    target: bool
    affected: bool
    mean_before: float
    mean_after: float
    mean_delta: float
    sigma_before: float
    sigma_after: float
    sigma_delta: float

    def __post_init__(self) -> None:
        values = (
            self.mean_before,
            self.mean_after,
            self.mean_delta,
            self.sigma_before,
            self.sigma_after,
            self.sigma_delta,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("variable impact values must be finite")

    def to_dict(self) -> dict[str, object]:
        return {
            "variable": str(self.var),
            "entity": {
                "ifc_class": self.var.entity.ifc_class,
                "global_id": self.var.entity.global_id,
            },
            "quantity": self.var.quantity,
            "role": self.role.value,
            "unit": self.unit,
            "target": self.target,
            "affected": self.affected,
            "mean_before": self.mean_before,
            "mean_after": self.mean_after,
            "mean_delta": self.mean_delta,
            "sigma_before": self.sigma_before,
            "sigma_after": self.sigma_after,
            "sigma_delta": self.sigma_delta,
        }


@dataclass(frozen=True)
class ChangeImpactReport:
    prior_world_digest: str
    candidate_world_digest: str
    transformation: Transformation
    transformation_payload: dict[str, object]
    disposition: ChangeDisposition
    targets: tuple[VarId, ...]
    affected: tuple[VarId, ...]
    impacted_entities: tuple[str, ...]
    impacts: tuple[VariableImpact, ...]
    verification_results: tuple[InvariantResult, ...]
    scope_digest: str

    @property
    def admissible(self) -> bool:
        return self.disposition is ChangeDisposition.ADMISSIBLE

    @property
    def failures(self) -> tuple[InvariantResult, ...]:
        return tuple(
            result
            for result in self.verification_results
            if result.status is Status.FAIL
        )

    @property
    def warnings(self) -> tuple[InvariantResult, ...]:
        return tuple(
            result
            for result in self.verification_results
            if result.status is Status.WARN
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "scope_digest": self.scope_digest,
            "prior_world_digest": self.prior_world_digest,
            "candidate_world_digest": self.candidate_world_digest,
            "disposition": self.disposition.value,
            "admissible": self.admissible,
            "transformation": self.transformation_payload,
            "targets": [str(var) for var in self.targets],
            "affected": [str(var) for var in self.affected],
            "impacted_entities": list(self.impacted_entities),
            "impact_count": len(self.impacts),
            "impacts": [impact.to_dict() for impact in self.impacts],
            "verification": {
                "failures": [_invariant_dict(item) for item in self.failures],
                "warnings": [_invariant_dict(item) for item in self.warnings],
            },
        }


def preview_change(world: World, transformation: Transformation) -> ChangeImpactReport:
    """Run the exact execution pipeline and expose its candidate as an RFI report."""
    candidate = preview(world, transformation)
    target_set = set(candidate.targets)
    affected_set = set(candidate.affected)
    before_variance = np.diag(world.full.sigma)
    after_variance = np.diag(candidate.candidate.full.sigma)

    impacts: list[VariableImpact] = []
    for row, var in enumerate(world.full.index.vars):
        mean_before = float(world.full.mu[row])
        mean_after = float(candidate.candidate.full.mu[row])
        sigma_before = math.sqrt(max(float(before_variance[row]), 0.0))
        sigma_after = math.sqrt(max(float(after_variance[row]), 0.0))
        if (
            mean_before == mean_after
            and sigma_before == sigma_after
            and var not in target_set
            and var not in affected_set
        ):
            continue
        slot = world.module.slot(var)
        impacts.append(
            VariableImpact(
                var=var,
                role=slot.role,
                unit=slot.unit.value,
                target=var in target_set,
                affected=var in affected_set,
                mean_before=mean_before,
                mean_after=mean_after,
                mean_delta=mean_after - mean_before,
                sigma_before=sigma_before,
                sigma_after=sigma_after,
                sigma_delta=sigma_after - sigma_before,
            )
        )

    impacted_entities = tuple(
        sorted(
            {
                world.module.entity(impact.var.entity).name
                for impact in impacts
            }
        )
    )
    payload = encode_transformation(transformation)
    scope_payload = {
        "prior_world_digest": candidate.prior_world_digest,
        "candidate_world_digest": candidate.candidate.digest(),
        "transformation": payload,
    }
    canonical = json.dumps(scope_payload, separators=(",", ":"), sort_keys=True)
    scope_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return ChangeImpactReport(
        prior_world_digest=candidate.prior_world_digest,
        candidate_world_digest=candidate.candidate.digest(),
        transformation=transformation,
        transformation_payload=payload,
        disposition=(
            ChangeDisposition.ADMISSIBLE
            if candidate.admissible
            else ChangeDisposition.BLOCKED
        ),
        targets=candidate.targets,
        affected=candidate.affected,
        impacted_entities=impacted_entities,
        impacts=tuple(impacts),
        verification_results=candidate.report.results,
        scope_digest=scope_digest,
    )


def _invariant_dict(result: InvariantResult) -> dict[str, object]:
    return {
        "invariant_id": result.invariant_id,
        "status": result.status.value,
        "subject": result.subject,
        "residual": result.residual,
        "detail": result.detail,
    }


__all__ = [
    "ChangeDisposition",
    "ChangeImpactReport",
    "VariableImpact",
    "preview_change",
]
