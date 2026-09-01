"""Identity-preserving probabilistic beam bending assurance.

The deterministic model is the bounded ANSI/AISC 360-22 F2-1 LRFD yielding
check validated in :mod:`gat.engineering.aisc360_22`:

    nominal moment capacity = 1e6 * fy[MPa] * Zx[m^3]
    design moment capacity  = 0.90 * nominal moment capacity

The Gaussian engine supplies the capacity distribution.  This module binds
that result to one beam, model contract, dependency slice, decision criterion,
and exact world digest.  Its evaluator caches only the engineering
calculation: an unrelated world change can be rebound without rerunning the
check, while a changed dependency necessarily produces a new computation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Iterable

import numpy as np

from gat.causal import AssessmentRecord
from gat.engine.decision import (
    DecisionAssessment,
    DecisionVerdict,
    MinimumDecision,
    assess_decision,
)
from gat.engine.executor import ExecutionResult, World
from gat.evidence import CalibratedObservation
from gat.engineering.aisc360_22 import (
    AISC360_22_F2_LRFD_METHOD,
    AISC360_22_F2_LRFD_ORACLE_ID,
    AISC360_22_F2_LRFD_ORACLE_RECORD_DIGEST,
    AISC360_22_F2_LRFD_VALIDATION_PROFILE,
    AISC360_22_F2_LRFD_VALIDATION_PROFILE_DIGEST,
)
from gat.ids import EntityId, VarId
from gat.ledger import verification_digest
from gat.engine.verify import run_invariants


BEAM_BENDING_METHOD = AISC360_22_F2_LRFD_METHOD
BEAM_DECISION_METHOD = "minimum-gaussian-aisc360-22-capacity-v1"


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _var_record(var: VarId) -> dict[str, object]:
    return {
        "entity": {
            "ifc_class": var.entity.ifc_class,
            "global_id": var.entity.global_id,
        },
        "quantity": var.quantity,
    }


def _var_label(var: VarId) -> str:
    return f"{var.entity.ifc_class}:{var.entity.global_id}.{var.quantity}"


@dataclass(frozen=True)
class BeamBendingCheck:
    """Decision contract for one opt-in structural beam."""

    beam: EntityId
    factored_demand_n_m: float
    confidence: float = 0.95
    label: str = ""

    def __post_init__(self) -> None:
        if self.beam.ifc_class != "IfcBeam":
            raise ValueError("beam bending check requires an IfcBeam identity")
        if not math.isfinite(self.factored_demand_n_m) or self.factored_demand_n_m < 0:
            raise ValueError("factored demand must be finite and non-negative")
        if not math.isfinite(self.confidence) or not 0.5 < self.confidence < 1.0:
            raise ValueError("confidence must be finite and between 0.5 and 1")

    @property
    def capacity_var(self) -> VarId:
        return VarId(self.beam, "DesignMomentCapacity")

    def decision(self) -> MinimumDecision:
        return MinimumDecision(
            self.capacity_var,
            self.factored_demand_n_m,
            self.confidence,
            self.label or f"{self.beam.global_id} bending resistance",
        )


@dataclass(frozen=True)
class BeamCheckResult:
    """One deterministic calculation/decision bound to an exact belief slice."""

    check: BeamBendingCheck
    assessment: DecisionAssessment
    dependency_vars: tuple[VarId, ...]
    dependency_digest: str
    model_contract_digest: str
    validation_profile_digest: str
    computation_digest: str
    recomputed: bool
    changed_inputs: tuple[VarId, ...] = ()
    affected_variables: tuple[VarId, ...] = ()

    @property
    def verdict(self) -> DecisionVerdict:
        return self.assessment.verdict

    def computation_details(self) -> dict[str, object]:
        return {
            "method": BEAM_BENDING_METHOD,
            "model_contract_digest": self.model_contract_digest,
            "validation_profile_digest": self.validation_profile_digest,
            "design_code_validation_profile_digest": (
                AISC360_22_F2_LRFD_VALIDATION_PROFILE_DIGEST
            ),
            "independent_oracle_id": AISC360_22_F2_LRFD_ORACLE_ID,
            "independent_oracle_record_digest": (
                AISC360_22_F2_LRFD_ORACLE_RECORD_DIGEST
            ),
            "dependency_digest": self.dependency_digest,
            "computation_digest": self.computation_digest,
            "dependency_variables": [_var_record(var) for var in self.dependency_vars],
            "recomputed": self.recomputed,
            "changed_inputs": [_var_record(var) for var in self.changed_inputs],
            "affected_variables": [
                _var_record(var) for var in self.affected_variables
            ],
        }


@dataclass(frozen=True)
class BeamDecisionChange:
    """Exact causal explanation for a beam decision revision."""

    evidence: CalibratedObservation
    prior: BeamCheckResult
    revised: BeamCheckResult
    changed_beliefs: tuple[dict[str, object], ...]
    covariance_changes: tuple[dict[str, object], ...]
    reason: str

    @property
    def verdict_changed(self) -> bool:
        return self.prior.verdict is not self.revised.verdict


@dataclass(frozen=True)
class _CachedCalculation:
    target_mean: float
    target_sigma: float
    p_satisfies: float
    verdict: DecisionVerdict
    computation_digest: str


class BeamBendingEvaluator:
    """Dependency-keyed evaluator proving when the beam check did or did not run."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str, str], _CachedCalculation] = {}

    def evaluate(
        self,
        world: World,
        check: BeamBendingCheck,
        *,
        changed_inputs: Iterable[VarId] = (),
        affected_variables: Iterable[VarId] = (),
    ) -> BeamCheckResult:
        entity = world.module.entities.get(check.beam)
        if entity is None:
            raise ValueError(f"beam {check.beam.global_id} is absent from the world")
        if entity.attrs.get("structural_method") != BEAM_BENDING_METHOD:
            raise ValueError("beam does not carry the supported structural contract")
        required = {
            "YieldStrengthMPa",
            "PlasticSectionModulusMajorM3",
            "NominalMomentCapacity",
            "DesignMomentCapacity",
        }
        if not required.issubset(entity.slots):
            raise ValueError("beam structural state is incomplete")
        expected_scope = AISC360_22_F2_LRFD_VALIDATION_PROFILE["required_scope"]
        actual_scope = {key: entity.attrs.get(key) for key in expected_scope}
        if actual_scope != expected_scope:
            raise ValueError("beam is outside the validated AISC 360-22 F2 scope")

        model_contract = self.model_contract(world, check)
        validation_profile = {
            "method": BEAM_DECISION_METHOD,
            "criterion": "P(design_capacity >= factored_demand)",
            "confidence": check.confidence,
            "covariance": "first-order-jacobian-pushforward",
            "verdicts": ["SATISFIED", "VIOLATED", "UNRESOLVED"],
            "design_code_validation_profile_digest": (
                AISC360_22_F2_LRFD_VALIDATION_PROFILE_DIGEST
            ),
            "independent_oracle_id": AISC360_22_F2_LRFD_ORACLE_ID,
        }
        model_digest = _canonical_digest(model_contract)
        validation_digest = _canonical_digest(validation_profile)
        dependencies = self._dependencies(world, check.capacity_var)
        dependency_payload = self._dependency_payload(world, dependencies)
        dependency_digest = _canonical_digest(dependency_payload)
        cache_key = (model_digest, validation_digest, dependency_digest)

        cached = self._cache.get(cache_key)
        recomputed = cached is None
        if cached is None:
            assessment = assess_decision(world, check.decision())
            output = {
                "model_contract_digest": model_digest,
                "validation_profile_digest": validation_digest,
                "dependency_digest": dependency_digest,
                "target_mean": assessment.target_mean,
                "target_sigma": assessment.target_sigma,
                "p_satisfies": assessment.p_satisfies,
                "verdict": assessment.verdict.value,
            }
            cached = _CachedCalculation(
                assessment.target_mean,
                assessment.target_sigma,
                assessment.p_satisfies,
                assessment.verdict,
                _canonical_digest(output),
            )
            self._cache[cache_key] = cached
        else:
            assessment = DecisionAssessment(
                check.decision(),
                cached.target_mean,
                cached.target_sigma,
                cached.p_satisfies,
                cached.verdict,
                world.digest(),
            )

        return BeamCheckResult(
            check,
            assessment,
            dependencies,
            dependency_digest,
            model_digest,
            validation_digest,
            cached.computation_digest,
            recomputed,
            tuple(sorted(set(changed_inputs))),
            tuple(sorted(set(affected_variables))),
        )

    @staticmethod
    def model_contract(world: World, check: BeamBendingCheck) -> dict[str, object]:
        entity = world.module.entities[check.beam]
        return {
            "method": BEAM_BENDING_METHOD,
            "module_semantics_digest": world.module.digest(),
            "subject": {
                "ifc_class": check.beam.ifc_class,
                "global_id": check.beam.global_id,
            },
            "expression": (
                "0.90 * 1e6 * YieldStrengthMPa * "
                "PlasticSectionModulusMajorM3"
            ),
            "resistance_factor": float(entity.attrs["resistance_factor"]),
            "design_code_validation_profile_digest": (
                AISC360_22_F2_LRFD_VALIDATION_PROFILE_DIGEST
            ),
            "scope": {
                key: entity.attrs[key]
                for key in AISC360_22_F2_LRFD_VALIDATION_PROFILE["required_scope"]
            },
            "factored_demand_n_m": check.factored_demand_n_m,
            "units": {
                "YieldStrengthMPa": "MPa",
                "PlasticSectionModulusMajorM3": "m3",
                "DesignMomentCapacity": "N*m",
            },
        }

    @staticmethod
    def _dependencies(world: World, target: VarId) -> tuple[VarId, ...]:
        raw: set[VarId] = set()
        frontier = [target]
        visited: set[VarId] = set()
        while frontier:
            var = frontier.pop()
            if var in visited:
                continue
            visited.add(var)
            if world.binding.is_raw(var):
                raw.add(var)
            else:
                frontier.extend(world.binding.deps.parents(var))
        return tuple(var for var in world.binding.raw_index.vars if var in raw)

    @staticmethod
    def _dependency_payload(
        world: World, dependencies: tuple[VarId, ...]
    ) -> dict[str, object]:
        rows = [world.binding.raw_index.row(var) for var in dependencies]
        covariance = world.belief.sigma[np.ix_(rows, rows)]
        return {
            "variables": [_var_record(var) for var in dependencies],
            "mean": [float(world.belief.mu[row]) for row in rows],
            "covariance": [float(value) for value in covariance.ravel()],
        }


def explain_beam_decision_change(
    before_world: World,
    after_world: World,
    evidence: CalibratedObservation,
    transition: ExecutionResult,
    prior: BeamCheckResult,
    revised: BeamCheckResult,
) -> BeamDecisionChange:
    """Compare exact state slices and explain why a beam verdict changed."""
    if transition.world.digest() != after_world.digest():
        raise ValueError("transition result and after_world differ")
    if prior.assessment.world_digest != before_world.digest():
        raise ValueError("prior beam result is not bound to before_world")
    if revised.assessment.world_digest != after_world.digest():
        raise ValueError("revised beam result is not bound to after_world")
    if evidence.subject not in transition.targets:
        raise ValueError("evidence subject is not a transition target")

    changed: list[dict[str, object]] = []
    for var in before_world.binding.raw_index.vars:
        old_mean = before_world.belief.mean(var)
        new_mean = after_world.belief.mean(var)
        old_sigma = before_world.belief.std(var)
        new_sigma = after_world.belief.std(var)
        row = before_world.binding.raw_index.row(var)
        covariance_row_changed = not np.array_equal(
            before_world.belief.sigma[row],
            after_world.belief.sigma[row],
        )
        if old_mean != new_mean or old_sigma != new_sigma or covariance_row_changed:
            changed.append(
                {
                    "variable": _var_record(var),
                    "prior_mean": old_mean,
                    "posterior_mean": new_mean,
                    "prior_sigma": old_sigma,
                    "posterior_sigma": new_sigma,
                    "covariance_row_changed": covariance_row_changed,
                }
            )

    covariance: list[dict[str, object]] = []
    tracked = tuple(dict.fromkeys((*revised.dependency_vars, revised.check.capacity_var)))
    for left_index, left in enumerate(tracked):
        left_row = before_world.full.index.row(left)
        for right in tracked[left_index:]:
            right_row = before_world.full.index.row(right)
            covariance.append(
                {
                    "left": _var_record(left),
                    "right": _var_record(right),
                    "prior_covariance": float(
                        before_world.full.sigma[left_row, right_row]
                    ),
                    "posterior_covariance": float(
                        after_world.full.sigma[left_row, right_row]
                    ),
                }
            )

    fy = evidence.subject
    fy_before = before_world.belief.mean(fy)
    fy_after = after_world.belief.mean(fy)
    fy_sigma_before = before_world.belief.std(fy)
    fy_sigma_after = after_world.belief.std(fy)
    a = prior.assessment
    b = revised.assessment
    reason = (
        f"{_var_label(fy)} was conditioned by {evidence.kind.value} evidence "
        f"{evidence.evidence_id} at {evidence.observed_value:g} {evidence.unit} "
        f"(noise sigma {evidence.noise_sigma:g}), changing its belief from "
        f"{fy_before:.6g} +/- {fy_sigma_before:.6g} to "
        f"{fy_after:.6g} +/- {fy_sigma_after:.6g} {evidence.unit}. "
        f"The affected {BEAM_BENDING_METHOD} computation changed design capacity "
        f"from {a.target_mean:.6g} +/- {a.target_sigma:.6g} to "
        f"{b.target_mean:.6g} +/- {b.target_sigma:.6g} N*m, so the "
        f"{prior.verdict.value} verdict became {revised.verdict.value} against "
        f"{revised.check.factored_demand_n_m:g} N*m at "
        f"{revised.check.confidence:.1%} confidence."
    )
    return BeamDecisionChange(
        evidence,
        prior,
        revised,
        tuple(changed),
        tuple(covariance),
        reason,
    )


def beam_assessment_record(
    world: World,
    result: BeamCheckResult,
    *,
    evidence_digest: str | None = None,
    change: BeamDecisionChange | None = None,
    assessment_id: str | None = None,
) -> AssessmentRecord:
    """Create the replayable verification record for one beam decision."""
    if result.assessment.world_digest != world.digest():
        raise ValueError("beam result is not bound to the supplied world")
    details: dict[str, object] = {
        "state_identity": result.assessment.world_digest,
        "criterion": {
            "kind": "minimum-design-moment-capacity",
            "target": _var_record(result.check.capacity_var),
            "factored_demand_n_m": result.check.factored_demand_n_m,
            "confidence": result.check.confidence,
        },
        "result": {
            "target_mean_n_m": result.assessment.target_mean,
            "target_sigma_n_m": result.assessment.target_sigma,
            "p_satisfies": result.assessment.p_satisfies,
            "p_violates": result.assessment.p_violates,
            "verdict": result.verdict.value,
        },
        "computation": result.computation_details(),
        "invariant_report_digest": verification_digest(run_invariants(world)),
    }
    if change is not None:
        if change.revised != result:
            raise ValueError("change.revised must be the recorded result")
        details["decision_change"] = {
            "prior_state_identity": change.prior.assessment.world_digest,
            "prior_verdict": change.prior.verdict.value,
            "revised_verdict": change.revised.verdict.value,
            "changed_beliefs": list(change.changed_beliefs),
            "covariance_changes": list(change.covariance_changes),
            "reason": change.reason,
        }
    stable = _canonical_digest(details)[:16]
    return AssessmentRecord(
        result.assessment.world_digest,
        assessment_id or f"beam-check:{stable}",
        "beam-bending-assurance",
        result.check.label or result.check.beam.global_id,
        result.verdict.value,
        BEAM_DECISION_METHOD,
        evidence_digest,
        details,
    )
__all__ = [
    "BEAM_BENDING_METHOD",
    "BeamBendingCheck",
    "BeamBendingEvaluator",
    "BeamCheckResult",
    "BeamDecisionChange",
    "beam_assessment_record",
    "explain_beam_decision_change",
]
