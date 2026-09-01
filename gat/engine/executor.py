"""The execution core: apply → propagate → verify → commit-or-rollback.

``World`` is the complete bound state — module, relationship graph,
binding, raw belief, and the pushforward full view.  ``execute`` applies a
transformation under the mandatory pipeline contract:

1. the operator produces a new raw belief (pure function of the old one),
2. the derived layer is rebuilt by pushforward (never mutated in place),
3. the full invariant registry runs on the candidate world,
4. strict mode: a FAIL raises :class:`~gat.errors.VerificationError` and
   the returned world is the untouched original (rollback is free because
   states are immutable values).

For non-observation operators the executor also machine-checks the
selectivity claim of README §8: variables outside the declared targets and
their DAG descendants must change by *exactly* zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

import numpy as np

from gat.engine.binding import GaussianBinding, bind
from gat.engine.propagate import (
    PropagationStats,
    push_forward_incremental,
    push_forward_with_jacobian,
)
from gat.engine.transform import (
    CompositeTransformation,
    ObservationTransformation,
    ObserveLinearized,
    Transformation,
)
from gat.engine.verify import VerificationReport, run_invariants
from gat.errors import BindingError, NumericalError, VerificationError
from gat.gaussian.state import GaussianState
from gat.ids import VarId
from gat.ir.core import Module
from gat.ir.graph import RelationshipGraph


@dataclass(frozen=True)
class World:
    """The complete architectural world state."""

    module: Module
    graph: RelationshipGraph
    binding: GaussianBinding
    belief: GaussianState
    full: GaussianState
    jacobian: np.ndarray | None = field(default=None, repr=False, compare=False)
    covariance_left: np.ndarray | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        expected = (self.binding.n_full, self.binding.n_raw)
        for field_name, source in (
            ("jacobian", self.jacobian),
            ("covariance_left", self.covariance_left),
        ):
            if source is None:
                continue
            value = np.asarray(source, dtype=np.float64)
            if value.shape != expected or not np.isfinite(value).all():
                raise ValueError(
                    f"world {field_name} must be finite with shape {expected}"
                )
            value = value.copy()
            value.setflags(write=False)
            object.__setattr__(self, field_name, value)

    @classmethod
    def compile(cls, module: Module) -> "World":
        binding, belief = bind(module)
        full, jacobian, covariance_left, _ = push_forward_with_jacobian(
            binding,
            belief,
        )
        return cls(
            module,
            RelationshipGraph.of(module),
            binding,
            belief,
            full,
            jacobian,
            covariance_left,
        )

    def with_belief(self, belief: GaussianState) -> "World":
        full, jacobian, covariance_left, _ = push_forward_with_jacobian(
            self.binding,
            belief,
        )
        return World(
            self.module,
            self.graph,
            self.binding,
            belief,
            full,
            jacobian,
            covariance_left,
        )

    def with_belief_incremental(
        self,
        belief: GaussianState,
    ) -> tuple["World", PropagationStats]:
        full, jacobian, covariance_left, stats = push_forward_incremental(
            self.binding,
            self.belief,
            belief,
            self.full,
            self.jacobian,
            self.covariance_left,
        )
        return (
            World(
                self.module,
                self.graph,
                self.binding,
                belief,
                full,
                jacobian,
                covariance_left,
            ),
            stats,
        )

    def digest(self) -> str:
        h = hashlib.sha256()
        h.update(self.module.digest().encode())
        h.update(self.full.mu.tobytes())
        h.update(self.full.sigma.tobytes())
        return h.hexdigest()


def _contains_observation(t: Transformation) -> bool:
    if isinstance(t, ObservationTransformation):
        return True
    if isinstance(t, CompositeTransformation):
        return any(_contains_observation(s) for s in t.steps)
    return False


@dataclass(frozen=True)
class ExecutionResult:
    """Outcome of one transformation execution."""

    world: World
    report: VerificationReport
    committed: bool
    transformation: Transformation
    targets: tuple[VarId, ...]
    affected: tuple[VarId, ...] = field(default=())
    deltas: tuple[tuple[VarId, float], ...] = field(default=())
    propagation: PropagationStats | None = None

    def describe(self) -> str:
        status = "committed" if self.committed else "REJECTED"
        return (
            f"{self.transformation.describe()} -> {status}; "
            f"{len(self.affected)} derived affected"
        )


@dataclass(frozen=True)
class ExecutionPreview:
    """Non-mutating candidate state produced by the execution pipeline.

    A preview runs the same apply, propagation, selectivity, and invariant
    checks as :func:`execute`, but exposes the candidate even when it fails
    verification.  It never authorizes or commits the candidate.  This is the
    evidence boundary used by design-change and RFI impact workflows.
    """

    prior_world_digest: str
    candidate: World
    report: VerificationReport
    transformation: Transformation
    targets: tuple[VarId, ...]
    affected: tuple[VarId, ...] = field(default=())
    deltas: tuple[tuple[VarId, float], ...] = field(default=())
    propagation: PropagationStats | None = None

    @property
    def admissible(self) -> bool:
        return self.report.passed


def preview(world: World, t: Transformation) -> ExecutionPreview:
    """Evaluate a transformation without committing or hiding a failed state."""
    _assert_observation_provenance(world, t)
    new_belief = t.apply(world.binding, world.belief)
    candidate, propagation = world.with_belief_incremental(new_belief)

    targets = t.target_vars()
    raw_targets = tuple(v for v in targets if world.binding.is_raw(v))
    affected = world.binding.deps.affected_set(targets)

    is_observation = _contains_observation(t)
    if not is_observation:
        _assert_selectivity(world, candidate, raw_targets, affected)

    deltas = _mean_deltas(world, candidate)
    report = run_invariants(candidate)
    return ExecutionPreview(
        prior_world_digest=world.digest(),
        candidate=candidate,
        report=report,
        transformation=t,
        targets=targets,
        affected=affected,
        deltas=deltas,
        propagation=propagation,
    )


def execute(world: World, t: Transformation, strict: bool = True) -> ExecutionResult:
    """Run one transformation under the mandatory propagate+verify contract."""
    candidate = preview(world, t)

    if not candidate.report.passed:
        if strict:
            raise VerificationError(candidate.report)
        return ExecutionResult(
            world,
            candidate.report,
            False,
            t,
            candidate.targets,
            candidate.affected,
            candidate.deltas,
            candidate.propagation,
        )

    return ExecutionResult(
        candidate.candidate,
        candidate.report,
        True,
        t,
        candidate.targets,
        candidate.affected,
        candidate.deltas,
        candidate.propagation,
    )


def _assert_observation_provenance(world: World, t: Transformation) -> None:
    """Refuse adapter likelihoods calibrated against another exact world."""
    if isinstance(t, ObserveLinearized):
        if t.expected_world_digest != world.digest():
            raise BindingError(
                "linearized observation is stale: the architectural world "
                "changed after calibration"
            )
    elif isinstance(t, CompositeTransformation):
        for step in t.steps:
            _assert_observation_provenance(world, step)


def _assert_selectivity(
    before: World,
    after: World,
    raw_targets: tuple[VarId, ...],
    affected: tuple[VarId, ...],
) -> None:
    """Machine-check README §8: nothing outside targets ∪ descendants moves."""
    allowed = set(raw_targets) | set(affected)
    moved = np.flatnonzero(after.full.mu != before.full.mu)
    for row in moved:
        var = after.full.index.var(int(row))
        if var not in allowed:
            raise NumericalError(
                f"selectivity violation: {var} changed but is outside the "
                f"affected set of the transformation"
            )


def _mean_deltas(before: World, after: World) -> tuple[tuple[VarId, float], ...]:
    diff = after.full.mu - before.full.mu
    rows = np.flatnonzero(diff)
    return tuple(
        (after.full.index.var(int(r)), float(diff[r])) for r in rows
    )
