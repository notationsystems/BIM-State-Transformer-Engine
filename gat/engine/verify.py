"""Verification: invariants checked as part of every execution.

Verification is not an afterthought — the executor runs the full invariant
registry after every transformation, and in strict mode a FAIL rejects the
transformation and rolls the world back (README §14 principle 6).

Structural invariants guard the symbolic layer; Gaussian invariants guard
the numerical layer; constraint invariants are compiled from the module's
typed constraint objects at check time, so *lowering* decides what must
hold for a given model and the verifier stays generic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from gat.errors import NumericalError
from gat.gaussian.linalg import chol_psd, max_asymmetry
from gat.ids import EntityId
from gat.ir.core import ExprEquals, LessEqual, NonNegative, RelKind, Role

if TYPE_CHECKING:  # pragma: no cover
    from gat.engine.executor import World


class Status(Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class InvariantResult:
    invariant_id: str
    status: Status
    subject: str
    residual: float
    detail: str


@dataclass(frozen=True)
class VerificationReport:
    results: tuple[InvariantResult, ...]

    @property
    def passed(self) -> bool:
        return all(r.status is not Status.FAIL for r in self.results)

    @property
    def failures(self) -> tuple[InvariantResult, ...]:
        return tuple(r for r in self.results if r.status is Status.FAIL)

    @property
    def warnings(self) -> tuple[InvariantResult, ...]:
        return tuple(r for r in self.results if r.status is Status.WARN)

    def counts(self) -> tuple[int, int, int]:
        p = sum(1 for r in self.results if r.status is Status.PASS)
        w = len(self.warnings)
        f = len(self.failures)
        return p, w, f

    def render(self) -> str:
        p, w, f = self.counts()
        lines = [f"verification: {p} pass, {w} warn, {f} fail"]
        for r in self.results:
            if r.status is Status.PASS:
                continue
            lines.append(
                f"  {r.status.value} {r.invariant_id} [{r.subject}] "
                f"residual={r.residual:+.6g}  {r.detail}"
            )
        return "\n".join(lines)


class Invariant(ABC):
    id: str = "invariant"
    description: str = ""

    @abstractmethod
    def check(self, world: "World") -> list[InvariantResult]: ...

    def _pass(self, subject: str, detail: str = "") -> InvariantResult:
        return InvariantResult(self.id, Status.PASS, subject, 0.0, detail)

    def _warn(self, subject: str, residual: float, detail: str) -> InvariantResult:
        return InvariantResult(self.id, Status.WARN, subject, residual, detail)

    def _fail(self, subject: str, residual: float, detail: str) -> InvariantResult:
        return InvariantResult(self.id, Status.FAIL, subject, residual, detail)


class RelEndpointsExist(Invariant):
    id = "STRUCT-01"
    description = "every relationship endpoint resolves to an entity"

    def check(self, world: "World") -> list[InvariantResult]:
        out = []
        for rel in world.module.rels:
            for end in (rel.source, rel.target):
                if end not in world.module.entities:
                    out.append(self._fail(str(end), 0.0, f"dangling endpoint of {rel.kind.value}"))
        if not out:
            out.append(self._pass("module", f"{len(world.module.rels)} edges"))
        return out


class SpatialAcyclicity(Invariant):
    id = "STRUCT-02"
    description = "AGGREGATES + CONTAINS edges form a DAG"

    def check(self, world: "World") -> list[InvariantResult]:
        edges = world.graph.spatial_edges()
        children: dict[EntityId, list[EntityId]] = {}
        indeg: dict[EntityId, int] = {}
        nodes: set[EntityId] = set()
        for rel in edges:
            children.setdefault(rel.source, []).append(rel.target)
            indeg[rel.target] = indeg.get(rel.target, 0) + 1
            nodes.add(rel.source)
            nodes.add(rel.target)
        ready = sorted(n for n in nodes if indeg.get(n, 0) == 0)
        seen = 0
        while ready:
            node = ready.pop()
            seen += 1
            for child in children.get(node, ()):
                indeg[child] -= 1
                if indeg[child] == 0:
                    ready.append(child)
        if seen != len(nodes):
            return [self._fail("spatial-tree", float(len(nodes) - seen), "cycle in spatial decomposition")]
        return [self._pass("spatial-tree", f"{len(nodes)} nodes")]


class OpeningWellFormed(Invariant):
    id = "STRUCT-03"
    description = "openings void exactly one wall; fillers fill exactly one opening"

    def check(self, world: "World") -> list[InvariantResult]:
        out = []
        for eid in world.module.entities:
            if eid.ifc_class == "IfcOpeningElement":
                hosts = world.graph.wall_of_opening(eid)
                if len(hosts) != 1:
                    out.append(self._fail(str(eid), float(len(hosts)), "opening must void exactly one wall"))
            if eid.ifc_class == "IfcDoor":
                fills = world.graph.opening_of_filler(eid)
                if len(fills) != 1:
                    out.append(self._fail(str(eid), float(len(fills)), "door must fill exactly one opening"))
        if not out:
            out.append(self._pass("openings"))
        return out


class SigmaSymmetric(Invariant):
    id = "GAUSS-01"
    description = "belief and full covariances are symmetric"

    TOL = 1e-9

    def check(self, world: "World") -> list[InvariantResult]:
        out = []
        for label, state in (("belief", world.belief), ("full", world.full)):
            asym = max_asymmetry(state.sigma)
            if asym > self.TOL:
                out.append(self._fail(label, asym, f"max|Sigma-Sigma^T| = {asym:.3e}"))
            else:
                out.append(self._pass(label))
        return out


class BeliefPSD(Invariant):
    id = "GAUSS-02"
    description = "raw belief covariance is PSD (Cholesky within jitter ladder)"

    DIAG_TOL = -1e-12

    def check(self, world: "World") -> list[InvariantResult]:
        out = []
        try:
            _, jitter = chol_psd(world.belief.sigma)
            detail = f"jitter={jitter:.1e}" if jitter else "jitter=0"
            out.append(self._pass("belief", detail))
        except NumericalError as exc:
            out.append(self._fail("belief", 0.0, str(exc)))
        # The full view is rank-deficient by construction; certify only that
        # no marginal variance is meaningfully negative.
        min_diag = float(np.diag(world.full.sigma).min())
        if min_diag < self.DIAG_TOL:
            out.append(self._fail("full", min_diag, f"negative marginal variance {min_diag:.3e}"))
        else:
            out.append(self._pass("full", "diag >= 0"))
        return out


class DerivedConsistent(Invariant):
    id = "GAUSS-03"
    description = "derived means equal exact re-evaluation of their expressions"

    def check(self, world: "World") -> list[InvariantResult]:
        env = world.full.env()
        worst = 0.0
        worst_var = None
        for var in world.binding.deps.derived_vars:
            slot = world.module.slot(var)
            assert slot.expr is not None
            expected = slot.expr.eval(env)
            actual = env[var]
            tol = 1e-9 * max(1.0, abs(expected))
            resid = abs(actual - expected)
            if resid > tol and resid > worst:
                worst = resid
                worst_var = var
        if worst_var is not None:
            return [self._fail(str(worst_var), worst, "derived mean drifted from definition")]
        return [self._pass("derived", f"{len(world.binding.deps.derived_vars)} vars")]


class NonNegativeQuantities(Invariant):
    id = "CONS-01"
    description = "NonNegative constraints: mean >= 0; warn when 2-sigma straddles zero"

    def check(self, world: "World") -> list[InvariantResult]:
        out = []
        any_bad = False
        for c in world.module.constraints:
            if not isinstance(c, NonNegative):
                continue
            mean = world.full.mean(c.var)
            std = world.full.std(c.var)
            if mean < -c.tol:
                out.append(self._fail(str(c.var), mean, f"mean {mean:.6f} < 0"))
                any_bad = True
            elif mean - 2.0 * std < 0.0:
                out.append(
                    self._warn(str(c.var), mean - 2.0 * std, f"2-sigma interval straddles zero (mu={mean:.6f}, sigma={std:.6f})")
                )
                any_bad = True
        if not any_bad:
            out.append(self._pass("nonneg"))
        return out


class BoundsRespected(Invariant):
    id = "CONS-02"
    description = "LessEqual constraints hold at the mean; warn on 2-sigma risk"

    def check(self, world: "World") -> list[InvariantResult]:
        out = []
        any_bad = False
        for c in world.module.constraints:
            if not isinstance(c, LessEqual):
                continue
            diff = world.full.mean(c.lhs) - world.full.mean(c.rhs)
            var_diff = (
                world.full.var_of(c.lhs)
                + world.full.var_of(c.rhs)
                - 2.0 * world.full.cov(c.lhs, c.rhs)
            )
            std_diff = float(np.sqrt(max(var_diff, 0.0)))
            subject = f"{c.lhs} <= {c.rhs}"
            if diff > c.tol:
                out.append(self._fail(subject, diff, f"violated by {diff:.6f} at the mean"))
                any_bad = True
            elif diff + 2.0 * std_diff > 0.0:
                out.append(
                    self._warn(subject, diff + 2.0 * std_diff, f"violation within 2 sigma (margin {-diff:.6f}, sigma {std_diff:.6f})")
                )
                any_bad = True
        if not any_bad:
            out.append(self._pass("bounds"))
        return out


class DefinitionsRestated(Invariant):
    id = "CONS-03"
    description = "ExprEquals constraints: independent restatements hold at the mean"

    def check(self, world: "World") -> list[InvariantResult]:
        out = []
        env = world.full.env()
        any_bad = False
        for c in world.module.constraints:
            if not isinstance(c, ExprEquals):
                continue
            expected = c.expr.eval(env)
            actual = env[c.var]
            resid = abs(actual - expected)
            if resid > c.tol * max(1.0, abs(expected)):
                out.append(self._fail(str(c.var), resid, "restated definition disagrees"))
                any_bad = True
        if not any_bad:
            out.append(self._pass("restatements"))
        return out


#: Rollup quantities cross-validated through the relationship graph rather
#: than the dependency DAG — a disagreement means lowering built the DAG
#: and the graph inconsistently.
ROLLUPS: dict[str, tuple[RelKind, str, str]] = {
    "TotalWallNetVolume": (RelKind.CONTAINS, "IfcWall", "NetVolume"),
    "TotalWallCost": (RelKind.CONTAINS, "IfcWall", "Cost"),
    "TotalFloorArea": (RelKind.AGGREGATES, "IfcSpace", "FloorArea"),
}


class AggregationConsistent(Invariant):
    id = "QTY-01"
    description = "storey rollups re-summed via the relationship graph match the DAG"

    TOL = 1e-9

    def check(self, world: "World") -> list[InvariantResult]:
        out = []
        any_bad = False
        for eid in world.module.entities:
            entity = world.module.entities[eid]
            for qname, slot in sorted(entity.slots.items()):
                if slot.role is not Role.DERIVED or qname not in ROLLUPS:
                    continue
                kind, member_class, member_qty = ROLLUPS[qname]
                members = [
                    m
                    for m in world.graph.targets(kind, eid)
                    if m.ifc_class == member_class
                    and member_qty in world.module.entities[m].slots
                ]
                total = sum(
                    world.full.mean(world.module.entities[m].var(member_qty))
                    for m in members
                )
                actual = world.full.mean(slot.var)
                resid = abs(actual - total)
                if resid > self.TOL * max(1.0, abs(total)):
                    out.append(
                        self._fail(str(slot.var), resid, f"graph re-sum {total!r} != DAG value {actual!r}")
                    )
                    any_bad = True
        if not any_bad:
            out.append(self._pass("rollups"))
        return out


#: The fixed invariant registry, in report order.  Extensions append.
ALL_INVARIANTS: tuple[Invariant, ...] = (
    RelEndpointsExist(),
    SpatialAcyclicity(),
    OpeningWellFormed(),
    SigmaSymmetric(),
    BeliefPSD(),
    DerivedConsistent(),
    NonNegativeQuantities(),
    BoundsRespected(),
    DefinitionsRestated(),
    AggregationConsistent(),
)


def run_invariants(world: "World") -> VerificationReport:
    results: list[InvariantResult] = []
    for inv in ALL_INVARIANTS:
        results.extend(inv.check(world))
    return VerificationReport(tuple(results))
