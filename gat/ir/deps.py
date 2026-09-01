"""Dependency DAG between raw parameters and derived quantities.

Nodes are :class:`~gat.ids.VarId`; there is an edge ``u -> v`` iff ``v`` is
a DERIVED slot and ``u`` appears free in ``v``'s defining expression.  The
graph provides:

* a deterministic topological order (Kahn's algorithm, ready set kept
  sorted by ``VarId``),
* ``affected_set`` — forward reachability from a set of changed variables,
  in topological order (README §8: identify affected downstream state),
* ``evaluate`` — exact mean evaluation of every derived variable,
* ``total_jacobian`` — the exact total derivative of every variable with
  respect to the raw vector, assembled by chain-rule accumulation along the
  topological order.  Rows for raw variables are unit vectors, so the full
  Jacobian ``J = [I; G]`` has an identity leading block by construction.
"""

from __future__ import annotations

import heapq

import numpy as np

from gat.errors import LoweringError
from gat.ids import VarId
from gat.ir.core import Module, QtySlot, Role


class DependencyGraph:
    """Immutable dependency structure compiled from a module's derived slots."""

    def __init__(self, module: Module):
        self._raw: tuple[VarId, ...] = module.raw_vars()
        self._derived_slots: dict[VarId, QtySlot] = {
            s.var: s for s in module.all_slots() if s.role is Role.DERIVED
        }
        known = set(self._raw) | set(self._derived_slots)

        # parents[v] = variables directly referenced by v's expression
        self._parents: dict[VarId, tuple[VarId, ...]] = {}
        self._children: dict[VarId, list[VarId]] = {v: [] for v in known}
        for var, slot in self._derived_slots.items():
            assert slot.expr is not None
            parents = slot.expr.free_vars()
            for p in parents:
                if p not in known:
                    raise LoweringError(
                        f"derived quantity {var} references unknown variable {p}"
                    )
            self._parents[var] = parents
            for p in parents:
                self._children[p].append(var)
        for v in self._children:
            self._children[v] = sorted(self._children[v])

        self._topo: tuple[VarId, ...] = self._topo_sort()

    # -- structure ---------------------------------------------------------

    @property
    def raw_vars(self) -> tuple[VarId, ...]:
        return self._raw

    @property
    def derived_vars(self) -> tuple[VarId, ...]:
        """Derived variables in topological evaluation order."""
        return self._topo

    def parents(self, var: VarId) -> tuple[VarId, ...]:
        return self._parents.get(var, ())

    def children(self, var: VarId) -> tuple[VarId, ...]:
        return tuple(self._children.get(var, ()))

    def _topo_sort(self) -> tuple[VarId, ...]:
        # Raw variables are sources; in-degree counts derived parents only.
        indeg = {
            v: sum(1 for p in self._parents[v] if p in self._derived_slots)
            for v in self._derived_slots
        }
        ready = sorted(v for v, d in indeg.items() if d == 0)
        heapq.heapify(ready)
        order: list[VarId] = []
        while ready:
            v = heapq.heappop(ready)
            order.append(v)
            for child in self._children.get(v, ()):
                if child in indeg:
                    indeg[child] -= 1
                    if indeg[child] == 0:
                        heapq.heappush(ready, child)
        if len(order) != len(self._derived_slots):
            unresolved = sorted(set(self._derived_slots) - set(order))
            raise LoweringError(
                f"cyclic dependency among derived quantities: {[str(v) for v in unresolved]}"
            )
        return tuple(order)

    def affected_set(self, changed: tuple[VarId, ...]) -> tuple[VarId, ...]:
        """Derived variables reachable from ``changed``, in topological order."""
        reached: set[VarId] = set()
        frontier = list(changed)
        while frontier:
            v = frontier.pop()
            for child in self._children.get(v, ()):
                if child not in reached:
                    reached.add(child)
                    frontier.append(child)
        return tuple(v for v in self._topo if v in reached)

    # -- evaluation --------------------------------------------------------

    def evaluate(self, raw_env: dict[VarId, float]) -> dict[VarId, float]:
        """Exact values of every derived variable given raw values.

        Returns a mapping containing *only* the derived variables, computed
        in topological order.  Means are exact nonlinear re-evaluations,
        never linearizations.
        """
        env: dict[VarId, float] = dict(raw_env)
        out: dict[VarId, float] = {}
        for var in self._topo:
            slot = self._derived_slots[var]
            assert slot.expr is not None
            value = slot.expr.eval(env)
            env[var] = value
            out[var] = value
        return out

    def total_jacobian(
        self, raw_order: tuple[VarId, ...], raw_env: dict[VarId, float]
    ) -> np.ndarray:
        """Total derivative ``G`` of derived variables w.r.t. the raw vector.

        ``G`` has one row per derived variable (topological order) and one
        column per raw variable (order given by ``raw_order``).  Assembled
        by chain rule: ``row(v) = sum_u (df_v/du) * row(u)`` with unit rows
        for raw ``u``.  All partials are analytic and evaluated at the
        supplied environment (the current mean).
        """
        n_raw = len(raw_order)
        col = {v: i for i, v in enumerate(raw_order)}
        rows: dict[VarId, np.ndarray] = {}
        env: dict[VarId, float] = dict(raw_env)

        G = np.zeros((len(self._topo), n_raw), dtype=np.float64)
        for r, var in enumerate(self._topo):
            slot = self._derived_slots[var]
            assert slot.expr is not None
            partials = slot.expr.grad(env)
            row = np.zeros(n_raw, dtype=np.float64)
            for parent, p in partials.items():
                if parent in rows:
                    row += p * rows[parent]
                else:
                    row[col[parent]] += p
            rows[var] = row
            G[r] = row
            env[var] = slot.expr.eval(env)
        return G

    def incremental_values_and_jacobian(
        self,
        raw_order: tuple[VarId, ...],
        raw_env: dict[VarId, float],
        previous_values: dict[VarId, float],
        previous_jacobian: np.ndarray,
        affected: tuple[VarId, ...],
    ) -> tuple[dict[VarId, float], np.ndarray]:
        """Re-evaluate only affected derived nodes and their Jacobian rows.

        Unaffected values and rows are copied from the preceding verified
        world. ``affected`` must be forward-closed under this dependency DAG;
        callers obtain that property from :meth:`affected_set`.
        """
        expected_shape = (len(self._topo), len(raw_order))
        if previous_jacobian.shape != expected_shape:
            raise ValueError(
                "previous derived Jacobian shape differs from the dependency graph"
            )
        if set(previous_values) != set(self._topo):
            raise ValueError("previous derived values differ from the dependency graph")
        affected_set = set(affected)
        unknown = affected_set - set(self._topo)
        if unknown:
            raise ValueError(f"affected set contains non-derived variables: {unknown}")

        col = {var: index for index, var in enumerate(raw_order)}
        topo_row = {var: index for index, var in enumerate(self._topo)}
        values = dict(previous_values)
        jacobian = np.asarray(previous_jacobian, dtype=np.float64).copy()
        env: dict[VarId, float] = dict(raw_env)
        env.update(values)

        for row_index, var in enumerate(self._topo):
            if var not in affected_set:
                continue
            slot = self._derived_slots[var]
            assert slot.expr is not None
            partials = slot.expr.grad(env)
            row = np.zeros(len(raw_order), dtype=np.float64)
            for parent, partial in partials.items():
                if parent in topo_row:
                    row += partial * jacobian[topo_row[parent]]
                else:
                    row[col[parent]] += partial
            value = slot.expr.eval(env)
            jacobian[row_index] = row
            values[var] = value
            env[var] = value
        return values, jacobian
