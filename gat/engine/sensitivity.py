"""Sensitivity analysis: the total Jacobian as a queryable artifact.

The dependency DAG says *what* depends on what; the total Jacobian says
*how much*.  This module exposes it as an API:

* ``sensitivities_of(world, var)`` — exact partial derivatives of one
  variable with respect to every raw parameter (a row of ``J``), i.e. the
  answer to "if this architectural state changes slightly, how does this
  dependent variable change?"
* ``variance_attribution(world, var)`` — the standard linear attribution
  of ``Var(y) = J Sigma J^T`` back to the raw parameters:
  ``a_i = J_i (Sigma J^T)_i / Var(y)`` — which sums to 1 and accounts for
  correlations (a parameter's share can be negative when correlations
  cancel).
* ``sensitivity_graph(world, var)`` — a rendered dependency tree with the
  local partial attached to every edge — the numerical upgrade of the
  plain dependency graph.

Everything here is read-only over the current linearization point; nothing
mutates state.
"""

from __future__ import annotations

import numpy as np

from gat.engine.executor import World
from gat.ids import VarId


def raw_jacobian_row(world: World, var: VarId) -> np.ndarray:
    """Exact total derivative of ``var`` w.r.t. the raw vector (1 x n_raw)."""
    binding = world.binding
    if binding.is_raw(var):
        row = np.zeros(binding.n_raw, dtype=np.float64)
        row[binding.raw_index.row(var)] = 1.0
        return row
    env = world.belief.env()
    G = binding.deps.total_jacobian(binding.raw_index.vars, env)
    topo_row = {v: r for r, v in enumerate(binding.deps.derived_vars)}
    return G[topo_row[var]].copy()


def sensitivities_of(
    world: World, var: VarId, nonzero_only: bool = True
) -> tuple[tuple[VarId, float], ...]:
    """``(raw_var, d var / d raw_var)`` pairs, largest magnitude first."""
    row = raw_jacobian_row(world, var)
    pairs = [
        (world.binding.raw_index.var(i), float(p))
        for i, p in enumerate(row)
        if not nonzero_only or p != 0.0
    ]
    pairs.sort(key=lambda item: (-abs(item[1]), item[0]))
    return tuple(pairs)


def variance_attribution(world: World, var: VarId) -> tuple[tuple[VarId, float], ...]:
    """Linear attribution of ``Var(var)`` to raw parameters (sums to 1).

    ``a_i = J_i * (Sigma J^T)_i / (J Sigma J^T)`` — the covariance between
    parameter ``i``'s contribution and the total, as a fraction of total
    variance.  Correlation-aware; shares may be negative.
    """
    row = raw_jacobian_row(world, var)
    sigma_jt = world.belief.sigma @ row
    total = float(row @ sigma_jt)
    if total <= 0.0:
        return ()
    shares = [
        (world.binding.raw_index.var(i), float(row[i] * sigma_jt[i] / total))
        for i in range(len(row))
        if row[i] != 0.0
    ]
    shares.sort(key=lambda item: (-abs(item[1]), item[0]))
    return tuple(shares)


def sensitivity_graph(world: World, var: VarId, _prefix: str = "", _local: float | None = None) -> str:
    """Render the dependency tree of ``var`` with local partials on edges."""
    lines: list[str] = []
    _render(world, var, "", None, lines, seen_depth=0)
    return "\n".join(lines)


def _render(
    world: World,
    var: VarId,
    prefix: str,
    local: float | None,
    lines: list[str],
    seen_depth: int,
) -> None:
    label = str(var)
    if local is not None:
        label = f"{label}   [d(child)/d(this) = {local:+.6g}]"
    lines.append(prefix + label)
    if seen_depth > 12:  # cycles are impossible; this guards runaway depth
        return
    binding = world.binding
    if binding.is_raw(var):
        return
    slot = world.module.slot(var)
    assert slot.expr is not None
    env = world.full.env()
    partials = slot.expr.grad(env)
    parents = sorted(partials)
    for k, parent in enumerate(parents):
        connector = "└── " if k == len(parents) - 1 else "├── "
        child_prefix = prefix + ("    " if k == len(parents) - 1 else "│   ")
        lines.append(prefix + connector.rstrip() + " ")
        lines.pop()
        _render(
            world,
            parent,
            prefix + connector if not prefix else child_prefix[: len(prefix)] + connector,
            partials[parent],
            lines,
            seen_depth + 1,
        )
