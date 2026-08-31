"""Pushforward of the raw belief through the dependency DAG.

The full continuous state is always the first-order pushforward of the raw
belief:

    mu_full    = [mu_r ; F(mu_r)]           (means are EXACT re-evaluations)
    Sigma_full = J Sigma_r J^T,  J = [I; G] (covariance is linearized at mu_r)

This module recomputes the entire derived layer on every commit — no
incremental mutation of derived entries ever occurs, so raw/derived
inconsistency is impossible by construction and numeric drift cannot
accumulate.  The affected subset of a change is still reported precisely
via :meth:`~gat.ir.deps.DependencyGraph.affected_set`.
"""

from __future__ import annotations

import numpy as np

from gat.engine.binding import GaussianBinding
from gat.gaussian.linalg import symmetrize
from gat.gaussian.state import GaussianState


def push_forward(binding: GaussianBinding, belief: GaussianState) -> GaussianState:
    """Compute the full (raw + derived) view from the raw belief."""
    deps = binding.deps
    raw_order = binding.raw_index.vars
    raw_env = belief.env()

    derived_values = deps.evaluate(raw_env)
    G = deps.total_jacobian(raw_order, raw_env)

    n_raw = binding.n_raw
    n_full = binding.n_full
    J = np.zeros((n_full, n_raw), dtype=np.float64)
    J[:n_raw, :] = np.eye(n_raw)

    mu_full = np.zeros(n_full, dtype=np.float64)
    mu_full[:n_raw] = belief.mu

    # G rows are in topological order; scatter them into full-index rows.
    topo = deps.derived_vars
    for r, var in enumerate(topo):
        row = binding.full_index.row(var)
        J[row, :] = G[r]
        mu_full[row] = derived_values[var]

    sigma_full = symmetrize(J @ belief.sigma @ J.T)
    return GaussianState(binding.full_index, mu_full, sigma_full)


def jacobian_rows(
    binding: GaussianBinding, belief: GaussianState, vars: tuple
) -> tuple[np.ndarray, np.ndarray]:
    """Observation rows ``H`` and exact predictions ``h(mu)`` for variables.

    For raw variables the row is a unit vector and the prediction the raw
    mean; for derived variables the row is the total-Jacobian row at the
    current mean and the prediction the exact nonlinear evaluation.
    """
    deps = binding.deps
    raw_order = binding.raw_index.vars
    raw_env = belief.env()
    derived_values = deps.evaluate(raw_env)
    G = deps.total_jacobian(raw_order, raw_env)
    topo_row = {v: r for r, v in enumerate(deps.derived_vars)}

    n_raw = binding.n_raw
    H = np.zeros((len(vars), n_raw), dtype=np.float64)
    predicted = np.zeros(len(vars), dtype=np.float64)
    for k, var in enumerate(vars):
        if binding.is_raw(var):
            H[k, binding.raw_index.row(var)] = 1.0
            predicted[k] = belief.mean(var)
        else:
            H[k, :] = G[topo_row[var]]
            predicted[k] = derived_values[var]
    return H, predicted
