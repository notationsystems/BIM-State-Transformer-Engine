"""Pushforward of the raw belief through the dependency DAG.

The full continuous state is always the first-order pushforward of the raw
belief:

    mu_full    = [mu_r ; F(mu_r)]           (means are EXACT re-evaluations)
    Sigma_full = J Sigma_r J^T,  J = [I; G] (covariance is linearized at mu_r)

Compilation computes the entire derived layer. Later commits can reuse the
preceding verified values/Jacobian and re-evaluate only the dependency-closed
affected rows. The dense full covariance is still materialized; only rows and
columns whose raw covariance or Jacobian support changed are recomputed.
Verification remains mandatory, so incremental reuse never becomes an
unverified mutation path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gat.engine.binding import GaussianBinding
from gat.errors import BindingError
from gat.gaussian.linalg import symmetrize
from gat.gaussian.state import GaussianState


@dataclass(frozen=True)
class PropagationStats:
    """Auditable work performed by one pushforward."""

    mode: str
    raw_mean_rows_changed: int
    raw_covariance_rows_changed: int
    derived_value_rows_recomputed: int
    derived_jacobian_rows_recomputed: int
    covariance_left_rows_recomputed: int
    full_covariance_rows_recomputed: int
    full_variable_count: int


def _full_jacobian(binding: GaussianBinding, derived_jacobian: np.ndarray) -> np.ndarray:
    result = np.zeros((binding.n_full, binding.n_raw), dtype=np.float64)
    result[: binding.n_raw, :] = np.eye(binding.n_raw)
    for source_row, var in enumerate(binding.deps.derived_vars):
        result[binding.full_index.row(var), :] = derived_jacobian[source_row]
    return result


def push_forward_with_jacobian(
    binding: GaussianBinding,
    belief: GaussianState,
) -> tuple[GaussianState, np.ndarray, np.ndarray, PropagationStats]:
    """Compute the complete full view and cacheable total Jacobian."""
    deps = binding.deps
    raw_env = belief.env()
    derived_values = deps.evaluate(raw_env)
    derived_jacobian = deps.total_jacobian(binding.raw_index.vars, raw_env)
    jacobian = _full_jacobian(binding, derived_jacobian)

    mu_full = np.zeros(binding.n_full, dtype=np.float64)
    mu_full[: binding.n_raw] = belief.mu
    for var, value in derived_values.items():
        mu_full[binding.full_index.row(var)] = value
    covariance_left = jacobian @ belief.sigma
    sigma_full = symmetrize(covariance_left @ jacobian.T)
    stats = PropagationStats(
        "full",
        binding.n_raw,
        binding.n_raw,
        len(deps.derived_vars),
        len(deps.derived_vars),
        binding.n_full,
        binding.n_full,
        binding.n_full,
    )
    return (
        GaussianState(binding.full_index, mu_full, sigma_full),
        jacobian,
        covariance_left,
        stats,
    )


def push_forward(binding: GaussianBinding, belief: GaussianState) -> GaussianState:
    """Compute the full (raw + derived) view from the raw belief."""
    return push_forward_with_jacobian(binding, belief)[0]


def push_forward_incremental(
    binding: GaussianBinding,
    previous_belief: GaussianState,
    belief: GaussianState,
    previous_full: GaussianState,
    previous_jacobian: np.ndarray | None,
    previous_covariance_left: np.ndarray | None,
) -> tuple[GaussianState, np.ndarray, np.ndarray, PropagationStats]:
    """Push forward by recomputing only rows invalidated by the raw update."""
    expected_cache_shape = (binding.n_full, binding.n_raw)
    if (
        previous_jacobian is None
        or previous_jacobian.shape != expected_cache_shape
        or previous_covariance_left is None
        or previous_covariance_left.shape != expected_cache_shape
    ):
        full, jacobian, covariance_left, stats = push_forward_with_jacobian(
            binding,
            belief,
        )
        return full, jacobian, covariance_left, PropagationStats(
            "full-cache-miss",
            stats.raw_mean_rows_changed,
            stats.raw_covariance_rows_changed,
            stats.derived_value_rows_recomputed,
            stats.derived_jacobian_rows_recomputed,
            stats.covariance_left_rows_recomputed,
            stats.full_covariance_rows_recomputed,
            stats.full_variable_count,
        )

    mean_rows = np.flatnonzero(belief.mu != previous_belief.mu)
    covariance_rows = np.flatnonzero(
        np.any(belief.sigma != previous_belief.sigma, axis=1)
    )
    mean_vars = tuple(binding.raw_index.var(int(row)) for row in mean_rows)
    covariance_vars = tuple(
        binding.raw_index.var(int(row)) for row in covariance_rows
    )
    affected_mean = binding.deps.affected_set(mean_vars)
    affected_covariance = binding.deps.affected_set(covariance_vars)

    previous_values = {
        var: previous_full.mean(var) for var in binding.deps.derived_vars
    }
    previous_derived_jacobian = np.vstack(
        [
            previous_jacobian[binding.full_index.row(var), :]
            for var in binding.deps.derived_vars
        ]
    ) if binding.deps.derived_vars else np.zeros((0, binding.n_raw))
    derived_values, derived_jacobian = (
        binding.deps.incremental_values_and_jacobian(
            binding.raw_index.vars,
            belief.env(),
            previous_values,
            previous_derived_jacobian,
            affected_mean,
        )
    )
    jacobian = _full_jacobian(binding, derived_jacobian)

    mu_full = previous_full.mu.copy()
    mu_full[: binding.n_raw] = belief.mu
    for var in affected_mean:
        mu_full[binding.full_index.row(var)] = derived_values[var]

    impacted_vars = set(covariance_vars)
    impacted_vars.update(affected_covariance)
    impacted_vars.update(affected_mean)
    impacted_rows = np.array(
        sorted(binding.full_index.row(var) for var in impacted_vars),
        dtype=np.intp,
    )
    if impacted_rows.size == binding.n_full:
        full, full_jacobian, covariance_left, stats = push_forward_with_jacobian(
            binding,
            belief,
        )
        return full, full_jacobian, covariance_left, PropagationStats(
            "full-invalidated",
            len(mean_rows),
            len(covariance_rows),
            stats.derived_value_rows_recomputed,
            stats.derived_jacobian_rows_recomputed,
            stats.covariance_left_rows_recomputed,
            stats.full_covariance_rows_recomputed,
            stats.full_variable_count,
        )

    sigma_full = previous_full.sigma.copy()
    if covariance_rows.size:
        covariance_left = jacobian @ belief.sigma
        covariance_left_rows = binding.n_full
    else:
        covariance_left = np.asarray(previous_covariance_left).copy()
        left_rows = np.array(
            sorted(binding.full_index.row(var) for var in affected_mean),
            dtype=np.intp,
        )
        if left_rows.size:
            covariance_left[left_rows, :] = (
                jacobian[left_rows, :] @ belief.sigma
            )
        covariance_left_rows = len(left_rows)
    if impacted_rows.size:
        forward = covariance_left[impacted_rows, :] @ jacobian.T
        reverse = (
            covariance_left @ jacobian[impacted_rows, :].T
        ).T
        block = 0.5 * (forward + reverse)
        sigma_full[impacted_rows, :] = block
        sigma_full[:, impacted_rows] = block.T
        sigma_full[np.ix_(impacted_rows, impacted_rows)] = block[:, impacted_rows]
    stats = PropagationStats(
        "incremental",
        len(mean_rows),
        len(covariance_rows),
        len(affected_mean),
        len(affected_mean),
        covariance_left_rows,
        len(impacted_rows),
        binding.n_full,
    )
    return (
        GaussianState(binding.full_index, mu_full, sigma_full),
        jacobian,
        covariance_left,
        stats,
    )


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
        elif var in topo_row:
            H[k, :] = G[topo_row[var]]
            predicted[k] = derived_values[var]
        else:
            raise BindingError(f"variable {var} is neither raw nor derived")
    return H, predicted
