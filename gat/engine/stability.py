"""Lyapunov-style stability analysis of transformation sequences.

The Jacobian answers *how a perturbation propagates locally*; this module
answers *what happens to perturbations over repeated transformations*.
Every shipped operator has an exact raw-space perturbation map:

    ShiftParameter      J = I                      (marginally stable)
    ScaleParameter      J = I with J_kk = factor
    SetParameter        J = I with row k zeroed    (the do-intervention
                        forgets the perturbation in the overridden channel)
    ObserveQuantity     J = I - K H                (a contraction of the
                        observed directions; K from the same conditioning
                        math the engine executes.  Exact for linear
                        observations; for nonlinear derived observations it
                        is the linearization at the current mean — the
                        mu-dependence of K and H contributes higher-order
                        terms when the innovation is nonzero)
    Composite           product of its steps' maps

For a sequence ``T_0 .. T_{n-1}`` applied at linearization points along the
trajectory, a raw perturbation evolves as ``delta_n = (prod J_i) delta_0``.
The singular spectrum of the product classifies the pipeline:

    contracting   sigma_max < 1 - tol     perturbations die out
    marginal      sigma_max ~ 1           perturbations persist
    amplifying    sigma_max > 1 + tol     perturbations grow

The derived layer adds gain on top: the full-state map is ``[I; G]``, so
``gain = sigma_max(G)`` bounds how strongly raw perturbations amplify into
derived quantities at the current linearization point.

An uncertainty-energy Lyapunov function ``V(Sigma) = tr(Sigma)`` is
tracked alongside: observations provably decrease it (Joseph form),
interventions inject fresh design variance, pushforward carries it into
the derived layer.  ``analyze`` simulates the sequence on a copy of the
world — nothing commits, and SVD lives here (an analysis API), never in
the execution path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gat.engine.executor import World
from gat.engine.propagate import jacobian_rows
from gat.engine.transform import (
    CompositeTransformation,
    ObserveQuantity,
    ScaleParameter,
    SetParameter,
    ShiftParameter,
    Transformation,
)
from gat.gaussian.linalg import chol_psd, chol_solve, symmetrize


def step_jacobian(world: World, t: Transformation) -> np.ndarray:
    """Exact raw-space perturbation map of one transformation at the
    current linearization point."""
    n = world.binding.n_raw
    if isinstance(t, ShiftParameter):
        return np.eye(n)
    if isinstance(t, ScaleParameter):
        J = np.eye(n)
        J[world.binding.raw_index.row(t.var), world.binding.raw_index.row(t.var)] = t.factor
        return J
    if isinstance(t, SetParameter):
        J = np.eye(n)
        row = world.binding.raw_index.row(t.var)
        J[row, row] = 0.0
        return J
    if isinstance(t, ObserveQuantity):
        vars = tuple(m.var for m in t.measurements)
        H, _ = jacobian_rows(world.binding, world.belief, vars)
        R = np.diag([m.noise_sigma**2 for m in t.measurements])
        S = symmetrize(H @ world.belief.sigma @ H.T + R)
        L, _ = chol_psd(S)
        K = chol_solve(L, H @ world.belief.sigma).T
        return np.eye(n) - K @ H
    if isinstance(t, CompositeTransformation):
        J = np.eye(n)
        current = world
        for step in t.steps:
            J = step_jacobian(current, step) @ J
            current = current.with_belief(step.apply(current.binding, current.belief))
        return J
    raise TypeError(f"no perturbation map for {type(t).__name__}")


@dataclass(frozen=True)
class StabilityReport:
    singular_values: tuple[float, ...]   # of the product map, descending
    sigma_max: float
    sigma_min: float
    derived_gain: float                  # sigma_max of [I; G] minus raw block: max gain into derived
    verdict: str                         # "contracting" | "marginal" | "amplifying"
    energy_trace: tuple[float, ...]      # V = tr(Sigma_raw) along the trajectory

    def render(self) -> str:
        sv = ", ".join(f"{s:.4f}" for s in self.singular_values[:6])
        deltas = " ".join(
            f"{'+' if d >= 0 else ''}{d:.6g}"
            for d in (
                self.energy_trace[i + 1] - self.energy_trace[i]
                for i in range(len(self.energy_trace) - 1)
            )
        )
        return (
            f"stability: {self.verdict}  (sigma_max {self.sigma_max:.6f}, "
            f"sigma_min {self.sigma_min:.6f}; derived gain {self.derived_gain:.3f})\n"
            f"  singular values: [{sv}{', ...' if len(self.singular_values) > 6 else ''}]\n"
            f"  uncertainty energy V=tr(Sigma): {self.energy_trace[0]:.6g}, "
            f"per-step dV: {deltas}"
        )


def analyze(
    world: World, transformations: list[Transformation], tol: float = 1e-9
) -> StabilityReport:
    """Simulate the sequence (without committing) and classify its
    perturbation dynamics."""
    n = world.binding.n_raw
    product = np.eye(n)
    current = world
    energy = [float(np.trace(current.belief.sigma))]
    for t in transformations:
        product = step_jacobian(current, t) @ product
        current = current.with_belief(t.apply(current.binding, current.belief))
        energy.append(float(np.trace(current.belief.sigma)))

    singular = np.linalg.svd(product, compute_uv=False)
    sigma_max = float(singular[0]) if singular.size else 1.0
    sigma_min = float(singular[-1]) if singular.size else 1.0

    env = current.belief.env()
    G = current.binding.deps.total_jacobian(current.binding.raw_index.vars, env)
    derived_gain = float(np.linalg.svd(G, compute_uv=False)[0]) if G.size else 0.0

    if sigma_max < 1.0 - tol:
        verdict = "contracting"
    elif sigma_max <= 1.0 + tol:
        verdict = "marginal"
    else:
        verdict = "amplifying"

    return StabilityReport(
        singular_values=tuple(float(s) for s in singular),
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        derived_gain=derived_gain,
        verdict=verdict,
        energy_trace=tuple(energy),
    )
