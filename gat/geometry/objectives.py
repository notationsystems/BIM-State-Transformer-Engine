"""Differentiable objectives over architectural state and its Gaussian field.

Two differentiability mechanisms, both real and both witnessed in tests:

* **State-space terms** (cost, daylight ratio, energy proxy, chance
  constraints) are functions of derived quantities, so their exact
  gradients w.r.t. raw parameters come from the engine's total Jacobian
  rows — the same chain-rule bridge the propagation layer uses.
* **Field terms** (ray transmittance through the Gaussian solid) have
  closed-form values; their gradients are obtained by forward-mode dual
  numbers (:mod:`gat.geometry.dual`) — exact to machine precision, no
  finite-difference step tuning.

Ray transmittance (the daylight/insulation field query): for a ray
``x(t) = o + t d``, ``t in [0, L]`` and Gaussian ``(mu, S)``:

    integral = N-normalizer * exp(-(c - b^2/a)/2) * sqrt(pi/(2a))
               * [erf((aL+b)/sqrt(2a)) - erf(b/sqrt(2a))]

with ``a = d^T S^-1 d``, ``b = d^T S^-1 (o-mu)``, ``c = (o-mu)^T S^-1 (o-mu)``.
Optical depth = kappa * sum_k w_k * integral_k; transmittance = exp(-depth).

Chance constraints reuse the module's LessEqual constraints: the margin
``m = rhs - lhs`` has mean and sigma under the full joint; the penalty
``softplus(-(mu_m - z_alpha * sigma_m) / s) * s`` is smooth and pushes the
design until the constraint holds with confidence ``Phi(z_alpha)`` —
compliance and optimization share one margin definition.

The optimizer is deterministic Armijo gradient descent over a chosen
subset of raw parameters; committing the result goes through ordinary
``SetParameter`` interventions, so verification gates the write-back like
any other transformation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from gat.engine.executor import World
from gat.engine.sensitivity import raw_jacobian_row
from gat.engine.transform import CompositeTransformation, SetParameter
from gat.gaussian.state import GaussianState
from gat.ids import VarId
from gat.ir.core import LessEqual

# ---------------------------------------------------------------------------
# Ray transmittance through the Gaussian field
# ---------------------------------------------------------------------------

_NORM3 = (2.0 * math.pi) ** -1.5


def ray_optical_depth(
    means: np.ndarray,
    covs: np.ndarray,
    weights: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
    length: float,
    kappa: float = 1.0,
) -> float:
    """Optical depth of a finite ray through a weighted Gaussian set."""
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    origin = np.asarray(origin, dtype=np.float64)

    inv = np.linalg.inv(covs)                              # (K, 3, 3)
    sign, logdet = np.linalg.slogdet(covs)
    delta = origin[None, :] - means                        # (K, 3)
    a = np.einsum("i,kij,j->k", direction, inv, direction)
    b = np.einsum("i,kij,kj->k", direction, inv, delta)
    c = np.einsum("ki,kij,kj->k", delta, inv, delta)

    erf_vec = np.vectorize(math.erf)
    root = np.sqrt(a / 2.0)
    seg = erf_vec((a * length + b) / np.sqrt(2.0 * a)) - erf_vec(b / np.sqrt(2.0 * a))
    integral = (
        _NORM3
        * np.exp(-0.5 * logdet)
        * np.exp(-0.5 * (c - b**2 / a))
        * np.sqrt(math.pi / (2.0 * a))
        * seg
    )
    del root
    return float(kappa * np.sum(weights * integral))


def ray_transmittance(*args, **kwargs) -> float:
    return math.exp(-ray_optical_depth(*args, **kwargs))


def scalar_ray_depth_dual(mu, cov, weight, origin, direction, length, kappa=1.0):
    """Dual-friendly single-Gaussian ray integral (gradient witness path).

    Written entirely in operations the Dual class supports, so tests can
    seed any of the inputs and read exact directional derivatives.
    """
    from gat.geometry import dual as D

    inv = D.inv3(D.Dual.lift(cov)) if isinstance(cov, D.Dual) else np.linalg.inv(cov)
    det = D.det3(D.Dual.lift(cov)) if isinstance(cov, D.Dual) else np.linalg.det(cov)
    delta = D.Dual.lift(origin) - D.Dual.lift(mu)
    d = D.Dual.lift(direction)
    a = (d @ (inv @ d)).sum()
    b = (d @ (inv @ delta)).sum()
    c = (delta @ (inv @ delta)).sum()
    seg = D.erf((a * length + b) / D.sqrt(a * 2.0)) - D.erf(b / D.sqrt(a * 2.0))
    integral = (
        _NORM3
        * (det ** -0.5)
        * D.exp((c - b * b / a) * -0.5)
        * D.sqrt(D.Dual.lift(math.pi) / (a * 2.0))
        * seg
    )
    return integral * weight * kappa


# ---------------------------------------------------------------------------
# The layout objective
# ---------------------------------------------------------------------------


def _softplus(x: float, s: float = 0.02) -> float:
    z = x / s
    if z > 40.0:
        return x
    return s * math.log1p(math.exp(z))


def _sigmoid(x: float, s: float = 0.02) -> float:
    z = x / s
    if z > 40.0:
        return 1.0
    if z < -40.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


@dataclass(frozen=True)
class ObjectiveTerm:
    label: str
    value: float
    weighted: float


@dataclass
class LayoutObjective:
    """C(x) = w_cost * cost + w_day * (daylight - target)^2 + w_energy * energy
    + chance-constraint penalties, with exact raw-space gradients."""

    cost_var: VarId
    daylight_area_var: VarId       # aperture area (e.g. opening.Area)
    daylight_floor_var: VarId      # reference floor area
    energy_terms: tuple[tuple[VarId, float], ...]  # (area var, U-value)
    daylight_target: float = 0.10
    w_cost: float = 1e-4
    w_daylight: float = 400.0
    w_energy: float = 0.02
    z_alpha: float = 2.0           # ~97.7% one-sided confidence
    penalty_weight: float = 200.0
    penalty_softness: float = 0.01
    constraints: tuple[LessEqual, ...] = field(default_factory=tuple)

    def evaluate(self, world: World) -> tuple[float, np.ndarray, tuple[ObjectiveTerm, ...]]:
        """Objective value, gradient w.r.t. the raw vector, and term report."""
        full = world.full
        n_raw = world.binding.n_raw
        grad = np.zeros(n_raw, dtype=np.float64)
        terms: list[ObjectiveTerm] = []

        # -- cost ----------------------------------------------------------
        cost = full.mean(self.cost_var)
        g_cost = raw_jacobian_row(world, self.cost_var)
        grad += self.w_cost * g_cost
        terms.append(ObjectiveTerm("cost", cost, self.w_cost * cost))

        # -- daylight ratio ------------------------------------------------
        area = full.mean(self.daylight_area_var)
        floor = full.mean(self.daylight_floor_var)
        ratio = area / floor
        g_area = raw_jacobian_row(world, self.daylight_area_var)
        g_floor = raw_jacobian_row(world, self.daylight_floor_var)
        g_ratio = g_area / floor - (area / floor**2) * g_floor
        resid = ratio - self.daylight_target
        grad += self.w_daylight * 2.0 * resid * g_ratio
        terms.append(
            ObjectiveTerm(f"daylight (ratio {ratio:.4f})", resid**2, self.w_daylight * resid**2)
        )

        # -- energy proxy --------------------------------------------------
        energy = 0.0
        for var, u_value in self.energy_terms:
            energy += u_value * full.mean(var)
            grad += self.w_energy * u_value * raw_jacobian_row(world, var)
        terms.append(ObjectiveTerm("energy", energy, self.w_energy * energy))

        # -- chance constraints -------------------------------------------
        penalty_total = 0.0
        for c in self.constraints:
            mu_m = full.mean(c.rhs) - full.mean(c.lhs)
            var_m = (
                full.var_of(c.rhs) + full.var_of(c.lhs) - 2.0 * full.cov(c.rhs, c.lhs)
            )
            sigma_m = math.sqrt(max(var_m, 0.0))
            shortfall = -(mu_m - self.z_alpha * sigma_m)
            penalty_total += _softplus(shortfall, self.penalty_softness)
            g_m = raw_jacobian_row(world, c.rhs) - raw_jacobian_row(world, c.lhs)
            grad += (
                self.penalty_weight
                * _sigmoid(shortfall, self.penalty_softness)
                * (-g_m)
            )
        terms.append(
            ObjectiveTerm("chance penalties", penalty_total, self.penalty_weight * penalty_total)
        )

        value = sum(t.weighted for t in terms)
        return value, grad, tuple(terms)


@dataclass(frozen=True)
class OptimizationStep:
    values: tuple[float, ...]      # parameter values after the step
    objective: float
    grad_norm: float
    step_size: float


@dataclass(frozen=True)
class OptimizationResult:
    params: tuple[VarId, ...]
    initial: tuple[float, ...]
    optimized: tuple[float, ...]
    objective_initial: float
    objective_final: float
    trajectory: tuple[OptimizationStep, ...]
    converged: bool


def optimize_layout(
    world: World,
    params: tuple[VarId, ...],
    objective: LayoutObjective,
    max_iter: int = 60,
    initial_step: float = 1e-3,
    grad_tol: float = 1e-6,
) -> OptimizationResult:
    """Deterministic Armijo gradient descent over the chosen raw parameters.

    Candidate states are ordinary belief updates (means moved, covariance
    untouched); nothing commits — the caller applies the result through
    interventions, which re-verifies.
    """
    binding = world.binding
    rows = [binding.raw_index.row(p) for p in params]
    current = world
    value, grad, _ = objective.evaluate(current)
    initial_values = tuple(float(current.belief.mu[r]) for r in rows)
    initial_value = value

    trajectory: list[OptimizationStep] = []
    step = initial_step
    converged = False
    for _ in range(max_iter):
        g = np.array([grad[r] for r in rows])
        gnorm = float(np.linalg.norm(g))
        if gnorm < grad_tol:
            converged = True
            break
        # Armijo backtracking on the true objective.
        improved = False
        for _try in range(25):
            mu_new = current.belief.mu.copy()
            for r, gr in zip(rows, g):
                mu_new[r] -= step * gr
            candidate = current.with_belief(
                GaussianState(current.belief.index, mu_new, current.belief.sigma)
            )
            cand_value, cand_grad, _ = objective.evaluate(candidate)
            if cand_value <= value - 1e-4 * step * gnorm**2:
                current, value, grad = candidate, cand_value, cand_grad
                improved = True
                trajectory.append(
                    OptimizationStep(
                        tuple(float(mu_new[r]) for r in rows), value, gnorm, step
                    )
                )
                step *= 1.5
                break
            step *= 0.5
        if not improved:
            converged = gnorm < 1e-3
            break

    return OptimizationResult(
        params=params,
        initial=initial_values,
        optimized=tuple(float(current.belief.mu[r]) for r in rows),
        objective_initial=initial_value,
        objective_final=value,
        trajectory=tuple(trajectory),
        converged=converged,
    )


def as_interventions(
    result: OptimizationResult, design_sigma: float = 0.005
) -> CompositeTransformation:
    """Package an optimization result as verifiable design interventions."""
    steps = tuple(
        SetParameter(var, value, design_sigma)
        for var, value in zip(result.params, result.optimized)
    )
    return CompositeTransformation(steps)
