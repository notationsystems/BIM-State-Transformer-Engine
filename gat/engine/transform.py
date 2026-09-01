"""First-class transformations over architectural state.

A :class:`Transformation` is an inspectable operation on the raw belief
(README §7, §14 principle 5): it declares the raw variables it targets,
describes its own parameters, and produces a new belief.  The executor
(:mod:`gat.engine.executor`) wraps every application with mandatory
pushforward propagation and verification.

Semantics of the shipped operators:

* ``SetParameter`` — a **do-intervention**: the designer overrides the
  value, so the old belief no longer carries information about the
  variable.  The marginal is replaced with ``N(value, design_sigma^2)``
  and cross-covariances into the variable are severed (row/col zeroed).
  Provably PSD-preserving: ``Sigma' = P Sigma P^T + q e e^T`` with ``P``
  the projection that zeroes the row.
* ``ShiftParameter`` / ``ScaleParameter`` — exact affine maps: the mean
  path and the covariance path share one Jacobian and cannot disagree.
* ``ObserveQuantity`` — Gaussian **conditioning** on measurements of raw
  or derived quantities (see :mod:`gat.gaussian.condition`).  Observation
  is the epistemic dual of intervention: it *sharpens* belief through
  correlations rather than severing them.
* ``CompositeTransformation`` — sequential composition via ``>>`` with
  atomic verification semantics (the executor verifies once, after the
  whole composite; any failure rolls back all steps).
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from gat.engine.binding import GaussianBinding
from gat.engine.propagate import jacobian_rows
from gat.errors import BindingError
from gat.gaussian.condition import ConditioningRecord, condition
from gat.gaussian.state import GaussianState
from gat.ids import VarId


@dataclass(frozen=True)
class Measurement:
    """One observed quantity: ``var`` measured as ``value +- noise_sigma``."""

    var: VarId
    value: float
    noise_sigma: float


class Transformation(ABC):
    """Base class for first-class state transformations."""

    name: str = "transformation"

    @abstractmethod
    def params(self) -> dict[str, object]:
        """Exact constructor arguments, for traces and reports."""

    @abstractmethod
    def target_vars(self) -> tuple[VarId, ...]:
        """Raw variables this operation touches (drives the affected set).

        For observations these are the raw ancestors that the conditioning
        update can move — resolved at apply time; the static declaration
        returns the observed variables themselves.
        """

    @abstractmethod
    def apply(
        self, binding: GaussianBinding, belief: GaussianState
    ) -> GaussianState:
        """Produce the new raw belief.  Must not mutate the input."""

    def describe(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in sorted(self.params().items()))
        return f"{self.name}({params})"

    def signature(self) -> str:
        payload = json.dumps(
            {"name": self.name, "params": {k: str(v) for k, v in self.params().items()}},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def then(self, other: "Transformation") -> "CompositeTransformation":
        steps: list[Transformation] = []
        for t in (self, other):
            if isinstance(t, CompositeTransformation):
                steps.extend(t.steps)
            else:
                steps.append(t)
        return CompositeTransformation(tuple(steps))

    def __rshift__(self, other: "Transformation") -> "CompositeTransformation":
        return self.then(other)


def _require_raw(binding: GaussianBinding, var: VarId, op: str) -> int:
    if not binding.is_raw(var):
        raise BindingError(f"{op} targets {var}, which is not a raw variable")
    return binding.raw_index.row(var)


class SetParameter(Transformation):
    """Design decision: ``var := value`` with fresh design uncertainty."""

    name = "set_parameter"

    def __init__(self, var: VarId, value: float, design_sigma: float):
        if design_sigma <= 0:
            raise ValueError("design_sigma must be positive")
        self.var = var
        self.value = float(value)
        self.design_sigma = float(design_sigma)

    def params(self) -> dict[str, object]:
        return {"var": str(self.var), "value": self.value, "design_sigma": self.design_sigma}

    def target_vars(self) -> tuple[VarId, ...]:
        return (self.var,)

    def apply(self, binding: GaussianBinding, belief: GaussianState) -> GaussianState:
        i = _require_raw(binding, self.var, self.name)
        mu = belief.mu.copy()
        sigma = belief.sigma.copy()
        mu[i] = self.value
        sigma[i, :] = 0.0
        sigma[:, i] = 0.0
        sigma[i, i] = self.design_sigma**2
        return belief.replace(mu=mu, sigma=sigma)


class ShiftParameter(Transformation):
    """Correlation-preserving mean shift: ``var += delta`` (exact affine)."""

    name = "shift_parameter"

    def __init__(self, var: VarId, delta: float):
        self.var = var
        self.delta = float(delta)

    def params(self) -> dict[str, object]:
        return {"var": str(self.var), "delta": self.delta}

    def target_vars(self) -> tuple[VarId, ...]:
        return (self.var,)

    def apply(self, binding: GaussianBinding, belief: GaussianState) -> GaussianState:
        i = _require_raw(binding, self.var, self.name)
        mu = belief.mu.copy()
        mu[i] += self.delta
        return belief.replace(mu=mu)


class ScaleParameter(Transformation):
    """Exact affine scaling: ``var *= factor``; ``Sigma' = D Sigma D^T``."""

    name = "scale_parameter"

    def __init__(self, var: VarId, factor: float):
        if factor == 0.0:
            raise ValueError(
                "factor must be nonzero: scaling to zero would collapse the "
                "variable's variance and break the full-rank belief guarantee"
            )
        self.var = var
        self.factor = float(factor)

    def params(self) -> dict[str, object]:
        return {"var": str(self.var), "factor": self.factor}

    def target_vars(self) -> tuple[VarId, ...]:
        return (self.var,)

    def apply(self, binding: GaussianBinding, belief: GaussianState) -> GaussianState:
        i = _require_raw(binding, self.var, self.name)
        mu = belief.mu.copy()
        sigma = belief.sigma.copy()
        mu[i] *= self.factor
        sigma[i, :] *= self.factor
        sigma[:, i] *= self.factor
        return belief.replace(mu=mu, sigma=sigma)


class ObserveQuantity(Transformation):
    """Condition the belief on one or more measurements (raw or derived)."""

    name = "observe_quantity"

    def __init__(self, measurements: tuple[Measurement, ...] | Measurement):
        if isinstance(measurements, Measurement):
            measurements = (measurements,)
        if not measurements:
            raise ValueError("at least one measurement required")
        self.measurements = tuple(measurements)
        self.record: ConditioningRecord | None = None

    @classmethod
    def single(cls, var: VarId, value: float, noise_sigma: float) -> "ObserveQuantity":
        return cls(Measurement(var, float(value), float(noise_sigma)))

    def params(self) -> dict[str, object]:
        return {
            f"m{k}": f"{m.var}={m.value}+-{m.noise_sigma}"
            for k, m in enumerate(self.measurements)
        }

    def target_vars(self) -> tuple[VarId, ...]:
        return tuple(m.var for m in self.measurements)

    def raw_targets(self, binding: GaussianBinding) -> tuple[VarId, ...]:
        """Raw ancestors the conditioning update can move: for a derived
        observation, every raw variable with a nonzero Jacobian entry; the
        update itself can move *any* correlated raw variable, so the
        affected set of an observation is conservatively all raw vars with
        nonzero gain — resolved by the executor from the actual update."""
        out: set[VarId] = set()
        for m in self.measurements:
            if binding.is_raw(m.var):
                out.add(m.var)
            else:
                for parent_var in binding.deps.raw_vars:
                    out.add(parent_var)
        return tuple(sorted(out))

    def apply(self, binding: GaussianBinding, belief: GaussianState) -> GaussianState:
        vars = tuple(m.var for m in self.measurements)
        H, predicted = jacobian_rows(binding, belief, vars)
        observed = np.array([m.value for m in self.measurements], dtype=np.float64)
        noise = np.array(
            [m.noise_sigma**2 for m in self.measurements], dtype=np.float64
        )
        posterior, record = condition(belief, H, predicted, observed, noise)
        self.record = record
        return posterior


class CompositeTransformation(Transformation):
    """Sequential composition with atomic verification semantics."""

    name = "composite"

    def __init__(self, steps: tuple[Transformation, ...]):
        if not steps:
            raise ValueError("composite requires at least one step")
        self.steps = steps

    def params(self) -> dict[str, object]:
        return {f"step{k}": t.describe() for k, t in enumerate(self.steps)}

    def target_vars(self) -> tuple[VarId, ...]:
        out: list[VarId] = []
        seen: set[VarId] = set()
        for t in self.steps:
            for var in t.target_vars():
                if var not in seen:
                    seen.add(var)
                    out.append(var)
        return tuple(out)

    def apply(self, binding: GaussianBinding, belief: GaussianState) -> GaussianState:
        current = belief
        for t in self.steps:
            current = t.apply(binding, current)
        return current

    def describe(self) -> str:
        return " >> ".join(t.describe() for t in self.steps)
