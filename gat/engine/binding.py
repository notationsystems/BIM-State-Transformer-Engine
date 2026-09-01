"""Binding the Architectural IR onto the Gaussian backend.

``bind`` compiles a module into a :class:`GaussianBinding`: the canonical
variable ordering (raw block first, each group sorted by ``VarId``), the
dependency DAG, and the initial raw belief assembled from slot priors.
The binding is immutable for the life of a world — v0 transformations
never add or remove variables, so array shapes are fixed after binding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gat.errors import BindingError
from gat.gaussian.state import GaussianState, VarIndex
from gat.ids import VarId
from gat.ir.core import Module, Role
from gat.ir.deps import DependencyGraph

#: Minimum admissible prior sigma for a raw variable, so the raw belief
#: starts strictly positive definite.
MIN_PRIOR_SIGMA = 1e-6


@dataclass(frozen=True)
class GaussianBinding:
    """The compiled bridge between IR variables and Gaussian rows."""

    raw_index: VarIndex
    full_index: VarIndex
    deps: DependencyGraph

    @property
    def n_raw(self) -> int:
        return len(self.raw_index)

    @property
    def n_full(self) -> int:
        return len(self.full_index)

    def is_raw(self, var: VarId) -> bool:
        return var in self.raw_index


def bind(module: Module) -> tuple[GaussianBinding, GaussianState]:
    """Compile the module's continuous fragment into (binding, prior belief)."""
    deps = DependencyGraph(module)
    raw_vars = tuple(sorted(module.raw_vars()))
    derived_sorted = tuple(sorted(module.derived_vars()))
    raw_index = VarIndex(raw_vars)
    full_index = VarIndex(raw_vars + derived_sorted)

    n = len(raw_vars)
    mu = np.zeros(n, dtype=np.float64)
    sigma = np.zeros((n, n), dtype=np.float64)
    for i, var in enumerate(raw_vars):
        slot = module.slot(var)
        if slot.role is not Role.RAW:
            raise BindingError(f"{var} is not a raw slot")
        if slot.prior_sigma < MIN_PRIOR_SIGMA:
            raise BindingError(
                f"raw slot {var} has prior sigma {slot.prior_sigma!r} < {MIN_PRIOR_SIGMA}"
            )
        mu[i] = slot.prior_mu
        sigma[i, i] = slot.prior_sigma**2

    belief = GaussianState(raw_index, mu, sigma)
    return GaussianBinding(raw_index, full_index, deps), belief
