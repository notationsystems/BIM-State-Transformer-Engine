"""Gaussian state containers.

``VarIndex`` is a frozen bijection between :class:`~gat.ids.VarId` and row
index; ``GaussianState`` couples an index with a mean vector and covariance
matrix.  Two instances appear in a bound world:

* the **belief** — the canonical joint over RAW variables only.  Always
  full-rank (priors have strictly positive sigma); every solve in the
  engine happens here.
* the **full view** — raw + derived variables, produced by pushforward
  ``mu = [mu_r; F(mu_r)]``, ``Sigma = J Sigma_r J^T`` with ``J = [I; G]``.
  Rank-deficient *by construction* (its null space encodes exact functional
  dependence between derived and raw state); it is read, printed, and
  queried, but never factored or inverted.

Arrays are float64, copied on construction, and frozen (`writeable=False`),
so states are values: transformations build new ones.
"""

from __future__ import annotations

import hashlib

import numpy as np

from gat.errors import BindingError
from gat.gaussian.linalg import assert_finite, symmetrize
from gat.ids import VarId


class VarIndex:
    """Frozen ordered bijection ``VarId <-> row``."""

    def __init__(self, vars_in_order: tuple[VarId, ...]):
        self._vars = tuple(vars_in_order)
        self._row = {v: i for i, v in enumerate(self._vars)}
        if len(self._row) != len(self._vars):
            raise BindingError("duplicate variable in index")

    def __len__(self) -> int:
        return len(self._vars)

    def __contains__(self, var: VarId) -> bool:
        return var in self._row

    @property
    def vars(self) -> tuple[VarId, ...]:
        return self._vars

    def row(self, var: VarId) -> int:
        try:
            return self._row[var]
        except KeyError:
            raise BindingError(f"variable {var} is not in this index") from None

    def var(self, row: int) -> VarId:
        return self._vars[row]

    def rows(self, vars: tuple[VarId, ...]) -> np.ndarray:
        return np.array([self.row(v) for v in vars], dtype=np.intp)


class GaussianState:
    """An immutable Gaussian ``N(mu, Sigma)`` over an ordered variable set."""

    def __init__(self, index: VarIndex, mu: np.ndarray, sigma: np.ndarray):
        n = len(index)
        mu = np.asarray(mu, dtype=np.float64).reshape(n).copy()
        sigma = symmetrize(np.asarray(sigma, dtype=np.float64).reshape(n, n))
        assert_finite(mu, "mu")
        assert_finite(sigma, "Sigma")
        mu.setflags(write=False)
        sigma.setflags(write=False)
        self.index = index
        self.mu = mu
        self.sigma = sigma

    # -- lookups -----------------------------------------------------------

    def mean(self, var: VarId) -> float:
        return float(self.mu[self.index.row(var)])

    def var_of(self, var: VarId) -> float:
        return float(self.sigma[self.index.row(var), self.index.row(var)])

    def std(self, var: VarId) -> float:
        return float(np.sqrt(max(self.var_of(var), 0.0)))

    def cov(self, a: VarId, b: VarId) -> float:
        return float(self.sigma[self.index.row(a), self.index.row(b)])

    def corr(self, a: VarId, b: VarId) -> float:
        sa, sb = self.std(a), self.std(b)
        if sa < 1e-150 or sb < 1e-150:
            return 0.0
        return self.cov(a, b) / (sa * sb)

    def env(self) -> dict[VarId, float]:
        """Mean environment mapping for expression evaluation."""
        return {v: float(self.mu[i]) for i, v in enumerate(self.index.vars)}

    def marginal(self, vars: tuple[VarId, ...]) -> "GaussianState":
        rows = self.index.rows(vars)
        return GaussianState(
            VarIndex(vars), self.mu[rows], self.sigma[np.ix_(rows, rows)]
        )

    # -- construction ------------------------------------------------------

    def replace(
        self, mu: np.ndarray | None = None, sigma: np.ndarray | None = None
    ) -> "GaussianState":
        return GaussianState(
            self.index,
            self.mu if mu is None else mu,
            self.sigma if sigma is None else sigma,
        )

    def digest(self) -> str:
        h = hashlib.sha256()
        h.update(self.mu.tobytes())
        h.update(self.sigma.tobytes())
        return h.hexdigest()
