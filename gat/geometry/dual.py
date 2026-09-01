"""Forward-mode dual numbers over numpy arrays.

A minimal batched forward-mode AD used as an *independent witness* for the
hand-derived analytic gradients in this subsystem: the test suite evaluates
every objective twice — analytically and through duals — and asserts
agreement (with central finite differences as a third witness).

``Dual`` carries ``val`` and ``eps`` (the directional derivative along one
seed direction) as numpy arrays of identical shape.  Operations are exact
forward-mode rules; nothing here is stochastic.
"""

from __future__ import annotations

import numpy as np


class Dual:
    __slots__ = ("val", "eps")

    #: Make numpy defer binary operations to Dual's reflected methods
    #: (otherwise ``ndarray @ Dual`` is consumed by numpy's own matmul).
    __array_priority__ = 1000

    def __init__(self, val, eps=None):
        self.val = np.asarray(val, dtype=np.float64)
        self.eps = (
            np.zeros_like(self.val) if eps is None else np.asarray(eps, dtype=np.float64)
        )
        if self.val.shape != self.eps.shape:
            raise ValueError("val/eps shape mismatch")

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def lift(x) -> "Dual":
        return x if isinstance(x, Dual) else Dual(x)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Dual(val={self.val!r}, eps={self.eps!r})"

    # -- arithmetic --------------------------------------------------------

    def __add__(self, other):
        other = Dual.lift(other)
        return Dual(self.val + other.val, self.eps + other.eps)

    __radd__ = __add__

    def __neg__(self):
        return Dual(-self.val, -self.eps)

    def __sub__(self, other):
        other = Dual.lift(other)
        return Dual(self.val - other.val, self.eps - other.eps)

    def __rsub__(self, other):
        return Dual.lift(other).__sub__(self)

    def __mul__(self, other):
        other = Dual.lift(other)
        return Dual(self.val * other.val, self.eps * other.val + self.val * other.eps)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = Dual.lift(other)
        val = self.val / other.val
        return Dual(val, (self.eps - val * other.eps) / other.val)

    def __rtruediv__(self, other):
        return Dual.lift(other).__truediv__(self)

    def __pow__(self, exponent: float):
        val = self.val**exponent
        return Dual(val, exponent * self.val ** (exponent - 1) * self.eps)

    def __matmul__(self, other):
        other = Dual.lift(other)
        return Dual(
            self.val @ other.val, self.eps @ other.val + self.val @ other.eps
        )

    def __rmatmul__(self, other):
        return Dual.lift(other).__matmul__(self)

    # -- reductions / reshaping -------------------------------------------

    def sum(self, axis=None):
        return Dual(self.val.sum(axis=axis), self.eps.sum(axis=axis))

    def mean(self, axis=None):
        return Dual(self.val.mean(axis=axis), self.eps.mean(axis=axis))

    @property
    def T(self):
        return Dual(self.val.T, self.eps.T)

    def reshape(self, *shape):
        return Dual(self.val.reshape(*shape), self.eps.reshape(*shape))

    def __getitem__(self, key):
        return Dual(self.val[key], self.eps[key])


def sqrt(x: Dual) -> Dual:
    root = np.sqrt(x.val)
    return Dual(root, 0.5 * x.eps / root)


def exp(x: Dual) -> Dual:
    e = np.exp(x.val)
    return Dual(e, e * x.eps)


def log(x: Dual) -> Dual:
    return Dual(np.log(x.val), x.eps / x.val)


def sin(x: Dual) -> Dual:
    return Dual(np.sin(x.val), np.cos(x.val) * x.eps)


def cos(x: Dual) -> Dual:
    return Dual(np.cos(x.val), -np.sin(x.val) * x.eps)


def erf(x: Dual) -> Dual:
    from math import erf as _erf, pi

    vec = np.vectorize(_erf)
    return Dual(vec(x.val), (2.0 / np.sqrt(pi)) * np.exp(-x.val**2) * x.eps)


def det3(m: Dual) -> Dual:
    """Determinant of a 3x3 dual matrix via cofactor expansion."""
    a, b, c = m[0, 0], m[0, 1], m[0, 2]
    d, e, f = m[1, 0], m[1, 1], m[1, 2]
    g, h, i = m[2, 0], m[2, 1], m[2, 2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def inv3(m: Dual) -> Dual:
    """Inverse of a 3x3 dual matrix: A^{-1} deriv = -A^{-1} dA A^{-1}."""
    inv_val = np.linalg.inv(m.val)
    return Dual(inv_val, -inv_val @ m.eps @ inv_val)


def directional_derivative(f, x: np.ndarray, direction: np.ndarray) -> float:
    """d/dt f(x + t*direction) at t=0, via one dual evaluation.

    ``f`` must be written in Dual-compatible operations and return a scalar
    Dual (0-d or shape-() value).
    """
    seed = Dual(np.asarray(x, dtype=np.float64), np.asarray(direction, dtype=np.float64))
    out = f(seed)
    return float(np.asarray(out.eps).reshape(()))
