"""Differentiable expression AST for derived architectural quantities.

Derived quantities in the Architectural IR are defined by small expression
trees over :class:`~gat.ids.VarId` references.  The grammar is deliberately
closed and tiny — every node has exact analytic partial derivatives, so the
dependency layer can assemble exact total Jacobians by the chain rule.
Numeric differentiation exists only in the test suite, as an oracle for
this code.

Every node supports:

* ``eval(env)``   — exact value given a mapping ``VarId -> float``
* ``grad(env)``   — partial derivatives w.r.t. each *directly referenced*
  variable (the chain rule across derived variables happens in
  :mod:`gat.ir.deps`, not here)
* ``free_vars()`` — the referenced variables, in deterministic sorted order
* ``to_str()``    — a stable, human-readable rendering used by the IR printer
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Mapping, Union

from gat.ids import VarId

Env = Union[Mapping[VarId, float], Callable[[VarId], float]]


def _lookup(env: Env, var: VarId) -> float:
    if callable(env):
        return env(var)
    return env[var]


class Expr(ABC):
    """Base class for expression nodes."""

    @abstractmethod
    def eval(self, env: Env) -> float:
        """Exact value of the expression at ``env``."""

    @abstractmethod
    def grad(self, env: Env) -> dict[VarId, float]:
        """Partials w.r.t. each directly referenced variable, at ``env``."""

    @abstractmethod
    def _collect(self, out: set[VarId]) -> None: ...

    @abstractmethod
    def to_str(self) -> str:
        """Deterministic textual rendering."""

    def free_vars(self) -> tuple[VarId, ...]:
        out: set[VarId] = set()
        self._collect(out)
        return tuple(sorted(out))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}({self.to_str()})"


def _merge(into: dict[VarId, float], frm: dict[VarId, float], scale: float = 1.0) -> None:
    for var, p in frm.items():
        into[var] = into.get(var, 0.0) + scale * p


@dataclass(frozen=True)
class Const(Expr):
    value: float

    def eval(self, env: Env) -> float:
        return self.value

    def grad(self, env: Env) -> dict[VarId, float]:
        return {}

    def _collect(self, out: set[VarId]) -> None:
        pass

    def to_str(self) -> str:
        return repr(self.value)


@dataclass(frozen=True)
class VarRef(Expr):
    var: VarId

    def eval(self, env: Env) -> float:
        return _lookup(env, self.var)

    def grad(self, env: Env) -> dict[VarId, float]:
        return {self.var: 1.0}

    def _collect(self, out: set[VarId]) -> None:
        out.add(self.var)

    def to_str(self) -> str:
        return str(self.var)


@dataclass(frozen=True)
class Add(Expr):
    left: Expr
    right: Expr

    def eval(self, env: Env) -> float:
        return self.left.eval(env) + self.right.eval(env)

    def grad(self, env: Env) -> dict[VarId, float]:
        g: dict[VarId, float] = {}
        _merge(g, self.left.grad(env))
        _merge(g, self.right.grad(env))
        return g

    def _collect(self, out: set[VarId]) -> None:
        self.left._collect(out)
        self.right._collect(out)

    def to_str(self) -> str:
        return f"({self.left.to_str()} + {self.right.to_str()})"


@dataclass(frozen=True)
class Sub(Expr):
    left: Expr
    right: Expr

    def eval(self, env: Env) -> float:
        return self.left.eval(env) - self.right.eval(env)

    def grad(self, env: Env) -> dict[VarId, float]:
        g: dict[VarId, float] = {}
        _merge(g, self.left.grad(env))
        _merge(g, self.right.grad(env), scale=-1.0)
        return g

    def _collect(self, out: set[VarId]) -> None:
        self.left._collect(out)
        self.right._collect(out)

    def to_str(self) -> str:
        return f"({self.left.to_str()} - {self.right.to_str()})"


@dataclass(frozen=True)
class Mul(Expr):
    left: Expr
    right: Expr

    def eval(self, env: Env) -> float:
        return self.left.eval(env) * self.right.eval(env)

    def grad(self, env: Env) -> dict[VarId, float]:
        lv = self.left.eval(env)
        rv = self.right.eval(env)
        g: dict[VarId, float] = {}
        _merge(g, self.left.grad(env), scale=rv)
        _merge(g, self.right.grad(env), scale=lv)
        return g

    def _collect(self, out: set[VarId]) -> None:
        self.left._collect(out)
        self.right._collect(out)

    def to_str(self) -> str:
        return f"({self.left.to_str()} * {self.right.to_str()})"


@dataclass(frozen=True)
class Neg(Expr):
    operand: Expr

    def eval(self, env: Env) -> float:
        return -self.operand.eval(env)

    def grad(self, env: Env) -> dict[VarId, float]:
        g: dict[VarId, float] = {}
        _merge(g, self.operand.grad(env), scale=-1.0)
        return g

    def _collect(self, out: set[VarId]) -> None:
        self.operand._collect(out)

    def to_str(self) -> str:
        return f"(-{self.operand.to_str()})"


@dataclass(frozen=True)
class ScaledSum(Expr):
    """``const + sum(coef_i * term_i)`` — assembly rollups and weighted sums.

    ``terms`` is a tuple of ``(coefficient, Expr)`` pairs.  An empty ``terms``
    tuple is legal and evaluates to ``const`` (a wall with no openings still
    gets a net-area definition through the uniform rule).
    """

    terms: tuple[tuple[float, Expr], ...]
    const: float = 0.0

    def eval(self, env: Env) -> float:
        total = self.const
        for coef, term in self.terms:
            total += coef * term.eval(env)
        return total

    def grad(self, env: Env) -> dict[VarId, float]:
        g: dict[VarId, float] = {}
        for coef, term in self.terms:
            _merge(g, term.grad(env), scale=coef)
        return g

    def _collect(self, out: set[VarId]) -> None:
        for _, term in self.terms:
            term._collect(out)

    def to_str(self) -> str:
        parts = [f"{coef!r}*{term.to_str()}" for coef, term in self.terms]
        body = " + ".join(parts) if parts else "0"
        if self.const != 0.0 or not parts:
            return f"({self.const!r} + {body})" if parts else f"({self.const!r})"
        return f"({body})"


@dataclass(frozen=True)
class Mean(Expr):
    """Arithmetic mean of one or more sub-expressions."""

    terms: tuple[Expr, ...]

    def __post_init__(self) -> None:
        if not self.terms:
            raise ValueError("Mean requires at least one term")

    def eval(self, env: Env) -> float:
        return sum(t.eval(env) for t in self.terms) / len(self.terms)

    def grad(self, env: Env) -> dict[VarId, float]:
        w = 1.0 / len(self.terms)
        g: dict[VarId, float] = {}
        for term in self.terms:
            _merge(g, term.grad(env), scale=w)
        return g

    def _collect(self, out: set[VarId]) -> None:
        for term in self.terms:
            term._collect(out)

    def to_str(self) -> str:
        return f"mean({', '.join(t.to_str() for t in self.terms)})"
