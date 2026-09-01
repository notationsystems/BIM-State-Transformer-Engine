"""Tests for gat/ir/exprs.py — the differentiable expression AST.

Covers: eval correctness on hand values for every node type; analytic
gradients against central finite differences at 3 fixed points per node
type; exact rational gradients for Mean/ScaledSum; sorted free_vars; and
the empty-ScaledSum-is-const rule.
"""

from __future__ import annotations

import math
import unittest

from gat.ids import EntityId, VarId
from gat.ir.exprs import Add, Const, Expr, Mean, Mul, Neg, ScaledSum, Sub, VarRef

# Fixed variable identities used throughout.  Sorted order of these VarIds
# (lexicographic on (entity.ifc_class, entity.global_id, quantity)) is:
# A < B < C  (same entity class/global_id ordering by quantity), see
# test_free_vars_sorted for an explicit check.
_E1 = EntityId("IfcWall", "W0000000000000000001")
_E2 = EntityId("IfcWall", "W0000000000000000002")
A = VarId(_E1, "Length")
B = VarId(_E1, "Width")
C = VarId(_E2, "Height")

FD_H = 1e-6
FD_RTOL = 1e-6


def fd_grad(expr: Expr, env: dict[VarId, float], h: float = FD_H) -> dict[VarId, float]:
    """Central finite-difference gradient — numeric oracle for grad()."""
    out: dict[VarId, float] = {}
    for var in expr.free_vars():
        env_p = dict(env)
        env_m = dict(env)
        env_p[var] += h
        env_m[var] -= h
        out[var] = (expr.eval(env_p) - expr.eval(env_m)) / (2.0 * h)
    return out


class TestEvalHandValues(unittest.TestCase):
    """Exact hand-computed values for every node type."""

    def test_const(self):
        self.assertEqual(Const(3.5).eval({}), 3.5)

    def test_varref(self):
        self.assertEqual(VarRef(A).eval({A: 2.25}), 2.25)

    def test_varref_callable_env(self):
        # Env may be a callable VarId -> float.
        self.assertEqual(VarRef(A).eval(lambda v: 7.5), 7.5)

    def test_add(self):
        # 2.0 + 3.5 = 5.5
        self.assertEqual(Add(VarRef(A), VarRef(B)).eval({A: 2.0, B: 3.5}), 5.5)

    def test_sub(self):
        # 2.0 - 3.5 = -1.5
        self.assertEqual(Sub(VarRef(A), VarRef(B)).eval({A: 2.0, B: 3.5}), -1.5)

    def test_mul(self):
        # 2.0 * 3.5 = 7.0
        self.assertEqual(Mul(VarRef(A), VarRef(B)).eval({A: 2.0, B: 3.5}), 7.0)

    def test_neg(self):
        self.assertEqual(Neg(VarRef(A)).eval({A: 2.0}), -2.0)

    def test_scaled_sum(self):
        # 1.5 + 2.0*2.0 + (-0.5)*4.0 = 1.5 + 4.0 - 2.0 = 3.5
        expr = ScaledSum(((2.0, VarRef(A)), (-0.5, VarRef(B))), const=1.5)
        self.assertEqual(expr.eval({A: 2.0, B: 4.0}), 3.5)

    def test_mean(self):
        # (1.0 + 2.0 + 3.0) / 3 = 2.0 exactly
        expr = Mean((VarRef(A), VarRef(B), VarRef(C)))
        self.assertEqual(expr.eval({A: 1.0, B: 2.0, C: 3.0}), 2.0)

    def test_mean_requires_terms(self):
        with self.assertRaises(ValueError):
            Mean(())

    def test_nested_composite(self):
        # (a + b) * (a - c) at a=2, b=3, c=0.5:  (5) * (1.5) = 7.5
        expr = Mul(Add(VarRef(A), VarRef(B)), Sub(VarRef(A), VarRef(C)))
        self.assertEqual(expr.eval({A: 2.0, B: 3.0, C: 0.5}), 7.5)


class TestGradFiniteDifferences(unittest.TestCase):
    """Analytic grad() vs central finite differences (h=1e-6, rtol 1e-6)
    at 3 fixed points per node type."""

    # Three fixed evaluation points, away from zeros of the partials.
    POINTS = (
        {A: 1.3, B: -0.7, C: 2.9},
        {A: -2.1, B: 0.45, C: -1.6},
        {A: 0.8, B: 3.2, C: 0.05},
    )

    def check(self, expr: Expr):
        for env in self.POINTS:
            analytic = expr.grad(env)
            numeric = fd_grad(expr, env)
            self.assertEqual(set(analytic), set(numeric), msg=expr.to_str())
            for var in numeric:
                self.assertTrue(
                    math.isclose(
                        analytic[var], numeric[var], rel_tol=FD_RTOL, abs_tol=1e-8
                    ),
                    msg=(
                        f"{expr.to_str()} d/d{var} at {env}: "
                        f"analytic {analytic[var]!r} vs fd {numeric[var]!r}"
                    ),
                )

    def test_const(self):
        for env in self.POINTS:
            self.assertEqual(Const(4.2).grad(env), {})

    def test_varref(self):
        self.check(VarRef(A))

    def test_add(self):
        # Repeated variable exercises _merge accumulation: d/da (a + a) = 2.
        self.check(Add(VarRef(A), VarRef(B)))
        self.check(Add(VarRef(A), VarRef(A)))

    def test_sub(self):
        self.check(Sub(VarRef(A), VarRef(B)))
        self.check(Sub(VarRef(A), VarRef(A)))  # cancels to 0

    def test_mul(self):
        self.check(Mul(VarRef(A), VarRef(B)))
        # Product with a shared variable: d/da (a*(a+c)) = 2a + c.
        self.check(Mul(VarRef(A), Add(VarRef(A), VarRef(C))))

    def test_neg(self):
        self.check(Neg(VarRef(B)))
        self.check(Neg(Mul(VarRef(A), VarRef(C))))

    def test_scaled_sum(self):
        expr = ScaledSum(
            ((2.0, VarRef(A)), (-0.5, Mul(VarRef(B), VarRef(C))), (3.0, VarRef(A))),
            const=1.25,
        )
        self.check(expr)

    def test_mean(self):
        expr = Mean((VarRef(A), Mul(VarRef(B), VarRef(C)), VarRef(A)))
        self.check(expr)

    def test_deep_composite(self):
        # -(a*b) + (a - c)*(b + 2)  — all binary node types plus Neg.
        expr = Add(
            Neg(Mul(VarRef(A), VarRef(B))),
            Mul(Sub(VarRef(A), VarRef(C)), Add(VarRef(B), Const(2.0))),
        )
        self.check(expr)


class TestExactGradients(unittest.TestCase):
    """Mean / ScaledSum gradients must be exact rationals (assertEqual)."""

    def test_mean_quarter_weights(self):
        # Mean of 4 distinct vars: each partial is exactly 1/4 = 0.25
        # (0.25 is exactly representable in binary floating point).
        D = VarId(_E2, "Width")
        expr = Mean((VarRef(A), VarRef(B), VarRef(C), VarRef(D)))
        g = expr.grad({A: 1.0, B: 2.0, C: 3.0, D: 4.0})
        self.assertEqual(g, {A: 0.25, B: 0.25, C: 0.25, D: 0.25})

    def test_mean_half_weights(self):
        # Mean of 2 terms: exactly 0.5 each.
        g = Mean((VarRef(A), VarRef(B))).grad({A: 0.0, B: 0.0})
        self.assertEqual(g, {A: 0.5, B: 0.5})

    def test_mean_repeated_term_accumulates_exactly(self):
        # mean(a, a) → d/da = 0.5 + 0.5 = 1.0 exactly.
        g = Mean((VarRef(A), VarRef(A))).grad({A: 3.0})
        self.assertEqual(g, {A: 1.0})

    def test_scaled_sum_coefficients_exact(self):
        # d/da = 2.0, d/db = -0.5 exactly; const contributes nothing.
        expr = ScaledSum(((2.0, VarRef(A)), (-0.5, VarRef(B))), const=7.0)
        g = expr.grad({A: 10.0, B: 20.0})
        self.assertEqual(g, {A: 2.0, B: -0.5})

    def test_scaled_sum_repeated_var_sums_exactly(self):
        # 1.5*a + 0.25*a → d/da = 1.75 exactly.
        expr = ScaledSum(((1.5, VarRef(A)), (0.25, VarRef(A))))
        self.assertEqual(expr.grad({A: 1.0}), {A: 1.75})


class TestFreeVars(unittest.TestCase):
    def test_free_vars_sorted(self):
        # Reference vars in reverse order; free_vars must come back sorted.
        expr = Add(VarRef(C), Mul(VarRef(B), VarRef(A)))
        fv = expr.free_vars()
        self.assertEqual(fv, tuple(sorted(fv)))
        # VarId sorts by (entity, quantity); _E1 < _E2 by global_id, and
        # within _E1 "Length" < "Width", so the exact order is (A, B, C).
        self.assertEqual(fv, (A, B, C))

    def test_free_vars_deduplicated(self):
        expr = Add(VarRef(A), Add(VarRef(A), VarRef(B)))
        self.assertEqual(expr.free_vars(), (A, B))

    def test_const_has_no_free_vars(self):
        self.assertEqual(Const(1.0).free_vars(), ())

    def test_scaled_sum_and_mean_free_vars(self):
        expr = ScaledSum(((1.0, Mean((VarRef(C), VarRef(A)))),), const=2.0)
        self.assertEqual(expr.free_vars(), (A, C))


class TestEmptyScaledSum(unittest.TestCase):
    def test_empty_scaled_sum_is_const(self):
        expr = ScaledSum((), const=5.5)
        self.assertEqual(expr.eval({}), 5.5)
        self.assertEqual(expr.grad({}), {})
        self.assertEqual(expr.free_vars(), ())

    def test_empty_scaled_sum_default_const_zero(self):
        self.assertEqual(ScaledSum(()).eval({}), 0.0)


if __name__ == "__main__":
    unittest.main()
