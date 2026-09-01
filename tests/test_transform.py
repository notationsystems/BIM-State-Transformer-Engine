"""Tests for gat/engine/transform.py — first-class transformations.

All beliefs come from the demo world (gat/demo/model.ifc, 24 raw vars).
Bitwise assertions use np.array_equal: SetParameter / ShiftParameter /
ScaleParameter touch exactly one raw row/column, so everything else must
be bit-identical, not merely close.
"""

from __future__ import annotations

import unittest

import numpy as np

from gat.engine.transform import (
    Measurement,
    ObserveQuantity,
    ScaleParameter,
    SetParameter,
    ShiftParameter,
)
from gat.engine.propagate import jacobian_rows
from gat.errors import BindingError
from gat.gaussian.condition import condition
from gat.session import GatSession


def _demo():
    return GatSession.load_ifc("gat/demo/model.ifc")


class TransformTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.session = _demo()
        cls.world = cls.session.world
        cls.binding = cls.world.binding
        cls.belief = cls.world.belief

    def var(self, name, qty):
        return self.session.var(name, qty)


class TestSetParameter(TransformTestBase):
    def test_zeroes_row_col_and_sets_diagonal(self):
        v = self.var("Wall-East", "Length")
        k = self.binding.raw_index.row(v)
        t = SetParameter(v, 4.8, design_sigma=0.02)
        out = t.apply(self.binding, self.belief)

        # Mean: only entry k changes, to exactly the set value.
        self.assertEqual(out.mu[k], 4.8)
        others = np.delete(np.arange(len(out.mu)), k)
        self.assertTrue(np.array_equal(out.mu[others], self.belief.mu[others]))

        # Sigma: row k and column k are exactly zero off-diagonal, the
        # diagonal is exactly design_sigma**2.
        self.assertEqual(out.sigma[k, k], 0.02**2)
        row = np.delete(out.sigma[k, :], k)
        col = np.delete(out.sigma[:, k], k)
        self.assertTrue(np.array_equal(row, np.zeros_like(row)))
        self.assertTrue(np.array_equal(col, np.zeros_like(col)))

        # The untouched (n-1)x(n-1) block is bit-identical.
        sub = np.ix_(others, others)
        self.assertTrue(np.array_equal(out.sigma[sub], self.belief.sigma[sub]))

    def test_nonpositive_design_sigma_rejected(self):
        v = self.var("Wall-East", "Length")
        with self.assertRaises(ValueError):
            SetParameter(v, 4.8, design_sigma=0.0)
        with self.assertRaises(ValueError):
            SetParameter(v, 4.8, design_sigma=-0.1)

    def test_derived_target_raises_binding_error(self):
        # Wall Height is a DERIVED slot (:= storey ClearHeight); a
        # do-intervention on it must be rejected at binding time.
        derived = self.var("Wall-East", "Height")
        t = SetParameter(derived, 3.2, design_sigma=0.01)
        with self.assertRaises(BindingError):
            t.apply(self.binding, self.belief)


class TestShiftParameter(TransformTestBase):
    def test_moves_only_target_mean_sigma_bit_identical(self):
        v = self.var("Office-A", "Length")
        k = self.binding.raw_index.row(v)
        t = ShiftParameter(v, 0.25)
        out = t.apply(self.binding, self.belief)

        self.assertAlmostEqual(out.mu[k], self.belief.mu[k] + 0.25, delta=1e-15)
        others = np.delete(np.arange(len(out.mu)), k)
        self.assertTrue(np.array_equal(out.mu[others], self.belief.mu[others]))
        # Covariance is completely untouched — every bit.
        self.assertTrue(np.array_equal(out.sigma, self.belief.sigma))


class TestScaleParameter(TransformTestBase):
    def test_sigma_is_d_sigma_d(self):
        """Sigma' = D Sigma D with D = diag(1,...,factor,...,1).

        factor = 2.0 is a power of two, so every product in both the
        implementation (row/col scaling) and the oracle (dense matmul) is
        exact in binary floating point: the two must agree bitwise.
        """
        v = self.var("Wall-Party", "Width")
        k = self.binding.raw_index.row(v)
        factor = 2.0
        t = ScaleParameter(v, factor)
        out = t.apply(self.binding, self.belief)

        d = np.ones(len(self.belief.mu))
        d[k] = factor
        D = np.diag(d)
        expected = D @ self.belief.sigma @ D
        self.assertTrue(np.array_equal(out.sigma, expected))

        self.assertEqual(out.mu[k], self.belief.mu[k] * factor)
        others = np.delete(np.arange(len(out.mu)), k)
        self.assertTrue(np.array_equal(out.mu[others], self.belief.mu[others]))

    def test_sigma_general_factor_close(self):
        # Non-dyadic factor: same identity, within one ulp-scale tolerance.
        v = self.var("Wall-Party", "Width")
        k = self.binding.raw_index.row(v)
        factor = 1.3
        out = ScaleParameter(v, factor).apply(self.binding, self.belief)
        d = np.ones(len(self.belief.mu))
        d[k] = factor
        expected = np.diag(d) @ self.belief.sigma @ np.diag(d)
        np.testing.assert_allclose(out.sigma, expected, rtol=1e-15, atol=0.0)


class TestComposite(TransformTestBase):
    def test_rshift_equals_manual_sequential_application(self):
        t1 = ShiftParameter(self.var("Office-A", "Length"), 0.1)
        t2 = ScaleParameter(self.var("Wall-Party", "Width"), 2.0)
        t3 = SetParameter(self.var("Door-1", "Width"), 0.95, design_sigma=0.004)
        composite = t1 >> t2 >> t3

        via_composite = composite.apply(self.binding, self.belief)
        manual = t3.apply(
            self.binding, t2.apply(self.binding, t1.apply(self.binding, self.belief))
        )
        self.assertTrue(np.array_equal(via_composite.mu, manual.mu))
        self.assertTrue(np.array_equal(via_composite.sigma, manual.sigma))

    def test_rshift_flattens_steps(self):
        t1 = ShiftParameter(self.var("Office-A", "Length"), 0.1)
        t2 = ScaleParameter(self.var("Wall-Party", "Width"), 2.0)
        t3 = SetParameter(self.var("Door-1", "Width"), 0.95, design_sigma=0.004)
        nested = (t1 >> t2) >> t3
        self.assertEqual(len(nested.steps), 3)
        self.assertEqual(nested.steps, (t1, t2, t3))
        # target_vars preserves first-seen order, de-duplicated.
        again = t1 >> t2 >> t3 >> ShiftParameter(t1.var, -0.1)
        self.assertEqual(again.target_vars(), (t1.var, t2.var, t3.var))


class TestApplyPurity(TransformTestBase):
    def test_apply_never_mutates_input_belief(self):
        mu_bytes = self.belief.mu.tobytes()
        sigma_bytes = self.belief.sigma.tobytes()
        transforms = [
            SetParameter(self.var("Wall-East", "Length"), 4.8, 0.02),
            ShiftParameter(self.var("Office-A", "Length"), 0.25),
            ScaleParameter(self.var("Wall-Party", "Width"), 1.1),
            ObserveQuantity.single(
                self.var("Office-A", "Length"), 5.01, noise_sigma=0.002
            ),
            ShiftParameter(self.var("Office-A", "Length"), 0.1)
            >> ScaleParameter(self.var("Wall-Party", "Width"), 2.0),
        ]
        for t in transforms:
            with self.subTest(t=t.describe()):
                t.apply(self.binding, self.belief)
                self.assertEqual(self.belief.mu.tobytes(), mu_bytes)
                self.assertEqual(self.belief.sigma.tobytes(), sigma_bytes)


class TestDescribeSignatureStability(TransformTestBase):
    def test_identical_constructions_agree(self):
        v = self.var("Wall-East", "Length")
        a = SetParameter(v, 4.8, design_sigma=0.02)
        b = SetParameter(v, 4.8, design_sigma=0.02)
        self.assertEqual(a.describe(), b.describe())
        self.assertEqual(a.signature(), b.signature())

        ca = a >> ShiftParameter(v, 0.1)
        cb = b >> ShiftParameter(v, 0.1)
        self.assertEqual(ca.describe(), cb.describe())
        self.assertEqual(ca.signature(), cb.signature())

        oa = ObserveQuantity.single(v, 4.61, noise_sigma=0.003)
        ob = ObserveQuantity.single(v, 4.61, noise_sigma=0.003)
        self.assertEqual(oa.describe(), ob.describe())
        self.assertEqual(oa.signature(), ob.signature())

    def test_different_params_give_different_signatures(self):
        v = self.var("Wall-East", "Length")
        a = SetParameter(v, 4.8, design_sigma=0.02)
        b = SetParameter(v, 4.9, design_sigma=0.02)
        self.assertNotEqual(a.signature(), b.signature())


class TestObserveQuantity(TransformTestBase):
    def _direct(self, measurements):
        """Direct condition() call using the same H/predicted assembly."""
        vars_ = tuple(m.var for m in measurements)
        H, predicted = jacobian_rows(self.binding, self.belief, vars_)
        observed = np.array([m.value for m in measurements])
        noise = np.array([m.noise_sigma**2 for m in measurements])
        post, _ = condition(self.belief, H, predicted, observed, noise)
        return post

    def test_raw_observation_matches_direct_condition(self):
        m = Measurement(self.var("Office-A", "Length"), 5.012, 0.002)
        t = ObserveQuantity(m)
        out = t.apply(self.binding, self.belief)
        direct = self._direct((m,))
        self.assertTrue(np.array_equal(out.mu, direct.mu))
        self.assertTrue(np.array_equal(out.sigma, direct.sigma))
        self.assertIsNotNone(t.record)

    def test_derived_observation_matches_direct_condition(self):
        # Office-A Volume is DERIVED (= Length * Width * ClearHeight);
        # observing it conditions the raw belief through its Jacobian row.
        m = Measurement(self.var("Office-A", "Volume"), 60.3, 0.5)
        t = ObserveQuantity(m)
        out = t.apply(self.binding, self.belief)
        direct = self._direct((m,))
        self.assertTrue(np.array_equal(out.mu, direct.mu))
        self.assertTrue(np.array_equal(out.sigma, direct.sigma))

    def test_joint_raw_and_derived_matches_direct_condition(self):
        ms = (
            Measurement(self.var("Level 1", "ClearHeight"), 3.005, 0.004),
            Measurement(self.var("Office-B", "FloorArea"), 16.1, 0.05),
        )
        t = ObserveQuantity(ms)
        out = t.apply(self.binding, self.belief)
        direct = self._direct(ms)
        self.assertTrue(np.array_equal(out.mu, direct.mu))
        self.assertTrue(np.array_equal(out.sigma, direct.sigma))

    def test_derived_observation_moves_correlated_raw_parents(self):
        # Observing the derived volume below its prediction must pull the
        # raw parents (Length, Width, ClearHeight) downward.
        m = Measurement(self.var("Office-A", "Volume"), 58.0, 0.2)
        out = ObserveQuantity(m).apply(self.binding, self.belief)
        for name, qty in (
            ("Office-A", "Length"),
            ("Office-A", "Width"),
            ("Level 1", "ClearHeight"),
        ):
            v = self.var(name, qty)
            self.assertLess(out.mean(v), self.belief.mean(v))

    def test_empty_measurements_rejected(self):
        with self.assertRaises(ValueError):
            ObserveQuantity(())


if __name__ == "__main__":
    unittest.main()
