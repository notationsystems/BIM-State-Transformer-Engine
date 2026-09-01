"""Tests for gat/engine/executor.py + gat/engine/verify.py.

Covers: the selectivity guarantee of execute() (bitwise), strict
rejection with rollback (unchanged world digest), strict=False rejection
semantics, purpose-built corrupted worlds for the invariant classes
(CONS-01 FAIL/WARN, CONS-02 FAIL/WARN, GAUSS-03, STRUCT-01), QTY-01 on
the pristine demo, and deterministic report rendering.

Demo model facts used below (gat/demo/model.ifc):
  * storey ClearHeight prior N(3.0, 0.01^2); every wall Height := ClearHeight
  * Opening-1 (raw Width=1.0 sigma 0.005, Height=2.1 sigma 0.005) voids
    Wall-Party (Height derived = 3.0)
  * Door-1 (raw Width=0.9 sigma 0.003, Height=2.0 sigma 0.003) fills Opening-1
  * lowering emits LessEqual(opening.Height, wall.Height) etc., and
    NonNegative on every slot.
"""

from __future__ import annotations

import unittest

import numpy as np

from gat.engine.executor import World, execute
from gat.engine.transform import SetParameter, ShiftParameter
from gat.engine.verify import Status, run_invariants
from gat.errors import VerificationError
from gat.gaussian.state import GaussianState
from gat.ids import EntityId, VarId
from gat.ir.core import Entity, Module, QtySlot, Rel, RelKind, Role, Unit
from gat.session import GatSession


def _demo():
    return GatSession.load_ifc("gat/demo/model.ifc")


class ExecutorTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.session = _demo()
        cls.world = cls.session.world

    def var(self, name, qty):
        return self.session.var(name, qty)


class TestSelectivity(ExecutorTestBase):
    def _outside_rows(self, result):
        allowed = set(result.targets) | set(result.affected)
        index = self.world.full.index
        return [i for i in range(len(index)) if index.var(i) not in allowed]

    def test_shift_touches_only_targets_and_descendants(self):
        # Office-A Length feeds FloorArea -> Volume and the storey
        # TotalFloorArea rollup; the other 59 variables must not move a bit.
        result = execute(self.world, ShiftParameter(self.var("Office-A", "Length"), 0.1))
        self.assertTrue(result.committed)
        expected_affected = {
            self.var("Office-A", "FloorArea"),
            self.var("Office-A", "Volume"),
            self.var("Level 1", "TotalFloorArea"),
        }
        self.assertEqual(set(result.affected), expected_affected)

        rows = self._outside_rows(result)
        self.assertEqual(len(rows), self.world.binding.n_full - 4)
        self.assertTrue(
            np.array_equal(result.world.full.mu[rows], self.world.full.mu[rows])
        )
        sub = np.ix_(rows, rows)
        self.assertTrue(
            np.array_equal(result.world.full.sigma[sub], self.world.full.sigma[sub])
        )
        # A pure mean shift leaves the raw covariance bit-identical.
        self.assertTrue(
            np.array_equal(result.world.belief.sigma, self.world.belief.sigma)
        )
        # Reported deltas cover exactly the moved variables.
        self.assertEqual(
            {v for v, _ in result.deltas},
            expected_affected | {self.var("Office-A", "Length")},
        )

    def test_set_parameter_selectivity_bitwise(self):
        result = execute(
            self.world, SetParameter(self.var("Wall-East", "Length"), 4.7, 0.005)
        )
        self.assertTrue(result.committed)
        rows = self._outside_rows(result)
        self.assertTrue(
            np.array_equal(result.world.full.mu[rows], self.world.full.mu[rows])
        )
        sub = np.ix_(rows, rows)
        self.assertTrue(
            np.array_equal(result.world.full.sigma[sub], self.world.full.sigma[sub])
        )


class TestRejectionAndRollback(ExecutorTestBase):
    # Opening-1.Height := 3.2 violates LessEqual(Opening-1.Height,
    # Wall-Party.Height): 3.2 > 3.0 by 0.2 at the mean -> CONS-02 FAIL.
    def _violating(self):
        return SetParameter(self.var("Opening-1", "Height"), 3.2, 0.005)

    def test_strict_raises_and_world_digest_unchanged(self):
        digest_before = self.world.digest()
        with self.assertRaises(VerificationError) as ctx:
            execute(self.world, self._violating(), strict=True)
        report = ctx.exception.report
        self.assertFalse(report.passed)
        fail_ids = {r.invariant_id for r in report.failures}
        self.assertIn("CONS-02", fail_ids)
        # The violation is by exactly 0.2 at the mean (3.2 - 3.0).
        cons02 = [r for r in report.failures if r.invariant_id == "CONS-02"]
        self.assertEqual(len(cons02), 1)
        self.assertAlmostEqual(cons02[0].residual, 0.2, delta=1e-9)
        # Rollback is total: the original world is untouched.
        self.assertEqual(self.world.digest(), digest_before)

    def test_non_strict_returns_uncommitted_original_world(self):
        result = execute(self.world, self._violating(), strict=False)
        self.assertFalse(result.committed)
        self.assertIs(result.world, self.world)
        self.assertFalse(result.report.passed)
        self.assertEqual(result.world.digest(), self.world.digest())

    def test_session_run_rolls_back_and_records_reject(self):
        session = _demo()
        digest_before = session.world.digest()
        with self.assertRaises(VerificationError):
            session.run(SetParameter(session.var("Opening-1", "Height"), 3.2, 0.005))
        self.assertEqual(session.world.digest(), digest_before)


class TestConstraintInvariants(ExecutorTestBase):
    def test_cons01_fail_on_negative_mean(self):
        # Door-1.Width := -0.5 makes NonNegative(Door-1.Width) fail at the
        # mean (and Door-1.Area = -0.5 * 2.0 = -1.0 fails too); LessEqual
        # door <= opening still holds, so the failures are pure CONS-01.
        t = SetParameter(self.var("Door-1", "Width"), -0.5, 0.001)
        with self.assertRaises(VerificationError) as ctx:
            execute(self.world, t)
        fails = ctx.exception.report.failures
        self.assertTrue(fails)
        self.assertEqual({r.invariant_id for r in fails}, {"CONS-01"})
        self.assertIn(str(self.var("Door-1", "Width")), {r.subject for r in fails})

    def test_cons01_warn_when_two_sigma_straddles_zero(self):
        # Door-1.Width := N(0.02, 0.02^2): mean >= 0 but mean - 2 sigma =
        # -0.02 < 0 -> WARN, and the transformation still commits.
        t = SetParameter(self.var("Door-1", "Width"), 0.02, 0.02)
        result = execute(self.world, t)
        self.assertTrue(result.committed)
        self.assertTrue(result.report.passed)
        warn_ids = {r.invariant_id for r in result.report.warnings}
        self.assertIn("CONS-01", warn_ids)
        warn_subjects = {
            r.subject for r in result.report.warnings if r.invariant_id == "CONS-01"
        }
        self.assertIn(str(self.var("Door-1", "Width")), warn_subjects)

    def test_cons02_fail_on_mean_violation(self):
        with self.assertRaises(VerificationError) as ctx:
            execute(self.world, SetParameter(self.var("Opening-1", "Height"), 3.2, 0.005))
        fails = ctx.exception.report.failures
        self.assertEqual({r.invariant_id for r in fails}, {"CONS-02"})

    def test_cons02_warn_within_two_sigma(self):
        # Opening-1.Height := N(2.99, 0.02^2) against Wall-Party.Height =
        # ClearHeight ~ N(3.0, 0.01^2).  SetParameter severs the
        # cross-covariance, so diff = -0.01 and
        # sigma_diff = sqrt(0.02^2 + 0.01^2) ~ 0.02236:
        # diff + 2 sigma ~ +0.0347 > 0 -> WARN but no FAIL; commits.
        t = SetParameter(self.var("Opening-1", "Height"), 2.99, 0.02)
        result = execute(self.world, t)
        self.assertTrue(result.committed)
        warns = [r for r in result.report.warnings if r.invariant_id == "CONS-02"]
        self.assertEqual(len(warns), 1)
        subject = warns[0].subject
        self.assertIn(str(self.var("Opening-1", "Height")), subject)
        self.assertIn(str(self.var("Wall-Party", "Height")), subject)
        # residual = diff + 2*sigma_diff = -0.01 + 2*sqrt(0.0005)
        expected_residual = -0.01 + 2.0 * np.sqrt(0.02**2 + 0.01**2)
        self.assertAlmostEqual(warns[0].residual, expected_residual, delta=1e-9)


class TestGaussianInvariants(ExecutorTestBase):
    def test_gauss03_fail_on_hand_built_inconsistent_full_state(self):
        # GAUSS-03 (derived means equal exact re-evaluation) is unreachable
        # through the public API: World.with_belief / World.compile always
        # rebuild the full view via push_forward, which re-evaluates every
        # derived mean exactly, so raw/derived inconsistency cannot arise
        # from any Transformation.  We therefore corrupt the full view by
        # hand: bump Door-1.Area (a derived leaf that no other expression,
        # rollup, or ExprEquals restatement references) so the *only*
        # failure is GAUSS-03.
        w = self.world
        area = self.var("Door-1", "Area")
        row = w.full.index.row(area)
        mu_bad = w.full.mu.copy()
        mu_bad[row] += 0.7
        full_bad = GaussianState(w.full.index, mu_bad, w.full.sigma)
        bad_world = World(w.module, w.graph, w.binding, w.belief, full_bad)

        report = run_invariants(bad_world)
        self.assertFalse(report.passed)
        self.assertEqual(
            [(r.invariant_id, r.subject) for r in report.failures],
            [("GAUSS-03", str(area))],
        )
        self.assertAlmostEqual(report.failures[0].residual, 0.7, delta=1e-9)

    def test_gauss03_passes_after_legal_transform(self):
        # Sanity check of the unreachability claim: after a legal transform
        # the rebuilt full view is consistent by construction.
        result = execute(
            self.world, SetParameter(self.var("Office-B", "Length"), 4.5, 0.005)
        )
        gauss03 = [
            r for r in result.report.results if r.invariant_id == "GAUSS-03"
        ]
        self.assertEqual([r.status for r in gauss03], [Status.PASS])


class TestStructuralInvariants(unittest.TestCase):
    def test_struct01_fail_on_dangling_relationship_endpoint(self):
        # Tiny hand-built module: one wall with a single raw slot and a
        # CONTAINS edge pointing at an entity that does not exist.
        wall = EntityId("IfcWall", "TINYWALL000000000001")
        ghost = EntityId("IfcSpace", "GHOST000000000000001")
        slot = QtySlot(
            var=VarId(wall, "Length"),
            role=Role.RAW,
            unit=Unit.M,
            prior_mu=2.0,
            prior_sigma=0.01,
        )
        module = Module(
            entities={wall: Entity(id=wall, name="tiny-wall", slots={"Length": slot})},
            rels=(Rel(RelKind.CONTAINS, wall, ghost),),
            constraints=(),
        )
        world = World.compile(module)
        report = run_invariants(world)
        self.assertFalse(report.passed)
        struct = [r for r in report.failures if r.invariant_id == "STRUCT-01"]
        self.assertEqual(len(struct), 1)
        self.assertEqual(struct[0].subject, str(ghost))


class TestDemoReportAndRollups(ExecutorTestBase):
    def test_pristine_demo_all_pass(self):
        report = run_invariants(self.world)
        self.assertTrue(report.passed)
        self.assertEqual(report.warnings, ())
        p, w, f = report.counts()
        self.assertEqual((w, f), (0, 0))
        self.assertGreaterEqual(p, 10)

    def test_aggregation_consistent_on_demo(self):
        report = run_invariants(self.world)
        qty = [r for r in report.results if r.invariant_id == "QTY-01"]
        self.assertEqual([r.status for r in qty], [Status.PASS])
        # Independent cross-check of one rollup: TotalFloorArea =
        # Office-A 5.0*4.0 + Office-B 4.0*4.0 = 20 + 16 = 36 m^2.
        total = self.world.full.mean(self.var("Level 1", "TotalFloorArea"))
        self.assertAlmostEqual(total, 36.0, delta=1e-9)

    def test_report_render_is_deterministic(self):
        # Same world -> identical string; a freshly loaded identical
        # session -> identical string as well (no ordering nondeterminism).
        r1 = run_invariants(self.world).render()
        r2 = run_invariants(self.world).render()
        self.assertEqual(r1, r2)
        fresh = _demo()
        self.assertEqual(run_invariants(fresh.world).render(), r1)
        self.assertEqual(r1.splitlines()[0], "verification: 12 pass, 0 warn, 0 fail")

    def test_failure_render_is_deterministic_and_lists_failures(self):
        result = execute(
            self.world,
            SetParameter(self.var("Opening-1", "Height"), 3.2, 0.005),
            strict=False,
        )
        text1 = result.report.render()
        result2 = execute(
            self.world,
            SetParameter(self.var("Opening-1", "Height"), 3.2, 0.005),
            strict=False,
        )
        self.assertEqual(text1, result2.report.render())
        self.assertIn("FAIL CONS-02", text1)
        self.assertIn(str(self.var("Opening-1", "Height")), text1)


if __name__ == "__main__":
    unittest.main()
