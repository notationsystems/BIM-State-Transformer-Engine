"""Tests for gat/ir/deps.py (DependencyGraph) and gat/engine/propagate.py.

Covers:

* topological order validity and determinism across rebuilds,
* affected_set forward reachability against hand-enumerated oracles,
* cycle detection (LoweringError) on a hand-built cyclic module,
* total_jacobian against central finite differences of evaluate(),
* push_forward exactness of derived means, the identity leading block of
  J (observed through the raw block of the full view), and a hand-computed
  pushforward variance for the linear storey rollup TotalWallNetVolume.

All numerical oracles here are computed independently of the analytic
gradient machinery in gat.ir.exprs / gat.ir.deps.
"""

from __future__ import annotations

import unittest

import numpy as np

from gat.errors import LoweringError
from gat.ids import EntityId, VarId
from gat.ir.core import Entity, Module, QtySlot, Role, Unit
from gat.ir.deps import DependencyGraph
from gat.ir.exprs import VarRef
from gat.session import GatSession

DEMO = "gat/demo/model.ifc"

STOREY = EntityId("IfcBuildingStorey", "GATSTY0000000000000015")
DOOR = EntityId("IfcDoor", "GATDOR0000000000000210")
OPENING = EntityId("IfcOpeningElement", "GATOPN0000000000000200")
SPACE_A = EntityId("IfcSpace", "GATSPC0000000000000300")
SPACE_B = EntityId("IfcSpace", "GATSPC0000000000000320")
WALLS = tuple(
    EntityId("IfcWall", f"GATWAL00000000000001{suffix}0")
    for suffix in ("0", "2", "4", "6", "8")
)
WALL_100 = WALLS[0]
WALL_180 = WALLS[4]  # the wall voided by the single opening

CLEAR_HEIGHT = VarId(STOREY, "ClearHeight")


class DemoModelMixin:
    """Loads the demo session once for the whole test class."""

    session: GatSession

    @classmethod
    def setUpClass(cls) -> None:
        cls.session = GatSession.load_ifc(DEMO)


class TestTopologicalOrder(DemoModelMixin, unittest.TestCase):
    def test_topo_is_valid_linearization(self) -> None:
        """Every derived parent of a derived var precedes it in topo order."""
        deps = self.session.world.binding.deps
        topo = deps.derived_vars
        self.assertEqual(len(topo), 39)
        self.assertEqual(len(set(topo)), 39, "topo order contains duplicates")
        derived = set(topo)
        position = {v: i for i, v in enumerate(topo)}
        for var in topo:
            for parent in deps.parents(var):
                if parent in derived:
                    self.assertLess(
                        position[parent],
                        position[var],
                        f"{parent} must precede {var} in topological order",
                    )

    def test_topo_identical_across_50_rebuilds(self) -> None:
        module = self.session.world.module
        reference = DependencyGraph(module).derived_vars
        for i in range(50):
            rebuilt = DependencyGraph(module).derived_vars
            self.assertEqual(rebuilt, reference, f"rebuild {i} changed the order")


class TestAffectedSet(DemoModelMixin, unittest.TestCase):
    def test_affected_set_clear_height_is_34_derived_vars(self) -> None:
        # Hand enumeration: ClearHeight feeds
        #   * each of the 2 space Volumes (Volume = FloorArea * ClearHeight),
        #   * each of the 5 walls' Height, and transitively GrossSideArea,
        #     GrossVolume, NetSideArea, NetVolume, Cost -> 6 per wall = 30,
        #   * the storey rollups TotalWallNetVolume and TotalWallCost.
        # Total = 2 + 30 + 2 = 34 of the 39 derived vars.  NOT affected:
        # door Area, opening Area, 2 space FloorAreas, TotalFloorArea (5).
        expected = {VarId(SPACE_A, "Volume"), VarId(SPACE_B, "Volume")}
        for wall in WALLS:
            for q in (
                "Height",
                "GrossSideArea",
                "GrossVolume",
                "NetSideArea",
                "NetVolume",
                "Cost",
            ):
                expected.add(VarId(wall, q))
        expected.add(VarId(STOREY, "TotalWallNetVolume"))
        expected.add(VarId(STOREY, "TotalWallCost"))
        self.assertEqual(len(expected), 34)

        deps = self.session.world.binding.deps
        affected = deps.affected_set((CLEAR_HEIGHT,))
        self.assertEqual(len(affected), 34)
        self.assertEqual(set(affected), expected)

    def test_affected_set_is_in_topo_order(self) -> None:
        deps = self.session.world.binding.deps
        affected = deps.affected_set((CLEAR_HEIGHT,))
        position = {v: i for i, v in enumerate(deps.derived_vars)}
        self.assertEqual(
            list(affected), sorted(affected, key=lambda v: position[v])
        )

    def test_affected_set_wall_width_hand_enumerated(self) -> None:
        # Hand enumeration for wall 100's Width: Width enters only
        # GrossVolume (= GrossSideArea * Width) and NetVolume
        # (= NetSideArea * Width); NetVolume feeds Cost (priced wall) and
        # TotalWallNetVolume; Cost feeds TotalWallCost.  Height /
        # GrossSideArea / NetSideArea do NOT depend on Width.
        expected = {
            VarId(WALL_100, "GrossVolume"),
            VarId(WALL_100, "NetVolume"),
            VarId(WALL_100, "Cost"),
            VarId(STOREY, "TotalWallNetVolume"),
            VarId(STOREY, "TotalWallCost"),
        }
        deps = self.session.world.binding.deps
        affected = deps.affected_set((VarId(WALL_100, "Width"),))
        self.assertEqual(set(affected), expected)


class TestCycleDetection(unittest.TestCase):
    def test_cyclic_module_raises_lowering_error(self) -> None:
        """Two mutually-referencing derived slots must be rejected."""
        eid = EntityId("IfcWall", "CYCLE00000000000000000")
        var_a = VarId(eid, "A")
        var_b = VarId(eid, "B")
        entity = Entity(
            id=eid,
            name="cyclic",
            slots={
                "A": QtySlot(
                    var=var_a, role=Role.DERIVED, unit=Unit.M, expr=VarRef(var_b)
                ),
                "B": QtySlot(
                    var=var_b, role=Role.DERIVED, unit=Unit.M, expr=VarRef(var_a)
                ),
            },
        )
        module = Module(entities={eid: entity}, rels=(), constraints=())
        with self.assertRaises(LoweringError):
            DependencyGraph(module)

    def test_unknown_reference_raises_lowering_error(self) -> None:
        eid = EntityId("IfcWall", "DANGL00000000000000000")
        ghost = VarId(eid, "Ghost")
        entity = Entity(
            id=eid,
            name="dangling",
            slots={
                "A": QtySlot(
                    var=VarId(eid, "A"),
                    role=Role.DERIVED,
                    unit=Unit.M,
                    expr=VarRef(ghost),
                ),
            },
        )
        module = Module(entities={eid: entity}, rels=(), constraints=())
        with self.assertRaises(LoweringError):
            DependencyGraph(module)


class TestTotalJacobian(DemoModelMixin, unittest.TestCase):
    def test_every_derived_row_matches_central_finite_differences(self) -> None:
        """G vs central differences of evaluate() at the demo linearization
        point, h = 1e-6, rtol 1e-6."""
        world = self.session.world
        deps = world.binding.deps
        raw_order = world.binding.raw_index.vars
        base_env = world.belief.env()
        topo = deps.derived_vars

        G = deps.total_jacobian(raw_order, base_env)
        self.assertEqual(G.shape, (39, 24))

        h = 1e-6
        fd = np.zeros_like(G)
        for j, raw_var in enumerate(raw_order):
            env_plus = dict(base_env)
            env_minus = dict(base_env)
            env_plus[raw_var] = base_env[raw_var] + h
            env_minus[raw_var] = base_env[raw_var] - h
            out_plus = deps.evaluate(env_plus)
            out_minus = deps.evaluate(env_minus)
            for r, var in enumerate(topo):
                fd[r, j] = (out_plus[var] - out_minus[var]) / (2.0 * h)

        # atol picks up exact-zero pairs; every nonzero entry is checked
        # relatively.  All demo exprs are polynomial, so central differences
        # are accurate to O(h^2) truncation plus rounding noise ~1e-9.
        np.testing.assert_allclose(G, fd, rtol=1e-6, atol=1e-8)

    def test_jacobian_row_count_and_order_match_topo(self) -> None:
        world = self.session.world
        deps = world.binding.deps
        raw_order = world.binding.raw_index.vars
        G = deps.total_jacobian(raw_order, world.belief.env())
        # Row for wall 100 Height (= ClearHeight) must be the unit vector
        # selecting the ClearHeight raw column: d Height / d ClearHeight = 1.
        r = deps.derived_vars.index(VarId(WALL_100, "Height"))
        col = raw_order.index(CLEAR_HEIGHT)
        expected_row = np.zeros(len(raw_order))
        expected_row[col] = 1.0
        np.testing.assert_array_equal(G[r], expected_row)


class TestPushForward(DemoModelMixin, unittest.TestCase):
    def test_derived_means_are_exact_reevaluations(self) -> None:
        world = self.session.world
        deps = world.binding.deps
        derived_values = deps.evaluate(world.belief.env())
        for var, value in derived_values.items():
            # Exact equality: push_forward stores evaluate()'s outputs
            # verbatim, never a linearized approximation.
            self.assertEqual(world.full.mean(var), value, str(var))

    def test_identity_leading_block(self) -> None:
        """J = [I; G] has an identity leading block, so the raw block of the
        full view must equal the belief exactly (mu and Sigma)."""
        world = self.session.world
        n_raw = world.binding.n_raw
        self.assertEqual(n_raw, 24)
        # Raw variables occupy the leading rows of the full index, in the
        # same order as the belief index.
        self.assertEqual(
            world.full.index.vars[:n_raw], world.belief.index.vars
        )
        np.testing.assert_array_equal(world.full.mu[:n_raw], world.belief.mu)
        np.testing.assert_array_equal(
            world.full.sigma[:n_raw, :n_raw], world.belief.sigma
        )

    def test_total_wall_net_volume_variance_quadratic_form(self) -> None:
        """Pushforward variance of the linear rollup TotalWallNetVolume vs
        the hand-computed quadratic form g^T Sigma_r g.

        The raw prior Sigma is diagonal, so the quadratic form is
        sum_i g_i^2 sigma_i^2.  Hand-derived gradient of
        TWNV = sum_w (L_w * H - [openings on w]) * W_w
        at the demo means (H = 3.0; opening 1.0 x 2.1 voids wall 180 only):

          dTWNV/dH   = sum_w L_w W_w
                     = 9.2*0.3 + 9.2*0.3 + 4.6*0.3 + 4.6*0.3 + 4.0*0.2 = 9.08
          dTWNV/dL_w = H * W_w          -> 0.9, 0.9, 0.9, 0.9, 0.6
          dTWNV/dW_w = NetSideArea_w    -> 27.6, 27.6, 13.8, 13.8,
                                           4.0*3.0 - 1.0*2.1 = 9.9
          dTWNV/d(opening Width)  = -oh * W_180 = -2.1*0.2 = -0.42
          dTWNV/d(opening Height) = -ow * W_180 = -1.0*0.2 = -0.20
          (UnitCost, door, space vars: 0)

        With prior sigmas (ClearHeight 0.01; wall Length 0.005 except wall
        180 at 0.004; wall Width 0.002; opening W/H 0.005):

          var = 9.08^2*0.01^2
              + 4*(0.9^2*0.005^2) + 0.6^2*0.004^2
              + 2*(27.6^2*0.002^2) + 2*(13.8^2*0.002^2) + 9.9^2*0.002^2
              + 0.42^2*0.005^2 + 0.2^2*0.005^2
              = 0.01634645
        """
        expected = (
            9.08**2 * 0.01**2
            + 4 * (0.9**2 * 0.005**2)
            + 0.6**2 * 0.004**2
            + 2 * (27.6**2 * 0.002**2)
            + 2 * (13.8**2 * 0.002**2)
            + 9.9**2 * 0.002**2
            + 0.42**2 * 0.005**2
            + 0.2**2 * 0.005**2
        )
        twnv = VarId(STOREY, "TotalWallNetVolume")
        actual = self.session.world.full.var_of(twnv)
        self.assertAlmostEqual(actual, expected, delta=1e-12)

    def test_push_forward_full_covariance_is_symmetric(self) -> None:
        sigma = self.session.world.full.sigma
        np.testing.assert_array_equal(sigma, sigma.T)


if __name__ == "__main__":
    unittest.main()
