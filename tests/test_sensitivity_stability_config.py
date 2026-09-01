"""Tests for gat/engine/sensitivity.py, stability.py, configuration.py.

All on the shipped demo model (two offices, 5 walls, 1 opening, 1 door,
1 storey; 24 raw + 39 derived vars).  Every numerical oracle is derived by
hand in a comment next to its assertion.  Fully deterministic - no
randomness anywhere.

stdlib unittest only.
"""

from __future__ import annotations

import os
import re
import unittest

from gat.engine.configuration import configuration_digest
from gat.engine.sensitivity import sensitivities_of, variance_attribution
from gat.engine.stability import analyze
from gat.engine.transform import (
    ObserveQuantity,
    ScaleParameter,
    SetParameter,
    ShiftParameter,
)
from gat.session import GatSession

DEMO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "gat", "demo", "model.ifc",
)


def _demo_text() -> str:
    with open(DEMO, "r", encoding="utf-8") as fh:
        return fh.read()


class SensitivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.session = GatSession.load_ifc(DEMO)

    def test_total_wall_cost_top_sensitivity_is_wall_width(self) -> None:
        # TotalWallCost = sum_w NetVolume_w * UnitCost_w with
        # NetVolume_w = (Length_w * ClearHeight - openings) * Width_w.
        # d TotalWallCost / d Width_w = NetSideArea_w * UnitCost_w.
        # Wall-South (and Wall-North): Length 9.2 m, Height = ClearHeight
        # 3.0 m, no openings -> NetSideArea = 9.2 * 3.0 = 27.6 m^2, at
        # UnitCost 320/m^3 -> 27.6 * 320 = 8832.0.  That beats
        # d/dClearHeight = sum_w Length_w*Width_w*UnitCost_w
        #   = 2*(9.2*0.3*320) + 2*(4.6*0.3*320) + 4.0*0.2*280
        #   = 1766.4 + 883.2 + 224.0 = 2873.6,
        # East/West widths (13.8*320 = 4416) and the Party width
        # ((12.0 - 2.1)*280 = 2772), so Width of Wall-South/North is top.
        total_cost = self.session.var("Level 1", "TotalWallCost")
        pairs = sensitivities_of(self.session.world, total_cost)
        top_var, top_val = pairs[0]
        self.assertEqual(top_var.quantity, "Width")
        south = self.session.entity_by_name("Wall-South")
        north = self.session.entity_by_name("Wall-North")
        self.assertIn(top_var.entity, (south, north))
        self.assertEqual(top_val, 8832.0)  # exact in float64: 27.6*320
        # The tie partner (the other of South/North) sits right behind it
        # with the same exact value.
        self.assertEqual(pairs[1][1], 8832.0)
        self.assertEqual(pairs[1][0].quantity, "Width")
        self.assertNotEqual(pairs[0][0].entity, pairs[1][0].entity)

    def test_variance_attribution_sums_to_one(self) -> None:
        vol = self.session.var("Office-A", "Volume")
        shares = variance_attribution(self.session.world, vol)
        self.assertGreater(len(shares), 0)
        self.assertAlmostEqual(sum(s for _, s in shares), 1.0, delta=1e-9)

    def test_clear_height_dominates_office_a_volume(self) -> None:
        # Volume = Length * Width * ClearHeight at (5, 4, 3) with
        # independent priors sigma_L = sigma_W = 0.005, sigma_H = 0.01.
        # J = (W*H, L*H, L*W) = (12, 15, 20), so
        # Var = (12*0.005)^2 + (15*0.005)^2 + (20*0.01)^2
        #     = 0.0036 + 0.005625 + 0.04 = 0.049225,
        # and ClearHeight's share = 0.04 / 0.049225 ~= 0.812595.
        vol = self.session.var("Office-A", "Volume")
        shares = variance_attribution(self.session.world, vol)
        top_var, top_share = shares[0]
        clear_height = self.session.var("Level 1", "ClearHeight")
        self.assertEqual(top_var, clear_height)
        self.assertAlmostEqual(top_share, 0.04 / 0.049225, delta=1e-9)
        self.assertGreater(top_share, 0.5)  # dominates outright


class StabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.session = GatSession.load_ifc(DEMO)
        cls.clear_height = cls.session.var("Level 1", "ClearHeight")
        cls.volume = cls.session.var("Office-A", "Volume")

    def test_shift_is_marginal_with_unit_spectrum(self) -> None:
        # ShiftParameter's perturbation map is exactly I (24 x 24): every
        # singular value is exactly 1.
        report = analyze(self.session.world, [ShiftParameter(self.clear_height, 0.1)])
        self.assertEqual(report.verdict, "marginal")
        self.assertEqual(len(report.singular_values), 24)
        for s in report.singular_values:
            self.assertEqual(s, 1.0)

    def test_scale_is_amplifying_with_sigma_max_factor(self) -> None:
        # ScaleParameter(factor=1.2) is I with one diagonal entry 1.2:
        # sigma_max = 1.2 exactly, everything else 1.
        report = analyze(self.session.world, [ScaleParameter(self.clear_height, 1.2)])
        self.assertEqual(report.verdict, "amplifying")
        self.assertAlmostEqual(report.sigma_max, 1.2, delta=1e-12)

    def test_observe_contracts_and_decreases_energy(self) -> None:
        # ObserveQuantity has map I - K H: it contracts the observed
        # direction (sigma_min < 1) and provably decreases the
        # uncertainty-energy Lyapunov function V = tr(Sigma).
        report = analyze(
            self.session.world,
            [ObserveQuantity.single(self.volume, 59.0, noise_sigma=0.05)],
        )
        self.assertLess(report.sigma_min, 1.0 - 1e-9)
        self.assertEqual(len(report.energy_trace), 2)
        self.assertLess(report.energy_trace[1], report.energy_trace[0])

    def test_set_parameter_has_zero_singular_value(self) -> None:
        # SetParameter is a do-intervention: its map is I with one row
        # zeroed, so the perturbation in that channel is forgotten and
        # sigma_min == 0 exactly.
        report = analyze(
            self.session.world, [SetParameter(self.clear_height, 3.4, 0.01)]
        )
        self.assertEqual(report.sigma_min, 0.0)


class ConfigurationDigestTests(unittest.TestCase):
    def test_identical_across_two_loads(self) -> None:
        d1 = configuration_digest(GatSession.load_ifc(DEMO).world)
        d2 = configuration_digest(GatSession.load_ifc(DEMO).world)
        self.assertEqual(d1, d2)

    def test_invariant_under_renumber_relabel_rename_rigid_motion(self) -> None:
        # One combined variant exercising the full quotient:
        # (a) all step ids shifted by +1000,
        # (b) every GlobalId replaced by a fresh unique string,
        # (c) every entity Name changed,
        # (d) the whole building rigidly moved: the storey placement
        #     (root of every lowered product's placement chain - IfcBuilding
        #     and IfcProject are not lowered) gets origin (17, -4, 0) and
        #     RefDirection (0, 1, 0), i.e. a +90 degree rotation, so all
        #     child placements translate and rotate together.
        base = _demo_text()
        variant = re.sub(r"#(\d+)", lambda m: f"#{int(m.group(1)) + 1000}", base)
        # (b) GlobalIds in the demo are exactly 'GAT' + 3 letters + 16
        # digits (22 chars); pset names like 'GAT_Material' don't match.
        gids = sorted(set(re.findall(r"'(GAT[A-Z]{3}\d{16})'", variant)))
        self.assertGreater(len(gids), 20)
        for i, gid in enumerate(gids):
            variant = variant.replace(f"'{gid}'", f"'Z{i:021d}'")
        # (c) entity names (quantity/pset names stay untouched).
        names = [
            "GAT Demo Project", "Demo Building", "Level 1",
            "Wall-South", "Wall-North", "Wall-West", "Wall-East",
            "Wall-Party", "Opening-1", "Door-1", "Office-A", "Office-B",
        ]
        for i, name in enumerate(names):
            self.assertIn(f"'{name}'", variant)
            variant = variant.replace(f"'{name}'", f"'Renamed-{i}'")
        # (d) storey placement #21/#22 are now #1021/#1022; #1030 is the
        # existing IFCDIRECTION((0.,1.,0.)) record.
        old_axis = "#1021=IFCAXIS2PLACEMENT3D(#1022,$,$);"
        self.assertIn(old_axis, variant)
        variant = variant.replace(
            old_axis, "#1021=IFCAXIS2PLACEMENT3D(#1022,$,#1030);"
        )
        old_origin = "#1022=IFCCARTESIANPOINT((0.,0.,0.));"
        self.assertIn(old_origin, variant)
        variant = variant.replace(
            old_origin, "#1022=IFCCARTESIANPOINT((17.,-4.,0.));", 1
        )

        original = GatSession.load_ifc(DEMO)
        rewritten = GatSession.from_text(variant, source="variant")
        # Sanity: the surgery really moved the model - the storey sits at
        # (17, -4) rotated by pi/2 in the rewritten world.
        storey_eid = [
            eid for eid in rewritten.world.module.entities
            if eid.ifc_class == "IfcBuildingStorey"
        ][0]
        placement = rewritten.world.module.entities[storey_eid].placement
        self.assertAlmostEqual(placement.x, 17.0, delta=1e-12)
        self.assertAlmostEqual(placement.y, -4.0, delta=1e-12)
        self.assertAlmostEqual(placement.angle, 1.5707963267948966, delta=1e-12)

        self.assertEqual(
            configuration_digest(original.world),
            configuration_digest(rewritten.world),
        )

    def test_different_after_clear_height_change(self) -> None:
        base = _demo_text()
        target = "#29=IFCQUANTITYLENGTH('ClearHeight',$,$,3.);"
        self.assertIn(target, base)
        variant = base.replace(
            target, "#29=IFCQUANTITYLENGTH('ClearHeight',$,$,3.1);"
        )
        d_base = configuration_digest(GatSession.load_ifc(DEMO).world)
        d_variant = configuration_digest(
            GatSession.from_text(variant, source="ch-3.1").world
        )
        self.assertNotEqual(d_base, d_variant)


if __name__ == "__main__":
    unittest.main()
