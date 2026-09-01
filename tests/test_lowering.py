"""Tests for gat/adapters/ifc/lower.py against the shipped demo model.

Machine-checks the lowered counts (10 entities, 24 raw, 39 derived,
18 relationship edges), the external-boundary attribute, the sigma
policy (defaults plus GAT_Uncertainty / GAT_Material overrides), and the
LoweringError paths, driven through the public GatSession entry points.
"""

from __future__ import annotations

import os
import unittest

import gat
from gat.errors import LoweringError
from gat.ir.core import RelKind, Role
from gat.session import GatSession

DEMO_PATH = os.path.join(os.path.dirname(gat.__file__), "demo", "model.ifc")

ENVELOPE_WALLS = ("Wall-South", "Wall-North", "Wall-West", "Wall-East")

_cache: dict[str, object] = {}


def demo_session() -> GatSession:
    """Load the demo model once for the whole module (it is read-only)."""
    if "session" not in _cache:
        _cache["session"] = GatSession.load_ifc(DEMO_PATH)
    return _cache["session"]


def demo_text() -> str:
    if "text" not in _cache:
        with open(DEMO_PATH, "r", encoding="utf-8") as fh:
            _cache["text"] = fh.read()
    return _cache["text"]


def edited_demo(old: str, new: str, case: unittest.TestCase) -> str:
    text = demo_text()
    case.assertIn(old, text, msg="demo model text changed; update this test")
    return text.replace(old, new)


class TestDemoCounts(unittest.TestCase):
    def setUp(self):
        self.module = demo_session().world.module

    def test_entity_count(self):
        # 1 storey + 5 walls + 1 opening + 1 door + 2 spaces = 10
        # (IfcProject / IfcBuilding are not lowered to IR entities).
        self.assertEqual(len(self.module.entities), 10)

    def test_entity_class_breakdown(self):
        by_class: dict[str, int] = {}
        for eid in self.module.entities:
            by_class[eid.ifc_class] = by_class.get(eid.ifc_class, 0) + 1
        self.assertEqual(
            by_class,
            {
                "IfcBuildingStorey": 1,
                "IfcWall": 5,
                "IfcOpeningElement": 1,
                "IfcDoor": 1,
                "IfcSpace": 2,
            },
        )

    def test_raw_and_derived_counts(self):
        # RAW: storey ClearHeight (1) + 5 walls x (Length, Width, UnitCost)
        # (15) + opening W,H (2) + door W,H (2) + 2 spaces x (L, W) (4) = 24.
        # DERIVED: opening Area (1) + door Area (1)
        # + 5 walls x (Height, GrossSideArea, NetSideArea, GrossVolume,
        #   NetVolume, Cost) (30) + 2 spaces x (FloorArea, Volume) (4)
        # + storey (TotalWallNetVolume, TotalWallCost, TotalFloorArea) (3)
        # = 39.
        roles = [slot.role for slot in self.module.all_slots()]
        self.assertEqual(roles.count(Role.RAW), 24)
        self.assertEqual(roles.count(Role.DERIVED), 39)

    def test_binding_agrees_with_module(self):
        binding = demo_session().world.binding
        self.assertEqual(binding.n_raw, 24)
        self.assertEqual(binding.n_full, 24 + 39)

    def test_rel_count_and_kinds(self):
        # AGGREGATES: storey -> Office-A, storey -> Office-B          = 2
        #   (project->building and building->storey drop out: those
        #    entities are not lowered)
        # CONTAINS: storey contains 5 walls + door                    = 6
        # VOIDS: Opening-1 -> Wall-Party                              = 1
        # FILLS: Door-1 -> Opening-1                                  = 1
        # BOUNDS: 8 IfcRelSpaceBoundary records                       = 8
        self.assertEqual(len(self.module.rels), 18)
        by_kind: dict[RelKind, int] = {}
        for rel in self.module.rels:
            by_kind[rel.kind] = by_kind.get(rel.kind, 0) + 1
        self.assertEqual(
            by_kind,
            {
                RelKind.AGGREGATES: 2,
                RelKind.CONTAINS: 6,
                RelKind.VOIDS: 1,
                RelKind.FILLS: 1,
                RelKind.BOUNDS: 8,
            },
        )


class TestExternalAttribute(unittest.TestCase):
    def setUp(self):
        self.session = demo_session()
        self.module = self.session.world.module

    def test_envelope_walls_marked_external(self):
        for name in ENVELOPE_WALLS:
            eid = self.session.entity_by_name(name)
            self.assertIs(
                self.module.entities[eid].attrs.get("external"), True, msg=name
            )

    def test_party_wall_not_external(self):
        eid = self.session.entity_by_name("Wall-Party")
        self.assertFalse(self.module.entities[eid].attrs.get("external", False))

    def test_no_other_entity_marked_external(self):
        external = {
            entity.name
            for entity in self.module.entities.values()
            if entity.attrs.get("external")
        }
        self.assertEqual(external, set(ENVELOPE_WALLS))


class TestSigmaPolicy(unittest.TestCase):
    def setUp(self):
        self.session = demo_session()
        self.module = self.session.world.module

    def slot(self, entity_name: str, qty: str):
        return self.module.slot(self.session.var(entity_name, qty))

    def test_party_wall_length_sigma_override(self):
        # GAT_Uncertainty pset (#191/#192) carries LengthSigma = 0.004.
        self.assertEqual(self.slot("Wall-Party", "Length").prior_sigma, 0.004)

    def test_party_wall_width_uses_default(self):
        # No WidthSigma override -> default (IfcWall, Width) = 0.002.
        self.assertEqual(self.slot("Wall-Party", "Width").prior_sigma, 0.002)

    def test_other_walls_use_defaults(self):
        for name in ENVELOPE_WALLS:
            self.assertEqual(self.slot(name, "Length").prior_sigma, 0.005, msg=name)
            self.assertEqual(self.slot(name, "Width").prior_sigma, 0.002, msg=name)

    def test_non_wall_defaults(self):
        self.assertEqual(self.slot("Level 1", "ClearHeight").prior_sigma, 0.01)
        self.assertEqual(self.slot("Opening-1", "Width").prior_sigma, 0.005)
        self.assertEqual(self.slot("Door-1", "Height").prior_sigma, 0.003)
        self.assertEqual(self.slot("Office-A", "Length").prior_sigma, 0.005)

    def test_unit_cost_slots_on_all_walls_with_relative_sigma(self):
        # No wall in the demo carries a UnitCostSigma property, so every
        # UnitCost sigma is exactly 8% of the mean:
        #   envelope walls: 320.0 * 0.08 = 25.6
        #   party wall:     280.0 * 0.08 = 22.4
        for name in ENVELOPE_WALLS:
            slot = self.slot(name, "UnitCost")
            self.assertEqual(slot.prior_mu, 320.0, msg=name)
            self.assertAlmostEqual(slot.prior_sigma, 25.6, delta=1e-12, msg=name)
        party = self.slot("Wall-Party", "UnitCost")
        self.assertEqual(party.prior_mu, 280.0)
        self.assertAlmostEqual(party.prior_sigma, 22.4, delta=1e-12)

    def test_unit_cost_sigma_from_material_pset_wins(self):
        # Edited copy: Wall-South's GAT_Material carries UnitCostSigma=12.5,
        # which must replace the 8% relative default for that wall only.
        text = edited_demo(
            "#115=IFCPROPERTYSET('GATPST0000000000000115',$,'GAT_Material',$,(#114));",
            "#115=IFCPROPERTYSET('GATPST0000000000000115',$,'GAT_Material',$,(#114,#117));\n"
            "#117=IFCPROPERTYSINGLEVALUE('UnitCostSigma',$,IFCREAL(12.5),$);",
            self,
        )
        session = GatSession.from_text(text)
        module = session.world.module
        south = module.slot(session.var("Wall-South", "UnitCost"))
        self.assertEqual(south.prior_sigma, 12.5)
        # Another wall keeps the relative default.
        north = module.slot(session.var("Wall-North", "UnitCost"))
        self.assertAlmostEqual(north.prior_sigma, 25.6, delta=1e-12)


class TestLoweringErrors(unittest.TestCase):
    def test_wall_missing_length_quantity(self):
        # Drop #110 (Wall-South Length) from its element quantity list.
        text = edited_demo(
            "'Qto_WallBaseQuantities',$,$,(#110,#111));",
            "'Qto_WallBaseQuantities',$,$,(#111));",
            self,
        )
        with self.assertRaises(LoweringError) as ctx:
            GatSession.from_text(text)
        self.assertIn("Length", str(ctx.exception))
        self.assertIn("Wall-South", str(ctx.exception))

    def test_two_storeys_rejected(self):
        extra = (
            "#500=IFCBUILDINGSTOREY('GATSTY0000000000000500',$,'Level 2',$,$,$,$,$,$,$);\n"
            "#501=IFCQUANTITYLENGTH('ClearHeight',$,$,3.);\n"
            "#502=IFCELEMENTQUANTITY('GATQTO0000000000000502',$,'Qto_StoreyBaseQuantities',$,$,(#501));\n"
            "#503=IFCRELDEFINESBYPROPERTIES('GATRDP0000000000000503',$,$,$,(#500),#502);\n"
        )
        text = edited_demo(
            "#400=IFCRELAGGREGATES", extra + "#400=IFCRELAGGREGATES", self
        )
        with self.assertRaises(LoweringError) as ctx:
            GatSession.from_text(text)
        self.assertIn("exactly one storey", str(ctx.exception))
        self.assertIn("2", str(ctx.exception))

    def test_millimetre_length_unit_is_normalized(self):
        text = edited_demo(
            "#2=IFCSIUNIT(*,.LENGTHUNIT.,$,.METRE.);",
            "#2=IFCSIUNIT(*,.LENGTHUNIT.,.MILLI.,.METRE.);",
            self,
        )
        session = GatSession.from_text(text)
        self.assertEqual(session.world.module.meta["ifc_length_unit"], "MILLI METRE")
        self.assertEqual(
            session.world.module.meta["ifc_length_scale_to_metres"], "0.001"
        )
        self.assertAlmostEqual(
            session.world.belief.mean(session.var("Level 1", "ClearHeight")),
            0.003,
            delta=1e-15,
        )


if __name__ == "__main__":
    unittest.main()
