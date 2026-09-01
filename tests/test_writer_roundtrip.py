"""Round-trip tests for gat/adapters/ifc/writer.py.

Pipeline under test: load the demo model, run a SetParameter intervention
followed by an ObserveQuantity conditioning, export to SPF, reload, and
check that every raw mean and every raw sigma (via the GAT_Posterior pset)
survives the trip.  Also covers determinism (byte-identical re-export),
the in-place patching of IfcQuantity* records, and ``format_real``.

stdlib unittest only; fully deterministic (no randomness anywhere).
"""

from __future__ import annotations

import filecmp
import os
import re
import tempfile
import unittest

from gat.adapters.ifc.writer import format_real
from gat.engine.transform import ObserveQuantity, SetParameter
from gat.session import GatSession

DEMO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "gat", "demo", "model.ifc",
)


class FormatRealTests(unittest.TestCase):
    def test_plain_decimal_form(self) -> None:
        # repr(3.4) == '3.4' has a dot already: emitted verbatim.
        self.assertEqual(format_real(3.4), "3.4")
        self.assertEqual(float(format_real(3.4)), 3.4)

    def test_integral_value_gets_trailing_dot(self) -> None:
        # STEP reals need a dot: 3.0 -> '3.0' (repr already carries it),
        # and the serializer never emits a bare integer literal.
        text = format_real(3.0)
        self.assertIn(".", text)
        self.assertEqual(float(text), 3.0)

    def test_exponent_form_has_dot_and_uppercase_e(self) -> None:
        # repr(1e-07) == '1e-07': mantissa '1' gains a dot, exponent is
        # normalized through int() and joined with uppercase 'E'.
        text = format_real(1e-07)
        self.assertEqual(text, "1.E-7")
        mantissa, _, exponent = text.partition("E")
        self.assertIn(".", mantissa)
        self.assertNotIn("e", text)  # uppercase E only
        self.assertEqual(float(text), 1e-07)

    def test_shortest_roundtrip_on_tricky_values(self) -> None:
        tricky = [
            0.1 + 0.2,        # 0.30000000000000004 - classic non-exact sum
            1e-7,             # exponent form
            123456.789,       # many significant digits
            9.2,              # demo wall length (non-dyadic)
            1.0 / 3.0,        # 17 significant digits needed
            5e-324,           # smallest subnormal
            1.7976931348623157e308,  # largest finite double
            -2.5,
            0.0,
            -0.0,
            1e300,
            2.675,            # rounds "wrong" in naive %.2f formatting
        ]
        for value in tricky:
            with self.subTest(value=value):
                text = format_real(value)
                # float(text) == value must hold *exactly*: format_real is
                # built on repr, Python's shortest round-tripping form.
                self.assertEqual(float(text), value)


class WriterRoundtripTests(unittest.TestCase):
    """Export after SetParameter + ObserveQuantity; reload; compare."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.session = GatSession.load_ifc(DEMO)
        cls.clear_height = cls.session.var("Level 1", "ClearHeight")
        volume = cls.session.var("Office-A", "Volume")
        # Design intervention then a derived-quantity observation.  The
        # observation conditions on Office-A Volume, which moves the raw
        # means of ClearHeight and Office-A Length/Width through their
        # correlations - all of them backed by IfcQuantityLength records,
        # so the export must carry the *posterior* means.
        cls.session.run(SetParameter(cls.clear_height, 3.4, design_sigma=0.01))
        cls.session.run(ObserveQuantity.single(volume, 67.5, noise_sigma=0.05))
        cls.out1 = os.path.join(cls.tmp.name, "out1.ifc")
        cls.out2 = os.path.join(cls.tmp.name, "out2.ifc")
        cls.patched, cls.appended = cls.session.export_ifc(cls.out1)
        cls.session.export_ifc(cls.out2)
        cls.reloaded = GatSession.load_ifc(cls.out1)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_export_counts(self) -> None:
        # 19 quantity-backed raw slots patched (24 raw vars minus the 5
        # UnitCost slots, which come from a property set, not a quantity).
        self.assertEqual(self.patched, 19)
        # Appended: per entity with raw slots, one property per raw slot
        # plus a pset and a rel.  storey(1)+2, 5 walls(2 or 3 raw each:
        # S/N/E/W have Length,Width,UnitCost=3; Party has 3)+2 each,
        # opening(2)+2, door(2)+2, 2 spaces(2)+2 each
        # = (1+2) + 5*(3+2) + (2+2) + (2+2) + 2*(2+2) = 3+25+4+4+8 = 44.
        self.assertEqual(self.appended, 44)

    def test_every_raw_mean_roundtrips(self) -> None:
        raw_vars = self.session.world.binding.raw_index.vars
        self.assertEqual(len(raw_vars), 24)
        for var in raw_vars:
            with self.subTest(var=str(var)):
                self.assertAlmostEqual(
                    self.reloaded.world.belief.mean(var),
                    self.session.world.belief.mean(var),
                    delta=1e-9,
                )

    def test_every_raw_sigma_recovered_via_gat_posterior(self) -> None:
        # The reloaded priors take their sigma from the appended
        # GAT_Posterior pset, so posterior marginal stds round-trip.
        for var in self.session.world.binding.raw_index.vars:
            with self.subTest(var=str(var)):
                self.assertAlmostEqual(
                    self.reloaded.world.belief.std(var),
                    self.session.world.belief.std(var),
                    delta=1e-9,
                )

    def test_export_twice_is_byte_identical(self) -> None:
        self.assertTrue(filecmp.cmp(self.out1, self.out2, shallow=False))

    def test_patched_quantity_record_carries_posterior_mean(self) -> None:
        # #29 is the storey ClearHeight IfcQuantityLength; the writer must
        # rewrite its value slot with the posterior mean (which the
        # observation moved off the SetParameter value 3.4).
        with open(self.out1, "r", encoding="utf-8") as fh:
            text = fh.read()
        match = re.search(
            r"#29=IFCQUANTITYLENGTH\('ClearHeight',\$,\$,([^)]+)\);", text
        )
        self.assertIsNotNone(match)
        written = float(match.group(1))
        posterior_mean = self.session.world.belief.mean(self.clear_height)
        # format_real round-trips exactly, so equality holds to the bit;
        # assert with an explicit (tight) tolerance anyway.
        self.assertAlmostEqual(written, posterior_mean, delta=1e-12)
        # And it is genuinely the posterior, not the intervened value.
        self.assertNotAlmostEqual(written, 3.4, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
