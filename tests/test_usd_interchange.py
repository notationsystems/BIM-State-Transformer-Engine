"""State-space interchange through OpenUSD: D(E(S)) ~= S, and continuation."""

from __future__ import annotations

import filecmp
import os
import tempfile
import unittest

from gat.adapters.usd_io import load_usd, state_equivalence
from gat.engine.transform import ObserveQuantity, SetParameter
from gat.errors import SpfParseError
from gat.session import GatSession

MODEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "gat", "demo", "model.ifc",
)


class UsdInterchangeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.session = GatSession.load_ifc(MODEL)
        vol_a = cls.session.var("Office-A", "Volume")
        cls.session.run(ObserveQuantity.single(vol_a, 59.4, 0.05))
        cls.usd_path = os.path.join(cls.tmp.name, "state.usda")
        cls.session.export_usd(cls.usd_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_invariant_suite_passes(self) -> None:
        world, _ = load_usd(self.usd_path)
        report = state_equivalence(self.session.world, world)
        for check in report.checks:
            self.assertTrue(check.passed, f"I_{check.name} failed: {check.detail}")
        self.assertTrue(report.equivalent)

    def test_belief_restored_bitwise(self) -> None:
        world, _ = load_usd(self.usd_path)
        self.assertEqual(
            world.belief.mu.tobytes(), self.session.world.belief.mu.tobytes()
        )
        self.assertEqual(
            world.belief.sigma.tobytes(), self.session.world.belief.sigma.tobytes()
        )

    def test_reconstructed_world_verifies(self) -> None:
        reconstructed = GatSession.load_usd(self.usd_path)
        self.assertTrue(reconstructed.verify().passed)

    def test_provenance_events_carried(self) -> None:
        _, trace = load_usd(self.usd_path)
        self.assertGreaterEqual(len(trace), 2)  # compile + observe
        self.assertEqual(trace[0]["stage"], "compile")

    def test_continuation_is_bitwise_identical(self) -> None:
        # Continuous runtime: T2 on the original session's world.
        continuous = GatSession.load_ifc(MODEL)
        vol_a = continuous.var("Office-A", "Volume")
        continuous.run(ObserveQuantity.single(vol_a, 59.4, 0.05))
        ch = continuous.var("Level 1", "ClearHeight")
        continuous.run(SetParameter(ch, 3.4, design_sigma=0.01))

        # Transferred runtime: reconstruct, then the same T2.
        transferred = GatSession.load_usd(self.usd_path)
        ch_b = transferred.var("Level 1", "ClearHeight")
        transferred.run(SetParameter(ch_b, 3.4, design_sigma=0.01))

        self.assertEqual(
            transferred.world.full.mu.tobytes(), continuous.world.full.mu.tobytes()
        )
        self.assertEqual(
            transferred.world.full.sigma.tobytes(),
            continuous.world.full.sigma.tobytes(),
        )
        self.assertTrue(
            state_equivalence(continuous.world, transferred.world).equivalent
        )

    def test_export_is_deterministic(self) -> None:
        other = os.path.join(self.tmp.name, "state2.usda")
        # A fresh session replays the same program; its trace digests match,
        # so the stage bytes must too.
        session = GatSession.load_ifc(MODEL)
        vol_a = session.var("Office-A", "Volume")
        session.run(ObserveQuantity.single(vol_a, 59.4, 0.05))
        session.export_usd(other)
        self.assertTrue(filecmp.cmp(self.usd_path, other, shallow=False))

    def test_unsupported_format_rejected(self) -> None:
        with open(self.usd_path, encoding="utf-8") as fh:
            text = fh.read()
        bad = os.path.join(self.tmp.name, "bad.usda")
        with open(bad, "w", encoding="utf-8") as fh:
            fh.write(text.replace("gat-usd v0", "gat-usd v999"))
        with self.assertRaises(SpfParseError):
            load_usd(bad)


if __name__ == "__main__":
    unittest.main()
