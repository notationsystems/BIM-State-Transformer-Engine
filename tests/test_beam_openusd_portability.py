"""Complete beam-chain transport and continuation through native OpenUSD."""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import os
import tempfile
import unittest

from gat.adapters.openusd import openusd_available
from gat.demo.beam_openusd_portability import run_demo


@unittest.skipUnless(openusd_available(), "optional usd-core runtime is not installed")
class BeamOpenUsdPortabilityTests(unittest.TestCase):
    def test_signed_beam_chain_authenticates_and_continues_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with redirect_stdout(StringIO()):
                summary = run_demo(directory, quiet=True)

            checkpoint = summary["checkpoint"]
            continuation = summary["continuation"]
            assurance = summary["assurance"]
            self.assertEqual(checkpoint["verdict"], "VIOLATED")
            self.assertTrue(checkpoint["carrier_signature_verified"])
            self.assertTrue(checkpoint["world_computationally_equivalent"])
            self.assertTrue(checkpoint["ledger_exactly_preserved"])
            self.assertEqual(continuation["verdict"], "SATISFIED")
            self.assertTrue(
                continuation["receiving_runtime_reproduced_checkpoint_computation"]
            )
            self.assertTrue(
                continuation["separate_process_world_matches_uninterrupted"]
            )
            self.assertTrue(
                continuation["separate_process_ledger_matches_uninterrupted"]
            )
            self.assertTrue(assurance["openusd_signature_verified"])
            self.assertEqual(
                assurance["openusd_trust_source"],
                "explicit-demo-resume-request",
            )
            self.assertFalse(assurance["material_certificate_signature_verified"])
            self.assertFalse(assurance["material_certificate_issuer_trust_verified"])
            self.assertFalse(assurance["may_authorize"])
            self.assertTrue(
                os.path.exists(
                    os.path.join(directory, "beam_checkpoint_signed.usdc")
                )
            )
            self.assertTrue(
                os.path.exists(
                    os.path.join(directory, "beam_continued_resumed.usdc")
                )
            )
            self.assertTrue(
                os.path.exists(
                    os.path.join(directory, "beam_openusd_portability_summary.json")
                )
            )


if __name__ == "__main__":
    unittest.main()
