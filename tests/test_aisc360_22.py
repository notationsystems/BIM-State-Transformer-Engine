"""Independent AISC design-example validation for the bounded F2-1 check."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from gat.engineering.aisc360_22 import (
    AISC360_22_F2_LRFD_METHOD,
    AISC360_22_F2_LRFD_ORACLE_ID,
    AISC360_22_F2_LRFD_ORACLE_RECORD_DIGEST,
    AISC360_22_F2_LRFD_VALIDATION_PROFILE_DIGEST,
    Aisc36022F2YieldingCheck,
    Aisc36022Verdict,
    evaluate_aisc36022_f2_yielding,
)


ORACLE = Path(__file__).parents[1] / "validation" / "aisc360-22-f1-1b-v1.json"
KSI_TO_MPA = 6.894757293168361
IN3_TO_M3 = 0.0254**3
KIP_FT_TO_N_M = 1000.0 * 4.4482216152605 * 0.3048


class Aisc36022F2OracleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.oracle = json.loads(ORACLE.read_text(encoding="utf-8"))
        inputs = self.oracle["published_inputs"]
        self.check = Aisc36022F2YieldingCheck(
            yield_strength_mpa=inputs["yield_strength_ksi"] * KSI_TO_MPA,
            plastic_section_modulus_major_m3=(
                inputs["plastic_section_modulus_major_in3"] * IN3_TO_M3
            ),
            factored_demand_n_m=(
                inputs["factored_demand_kip_ft"] * KIP_FT_TO_N_M
            ),
        )

    def test_pinned_official_example_reproduces_published_lrfd_results(self) -> None:
        result = evaluate_aisc36022_f2_yielding(self.check)
        expected = self.oracle["published_results"]
        tolerance = self.oracle["comparison"]["absolute_tolerance_kip_ft"]

        self.assertEqual(self.oracle["oracle_id"], AISC360_22_F2_LRFD_ORACLE_ID)
        canonical_oracle = json.dumps(
            self.oracle,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        self.assertEqual(
            hashlib.sha256(canonical_oracle).hexdigest(),
            AISC360_22_F2_LRFD_ORACLE_RECORD_DIGEST,
        )
        self.assertAlmostEqual(
            result.nominal_moment_n_m / KIP_FT_TO_N_M,
            expected["nominal_moment_kip_ft"],
            delta=tolerance,
        )
        self.assertAlmostEqual(
            result.available_moment_n_m / KIP_FT_TO_N_M,
            expected["available_moment_kip_ft"],
            delta=tolerance,
        )
        self.assertEqual(result.verdict.value, expected["verdict"])
        self.assertLess(result.utilization, 1.0)
        self.assertEqual(result.to_dict()["method"], AISC360_22_F2_LRFD_METHOD)
        self.assertEqual(
            result.to_dict()["validation_profile_digest"],
            AISC360_22_F2_LRFD_VALIDATION_PROFILE_DIGEST,
        )

    def test_scope_guard_rejects_unbraced_or_noncompact_members(self) -> None:
        for override in (
            {"bracing": "unbraced"},
            {"section_classification": "noncompact"},
            {"section_property_kind": "elastic-section-modulus-sx"},
        ):
            with self.subTest(override=override):
                with self.assertRaisesRegex(ValueError, "outside the implemented"):
                    Aisc36022F2YieldingCheck(
                        self.check.yield_strength_mpa,
                        self.check.plastic_section_modulus_major_m3,
                        self.check.factored_demand_n_m,
                        **override,
                    )

    def test_demand_above_available_strength_fails(self) -> None:
        passing = evaluate_aisc36022_f2_yielding(self.check)
        failing = evaluate_aisc36022_f2_yielding(
            Aisc36022F2YieldingCheck(
                self.check.yield_strength_mpa,
                self.check.plastic_section_modulus_major_m3,
                passing.available_moment_n_m + 1.0,
            )
        )

        self.assertEqual(failing.verdict, Aisc36022Verdict.FAIL)
        self.assertGreater(failing.utilization, 1.0)


if __name__ == "__main__":
    unittest.main()
