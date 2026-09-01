"""Closed JSON contract for Blender, CI, and future Kit clients."""

from __future__ import annotations

import hashlib
import os
import tempfile
import unittest

import gat.demo
import numpy as np
from gat.adapters.openusd import generate_openusd_keypair, openusd_available
from gat.engine.transform import ObserveLinearized
from gat.headless import REQUEST_FORMAT, RESPONSE_FORMAT, handle_request
from gat.session import GatSession


MODEL = os.path.join(os.path.dirname(gat.demo.__file__), "model.ifc")


class HeadlessContractTests(unittest.TestCase):
    def request(self, operation: str, payload: dict) -> dict:
        return {
            "format": REQUEST_FORMAT,
            "request_id": f"test-{operation}",
            "operation": operation,
            "state": {"kind": "ifc", "path": MODEL},
            "payload": payload,
        }

    @staticmethod
    def ref(entity_name: str, quantity: str) -> dict[str, str]:
        return {"entity_name": entity_name, "quantity": quantity}

    def opening_payload(self) -> dict:
        return {
            "case_id": "opening-fit-1",
            "workflow": "OPENING_VERIFICATION",
            "subject": "Door-1 into Opening-1",
            "checks": [
                {
                    "kind": "difference",
                    "check_id": "width",
                    "lhs": self.ref("Opening-1", "Width"),
                    "rhs": self.ref("Door-1", "Width"),
                    "minimum_margin": 0.05,
                    "confidence": 0.95,
                    "label": "opening width fit",
                },
                {
                    "kind": "difference",
                    "check_id": "height",
                    "lhs": self.ref("Opening-1", "Height"),
                    "rhs": self.ref("Door-1", "Height"),
                    "minimum_margin": 0.05,
                    "confidence": 0.95,
                    "label": "opening height fit",
                },
            ],
        }

    def test_summary_is_stable_and_verified(self) -> None:
        response = handle_request(self.request("summary", {}))
        self.assertEqual(response["format"], RESPONSE_FORMAT)
        self.assertEqual(response["result"]["entities"], 10)
        self.assertEqual(response["result"]["raw_variables"], 24)
        self.assertTrue(response["result"]["verification"]["passed"])

    def test_opening_workflow_requests_as_built_evidence(self) -> None:
        response = handle_request(
            self.request("acceptance", self.opening_payload())
        )
        result = response["result"]
        self.assertEqual(result["disposition"], "REQUEST_EVIDENCE")
        self.assertEqual(result["uncovered_check_ids"], ["width", "height"])
        self.assertFalse(result["may_authorize"])

    def test_headless_receipt_must_exist_in_the_loaded_state_ledger(self) -> None:
        payload = self.opening_payload()
        world_digest = GatSession.load_ifc(MODEL).world.digest()
        payload["evidence_receipts"] = [
            {
                "receipt_id": "forged",
                "evidence_kind": "calibrated-scan-clearance-likelihood",
                "evidence_digest": hashlib.sha256(b"evidence").hexdigest(),
                "prior_world_digest": hashlib.sha256(b"prior").hexdigest(),
                "result_world_digest": world_digest,
                "calibration_id": "survey-control-A",
                "check_ids": ["width", "height"],
                "ledger_event_hash": hashlib.sha256(b"not-in-ledger").hexdigest(),
                "verification_passed": True,
            }
        ]
        with self.assertRaisesRegex(ValueError, "not in the state ledger"):
            handle_request(self.request("acceptance", payload))

    def test_change_impact_is_read_only(self) -> None:
        response = handle_request(
            self.request(
                "change_impact",
                {
                    "change": {
                        "op": "set_parameter",
                        "target": self.ref("Level 1", "ClearHeight"),
                        "value": 3.4,
                        "design_sigma": 0.01,
                    }
                },
            )
        )
        result = response["result"]
        self.assertEqual(result["disposition"], "ADMISSIBLE")
        self.assertEqual(len(result["affected"]), 34)
        self.assertNotEqual(
            result["candidate_world_digest"], response["world_digest"]
        )

    def test_infeasible_change_is_reported_not_committed(self) -> None:
        response = handle_request(
            self.request(
                "change_impact",
                {
                    "change": {
                        "op": "set_parameter",
                        "target": self.ref("Opening-1", "Height"),
                        "value": 3.6,
                        "design_sigma": 0.005,
                    }
                },
            )
        )
        self.assertEqual(response["result"]["disposition"], "BLOCKED")
        self.assertTrue(response["result"]["verification"]["failures"])

    def test_unknown_fields_and_operations_fail_closed(self) -> None:
        request = self.request("summary", {})
        request["execute_python"] = "pass"
        with self.assertRaisesRegex(ValueError, "fields differ"):
            handle_request(request)
        with self.assertRaisesRegex(ValueError, "unsupported headless operation"):
            handle_request(self.request("shell", {}))

        request = self.request("summary", {})
        request["state"]["trusted_public_keys"] = {"request-chosen-key": "AAAA"}
        with self.assertRaisesRegex(ValueError, "fields differ"):
            handle_request(request)

    @unittest.skipUnless(openusd_available(), "optional OpenUSD runtime not installed")
    def test_signed_ledger_bound_evidence_can_close_headless_acceptance(self) -> None:
        session = GatSession.load_ifc(MODEL)
        before = session.world.digest()
        target = session.var("Level 1", "ClearHeight")
        row = np.zeros(session.world.binding.n_raw, dtype=np.float64)
        row[session.world.binding.raw_index.row(target)] = 1.0
        evidence_digest = hashlib.sha256(b"controlled-clear-height-scan").hexdigest()
        observation = ObserveLinearized(
            row=row,
            predicted=session.world.belief.mean(target),
            observed=session.world.belief.mean(target),
            noise_sigma=0.001,
            raw_targets=(target,),
            expected_raw_order=session.world.binding.raw_index.vars,
            expected_belief_digest=session.world.belief.digest(),
            expected_world_digest=before,
            evidence_digest=evidence_digest,
            label="controlled clear-height support",
        )
        result = session.run(
            observation,
            provenance={
                "evidence_kind": "calibrated-scan-clearance-likelihood",
                "calibration_id": "survey-control-A",
                "check_ids": ["route-clearance"],
            },
        )
        event = session.ledger.events[-1]
        key = generate_openusd_keypair("headless-test-key")

        with tempfile.TemporaryDirectory() as directory:
            carrier = os.path.join(directory, "accepted-state.usda")
            session.export_openusd(carrier, signing_key=key)
            response = handle_request(
                {
                    "format": REQUEST_FORMAT,
                    "request_id": "signed-acceptance",
                    "operation": "acceptance",
                    "state": {
                        "kind": "openusd",
                        "path": carrier,
                        "require_signature": True,
                    },
                    "payload": {
                        "case_id": "clear-route-1",
                        "workflow": "AS_BUILT_CLEARANCE",
                        "subject": "clear duct route",
                        "checks": [
                            {
                                "kind": "clearance",
                                "check_id": "route-clearance",
                                "proposal": {
                                    "origin": [4.0, 1.8, 3.55],
                                    "angle": 0.0,
                                    "extents": [3.0, 0.4, 0.4],
                                },
                                "required_clearance": 0.05,
                                "confidence": 0.95,
                                "position_sigma": 0.002,
                                "label": "clear duct route",
                            }
                        ],
                        "evidence_receipts": [
                            {
                                "receipt_id": "controlled-scan-1",
                                "evidence_kind": "calibrated-scan-clearance-likelihood",
                                "evidence_digest": evidence_digest,
                                "prior_world_digest": before,
                                "result_world_digest": result.world.digest(),
                                "calibration_id": "survey-control-A",
                                "check_ids": ["route-clearance"],
                                "ledger_event_hash": event.event_hash,
                                "verification_passed": True,
                            }
                        ],
                    },
                },
                trusted_public_keys={key.key_id: key.public_key},
            )

        self.assertEqual(response["result"]["disposition"], "ACCEPT")
        self.assertTrue(response["result"]["may_authorize"])


if __name__ == "__main__":
    unittest.main()
