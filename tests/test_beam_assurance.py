"""Reference evidence -> state -> computation -> decision -> verification chain."""

from __future__ import annotations

import hashlib
import os
import tempfile
import unittest

import gat.demo
from gat.engine.transform import ObserveQuantity
from gat.engineering import (
    BeamBendingCheck,
    BeamBendingEvaluator,
    beam_assessment_record,
    explain_beam_decision_change,
)
from gat.evidence import CalibratedObservation, EvidenceKind
from gat.errors import LoweringError
from gat.ledger import replay_ledger
from gat.proof_manifest import (
    NumericContract,
    create_computation_proof_manifest,
    verify_computation_proof_manifest,
)
from gat.session import GatSession
from gat.state_snapshot import computational_equivalence


BEAM_MODEL = os.path.join(os.path.dirname(gat.demo.__file__), "beam_model.ifc")
CERTIFICATE = os.path.join(
    os.path.dirname(gat.demo.__file__), "material_certificate.json"
)


def material_evidence(session: GatSession) -> CalibratedObservation:
    with open(CERTIFICATE, "rb") as stream:
        source = stream.read()
    return CalibratedObservation.from_source_bytes(
        "MAT-CERT-B1-325",
        session.var("Beam-B1", "YieldStrengthMPa"),
        EvidenceKind.MEASURED,
        325.0,
        2.0,
        "MPa",
        source,
        "coupon-test-material-certificate-v1",
        hashlib.sha256(b"coupon-test-calibration-v1").hexdigest(),
    )


class BeamLoweringTests(unittest.TestCase):
    def test_annotated_beam_lowers_closed_structural_contract(self) -> None:
        session = GatSession.load_ifc(BEAM_MODEL)
        beam = session.entity_by_name("Beam-B1")
        entity = session.world.module.entities[beam]

        self.assertEqual(entity.attrs["structural_method"], "elastic-section-yield-v1")
        self.assertEqual(entity.attrs["resistance_factor"], 0.9)
        self.assertEqual(
            set(entity.slots),
            {
                "Length",
                "YieldStrengthMPa",
                "SectionModulusM3",
                "NominalMomentCapacity",
                "DesignMomentCapacity",
            },
        )
        self.assertEqual(entity.slots["YieldStrengthMPa"].unit.value, "MPa")
        self.assertEqual(entity.slots["DesignMomentCapacity"].unit.value, "N*m")
        self.assertAlmostEqual(
            session.world.full.mean(session.var("Beam-B1", "DesignMomentCapacity")),
            315_000.0,
        )
        self.assertTrue(session.verify().passed)

    def test_unannotated_ifc_beam_remains_opaque(self) -> None:
        with open(BEAM_MODEL, "r", encoding="utf-8") as stream:
            text = stream.read()
        text = text.replace("'GAT_Structural'", "'External_Structural_Data'")
        session = GatSession.from_text(text, "unannotated-beam.ifc")

        self.assertFalse(
            any(entity.id.ifc_class == "IfcBeam" for entity in session.world.module.entities.values())
        )
        self.assertTrue(session.verify().passed)

    def test_incomplete_opt_in_structural_contract_fails_closed(self) -> None:
        with open(BEAM_MODEL, "r", encoding="utf-8") as stream:
            text = stream.read()
        text = text.replace("'SectionModulusM3Sigma'", "'UnknownSigma'")
        with self.assertRaisesRegex(LoweringError, "SectionModulusM3"):
            GatSession.from_text(text, "incomplete-beam.ifc")


class CalibratedEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session = GatSession.load_ifc(BEAM_MODEL)

    def test_evidence_identity_includes_epistemic_kind_and_source(self) -> None:
        measured = material_evidence(self.session)
        inferred = CalibratedObservation(
            measured.evidence_id,
            measured.subject,
            EvidenceKind.INFERRED,
            measured.observed_value,
            measured.noise_sigma,
            measured.unit,
            measured.source_digest,
            measured.method,
            measured.calibration_digest,
        )

        self.assertNotEqual(measured.digest(), inferred.digest())
        self.assertEqual(measured.provenance()["evidence"]["kind"], "MEASURED")

    def test_unit_mismatch_is_rejected_before_conditioning(self) -> None:
        evidence = material_evidence(self.session)
        wrong = CalibratedObservation(
            evidence.evidence_id,
            evidence.subject,
            evidence.kind,
            evidence.observed_value,
            evidence.noise_sigma,
            "Pa",
            evidence.source_digest,
            evidence.method,
            evidence.calibration_digest,
        )
        with self.assertRaisesRegex(ValueError, "canonical unit"):
            wrong.transformation(self.session.world)


class BeamAssuranceChainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session = GatSession.load_ifc(BEAM_MODEL)
        self.initial_world = self.session.world
        self.beam = self.session.entity_by_name("Beam-B1")
        self.check = BeamBendingCheck(
            self.beam,
            factored_demand_n_m=301_000.0,
            confidence=0.95,
            label="Beam-B1 factored bending",
        )
        self.evaluator = BeamBendingEvaluator()

    def test_material_evidence_flips_decision_and_records_exact_cause(self) -> None:
        prior = self.evaluator.evaluate(self.session.world, self.check)
        self.assertEqual(prior.verdict.value, "SATISFIED")
        self.assertAlmostEqual(prior.assessment.target_mean, 315_000.0)
        self.assertGreaterEqual(prior.assessment.p_satisfies, 0.95)
        self.session.record_assessment(
            beam_assessment_record(self.session.world, prior),
            provenance={"phase": "prior"},
        )

        evidence = material_evidence(self.session)
        before = self.session.world
        transition = self.session.run(
            evidence.transformation(self.session.world),
            provenance=evidence.provenance(),
        )
        revised = self.evaluator.evaluate(
            self.session.world,
            self.check,
            changed_inputs=transition.targets,
            affected_variables=transition.affected,
        )
        change = explain_beam_decision_change(
            before,
            self.session.world,
            evidence,
            transition,
            prior,
            revised,
        )
        record = beam_assessment_record(
            self.session.world,
            revised,
            evidence_digest=evidence.digest(),
            change=change,
        )
        self.session.record_assessment(record)

        fy = self.session.var("Beam-B1", "YieldStrengthMPa")
        self.assertEqual(transition.targets, (fy,))
        self.assertEqual(
            transition.affected,
            (
                self.session.var("Beam-B1", "NominalMomentCapacity"),
                self.session.var("Beam-B1", "DesignMomentCapacity"),
            ),
        )
        # 325 MPa is the observation; Bayesian conditioning produces the
        # posterior, rather than pretending the observation is exact state.
        self.assertAlmostEqual(self.session.world.belief.mean(fy), 326.47058823529414)
        self.assertAlmostEqual(self.session.world.belief.std(fy), 1.940285000290664)
        self.assertLess(
            self.session.world.belief.std(fy),
            before.belief.std(fy),
        )
        self.assertEqual(revised.verdict.value, "VIOLATED")
        self.assertLessEqual(revised.assessment.p_satisfies, 0.05)
        self.assertTrue(revised.recomputed)
        self.assertTrue(change.verdict_changed)
        self.assertIn("SATISFIED verdict became VIOLATED", change.reason)
        self.assertEqual(record.details["decision_change"]["reason"], change.reason)
        fy_capacity_covariance = next(
            item
            for item in change.covariance_changes
            if item["left"]["quantity"] == "YieldStrengthMPa"
            and item["right"]["quantity"] == "DesignMomentCapacity"
        )
        self.assertLess(
            fy_capacity_covariance["posterior_covariance"],
            fy_capacity_covariance["prior_covariance"],
        )
        self.assertTrue(change.changed_beliefs[0]["covariance_row_changed"])
        self.assertEqual(
            [event.kind for event in self.session.ledger.events],
            ["genesis", "assessment", "transition", "assessment"],
        )
        self.assertEqual(
            self.session.ledger.events[2].provenance["evidence_digest"],
            evidence.digest(),
        )
        replayed = replay_ledger(self.initial_world, self.session.ledger)
        self.assertEqual(replayed.world.digest(), self.session.world.digest())
        self.assertEqual(replayed.accepted, 1)
        self.assertEqual(replayed.non_state, 2)

        # Opaque fixture bytes exercise statement binding only.  Without an
        # SP1 backend verifier this must never become a verified-proof claim.
        proof_bytes = b"not-an-sp1-proof-reference-fixture"
        manifest = create_computation_proof_manifest(
            self.session.ledger,
            2,
            numeric_contract=NumericContract(
                "beam-binary64-v1",
                hashlib.sha256(b"beam-binary64-profile-v1").hexdigest(),
                "ieee754-binary64",
                "nearest-ties-to-even",
                "reject-nonfinite",
            ),
            model_contract_digest=revised.model_contract_digest,
            validation_profile_digest=revised.validation_profile_digest,
            computation_result_digest=revised.computation_digest,
            evidence_commitments=(evidence.digest(), evidence.source_digest),
            proof_system="sp1-test-fixture",
            proof_type="unverified-opaque-fixture",
            program_digest=hashlib.sha256(b"beam-guest-fixture").hexdigest(),
            verifying_key_digest=hashlib.sha256(b"beam-vkey-fixture").hexdigest(),
            proof_artifact=proof_bytes,
        )
        proof_report = verify_computation_proof_manifest(
            manifest,
            self.session.ledger,
            proof_bytes,
        )
        self.assertTrue(proof_report.bound)
        self.assertFalse(proof_report.proof_verified)
        self.assertEqual(
            manifest.computation_result_digest,
            revised.computation_digest,
        )

    def test_only_dependency_changes_rerun_engineering_computation(self) -> None:
        first = self.evaluator.evaluate(self.session.world, self.check)
        length = self.session.var("Beam-B1", "Length")
        unrelated = self.session.run(ObserveQuantity.single(length, 6.001, 0.001))
        rebound = self.evaluator.evaluate(
            self.session.world,
            self.check,
            changed_inputs=unrelated.targets,
            affected_variables=unrelated.affected,
        )

        self.assertFalse(rebound.recomputed)
        self.assertEqual(rebound.computation_digest, first.computation_digest)
        self.assertEqual(rebound.dependency_digest, first.dependency_digest)
        self.assertNotEqual(
            rebound.assessment.world_digest,
            first.assessment.world_digest,
        )

        evidence = material_evidence(self.session)
        changed = self.session.run(
            evidence.transformation(self.session.world),
            provenance=evidence.provenance(),
        )
        revised = self.evaluator.evaluate(
            self.session.world,
            self.check,
            changed_inputs=changed.targets,
            affected_variables=changed.affected,
        )
        self.assertTrue(revised.recomputed)
        self.assertNotEqual(revised.dependency_digest, first.dependency_digest)
        self.assertNotEqual(revised.computation_digest, first.computation_digest)

    def test_ifc_and_snapshot_transport_preserve_continuation_state(self) -> None:
        evidence = material_evidence(self.session)
        self.session.run(
            evidence.transformation(self.session.world),
            provenance=evidence.provenance(),
        )
        expected = self.evaluator.evaluate(self.session.world, self.check)

        with tempfile.TemporaryDirectory() as directory:
            ifc_path = os.path.join(directory, "beam_posterior.ifc")
            snapshot_path = os.path.join(directory, "beam_state.gat.json")
            self.session.export_ifc(ifc_path)
            self.session.export_snapshot(snapshot_path)
            ifc_reloaded = GatSession.load_ifc(ifc_path)
            resumed = GatSession.load_snapshot(snapshot_path)

        fy = self.session.var("Beam-B1", "YieldStrengthMPa")
        self.assertAlmostEqual(
            ifc_reloaded.world.belief.mean(fy),
            self.session.world.belief.mean(fy),
        )
        self.assertAlmostEqual(
            ifc_reloaded.world.belief.std(fy),
            self.session.world.belief.std(fy),
        )
        self.assertTrue(
            computational_equivalence(self.session.world, resumed.world).passed
        )
        resumed_result = BeamBendingEvaluator().evaluate(resumed.world, self.check)
        self.assertEqual(resumed_result.computation_digest, expected.computation_digest)
        self.assertEqual(resumed_result.verdict, expected.verdict)

        # Another runtime can continue the exact Gaussian state.
        section = self.session.var("Beam-B1", "SectionModulusM3")
        follow_up = ObserveQuantity.single(section, 0.00099, 0.000002)
        self.session.run(follow_up)
        resumed.run(follow_up)
        self.assertEqual(self.session.world.digest(), resumed.world.digest())


if __name__ == "__main__":
    unittest.main()
