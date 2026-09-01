"""Complete evidence-to-verification reference experiment for one IFC beam."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gat.demo
from gat.engineering import (
    BeamBendingCheck,
    BeamBendingEvaluator,
    beam_assessment_record,
    explain_beam_decision_change,
    read_material_certificate,
)
from gat.ledger import read_ledger, replay_ledger
from gat.session import GatSession
from gat.sp1_beam import (
    build_sp1_beam_claim,
    build_sp1_beam_request,
    sp1_beam_assessment_record,
    write_sp1_beam_request,
)
from gat.state_snapshot import computational_equivalence


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_beam_assurance(
    output_directory: str | Path,
    *,
    quiet: bool = False,
) -> dict[str, object]:
    """Execute, record, transport, and replay the reference beam chain."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    package = Path(gat.demo.__file__).parent
    model_path = package / "beam_model.ifc"
    certificate_path = package / "material_certificate.json"

    session = GatSession.load_ifc(str(model_path))
    initial_world = session.world
    beam = session.entity_by_name("Beam-B1")
    fy = session.var("Beam-B1", "YieldStrengthMPa")
    check = BeamBendingCheck(
        beam,
        301_000.0,
        0.95,
        "Beam-B1 factored bending",
    )
    evaluator = BeamBendingEvaluator()

    prior = evaluator.evaluate(session.world, check)
    session.record_assessment(
        beam_assessment_record(session.world, prior),
        provenance={"phase": "prior-design-belief"},
    )

    certificate_evidence = read_material_certificate(certificate_path).to_evidence(
        session.world
    )
    evidence = certificate_evidence.observation
    before_observation = session.world
    transition = session.run(
        evidence.transformation(session.world),
        provenance=certificate_evidence.provenance(),
    )
    transition_event = session.ledger.events[-1]
    revised = evaluator.evaluate(
        session.world,
        check,
        changed_inputs=transition.targets,
        affected_variables=transition.affected,
    )
    change = explain_beam_decision_change(
        before_observation,
        session.world,
        evidence,
        transition,
        prior,
        revised,
    )
    session.record_assessment(
        beam_assessment_record(
            session.world,
            revised,
            evidence_digest=evidence.digest(),
            change=change,
        ),
        provenance={"phase": "post-evidence-engineering-verification"},
    )
    fixed_claim = build_sp1_beam_claim(session.world, revised, evidence)
    fixed_assessment_event = session.record_assessment(
        sp1_beam_assessment_record(
            session.world,
            revised,
            evidence,
            fixed_claim,
        ),
        provenance={
            "phase": "sp1-bounded-fixed-point-claim",
            "proof_status": "BACKEND_REQUIRED",
        },
    )
    proof_request = build_sp1_beam_request(
        session.ledger,
        transition_event.seq,
        fixed_claim,
    )

    ifc_path = output / "beam_posterior.ifc"
    snapshot_path = output / "beam_state.gat.json"
    ledger_path = output / "beam_ledger.json"
    proof_request_path = output / "beam_sp1_request.json"
    summary_path = output / "beam_assurance_summary.json"
    session.export_ifc(str(ifc_path))
    snapshot_digest = session.export_snapshot(str(snapshot_path))
    ledger_head = session.export_ledger(str(ledger_path))
    write_sp1_beam_request(proof_request, proof_request_path)

    resumed = GatSession.load_snapshot(str(snapshot_path))
    equivalence = computational_equivalence(session.world, resumed.world)
    replayed = replay_ledger(initial_world, read_ledger(ledger_path))
    if not equivalence.passed or replayed.world.digest() != session.world.digest():
        raise RuntimeError("reference chain did not survive transport/replay")
    resumed_result = BeamBendingEvaluator().evaluate(resumed.world, check)
    if resumed_result.computation_digest != revised.computation_digest:
        raise RuntimeError("receiving runtime produced a different beam computation")

    summary: dict[str, object] = {
        "chain": [
            "IFC Beam",
            "Canonical State",
            "Material Observation",
            "Gaussian Update",
            "Structural Calculation",
            "Decision",
            "Verification Record",
            "Optional Proof Request",
        ],
        "identity": {
            "beam": {
                "ifc_class": beam.ifc_class,
                "global_id": beam.global_id,
            },
            "evidence_digest": evidence.digest(),
            "source_digest": evidence.source_digest,
            "prior_world_digest": prior.assessment.world_digest,
            "result_world_digest": revised.assessment.world_digest,
            "ledger_head": ledger_head,
            "snapshot_digest": snapshot_digest,
            "computation_digest": revised.computation_digest,
            "fixed_point_computation_digest": fixed_claim.computation_digest,
        },
        "belief_update": {
            "variable": str(fy),
            "evidence_kind": evidence.kind.value,
            "observed_value_mpa": evidence.observed_value,
            "prior_mean_mpa": before_observation.belief.mean(fy),
            "prior_sigma_mpa": before_observation.belief.std(fy),
            "posterior_mean_mpa": session.world.belief.mean(fy),
            "posterior_sigma_mpa": session.world.belief.std(fy),
        },
        "selective_computation": {
            "changed_inputs": [str(var) for var in transition.targets],
            "affected_variables": [str(var) for var in transition.affected],
            "dependency_variables": [str(var) for var in revised.dependency_vars],
            "recomputed": revised.recomputed,
        },
        "decision": {
            "demand_n_m": check.factored_demand_n_m,
            "confidence": check.confidence,
            "prior_capacity_mean_n_m": prior.assessment.target_mean,
            "prior_capacity_sigma_n_m": prior.assessment.target_sigma,
            "prior_p_satisfies": prior.assessment.p_satisfies,
            "prior_verdict": prior.verdict.value,
            "revised_capacity_mean_n_m": revised.assessment.target_mean,
            "revised_capacity_sigma_n_m": revised.assessment.target_sigma,
            "revised_p_satisfies": revised.assessment.p_satisfies,
            "revised_verdict": revised.verdict.value,
            "reason": change.reason,
        },
        "verification": {
            "invariants_passed": session.verify().passed,
            "ledger_replay_matches": replayed.world.digest() == session.world.digest(),
            "snapshot_computationally_equivalent": equivalence.passed,
            "receiving_runtime_computation_matches": (
                resumed_result.computation_digest == revised.computation_digest
            ),
            "sp1_proof_status": "BACKEND_REQUIRED",
            "sp1_proof_verified": False,
            "sp1_guest_implemented": True,
            "sp1_fixed_assessment_event_seq": fixed_assessment_event.seq,
            "sp1_public_statement_digest": proof_request.public_statement_digest,
        },
        "artifacts": {
            "posterior_ifc": ifc_path.name,
            "state_snapshot": snapshot_path.name,
            "execution_ledger": ledger_path.name,
            "sp1_request": proof_request_path.name,
        },
    }
    _write_json(summary_path, summary)

    if not quiet:
        print("=== GAT BEAM ASSURANCE REFERENCE CHAIN ============================")
        print(f"beam identity: {beam.ifc_class}:{beam.global_id}")
        print(
            f"prior:   fy={before_observation.belief.mean(fy):.3f} +/- "
            f"{before_observation.belief.std(fy):.3f} MPa -> "
            f"{prior.assessment.target_mean:.1f} +/- "
            f"{prior.assessment.target_sigma:.1f} N*m -> {prior.verdict.value}"
        )
        print(
            f"evidence: {evidence.kind.value} {evidence.observed_value:.1f} +/- "
            f"{evidence.noise_sigma:.1f} MPa ({evidence.digest()[:12]})"
        )
        print(
            f"posterior: fy={session.world.belief.mean(fy):.3f} +/- "
            f"{session.world.belief.std(fy):.3f} MPa -> "
            f"{revised.assessment.target_mean:.1f} +/- "
            f"{revised.assessment.target_sigma:.1f} N*m -> {revised.verdict.value}"
        )
        print(
            "selective recomputation: "
            + ", ".join(var.quantity for var in transition.affected)
        )
        print(f"reason: {change.reason}")
        print(
            "verification: invariants PASS; ledger replay PASS; "
            "snapshot continuation PASS"
        )
        print("SP1: BACKEND_REQUIRED (no proof claimed or verified)")
        print(f"artifacts: {output.resolve()}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the complete GAT beam assurance reference chain"
    )
    parser.add_argument(
        "output_directory",
        nargs="?",
        default="beam-assurance-output",
    )
    args = parser.parse_args()
    run_beam_assurance(args.output_directory)


if __name__ == "__main__":
    main()
