"""Signed OpenUSD transport and continuation proof for the beam assurance chain.

Run with::

    python -m gat.demo.beam_openusd_portability [output-directory]

The publisher conditions the beam with the first material certificate, records
the revised assessment, and signs an OpenUSD carrier. A separate Python process
authenticates that carrier, reproduces the transported computation, applies a
follow-up certificate, records the next assessment, and exports its result.
The publisher also follows the same second step without interruption. Exact
world and ledger equality are the success condition.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Mapping

import gat.demo
from gat.adapters.openusd import generate_openusd_keypair, read_openusd
from gat.engineering import (
    BeamBendingCheck,
    BeamBendingEvaluator,
    BeamCheckResult,
    beam_assessment_record,
    explain_beam_decision_change,
    read_material_certificate,
)
from gat.session import GatSession
from gat.state_snapshot import computational_equivalence


@dataclass(frozen=True)
class _EvidenceStep:
    revised: BeamCheckResult
    evidence_digest: str
    certificate_source_digest: str
    transition_event_hash: str
    assessment_event_hash: str


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_object(path: str | Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _object(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be non-empty text")
    return value


def _beam_check(session: GatSession) -> BeamBendingCheck:
    return BeamBendingCheck(
        session.entity_by_name("Beam-B1"),
        factored_demand_n_m=301_000.0,
        confidence=0.95,
        label="Beam-B1 factored bending",
    )


def _apply_certificate(
    session: GatSession,
    evaluator: BeamBendingEvaluator,
    check: BeamBendingCheck,
    prior: BeamCheckResult,
    certificate_path: str | Path,
    *,
    phase: str,
) -> _EvidenceStep:
    certificate_evidence = read_material_certificate(certificate_path).to_evidence(
        session.world
    )
    evidence = certificate_evidence.observation
    if evidence.subject.entity != check.beam:
        raise RuntimeError("certificate and beam-check subjects differ")
    before = session.world
    transition = session.run(
        evidence.transformation(before),
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
        before,
        session.world,
        evidence,
        transition,
        prior,
        revised,
    )
    assessment_event = session.record_assessment(
        beam_assessment_record(
            session.world,
            revised,
            evidence_digest=evidence.digest(),
            change=change,
        ),
        provenance={"phase": phase},
    )
    return _EvidenceStep(
        revised,
        evidence.digest(),
        evidence.source_digest,
        transition_event.event_hash,
        assessment_event.event_hash,
    )


def _resume_worker(request_path: str) -> int:
    request = _read_object(request_path)
    if request.get("format") != "gat-beam-openusd-resume-request-v1":
        raise ValueError("unsupported beam OpenUSD resume request")
    trust = _object(request.get("trust"), "trust")
    expected = _object(request.get("expected_checkpoint"), "expected_checkpoint")
    key_id = _text(trust.get("key_id"), "trust.key_id")
    try:
        public_key = base64.b64decode(
            _text(trust.get("public_key_base64"), "trust.public_key_base64"),
            validate=True,
        )
    except ValueError as exc:
        raise ValueError("trust.public_key_base64 is invalid") from exc
    if len(public_key) != 32:
        raise ValueError("trusted Ed25519 public key must contain 32 bytes")

    session = GatSession.load_openusd(
        _text(request.get("checkpoint"), "checkpoint"),
        trusted_public_keys={key_id: public_key},
        require_signature=True,
    )
    if not session.carrier_signature_verified:
        raise RuntimeError("receiving runtime did not authenticate the carrier")
    if session.world.digest() != _text(expected.get("world_digest"), "world_digest"):
        raise RuntimeError("transported world identity differs")
    if session.ledger.head != _text(expected.get("ledger_head"), "ledger_head"):
        raise RuntimeError("transported ledger identity differs")
    if len(session.ledger.events) != 4:
        raise RuntimeError("checkpoint ledger does not contain the closed beam chain")
    transported_transition = session.ledger.events[-2]
    transported_assessment = session.ledger.events[-1]
    if (
        transported_transition.kind != "transition"
        or transported_transition.event_hash
        != _text(expected.get("transition_event_hash"), "transition_event_hash")
        or transported_assessment.kind != "assessment"
        or transported_assessment.event_hash
        != _text(expected.get("assessment_event_hash"), "assessment_event_hash")
    ):
        raise RuntimeError("transported transition or assessment identity differs")
    expected_evidence = _text(expected.get("evidence_digest"), "evidence_digest")
    certificate = _object(
        transported_transition.provenance.get("material_certificate"),
        "transition.material_certificate",
    )
    assessment_details = _object(
        transported_assessment.operation.get("details"),
        "assessment.details",
    )
    if (
        transported_transition.provenance.get("evidence_digest") != expected_evidence
        or certificate.get("source_digest")
        != _text(
            expected.get("certificate_source_digest"),
            "certificate_source_digest",
        )
        or transported_assessment.operation.get("world_digest")
        != session.world.digest()
        or assessment_details.get("state_identity") != session.world.digest()
    ):
        raise RuntimeError("transported evidence-to-state identity differs")

    check = _beam_check(session)
    evaluator = BeamBendingEvaluator()
    transported = evaluator.evaluate(session.world, check)
    computation = _object(
        assessment_details.get("computation"),
        "details.computation",
    )
    expected_computation = _text(
        expected.get("computation_digest"), "computation_digest"
    )
    if (
        transported.computation_digest != expected_computation
        or computation.get("computation_digest") != expected_computation
        or transported.verdict.value != _text(expected.get("verdict"), "verdict")
        or transported_assessment.operation.get("evidence_digest")
        != expected_evidence
    ):
        raise RuntimeError("receiving runtime did not reproduce the transported verdict")

    continued = _apply_certificate(
        session,
        evaluator,
        check,
        transported,
        _text(request.get("followup_certificate"), "followup_certificate"),
        phase="receiving-runtime-followup-certificate",
    )
    output = _text(request.get("output"), "output")
    session.export_openusd(output)
    receipt_path = Path(_text(request.get("receipt"), "receipt"))
    _write_json(
        receipt_path,
        {
            "format": "gat-beam-openusd-resume-receipt-v1",
            "carrier_signature_verified": True,
            "carrier_signing_key_id": session.carrier_signing_key_id,
            "transported_world_digest": _text(
                expected.get("world_digest"), "world_digest"
            ),
            "transported_ledger_head": _text(
                expected.get("ledger_head"), "ledger_head"
            ),
            "transported_transition_event_hash": transported_transition.event_hash,
            "transported_assessment_event_hash": transported_assessment.event_hash,
            "transported_computation_digest": transported.computation_digest,
            "transported_verdict": transported.verdict.value,
            "continuation_world_digest": session.world.digest(),
            "continuation_ledger_head": session.ledger.head,
            "continuation_assessment_event_hash": continued.assessment_event_hash,
            "continuation_computation_digest": continued.revised.computation_digest,
            "continuation_verdict": continued.revised.verdict.value,
            "may_authorize": False,
        },
    )
    return 0


def run_demo(
    output_directory: str | Path,
    *,
    quiet: bool = False,
) -> dict[str, object]:
    """Prove authenticated beam-chain transport and exact continuation."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    package = Path(gat.demo.__file__).parent
    model_path = package / "beam_model.ifc"
    first_certificate_path = package / "material_certificate.json"
    followup_certificate_path = package / "material_certificate_followup.json"
    checkpoint_path = output / "beam_checkpoint_signed.usdc"
    resumed_path = output / "beam_continued_resumed.usdc"
    uninterrupted_path = output / "beam_continued_uninterrupted.usdc"
    request_path = output / "beam_openusd_resume_request.json"
    receipt_path = output / "beam_openusd_resume_receipt.json"
    summary_path = output / "beam_openusd_portability_summary.json"

    source = GatSession.load_ifc(str(model_path))
    check = _beam_check(source)
    evaluator = BeamBendingEvaluator()
    initial = evaluator.evaluate(source.world, check)
    if initial.verdict.value != "SATISFIED":
        raise RuntimeError("reference beam prior no longer satisfies the check")
    source.record_assessment(
        beam_assessment_record(source.world, initial),
        provenance={"phase": "publisher-prior-design-belief"},
    )
    checkpoint_step = _apply_certificate(
        source,
        evaluator,
        check,
        initial,
        first_certificate_path,
        phase="publisher-material-certificate",
    )
    if checkpoint_step.revised.verdict.value != "VIOLATED":
        raise RuntimeError("first certificate no longer produces the reference violation")
    checkpoint_world = source.world
    checkpoint_ledger = source.ledger.to_dict()
    checkpoint_head = source.ledger.head

    publisher = generate_openusd_keypair("beam-portability-publisher-v1")
    snapshot_digest = source.export_openusd(
        str(checkpoint_path),
        signing_key=publisher,
    )
    _write_json(
        request_path,
        {
            "format": "gat-beam-openusd-resume-request-v1",
            "checkpoint": str(checkpoint_path.resolve()),
            "followup_certificate": str(followup_certificate_path.resolve()),
            "output": str(resumed_path.resolve()),
            "receipt": str(receipt_path.resolve()),
            "trust": {
                "key_id": publisher.key_id,
                "public_key_base64": base64.b64encode(publisher.public_key).decode(
                    "ascii"
                ),
                "require_signature": True,
            },
            "expected_checkpoint": {
                "world_digest": checkpoint_world.digest(),
                "ledger_head": checkpoint_head,
                "transition_event_hash": checkpoint_step.transition_event_hash,
                "assessment_event_hash": checkpoint_step.assessment_event_hash,
                "computation_digest": checkpoint_step.revised.computation_digest,
                "verdict": checkpoint_step.revised.verdict.value,
                "evidence_digest": checkpoint_step.evidence_digest,
                "certificate_source_digest": (
                    checkpoint_step.certificate_source_digest
                ),
            },
        },
    )

    repository_root = Path(__file__).resolve().parents[2]
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "gat.demo.beam_openusd_portability",
            "--resume-worker",
            str(request_path.resolve()),
        ],
        cwd=repository_root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if process.returncode != 0:
        raise RuntimeError(
            "beam OpenUSD continuation worker failed:\n"
            f"stdout:\n{process.stdout}\nstderr:\n{process.stderr}"
        )

    uninterrupted_step = _apply_certificate(
        source,
        evaluator,
        check,
        checkpoint_step.revised,
        followup_certificate_path,
        phase="receiving-runtime-followup-certificate",
    )
    source.export_openusd(str(uninterrupted_path))

    trusted_checkpoint = read_openusd(
        checkpoint_path,
        trusted_public_keys={publisher.key_id: publisher.public_key},
        require_signature=True,
    )
    resumed = read_openusd(resumed_path)
    uninterrupted = read_openusd(uninterrupted_path)
    receipt = _read_object(receipt_path)
    checkpoint_equivalence = computational_equivalence(
        checkpoint_world,
        trusted_checkpoint.world,
    )
    continuation_equivalence = computational_equivalence(
        uninterrupted.world,
        resumed.world,
    )
    ledgers_match = (
        trusted_checkpoint.ledger is not None
        and resumed.ledger is not None
        and uninterrupted.ledger is not None
        and trusted_checkpoint.ledger.to_dict() == checkpoint_ledger
        and resumed.ledger.to_dict() == uninterrupted.ledger.to_dict()
    )
    final_result = BeamBendingEvaluator().evaluate(resumed.world, _beam_check(source))
    if (
        not trusted_checkpoint.signature.verified
        or not checkpoint_equivalence.passed
        or not continuation_equivalence.passed
        or not ledgers_match
        or final_result.computation_digest
        != uninterrupted_step.revised.computation_digest
        or final_result.verdict.value != "SATISFIED"
        or receipt.get("carrier_signature_verified") is not True
        or receipt.get("continuation_ledger_head") != resumed.ledger.head
    ):
        raise RuntimeError("beam assurance chain did not survive OpenUSD continuation")

    fy = source.var("Beam-B1", "YieldStrengthMPa")
    summary: dict[str, object] = {
        "chain": [
            "IFC Beam",
            "Material Certificate",
            "Gaussian Update",
            "AISC Calculation",
            "Verdict Record",
            "Signed OpenUSD Carrier",
            "Authenticated Runtime Resume",
            "Follow-up Certificate",
            "Exact Continuation",
        ],
        "checkpoint": {
            "snapshot_digest": snapshot_digest,
            "world_digest": checkpoint_world.digest(),
            "ledger_head": checkpoint_head,
            "transition_event_hash": checkpoint_step.transition_event_hash,
            "assessment_event_hash": checkpoint_step.assessment_event_hash,
            "evidence_digest": checkpoint_step.evidence_digest,
            "certificate_source_digest": checkpoint_step.certificate_source_digest,
            "computation_digest": checkpoint_step.revised.computation_digest,
            "verdict": checkpoint_step.revised.verdict.value,
            "carrier_signature_verified": trusted_checkpoint.signature.verified,
            "carrier_signing_key_id": trusted_checkpoint.signature.key_id,
            "world_computationally_equivalent": checkpoint_equivalence.passed,
            "ledger_exactly_preserved": trusted_checkpoint.ledger.to_dict()
            == checkpoint_ledger,
        },
        "continuation": {
            "world_digest": resumed.world.digest(),
            "ledger_head": resumed.ledger.head,
            "assessment_event_hash": uninterrupted_step.assessment_event_hash,
            "evidence_digest": uninterrupted_step.evidence_digest,
            "certificate_source_digest": (
                uninterrupted_step.certificate_source_digest
            ),
            "computation_digest": final_result.computation_digest,
            "verdict": final_result.verdict.value,
            "posterior_yield_strength_mean_mpa": resumed.world.belief.mean(fy),
            "posterior_yield_strength_sigma_mpa": resumed.world.belief.std(fy),
            "separate_process_world_matches_uninterrupted": (
                continuation_equivalence.passed
            ),
            "separate_process_ledger_matches_uninterrupted": ledgers_match,
            "receiving_runtime_reproduced_checkpoint_computation": (
                receipt.get("transported_computation_digest")
                == checkpoint_step.revised.computation_digest
            ),
        },
        "assurance": {
            "openusd_signature_verified": True,
            "openusd_trust_source": "explicit-demo-resume-request",
            "material_certificate_signature_verified": False,
            "material_certificate_issuer_trust_verified": False,
            "scope_assertions_independently_verified": False,
            "may_authorize": False,
        },
        "artifacts": {
            "signed_checkpoint": checkpoint_path.name,
            "resumed_result": resumed_path.name,
            "uninterrupted_result": uninterrupted_path.name,
            "resume_request": request_path.name,
            "resume_receipt": receipt_path.name,
        },
    }
    _write_json(summary_path, summary)

    if not quiet:
        print("GAT BEAM OPENUSD PORTABILITY")
        print(
            f"checkpoint: {checkpoint_step.revised.verdict.value} "
            f"world={checkpoint_world.digest()[:12]} ledger={checkpoint_head[:12]}"
        )
        print(
            f"carrier: signature verified by {trusted_checkpoint.signature.key_id}; "
            "state, evidence transition, calculation, and verdict record preserved"
        )
        print(
            "runtime boundary: separate process authenticated the stage, reproduced "
            "the checkpoint calculation, and applied the follow-up certificate"
        )
        print(
            f"continuation: {final_result.verdict.value} "
            f"world={resumed.world.digest()[:12]} ledger={resumed.ledger.head[:12]}"
        )
        print("result: exact world and ledger continuation match uninterrupted execution")
        print("authorization: FALSE (certificate identity/trust remains unverified)")
        print(f"artifacts: {output.resolve()}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_directory",
        nargs="?",
        default="out-beam-openusd-portability",
    )
    parser.add_argument(
        "--resume-worker",
        metavar="REQUEST",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    if args.resume_worker is not None:
        return _resume_worker(args.resume_worker)
    run_demo(args.output_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
