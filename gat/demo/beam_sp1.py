"""Generate and independently verify the real SP1 beam proof."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from gat.demo.beam_assurance import run_beam_assurance
from gat.ledger import read_ledger
from gat.proof_manifest import (
    create_computation_proof_manifest,
    verify_computation_proof_manifest,
    write_computation_proof_manifest,
)
from gat.sp1_beam import (
    SP1_BEAM_MEDIA_TYPE,
    SP1_BEAM_PROOF_SYSTEM,
    SP1_BEAM_PROOF_TYPE,
    read_sp1_beam_receipt,
    read_sp1_beam_request,
    sp1_beam_subprocess_verifier,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def run_beam_sp1(
    output_directory: str | Path,
    executable: str | Path,
    *,
    timeout_seconds: float = 3600.0,
    quiet: bool = False,
) -> dict[str, object]:
    """Run the beam chain, prove its fixed claim, and verify the proof."""
    output = Path(output_directory)
    summary = run_beam_assurance(output, quiet=True)
    executable_path = Path(executable).resolve()
    request_path = output / "beam_sp1_request.json"
    proof_path = output / "beam.sp1-proof"
    receipt_path = output / "beam_sp1_receipt.json"
    manifest_path = output / "beam_sp1_manifest.json"
    ledger_path = output / "beam_ledger.json"

    completed = subprocess.run(
        [
            str(executable_path),
            "prove",
            "--request",
            str(request_path.resolve()),
            "--proof",
            str(proof_path.resolve()),
            "--receipt",
            str(receipt_path.resolve()),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "SP1 beam proof generation failed:\n"
            f"stdout:\n{completed.stdout[-4000:]}\n"
            f"stderr:\n{completed.stderr[-4000:]}"
        )

    request = read_sp1_beam_request(request_path)
    receipt = read_sp1_beam_receipt(receipt_path)
    proof_bytes = proof_path.read_bytes()
    ledger = read_ledger(ledger_path)
    manifest = create_computation_proof_manifest(
        ledger,
        request.transition_event_seq,
        numeric_contract=request.numeric_contract,
        model_contract_digest=request.claim.input.model_contract_digest,
        validation_profile_digest=request.claim.input.validation_profile_digest,
        computation_result_digest=request.claim.computation_digest,
        evidence_commitments=(
            request.claim.input.evidence_digest,
            request.claim.input.evidence_source_digest,
        ),
        proof_system=SP1_BEAM_PROOF_SYSTEM,
        proof_type=SP1_BEAM_PROOF_TYPE,
        program_digest=receipt.program_digest,
        verifying_key_digest=receipt.verifying_key_digest,
        proof_artifact=proof_bytes,
        media_type=SP1_BEAM_MEDIA_TYPE,
        locator=proof_path.name,
    )
    verifier = sp1_beam_subprocess_verifier(
        executable_path,
        request_path,
        timeout_seconds=timeout_seconds,
    )
    report = verify_computation_proof_manifest(
        manifest,
        ledger,
        proof_bytes,
        verifier=verifier,
    )
    if not report.proof_verified:
        raise RuntimeError(report.render())
    write_computation_proof_manifest(manifest, manifest_path)

    verification = summary["verification"]
    assert isinstance(verification, dict)
    verification.update(
        {
            "sp1_proof_status": "PROOF_VERIFIED",
            "sp1_proof_verified": True,
            "sp1_program_digest": receipt.program_digest,
            "sp1_verifying_key_digest": receipt.verifying_key_digest,
            "sp1_proof_artifact_digest": receipt.proof_artifact_digest,
        }
    )
    artifacts = summary["artifacts"]
    assert isinstance(artifacts, dict)
    artifacts.update(
        {
            "sp1_proof": proof_path.name,
            "sp1_receipt": receipt_path.name,
            "sp1_manifest": manifest_path.name,
        }
    )
    _write_json(output / "beam_assurance_summary.json", summary)

    if not quiet:
        print("=== GAT SP1 BEAM PROOF ===========================================")
        print(f"fixed computation: {request.claim.computation_digest}")
        print(f"public statement:  {request.public_statement_digest}")
        print(f"program:           {receipt.program_digest}")
        print(f"proof artifact:    {receipt.proof_artifact_digest}")
        print("verification:      PROOF VERIFIED")
        print(f"artifacts:         {output.resolve()}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and verify the GAT SP1 beam proof")
    parser.add_argument("output_directory")
    parser.add_argument("--executable", required=True)
    parser.add_argument("--timeout", type=float, default=3600.0)
    args = parser.parse_args()
    run_beam_sp1(
        args.output_directory,
        args.executable,
        timeout_seconds=args.timeout,
    )


if __name__ == "__main__":
    main()
