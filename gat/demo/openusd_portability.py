"""Separate-process continuation proof through the OpenUSD carrier.

Run with::

    python -m gat.demo.openusd_portability [output-directory]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from gat.adapters.openusd import read_openusd
from gat.engine.transform import ObserveQuantity, ShiftParameter
from gat.session import GatSession
from gat.state_snapshot import computational_equivalence


MODEL = os.path.join(os.path.dirname(__file__), "model.ifc")


def _first_transition(session: GatSession) -> None:
    session.run(
        ObserveQuantity.single(
            session.var("Office-A", "Volume"),
            59.4,
            noise_sigma=0.05,
        )
    )


def _second_transition(session: GatSession) -> None:
    session.run(
        ShiftParameter(session.var("Level 1", "ClearHeight"), 0.10)
    )


def _resume_worker(checkpoint: str, output: str) -> int:
    session = GatSession.load_openusd(checkpoint)
    _second_transition(session)
    session.export_openusd(output)
    return 0


def run_demo(output_directory: str) -> str:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "state_after_t1.usdc"
    uninterrupted_path = output / "state_t2_uninterrupted.usdc"
    resumed_path = output / "state_t2_resumed.usdc"

    source = GatSession.load_ifc(MODEL)
    _first_transition(source)
    after_t1 = source.world
    carrier_digest = source.export_openusd(str(checkpoint))

    repository_root = Path(__file__).resolve().parents[2]
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "gat.demo.openusd_portability",
            "--resume-worker",
            str(checkpoint),
            str(resumed_path),
        ],
        cwd=repository_root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if process.returncode != 0:
        raise RuntimeError(
            "OpenUSD continuation worker failed:\n"
            f"stdout:\n{process.stdout}\nstderr:\n{process.stderr}"
        )

    _second_transition(source)
    source.export_openusd(str(uninterrupted_path))

    checkpoint_loaded = read_openusd(checkpoint)
    resumed_loaded = read_openusd(resumed_path)
    uninterrupted_loaded = read_openusd(uninterrupted_path)
    checkpoint_report = computational_equivalence(after_t1, checkpoint_loaded.world)
    resumed = resumed_loaded.world
    uninterrupted = uninterrupted_loaded.world
    continuation_report = computational_equivalence(uninterrupted, resumed)
    if not checkpoint_report.passed or not continuation_report.passed:
        raise RuntimeError(
            checkpoint_report.render() + "\n" + continuation_report.render()
        )
    if (
        checkpoint_loaded.ledger is None
        or resumed_loaded.ledger is None
        or uninterrupted_loaded.ledger is None
        or resumed_loaded.ledger.head != uninterrupted_loaded.ledger.head
    ):
        raise RuntimeError("OpenUSD did not preserve exact ledger continuation")

    print("GAT OPENUSD COMPUTATIONAL-STATE PORTABILITY")
    print(
        f"T1 carrier: {checkpoint.name}  "
        f"snapshot={carrier_digest[:12]} world={after_t1.digest()[:12]}"
    )
    print("runtime boundary: a separate process restored the USD stage and applied T2")
    print(continuation_report.render())
    print(f"ledger continuation: {resumed_loaded.ledger.head[:12]}")
    print("result: OpenUSD preserved exact verified state and causal continuation")
    return resumed.digest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_directory", nargs="?", default="out-openusd-portability")
    parser.add_argument(
        "--resume-worker",
        nargs=2,
        metavar=("CHECKPOINT", "OUTPUT"),
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    if args.resume_worker is not None:
        return _resume_worker(args.resume_worker[0], args.resume_worker[1])
    run_demo(args.output_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
