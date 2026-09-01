"""Operational-portability demonstration for ``GatStateSnapshot v1``.

The main process applies T1 and writes a restart checkpoint.  A separate
Python process loads that checkpoint, applies T2, and writes its result.
The original process applies the same T2 without interruption.  The two
resulting worlds must be computationally equivalent.

Run with::

    python -m gat.demo.portability [output-directory]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from gat.engine.transform import ObserveQuantity, ShiftParameter
from gat.session import GatSession
from gat.state_snapshot import computational_equivalence, read_snapshot


MODEL = os.path.join(os.path.dirname(__file__), "model.ifc")


def _apply_first(session: GatSession) -> None:
    """T1: assimilate evidence, creating non-diagonal posterior covariance."""
    volume = session.var("Office-A", "Volume")
    session.run(ObserveQuantity.single(volume, 59.4, noise_sigma=0.05))


def _apply_second(session: GatSession) -> None:
    """T2: continue evolution from either uninterrupted or restored state."""
    clear_height = session.var("Level 1", "ClearHeight")
    session.run(ShiftParameter(clear_height, 0.10))


def _resume_worker(checkpoint: str, output: str) -> int:
    session = GatSession.load_snapshot(checkpoint)
    _apply_second(session)
    session.export_snapshot(output)
    return 0


def run_demo(output_directory: str) -> str:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "state_after_t1.gat.json"
    uninterrupted_path = output / "state_t2_uninterrupted.gat.json"
    resumed_path = output / "state_t2_resumed.gat.json"

    source = GatSession.load_ifc(MODEL)
    _apply_first(source)
    world_after_t1 = source.world
    checkpoint_digest = source.export_snapshot(str(checkpoint))

    repository_root = Path(__file__).resolve().parents[2]
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "gat.demo.portability",
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
            "snapshot resume worker failed:\n"
            f"stdout:\n{process.stdout}\nstderr:\n{process.stderr}"
        )

    # The original process continues without decoding the checkpoint.
    _apply_second(source)
    source.export_snapshot(str(uninterrupted_path))

    checkpoint_result = read_snapshot(checkpoint)
    checkpoint_report = computational_equivalence(
        world_after_t1, checkpoint_result.world
    )
    resumed = read_snapshot(resumed_path)
    uninterrupted = read_snapshot(uninterrupted_path)
    continuation_report = computational_equivalence(
        uninterrupted.world, resumed.world
    )
    if not checkpoint_report.passed or not continuation_report.passed:
        raise RuntimeError(
            checkpoint_report.render() + "\n" + continuation_report.render()
        )

    print("GAT STATE-SPACE PORTABILITY")
    print(
        f"T1 checkpoint: {checkpoint.name}  "
        f"snapshot={checkpoint_digest[:12]} world={world_after_t1.digest()[:12]}"
    )
    print(
        "runtime boundary: separate Python process loaded the snapshot and "
        "continued with T2"
    )
    print(continuation_report.render())
    print(
        "result: T2(restore(snapshot(T1(S0)))) == T2(T1(S0)); "
        "operational identity preserved"
    )
    return resumed.world.digest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_directory", nargs="?", default="out-portability")
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
