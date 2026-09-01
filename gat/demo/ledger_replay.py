"""Build, persist, and independently replay an authoritative GAT ledger.

Run with::

    python -m gat.demo.ledger_replay execution-ledger.json
"""

from __future__ import annotations

import argparse
import hashlib
import os

import numpy as np

import gat.demo
from gat.engine.transform import ObserveLinearized, SetParameter, ShiftParameter
from gat.errors import VerificationError
from gat.ledger import read_ledger, replay_ledger
from gat.session import GatSession


MODEL = os.path.join(os.path.dirname(gat.demo.__file__), "model.ifc")


def _scan_support_observation(session: GatSession) -> ObserveLinearized:
    """A compact stand-in for an already calibrated scan adapter result."""
    world = session.world
    clear_height = session.var("Level 1", "ClearHeight")
    row = np.zeros(world.binding.n_raw)
    row[world.binding.raw_index.row(clear_height)] = 1.0
    return ObserveLinearized(
        row=row,
        predicted=world.belief.mean(clear_height),
        observed=2.992,
        noise_sigma=0.004,
        raw_targets=(clear_height,),
        expected_raw_order=world.binding.raw_index.vars,
        expected_belief_digest=world.belief.digest(),
        expected_world_digest=world.digest(),
        evidence_digest=hashlib.sha256(b"scan-42/survey-control-A/cal-v3").hexdigest(),
        label="calibrated scan support face Wall-Party",
    )


def run(path: str) -> None:
    session = GatSession.load_ifc(MODEL)
    initial_world = session.world
    observation = _scan_support_observation(session)
    session.run(
        observation,
        provenance={
            "evidence_kind": "scan-to-clearance-likelihood",
            "scan_digest": observation.evidence_digest,
            "pose_source": "survey-control-A",
            "calibration": "cal-v3",
        },
    )
    session.run(
        ShiftParameter(session.var("Level 1", "ClearHeight"), 0.001),
        provenance={"action": "accepted design adjustment", "approval": "design-review-17"},
    )
    try:
        session.run(
            SetParameter(session.var("Opening-1", "Height"), 3.2, 0.005),
            provenance={"action": "rejected opening revision", "proposal": "change-91"},
        )
    except VerificationError:
        pass

    head = session.export_ledger(path)
    independent_checkpoint = GatSession.load_ifc(MODEL).world
    replay = replay_ledger(independent_checkpoint, read_ledger(path))
    if replay.world.digest() != session.world.digest():
        raise RuntimeError("replay did not reconstruct the exact final world")

    print(f"ledger head: {head}")
    print(
        f"replayed exactly: {replay.accepted} accepted, "
        f"{replay.rejected} rejected, world {replay.world.digest()}"
    )
    print(f"initial checkpoint: {initial_world.digest()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", help="output JSON ledger path")
    args = parser.parse_args()
    run(args.ledger)


if __name__ == "__main__":
    main()
