"""Predict -> observe -> verify -> replay with explicit process dynamics."""

from __future__ import annotations

import hashlib
import os

import numpy as np

import gat.demo
from gat.engine.dynamics import EvolveLinearGaussian, forecast_process
from gat.engine.transform import ObserveQuantity
from gat.ledger import replay_ledger
from gat.session import GatSession


MODEL = os.path.join(os.path.dirname(gat.demo.__file__), "model.ifc")


def run() -> str:
    session = GatSession.load_ifc(MODEL)
    initial = session.world
    height = session.var("Level 1", "ClearHeight")
    volume = session.var("Office-A", "Volume")
    process = EvolveLinearGaussian(
        (height,),
        transition=np.array([[1.0]]),
        offset=np.array([-0.0005]),
        process_covariance=np.array([[0.0002**2]]),
        elapsed_seconds=86400.0,
        model_id="daily-settlement-monitor-v1",
        calibration_digest=hashlib.sha256(
            b"survey/calibration/daily-settlement-monitor-v1"
        ).hexdigest(),
    )

    forecast = forecast_process(initial, process, steps=7)
    if session.world.digest() != initial.digest():
        raise RuntimeError("forecast mutated the session")

    session.run(
        process,
        provenance={"clock": "building-controller-A", "interval": "day-1"},
    )
    predicted_volume = session.world.full.mean(volume)
    session.run(
        ObserveQuantity.single(volume, predicted_volume - 0.02, 0.05),
        provenance={"sensor": "volume-laser-A", "reading_interval": "day-1"},
    )
    replay = replay_ledger(initial, session.ledger)
    if replay.world.digest() != session.world.digest():
        raise RuntimeError("temporal ledger replay diverged")

    print("GAT TEMPORAL PREDICT-UPDATE")
    print(
        f"7-day forecast: height={forecast.final_world.belief.mean(height):.6f} m, "
        f"sigma={forecast.final_world.belief.std(height):.6f} m"
    )
    print(
        f"day-1 posterior: height={session.world.belief.mean(height):.6f} m, "
        f"sigma={session.world.belief.std(height):.6f} m"
    )
    print(
        f"replay: {replay.accepted} accepted transitions, "
        f"ledger={replay.head[:12]}, world={replay.world.digest()[:12]}"
    )
    return replay.world.digest()


if __name__ == "__main__":
    run()
