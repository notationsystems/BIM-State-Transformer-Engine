"""Decision-focused evidence planning over the shipped GAT BIM model.

Run with ``python -m gat.demo.active_inference``.  The demo asks which
measurement is worth acquiring to resolve ``Office-A.Volume >= 60 m3`` at
95% confidence.  It then commits a real laser-volume reading through the
standard verified observation pathway and stops once the decision resolves.
"""

from __future__ import annotations

import os

from gat import (
    MinimumDecision,
    ObservationCandidate,
    assess_decision,
    plan_decision_evidence,
)
from gat.session import GatSession


MODEL = os.path.join(os.path.dirname(__file__), "model.ifc")


def main() -> int:
    session = GatSession.load_ifc(MODEL)
    target = session.var("Office-A", "Volume")
    candidates = (
        ObservationCandidate(
            session.var("Wall-South", "UnitCost"),
            10.0,
            "cost survey",
            cost_nats=0.01,
        ),
        ObservationCandidate(
            session.var("Level 1", "ClearHeight"),
            0.01,
            "laser height",
            cost_nats=0.05,
        ),
        ObservationCandidate(target, 0.05, "laser volume", cost_nats=0.10),
    )
    decision = MinimumDecision(
        target,
        minimum=60.0,
        confidence=0.95,
        label="Office-A usable volume >= 60 m3",
    )
    plan = plan_decision_evidence(session.world, decision, candidates)

    print("=== DECIDE: is the criterion already resolved? ===")
    print(plan.render())
    assert plan.selected is not None

    reading = 60.2
    print(f"\n=== OBSERVE: {plan.selected.candidate.name} = {reading:.6f} ===")
    result = session.run(plan.selected.candidate.observe(reading))
    print(result.describe())
    posterior = assess_decision(session.world, decision)
    print(posterior.render())
    print(f"verification: {'PASS' if session.verify().passed else 'FAIL'}")
    assert posterior.resolved
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
