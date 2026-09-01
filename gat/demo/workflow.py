"""Self-asserting construction acceptance and RFI workflow demonstration."""

from __future__ import annotations

import os

import gat.demo
from gat.engine.transform import SetParameter
from gat.session import GatSession
from gat.workflows import (
    AcceptanceCase,
    AcceptanceDisposition,
    AcceptancePolicy,
    ChangeDisposition,
    DifferenceDecision,
    WorkflowKind,
    assess_difference,
    difference_check,
    evaluate_acceptance_case,
    preview_change,
)


def run() -> None:
    model = os.path.join(os.path.dirname(gat.demo.__file__), "model.ifc")
    session = GatSession.load_ifc(model)
    original = session.world.digest()

    print("=== ACCEPTANCE: opening fit under the joint BIM belief ============")
    checks = []
    for check_id, quantity in (("width", "Width"), ("height", "Height")):
        assessment = assess_difference(
            session.world,
            DifferenceDecision(
                session.var("Opening-1", quantity),
                session.var("Door-1", quantity),
                minimum_margin=0.05,
                confidence=0.95,
                label=f"Door-1 {quantity.lower()} fit",
            ),
        )
        check = difference_check(check_id, assessment)
        checks.append(check)
        print(
            f"{check_id}: margin {assessment.margin_mean:.4f} +- "
            f"{assessment.margin_sigma:.4f} m -> {assessment.verdict.value}"
        )
    case = AcceptanceCase(
        "opening-fit-demo",
        WorkflowKind.OPENING_VERIFICATION,
        "Door-1 into Opening-1",
        tuple(checks),
    )
    as_built = evaluate_acceptance_case(case)
    assert as_built.disposition is AcceptanceDisposition.REQUEST_EVIDENCE
    assert set(as_built.uncovered_check_ids) == {"width", "height"}
    print(f"as-built disposition: {as_built.disposition.value}")
    print("reason: numerical fit is not a substitute for verified field evidence")

    design_only = evaluate_acceptance_case(
        case,
        policy=AcceptancePolicy(
            "design-review-v1",
            require_verified_evidence_for_accept=False,
        ),
    )
    assert design_only.disposition is AcceptanceDisposition.ACCEPT
    print(f"design-review disposition: {design_only.disposition.value}")
    print("scope remains a recommendation; no approval or BIM mutation occurred")

    print("\n=== RFI: preview a design change without committing it ===========")
    admissible = preview_change(
        session.world,
        SetParameter(session.var("Level 1", "ClearHeight"), 3.4, 0.01),
    )
    assert admissible.disposition is ChangeDisposition.ADMISSIBLE
    assert len(admissible.affected) == 34
    assert session.world.digest() == original
    print(
        f"clear height 3.40 m -> {admissible.disposition.value}; "
        f"{len(admissible.impacted_entities)} entities, "
        f"{len(admissible.impacts)} variable impacts"
    )

    blocked = preview_change(
        session.world,
        SetParameter(session.var("Opening-1", "Height"), 3.6, 0.005),
    )
    assert blocked.disposition is ChangeDisposition.BLOCKED
    assert blocked.failures
    assert session.world.digest() == original
    print(
        f"opening height 3.60 m -> {blocked.disposition.value}; "
        f"failed {[item.invariant_id for item in blocked.failures]}"
    )
    print("canonical state digest unchanged across both previews")

    print("\n=== WORKFLOW VERDICT ===============================================")
    print("GAT now separates three decisions: probabilistic fit, evidence-")
    print("bound construction acceptance, and explicit human authorization.")
    print("Blender/CI can consume the same read-only headless response.")


if __name__ == "__main__":
    run()
