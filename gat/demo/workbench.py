"""One worked problem, one instrument: the clearance review walkthrough.

Does the available building evidence support routing a duct through the
party wall at the required clearance?  The engine decides; this demo binds
its decision, the execution history and the corpus audit into a single
offline Notation Workbench page, then asserts what the page must show.

Run with::

    python -m gat.demo.workbench out-dir
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import gat.demo
from gat.causal import AssessmentRecord
from gat.headless import handle_request
from gat.session import GatSession


MODEL = os.path.join(os.path.dirname(gat.demo.__file__), "model.ifc")


def clearance_request() -> dict:
    """A prefabricated duct crossing the party wall, cleared at 95 %."""
    return {
        "format": "gat-headless-request-v1",
        "request_id": "walkthrough-duct-route",
        "operation": "acceptance",
        "state": {"kind": "ifc", "path": MODEL},
        "payload": {
            "case_id": "duct-route-1",
            "workflow": "AS_BUILT_CLEARANCE",
            "subject": "crossing duct",
            "checks": [
                {
                    "kind": "clearance",
                    "check_id": "route-clearance",
                    "proposal": {
                        "origin": [4.0, 1.8, 2.6],
                        "angle": 0.0,
                        "extents": [3.0, 0.4, 0.4],
                    },
                    "required_clearance": 0.05,
                    "confidence": 0.95,
                    "position_sigma": 0.02,
                    "label": "crossing duct",
                }
            ],
        },
    }


def run(out_dir: str) -> dict[str, str]:
    from gat.cli import main as cli_main

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. The question, bound to a specific proposal and this exact model.
    request = clearance_request()
    response = handle_request(request)
    (out / "request.json").write_text(json.dumps(request, indent=1), encoding="utf-8")
    (out / "response.json").write_text(json.dumps(response, indent=1), encoding="utf-8")
    disposition = response["result"]["disposition"]

    # 2. The history: the engine's own record of the assessment.
    session = GatSession.load_ifc(MODEL)
    session.record_assessment(
        AssessmentRecord(
            world_digest=session.world.digest(),
            assessment_id="duct-route-1",
            assessment_type="as-built-clearance",
            subject="Wall-Party",
            verdict="VIOLATED" if disposition == "REJECT" else "SATISFIED",
            method="gat-headless acceptance v1",
        ),
        provenance={"phase": "walkthrough", "request_id": request["request_id"]},
    )
    ledger_path = out / "ledger.json"
    session.export_ledger(str(ledger_path))

    # 3. One page: decision at the spot it was decided, evidence, history, audit.
    page = out / "workbench.html"
    code = cli_main(
        [
            "workbench", MODEL, "-o", str(page), "--variations", "4",
            "--decision", str(out / "response.json"),
            "--request", str(out / "request.json"),
            "--ledger", str(ledger_path),
        ]
    )
    if code != 0:
        raise SystemExit(f"gat workbench exited {code}")
    html = page.read_text(encoding="utf-8")

    # 4. Self-asserting: the page shows the verdict, the history and the corpus limits.
    expected = {
        "verdict": f'"disposition":"{disposition}"',
        "subject": "Wall-Party",
        "history": "hash chain verified",
        "audit": "gat-ifc-audit-v1",
        "identity": session.world.digest(),
        "map honesty": 'data-mode="MAP" class="unavailable"',
    }
    missing = [label for label, needle in expected.items() if needle not in html]
    if missing:
        raise SystemExit(f"workbench page is missing: {', '.join(missing)}")
    if disposition != "REJECT":
        raise SystemExit(f"walkthrough expects REJECT for the crossing duct, got {disposition}")
    return {"disposition": disposition, "page": str(page), "ledger": str(ledger_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir", help="directory for the request, response, ledger and page")
    args = parser.parse_args(argv)
    result = run(args.out_dir)
    print(
        f"clearance review: {result['disposition']} — open {result['page']} "
        "(STRUCTURE shows the duct where it was refused; EVIDENCE lists the "
        "next evidence; TIME holds the record)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
