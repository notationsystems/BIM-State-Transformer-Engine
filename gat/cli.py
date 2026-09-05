"""Stable command-line surface over the headless GAT core.

    gat audit   model.ifc                        IFC compatibility inventory (fail-closed)
    gat check   model.ifc [--proposed duct.json] probabilistic clash report
    gat verify  model.ifc                        invariants + compliance
    gat inspect model.ifc [--var Wall-Party.Length] state, sensitivities
    gat splats  model.ifc out/                   splat PLYs (+ --variations N)
    gat sample  model.ifc [--n 500]              realization / violation rates
    gat report  response.json [--html]           render a gat-headless decision
    gat ledger  ledger.json [--html]             render an execution-ledger timeline
    gat view    model.ifc -o viewer.html         offline 3D viewer (+ --decision overlay)

Every command is deterministic and never mutates the model.  ``--json``
(where offered) switches to machine-readable output.

Exit codes are per-command contracts: ``audit`` returns 0 when the model is
pipeline-ready, 2 when unsupported content blocks ingestion, and 3 on I/O
errors; ``report`` returns 0 for a rendered decision, 1 when the response
is a rendered headless error, 2 for invalid input, and 3 on I/O errors;
``ledger`` returns 0 for a rendered timeline, 2 for an invalid or
tampered chain, and 3 on I/O errors;
the state commands return 0 clean, 1 findings (a likely clash, a failed
verification), and 2 on usage or input errors.

A proposed-element spec (for ``check --proposed``) is a small JSON file:

    {"origin": [4.0, 1.8, 2.6], "extents": [3.0, 0.4, 0.4],
     "angle_deg": 0.0, "position_sigma": 0.02}
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Sequence

from gat.engine.sampling import sample_report
from gat.engine.sensitivity import sensitivities_of, variance_attribution
from gat.errors import GatError
from gat.geometry.clash import detect, score_proposed_box
from gat.geometry.compliance import check_compliance
from gat.geometry.gaussianize import OrientedBox
from gat.geometry.splat_io import export_splat_ply
from gat.geometry.stateio import derive_scene
from gat.geometry.variations import export_variations, variation_spread
from gat.ids import VarId
from gat.ifc_audit import audit_ifc_file
from gat.report import decode_ledger, decode_response, render_html, render_text
from gat.session import GatSession


def _write_report(rendered: str, output: str | None) -> None:
    if output is None:
        sys.stdout.write(rendered)
    else:
        Path(output).write_text(rendered, encoding="utf-8")


def _run_audit(args: argparse.Namespace) -> int:
    try:
        report = audit_ifc_file(args.model)
        if args.html:
            rendered = render_html(decode_response(report.to_dict()))
        elif args.text:
            rendered = report.render() + "\n"
        else:
            rendered = report.to_json(pretty=not args.compact)
        _write_report(rendered, args.output)
    except OSError as exc:
        sys.stderr.write(f"gat audit: {exc}\n")
        return 3
    if report.pipeline_ready:
        return 0
    return 2


def _run_report(args: argparse.Namespace) -> int:
    try:
        if args.response == "-":
            value = json.load(sys.stdin)
        else:
            with open(args.response, "r", encoding="utf-8") as handle:
                value = json.load(handle)
        report = decode_response(value)
        rendered = render_html(report) if args.html else render_text(report) + "\n"
        _write_report(rendered, args.output)
    except OSError as exc:
        sys.stderr.write(f"gat report: {exc}\n")
        return 3
    return 1 if report.operation == "error" else 0


def _bind_decision(command: str, args: argparse.Namespace):
    """Load the model and bind an optional headless decision to it, fail-closed.

    Returns ``(session, model_path, decision, request, response)``; the
    decision overlay is ``None`` when no ``--decision`` was given.
    """
    from gat.geometry.viewer import decision_overlay

    model_path = args.model
    decision = None
    request = None
    response = None
    if args.request and not args.decision:
        raise ValueError(f"{command}: --request requires --decision")
    if args.request:
        with open(args.request, "r", encoding="utf-8") as handle:
            request = json.load(handle)
        # The world digest carries the source path string, so load the model
        # exactly as the request did when both name the same file.
        state = request.get("state") if isinstance(request, dict) else None
        request_path = state.get("path") if isinstance(state, dict) else None
        if (
            isinstance(request_path, str)
            and os.path.exists(request_path)
            and os.path.samefile(request_path, args.model)
        ):
            model_path = request_path
    session = _load(model_path)
    if args.decision:
        with open(args.decision, "r", encoding="utf-8") as handle:
            response = json.load(handle)
        try:
            decision = decision_overlay(session.world, response, request)
        except ValueError as exc:
            if "different world" in str(exc):
                raise ValueError(
                    f"{exc}; the world digest includes the model's path string, so "
                    "load it with the same path form the headless request used "
                    f"(or pass --request so {command} can match it)"
                ) from exc
            raise
    return session, model_path, decision, request, response


def _run_view(args: argparse.Namespace) -> int:
    from gat.geometry.viewer import export_viewer_html

    session, _, decision, _, _ = _bind_decision("gat view", args)
    count = export_viewer_html(
        session.world,
        args.output,
        n=args.variations,
        seed=args.seed,
        spacing=args.spacing,
        model_name=args.model,
        decision=decision,
    )
    suffix = f" + decision {decision['disposition']}" if decision else ""
    print(f"wrote {args.output}: offline viewer with {count} realizations{suffix}")
    return 0


def _run_workbench(args: argparse.Namespace) -> int:
    from gat.workbench import export_workbench_html

    session, model_path, decision, _, response = _bind_decision("gat workbench", args)
    decision_report = decode_response(response) if response is not None else None
    ledger = decode_ledger(args.ledger) if args.ledger else None
    audit = None
    audit_reason = ""
    if args.no_audit:
        audit_reason = "The IFC audit was skipped (--no-audit)."
    elif model_path.lower().endswith(".ifc"):
        audit_document = audit_ifc_file(model_path)
        if audit_document.world_digest not in (None, session.world.digest()):
            raise ValueError(
                "gat workbench: the audit lowered a different world than the model "
                f"({audit_document.world_digest} != {session.world.digest()})"
            )
        audit = decode_response(audit_document.to_dict())
    else:
        suffix = Path(model_path).suffix or "(no extension)"
        audit_reason = (
            "The IFC audit applies to IFC sources; this model was loaded from "
            f"a {suffix} carrier."
        )
    availability = export_workbench_html(
        session.world,
        args.output,
        model_name=args.model,
        n=args.variations,
        seed=args.seed,
        spacing=args.spacing,
        decision=decision,
        decision_report=decision_report,
        ledger=ledger,
        audit=audit,
        audit_reason=audit_reason,
    )
    modes = ", ".join(
        mode if state == "available" else f"{mode} ({state})"
        for mode, state in availability.items()
    )
    print(f"wrote {args.output}: workbench with modes {modes}")
    return 0


def _run_ledger(args: argparse.Namespace) -> int:
    try:
        report = decode_ledger(args.ledger)
        rendered = render_html(report) if args.html else render_text(report) + "\n"
        _write_report(rendered, args.output)
    except OSError as exc:
        sys.stderr.write(f"gat ledger: {exc}\n")
        return 3
    return 0


# -- state commands ---------------------------------------------------------


def _load(path: str) -> GatSession:
    if path.endswith((".usda", ".usdc")):
        return GatSession.load_usd(path)
    return GatSession.load_ifc(path)


def _pretty_var(world, var: VarId) -> str:
    """Entity-name form of a variable when the name is unique."""
    entity = world.module.entities.get(var.entity)
    if entity is not None and entity.name:
        same_name = sum(
            1 for e in world.module.entities.values() if e.name == entity.name
        )
        if same_name == 1:
            return f"{entity.name}.{var.quantity}"
    return str(var)


def _emit(data: dict, as_json: bool, human: str) -> None:
    if as_json:
        json.dump(data, sys.stdout, indent=1, sort_keys=True)
        sys.stdout.write("\n")
    else:
        print(human)


def _clash_item_dict(item) -> dict:
    return {
        "a": item.element_a,
        "b": item.element_b,
        "clearance": item.clearance,
        "sigma": item.sigma,
        "p_clash": item.p_clash,
        "overlap_mass": item.overlap_mass,
        "separation_significance": item.separation_sig,
        "witness": list(item.witness),
    }


def _run_check(args: argparse.Namespace) -> int:
    session = _load(args.model)
    scene = derive_scene(session.world)
    report = detect(scene, max_clearance=args.max_clearance)
    items = list(report.items)

    proposed_items = []
    if args.proposed:
        with open(args.proposed, encoding="utf-8") as fh:
            spec = json.load(fh)
        box = OrientedBox(
            origin=tuple(spec["origin"]),
            angle=math.radians(float(spec.get("angle_deg", 0.0))),
            extents=tuple(spec["extents"]),
        )
        proposed = score_proposed_box(
            scene,
            box,
            position_sigma=float(spec.get("position_sigma", 0.0)),
            max_clearance=args.max_clearance,
        )
        proposed_items = list(proposed.items)

    worst = max(
        (it.p_clash for it in items + proposed_items), default=0.0
    )
    data = {
        "model": args.model,
        "existing_pairs": [_clash_item_dict(it) for it in items],
        "proposed": [_clash_item_dict(it) for it in proposed_items],
        "worst_p_clash": worst,
        "fail_above": args.fail_above,
    }
    human = [report.render()]
    if args.proposed:
        human.append("")
        human.append(f"proposed element ({args.proposed}):")
        if proposed_items:
            human.extend(it.render() for it in proposed_items)
        else:
            human.append("  no clashes within broad-phase range")
    _emit(data, args.json, "\n".join(human))
    return 1 if worst >= args.fail_above else 0


def _run_verify(args: argparse.Namespace) -> int:
    session = _load(args.model)
    invariants = session.verify()
    compliance = check_compliance(session.world)
    passed = invariants.passed and compliance.passed
    p, w, f = invariants.counts()
    data = {
        "model": args.model,
        "invariants": {"pass": p, "warn": w, "fail": f},
        "compliance": [
            {
                "rule": r.rule,
                "subject": r.subject,
                "margin_mean": r.margin_mean,
                "margin_sigma": r.margin_sigma,
                "p_satisfied": r.p_satisfied,
                "status": r.status,
            }
            for r in compliance.rows
        ],
        "passed": passed,
    }
    human = invariants.render() + "\n" + compliance.render()
    _emit(data, args.json, human)
    return 0 if passed else 1


def _run_inspect(args: argparse.Namespace) -> int:
    session = _load(args.model)
    world = session.world
    if args.var is None:
        census: dict[str, int] = {}
        for eid in world.module.entities:
            census[eid.ifc_class] = census.get(eid.ifc_class, 0) + 1
        data = {
            "model": args.model,
            "entities": census,
            "n_raw": world.binding.n_raw,
            "n_derived": world.binding.n_full - world.binding.n_raw,
            "digest": world.digest(),
            "spread": variation_spread(world),
        }
        human = [
            f"{args.model}: "
            + ", ".join(f"{n} {c}" for c, n in sorted(census.items())),
            f"variables: {world.binding.n_raw} raw + "
            f"{world.binding.n_full - world.binding.n_raw} derived",
            f"digest: {world.digest()[:16]}...",
        ]
        _emit(data, args.json, "\n".join(human))
        return 0

    entity_name, _, quantity = args.var.rpartition(".")
    if not entity_name:
        print(f"--var must be Entity-Name.Quantity, got {args.var!r}", file=sys.stderr)
        return 2
    var = VarId(session.entity_by_name(entity_name), quantity)
    mean = world.full.mean(var)
    sigma = world.full.std(var)
    sens = sensitivities_of(world, var)[: args.top]
    attribution = variance_attribution(world, var)[: args.top]
    data = {
        "var": str(var),
        "mean": mean,
        "sigma": sigma,
        "sensitivities": [{"wrt": str(v), "d": d} for v, d in sens],
        "variance_attribution": [
            {"source": str(v), "share": s} for v, s in attribution
        ],
    }
    human = [f"{args.var}: {mean:.6f} +- {sigma:.6f}"]
    if sens:
        human.append("sensitivities (d target / d raw):")
        human.extend(f"  {d:+12.4f}  {_pretty_var(world, v)}" for v, d in sens)
    if attribution:
        human.append("variance attribution:")
        human.extend(
            f"  {s*100:6.1f}%  {_pretty_var(world, v)}" for v, s in attribution
        )
    _emit(data, args.json, "\n".join(human))
    return 0


def _run_splats(args: argparse.Namespace) -> int:
    import os

    session = _load(args.model)
    os.makedirs(args.out_dir, exist_ok=True)
    scene = derive_scene(session.world, spacing=args.spacing)
    nominal = os.path.join(args.out_dir, "building.ply")
    count = export_splat_ply(scene.cloud, nominal)
    lines = [f"wrote {nominal}: {count} splats"]
    if args.variations:
        manifest = export_variations(
            session.world,
            args.out_dir,
            n=args.variations,
            seed=args.seed,
            spacing=args.spacing,
        )
        passed = sum(1 for s in manifest["samples"] if s["passed"])
        lines.append(
            f"wrote {args.variations} belief-sampled variations "
            f"(seed {args.seed}; {passed}/{args.variations} pass invariants) "
            f"+ manifest.json"
        )
    print("\n".join(lines))
    return 0


def _run_sample(args: argparse.Namespace) -> int:
    session = _load(args.model)
    report = sample_report(session.world, n=args.n, seed=args.seed)
    data = {
        "model": args.model,
        "n": report.n,
        "seed": report.seed,
        "pass_rate": report.pass_rate,
        "violation_rates": [
            {"failure": key, "rate": rate} for key, rate in report.violation_rates
        ],
    }
    _emit(data, args.json, report.render())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gat",
        description="Decision-focused, uncertainty-aware BIM state tools",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser(
        "audit",
        help="inventory IFC compatibility without mutating or partially importing it",
    )
    audit.add_argument("model", help="IFC STEP file to inspect")
    audit.add_argument("-o", "--output", help="write the report to this path")
    audit.add_argument("--compact", action="store_true", help="emit compact canonical JSON")
    audit.add_argument("--text", action="store_true", help="emit a concise human-readable summary")
    audit.add_argument(
        "--html",
        action="store_true",
        help="emit a self-contained, script-free HTML report",
    )
    audit.set_defaults(handler=_run_audit)

    p = commands.add_parser("check", help="probabilistic clash report")
    p.add_argument("model")
    p.add_argument("--proposed", help="JSON spec of a proposed element")
    p.add_argument("--max-clearance", type=float, default=0.5)
    p.add_argument("--fail-above", type=float, default=0.5,
                   help="exit 1 when any P(clash) reaches this (default 0.5)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(handler=_run_check)

    p = commands.add_parser("verify", help="invariants + compliance under uncertainty")
    p.add_argument("model")
    p.add_argument("--json", action="store_true")
    p.set_defaults(handler=_run_verify)

    p = commands.add_parser("inspect", help="state summary or one variable in depth")
    p.add_argument("model")
    p.add_argument("--var", help="Entity-Name.Quantity, e.g. Wall-Party.Length")
    p.add_argument("--top", type=int, default=6)
    p.add_argument("--json", action="store_true")
    p.set_defaults(handler=_run_inspect)

    p = commands.add_parser("splats", help="3DGS splat export (+ sampled variations)")
    p.add_argument("model")
    p.add_argument("out_dir")
    p.add_argument("--variations", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--spacing", type=float, default=0.75)
    p.set_defaults(handler=_run_splats)

    p = commands.add_parser("sample", help="invariant checking over belief realizations")
    p.add_argument("model")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", action="store_true")
    p.set_defaults(handler=_run_sample)

    p = commands.add_parser(
        "report",
        help="render a gat-headless response for humans (terminal or HTML)",
    )
    p.add_argument("response", help="headless response JSON path, or - for stdin")
    p.add_argument("-o", "--output", help="write the rendering to this path")
    p.add_argument(
        "--html",
        action="store_true",
        help="emit a self-contained, script-free HTML report",
    )
    p.set_defaults(handler=_run_report)

    p = commands.add_parser(
        "ledger",
        help="render a hash-chained execution ledger as a human timeline",
    )
    p.add_argument("ledger", help="execution ledger JSON path")
    p.add_argument("-o", "--output", help="write the rendering to this path")
    p.add_argument(
        "--html",
        action="store_true",
        help="emit a self-contained, script-free HTML timeline",
    )
    p.set_defaults(handler=_run_ledger)

    p = commands.add_parser(
        "view",
        help="self-contained offline 3D viewer of the belief and its samples",
    )
    p.add_argument("model")
    p.add_argument("-o", "--output", required=True, help="viewer HTML path")
    p.add_argument("--variations", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--spacing", type=float, default=0.75)
    p.add_argument(
        "--decision",
        help="gat-headless response to overlay (must be evaluated on this model)",
    )
    p.add_argument(
        "--request",
        help="the matching gat-headless request; draws proposed clearance geometry",
    )
    p.set_defaults(handler=_run_view)

    p = commands.add_parser(
        "workbench",
        help="the Notation Workbench: one offline instrument, eight projection modes",
    )
    p.add_argument("model")
    p.add_argument("-o", "--output", required=True, help="workbench HTML path")
    p.add_argument("--variations", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--spacing", type=float, default=0.75)
    p.add_argument(
        "--decision",
        help="gat-headless response to bind (EVIDENCE mode + STRUCTURE overlay)",
    )
    p.add_argument(
        "--request",
        help="the matching gat-headless request; draws proposed clearance geometry",
    )
    p.add_argument("--ledger", help="execution ledger JSON to bind (TIME mode)")
    p.add_argument(
        "--no-audit",
        action="store_true",
        help="skip the IFC compatibility audit (COMPLEXITY mode stays empty)",
    )
    p.set_defaults(handler=_run_workbench)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 2 if exc.code not in (0, None) else 0
    try:
        return int(args.handler(args))
    except FileNotFoundError as exc:
        print(f"gat: {exc}", file=sys.stderr)
        return 2
    except (GatError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"gat: {exc}", file=sys.stderr)
        return 2


def ifc_audit_main(argv: Sequence[str] | None = None) -> int:
    return main(["audit", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    raise SystemExit(main())
