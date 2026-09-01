"""Stable command-line surface over the headless GAT core.

    gat audit   model.ifc                        IFC compatibility inventory (fail-closed)
    gat check   model.ifc [--proposed duct.json] probabilistic clash report
    gat verify  model.ifc                        invariants + compliance
    gat inspect model.ifc [--var Wall-Party.Length] state, sensitivities
    gat splats  model.ifc out/                   splat PLYs (+ --variations N)
    gat sample  model.ifc [--n 500]              realization / violation rates

Every command is deterministic and never mutates the model.  ``--json``
(where offered) switches to machine-readable output.

Exit codes are per-command contracts: ``audit`` returns 0 when the model is
pipeline-ready, 2 when unsupported content blocks ingestion, and 3 on I/O
errors; the state commands return 0 clean, 1 findings (a likely clash, a
failed verification), and 2 on usage or input errors.

A proposed-element spec (for ``check --proposed``) is a small JSON file:

    {"origin": [4.0, 1.8, 2.6], "extents": [3.0, 0.4, 0.4],
     "angle_deg": 0.0, "position_sigma": 0.02}
"""

from __future__ import annotations

import argparse
import json
import math
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
from gat.session import GatSession


def _write_report(rendered: str, output: str | None) -> None:
    if output is None:
        sys.stdout.write(rendered)
    else:
        Path(output).write_text(rendered, encoding="utf-8")


def _run_audit(args: argparse.Namespace) -> int:
    try:
        report = audit_ifc_file(args.model)
        rendered = report.render() + "\n" if args.text else report.to_json(pretty=not args.compact)
        _write_report(rendered, args.output)
    except OSError as exc:
        sys.stderr.write(f"gat audit: {exc}\n")
        return 3
    if report.pipeline_ready:
        return 0
    return 2


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
