"""Stable command-line surface over the headless GAT core."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from gat.ifc_audit import audit_ifc_file


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


def ifc_audit_main(argv: Sequence[str] | None = None) -> int:
    return main(["audit", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    raise SystemExit(main())
