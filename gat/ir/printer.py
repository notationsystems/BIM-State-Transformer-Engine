"""Deterministic textual dump of the Architectural IR.

The printer output is byte-stable for a given module: entities, slots,
relationships, and constraints are emitted in their canonical sorted
orders with fixed float formatting.  The module digest is the SHA-256 of
this text, and a golden-file test pins the demo model's dump.
"""

from __future__ import annotations

from gat.ir.core import (
    ExprEquals,
    LessEqual,
    Module,
    NonNegative,
    Role,
)


def _fmt(value: float) -> str:
    return repr(float(value))


def print_module(module: Module) -> str:
    lines: list[str] = ["gat-ir v0"]
    for key in sorted(module.meta):
        lines.append(f"meta {key} = {module.meta[key]}")

    for eid in module.entities:
        entity = module.entities[eid]
        lines.append(f"entity {eid} name={entity.name!r}")
        if entity.placement is not None:
            p = entity.placement
            lines.append(
                f"  placement x={_fmt(p.x)} y={_fmt(p.y)} z={_fmt(p.z)} angle={_fmt(p.angle)}"
            )
        for akey in sorted(entity.attrs):
            lines.append(f"  attr {akey} = {entity.attrs[akey]!r}")
        for qname in sorted(entity.slots):
            slot = entity.slots[qname]
            if slot.role is Role.RAW:
                lines.append(
                    f"  raw {qname} : {slot.unit.value} "
                    f"~ N({_fmt(slot.prior_mu)}, {_fmt(slot.prior_sigma)}^2)"
                )
            else:
                assert slot.expr is not None
                lines.append(
                    f"  derived {qname} : {slot.unit.value} := {slot.expr.to_str()}"
                )

    for rel in module.rels:
        lines.append(f"rel {rel.kind.value} {rel.source} -> {rel.target}")

    for c in module.constraints:
        if isinstance(c, NonNegative):
            lines.append(f"constraint nonneg {c.var}")
        elif isinstance(c, LessEqual):
            lines.append(f"constraint lesseq {c.lhs} <= {c.rhs}")
        elif isinstance(c, ExprEquals):
            lines.append(f"constraint expreq {c.var} == {c.expr.to_str()}")

    return "\n".join(lines) + "\n"
