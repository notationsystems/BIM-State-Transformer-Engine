"""The Notation Workbench: one instrument, eight projection modes, one identity.

The platform synthesis names a projection triad — kepler.gl for analytical
geography (*where is the pattern?*), CesiumJS for geodetic reality (*where
does it exist?*), Three.js for computational structure (*how is it
constituted?*) — behind a mode toolbar: MAP, GLOBE, STRUCTURE, GRAPH, STATE,
TIME, EVIDENCE, COMPLEXITY.  This module composes the surfaces GAT already
produces into that instrument, one self-contained HTML file:

* STRUCTURE embeds the offline 3D viewer (the Three.js seat, filled today
  by the hand-rolled WebGL renderer) in a sandboxed frame;
* GRAPH draws the IR relationship graph ``G`` with a deterministic layout
  and its provenance;
* STATE lists the belief ``N(mu, Sigma)`` per entity — every quantity's
  mean and sigma, raw or derived;
* TIME, EVIDENCE and COMPLEXITY host the ledger timeline, the bound
  decision report and the IFC audit in the report grammar;
* MAP and GLOBE are declared **unavailable** with the reason: this corpus
  release lowers no geographic coordinate semantics, and a globe would be a
  connected instrument, a surface class not yet defined.

Three rules carry across every mode.  Projection never mutates its source:
the page renders and re-checks, it never writes.  Identity survives
representation: the same ``EntityId`` names an element in every mode, the
same world digest names the world, and the frames exchange only ids.
Visual adjacency is never evidence: the graph layout is a reading order,
not a measurement, and says so.

Every ``ProjectionSpec`` — what a mode projects, from which source, whether
it is available and why not — is embedded verbatim, so the instrument
states what it is not.  See ``docs/projection-spec-v1.md``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import html as html_mod
import json
from pathlib import Path
from typing import Mapping

from gat.engine.executor import World
from gat.geometry.viewer import (
    VIEWER_SCENE_FORMAT,
    frame_record,
    render_viewer_html,
    viewer_payload,
)
from gat.ir.core import RelKind
from gat.report import (
    _HTML_STYLE,
    DecisionReport,
    NON_AUTHORIZING_FOOTER,
    READ_ONLY_FOOTER,
    disposition_hex,
    format_digest,
    render_html_fragment,
)

WORKBENCH_FORMAT = "gat-workbench-v1"
MESSAGE_FORMAT = "gat-workbench-message-v1"
PROJECTION_SPEC_VERSION = "gat-projection-spec-v1"

#: The mode toolbar, in the order the synthesis names it.
MODES = ("MAP", "GLOBE", "STRUCTURE", "GRAPH", "STATE", "TIME", "EVIDENCE", "COMPLEXITY")

AVAILABLE = "available"
EMPTY = "empty"            # the mode exists here; nothing is bound to it yet
UNAVAILABLE = "unavailable"  # the corpus cannot fill it; the reason is stated

#: Reading-order rank of IFC classes for the GRAPH layout: containers first,
#: then spaces, then the elements that bound them, then what voids and fills
#: them.  Position on the canvas is a reading order, never a measurement.
_GRAPH_RANKS = (
    "IfcProject",
    "IfcSite",
    "IfcBuilding",
    "IfcBuildingStorey",
    "IfcSpace",
    "IfcWall",
    "IfcWallStandardCase",
    "IfcSlab",
    "IfcColumn",
    "IfcBeam",
    "IfcOpeningElement",
    "IfcDoor",
    "IfcWindow",
)

_RULES = (
    "projection never mutates its source",
    "identity survives representation",
    "visual adjacency is never evidence",
)


@dataclass(frozen=True)
class ProjectionSpec:
    """What one Workbench mode projects, from where, and whether it can."""

    mode: str
    seat: str
    question: str
    surface_class: str
    #: Every representation identifies its source, its transformation, its
    #: supported meaning and its information loss — and the identity, frame
    #: and time in which it may be compared with another representation.
    source: str
    transformation: str
    meaning: str
    loss: str
    identity: str
    frame: str
    time: str
    availability: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {"version": PROJECTION_SPEC_VERSION, **asdict(self), "mutates_source": False}


def projection_specs(
    world: World,
    *,
    decision_bound: bool,
    ledger_bound: bool,
    audit_bound: bool,
    audit_reason: str = "",
) -> tuple[ProjectionSpec, ...]:
    """The eight ProjectionSpecs for this world and what is bound to it."""
    module = world.module
    geographic = (
        "This corpus release lowers no geographic coordinate semantics: no "
        "IfcSite placement, IfcMapConversion or coordinate reference system "
        "reaches the IR, so nothing can be placed on a map without inventing "
        "a position — and an invented position would be visual adjacency "
        "presented as evidence."
    )
    one_world = "one belief state, named by its world digest"
    frame = frame_record(world)
    model_frame = (
        f"{frame['id']} frame: {frame['convention']}; {frame['units']}, {frame['up']} up, "
        f"{frame['handedness']}-handed; no geodetic frame (CRS none); {frame['uncertainty']}"
    )
    ir_units = (
        f"IR units as lowered: source unit {frame['source_unit']} x "
        f"{frame['scale_to_metres']} -> m (m, m2, m3, cur)"
    )
    return (
        ProjectionSpec(
            mode="MAP",
            seat="kepler.gl — analytical geography",
            question="Where is the pattern?",
            surface_class="instrument",
            source="geographic coordinates and aggregates (none in this corpus)",
            transformation="none performed",
            meaning="would be: patterns and aggregates over geography",
            loss="not assessable until a source exists",
            identity="EntityId (would be)",
            frame="none: no coordinate reference system is lowered",
            time="none",
            availability=UNAVAILABLE,
            reason=geographic,
        ),
        ProjectionSpec(
            mode="GLOBE",
            seat="CesiumJS — geodetic reality",
            question="Where does it exist?",
            surface_class="connected instrument (not yet defined)",
            source="geodetic placement plus terrain and imagery tiles (none in this corpus)",
            transformation="none performed",
            meaning="would be: the element at its place on the earth",
            loss="not assessable until a source exists",
            identity="EntityId (would be)",
            frame="none: no geodetic datum is lowered",
            time="none",
            availability=UNAVAILABLE,
            reason=geographic
            + " A globe also fetches terrain and imagery over the network: a "
            "connected instrument, a surface class this release does not define.",
        ),
        ProjectionSpec(
            mode="STRUCTURE",
            seat="Three.js seat — computational structure (self-contained WebGL viewer today)",
            question="How is it constituted?",
            surface_class="instrument",
            source=f"gat view scene ({VIEWER_SCENE_FORMAT}) derived from the world"
            + (", with the bound decision overlay" if decision_bound else ""),
            transformation="oriented boxes -> moment-matched Gaussian tiles -> k-sigma "
            "ellipsoids; N realizations drawn from N(mu, Sigma) with the stated seed; "
            "EXPLODE displaces pieces along the IR hierarchy (a reading offset with "
            "leader lines, never a position)",
            meaning="the belief's marginal geometry and correlated sampled realizations; "
            "a painted colour is a verdict only under a bound decision; an outline in "
            "an audit status colour marks what the corpus could not fully represent",
            loss="cross-element covariance appears only through sampled realizations, "
            "never as a shape; placements are exact metadata, not belief; tiles "
            "approximate boxes; exploded positions carry no information",
            identity="EntityId per element; world digest per scene",
            frame=model_frame,
            time=one_world + "; realizations are draws, not moments in time",
            availability=AVAILABLE,
            reason="",
        ),
        ProjectionSpec(
            mode="GRAPH",
            seat="IR relationship graph G",
            question="What relates to what, and on whose authority?",
            surface_class="instrument",
            source=f"{len(module.entities)} entities, {len(module.rels)} typed edges "
            "with IFC source references",
            transformation="rank by IFC class, order by EntityId; edges typed and "
            "referenced",
            meaning="which entities relate, how, and which IFC record asserts it",
            loss="no geometry, no quantities; position and distance on the canvas "
            "carry no information",
            identity="EntityId",
            frame="none: a reading order",
            time=one_world + "; the symbolic structure is immutable in v0",
            availability=AVAILABLE,
            reason="",
        ),
        ProjectionSpec(
            mode="STATE",
            seat="IR entities X over the belief N(mu, Sigma)",
            question="What is believed, and how surely?",
            surface_class="instrument",
            source=f"{len(tuple(module.all_slots()))} quantity slots "
            f"({len(module.raw_vars())} raw, {len(module.derived_vars())} derived)",
            transformation="marginal mean and sigma per quantity at six significant digits",
            meaning="what is believed about each quantity: raw (canonical belief) or "
            "derived (pushforward), with the IFC record it came from",
            loss="correlations between quantities are not shown (marginals only); "
            "rounding at six significant digits",
            identity="EntityId plus quantity name (VarId)",
            frame=ir_units,
            time=one_world,
            availability=AVAILABLE,
            reason="",
        ),
        ProjectionSpec(
            mode="TIME",
            seat="execution ledger timeline",
            question="What happened, in what order, and did the chain hold?",
            surface_class="report",
            source="hash-chained execution ledger (gat ledger)"
            if ledger_bound
            else "no ledger bound",
            transformation="chain verified in full, then one card per event with its "
            "scalar fields, provenance, world digests and verification counts",
            meaning="the recorded sequence of transitions, assessments, policies, "
            "approvals and rejections",
            loss="nested JSON is elided; timelines over 50 events elide the middle "
            "explicitly",
            identity="event seq and hash; prior and result world digests",
            frame="none",
            time="ledger sequence order; wall-clock only where provenance recorded it",
            availability=AVAILABLE if ledger_bound else EMPTY,
            reason=""
            if ledger_bound
            else "No execution ledger is bound. Export one (session.export_ledger, or "
            "python -m gat.demo.ledger_replay ledger.json) and pass --ledger.",
        ),
        ProjectionSpec(
            mode="EVIDENCE",
            seat="decision report",
            question="What was decided, on what evidence, and what is still missing?",
            surface_class="report",
            source="gat-headless response (gat report)" if decision_bound else "no decision bound",
            transformation="identities re-checked, then the report grammar: headline, "
            "reasons, cards, tables, footers",
            meaning="the engine's disposition, reasons, risks and requested evidence, "
            "verbatim",
            loss="none intended: probabilities at five decimals, digests abbreviated "
            "with the full value preserved",
            identity="request id and world digest; subjects are named by entity name, "
            "not EntityId (an engine contract, noted)",
            frame="as evaluated by the engine",
            time="the world the response was evaluated on",
            availability=AVAILABLE if decision_bound else EMPTY,
            reason=""
            if decision_bound
            else "No decision is bound. Evaluate a request with gat-headless and pass "
            "--decision response.json [--request request.json].",
        ),
        ProjectionSpec(
            mode="COMPLEXITY",
            seat="IFC compatibility audit",
            question="What can this corpus represent, and what can it not?",
            surface_class="report",
            source="gat-ifc-audit-v1 (gat audit)" if audit_bound else "no audit bound",
            transformation="readiness re-checked against stage statuses, then the report "
            "grammar",
            meaning="pipeline stages, supported products and issues with their own "
            "statuses; unsupported structures listed, never skipped",
            loss="supported-product scope only (the audit's coverage boundary)",
            identity="source file sha256 and the lowered world digest",
            frame="the IFC length unit as audited",
            time="the audited file version",
            availability=AVAILABLE if audit_bound else EMPTY,
            reason="" if audit_bound else (audit_reason or "No IFC audit is bound."),
        ),
    )


def graph_payload(world: World) -> dict[str, object]:
    """The relationship graph with a deterministic reading-order layout.

    Nodes are ranked by IFC class (containers above spaces above elements
    above openings and fills) and ordered within a rank by ``EntityId``;
    rows are compacted to the ranks present.  Edges carry their kind and
    their IFC source reference, so an edge without provenance is visible.
    """
    module = world.module
    rank_of = {name: index for index, name in enumerate(_GRAPH_RANKS)}
    ranks = sorted(
        {rank_of.get(eid.ifc_class, len(_GRAPH_RANKS)) for eid in module.entities}
    )
    row_of = {rank: row for row, rank in enumerate(ranks)}
    per_row: dict[int, list] = {row: [] for row in row_of.values()}
    for eid, entity in module.entities.items():
        row = row_of[rank_of.get(eid.ifc_class, len(_GRAPH_RANKS))]
        per_row[row].append((eid, entity))
    nodes = []
    for row in sorted(per_row):
        members = per_row[row]
        for column, (eid, entity) in enumerate(members):
            nodes.append(
                {
                    "entity": str(eid),
                    "name": entity.name,
                    "class": eid.ifc_class,
                    "row": row,
                    "column": column,
                    "columns": len(members),
                    "source_ref": entity.source_ref,
                    "quantities": len(entity.slots),
                }
            )
    edges = [
        {
            "kind": rel.kind.value,
            "source": str(rel.source),
            "target": str(rel.target),
            "source_ref": rel.source_ref,
        }
        for rel in module.rels
    ]
    kinds = [
        {"kind": kind.value, "count": sum(1 for rel in module.rels if rel.kind is kind)}
        for kind in RelKind
        if any(rel.kind is kind for rel in module.rels)
    ]
    return {
        "rows": len(ranks),
        "row_classes": [
            sorted({eid.ifc_class for eid, _ in per_row[row]}) for row in sorted(per_row)
        ],
        "nodes": nodes,
        "edges": edges,
        "kinds": kinds,
    }


def state_payload(world: World) -> dict[str, object]:
    """Every entity's quantities with their believed mean and sigma."""
    module = world.module
    entities = []
    for eid, entity in module.entities.items():
        quantities = []
        for name in sorted(entity.slots):
            slot = entity.slots[name]
            quantities.append(
                {
                    "name": name,
                    "role": slot.role.value,
                    "unit": slot.unit.value,
                    "mean": float(f"{world.full.mean(slot.var):.6g}"),
                    "sigma": float(f"{world.full.std(slot.var):.6g}"),
                    "source_ref": slot.source_ref,
                }
            )
        entities.append(
            {
                "entity": str(eid),
                "name": entity.name,
                "class": eid.ifc_class,
                "source_ref": entity.source_ref,
                "attrs": {key: value for key, value in sorted(entity.attrs.items())},
                "quantities": quantities,
            }
        )
    meta = dict(module.meta)
    return {
        "world_digest": world.digest(),
        "frame": frame_record(world),
        "entities": entities,
        "raw": len(module.raw_vars()),
        "derived": len(module.derived_vars()),
        "relationships": len(module.rels),
        "constraints": len(module.constraints),
        "meta": {key: str(meta[key]) for key in sorted(meta)},
    }


def workbench_payload(
    world: World,
    *,
    model_name: str = "",
    n: int = 8,
    seed: int = 0,
    spacing: float = 0.75,
    decision: Mapping[str, object] | None = None,
    decision_report: DecisionReport | None = None,
    ledger: DecisionReport | None = None,
    audit: DecisionReport | None = None,
    audit_reason: str = "",
    audit_statuses: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """The instrument's data: specs, the STRUCTURE scene, GRAPH, STATE.

    ``decision`` is the viewer overlay from :func:`decision_overlay`, already
    bound fail-closed to ``world``; ``decision_report``, ``ledger`` and
    ``audit`` are decoded reports for EVIDENCE, TIME and COMPLEXITY.  A
    decision report whose world differs from the overlay's is refused.
    """
    if (decision is None) != (decision_report is None):
        raise ValueError("a decision overlay and its report must be bound together")
    if decision_report is not None and decision is not None:
        if decision_report.disposition != decision["disposition"]:
            raise ValueError("decision report and overlay disagree on the disposition")
    if audit_statuses is not None and audit is None:
        raise ValueError("audit statuses come from the bound audit; bind both or neither")
    scene = viewer_payload(
        world,
        n=n,
        seed=seed,
        spacing=spacing,
        model_name=model_name,
        decision=decision,
        audit_statuses=audit_statuses,
    )
    specs = projection_specs(
        world,
        decision_bound=decision is not None,
        ledger_bound=ledger is not None,
        audit_bound=audit is not None,
        audit_reason=audit_reason,
    )
    return {
        "format": WORKBENCH_FORMAT,
        "message_format": MESSAGE_FORMAT,
        "model": model_name,
        "world_digest": world.digest(),
        "rules": list(_RULES),
        "modes": [spec.to_dict() for spec in specs],
        "structure": scene,
        "graph": graph_payload(world),
        "state": state_payload(world),
        "decision": None
        if decision is None
        else {
            "disposition": decision["disposition"],
            "headline": decision["headline"],
            "color": decision["color"],
            "subjects": list(decision["subjects"]),
        },
    }


def render_workbench_html(
    payload: Mapping[str, object],
    *,
    decision_report: DecisionReport | None = None,
    ledger: DecisionReport | None = None,
    audit: DecisionReport | None = None,
) -> str:
    """Compose the single-file instrument from a payload and its reports."""
    esc = html_mod.escape
    specs = {spec["mode"]: spec for spec in payload["modes"]}
    viewer_document = render_viewer_html(payload["structure"])
    page_payload = {key: value for key, value in payload.items() if key != "structure"}
    # The data block is JSON, not markup: every angle bracket and ampersand
    # is escaped so no untrusted string can ever read as a tag.
    encoded = (
        json.dumps(page_payload, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )

    def spec_strip(mode: str) -> str:
        spec = specs[mode]
        rows = [
            ("seat", spec["seat"]),
            ("question", spec["question"]),
            ("surface class", spec["surface_class"]),
            ("source", spec["source"]),
            ("transformation", spec["transformation"]),
            ("meaning", spec["meaning"]),
            ("information loss", spec["loss"]),
            ("identity", spec["identity"]),
            ("frame", spec["frame"]),
            ("time", spec["time"]),
            ("availability", spec["availability"]),
        ]
        if spec["reason"]:
            rows.append(("reason", spec["reason"]))
        body = "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in rows)
        return (
            f'<details class="spec {esc(spec["availability"])}">'
            f"<summary>ProjectionSpec — {esc(mode)} · {esc(spec['availability'])}"
            "</summary>"
            f"<dl>{body}</dl></details>"
        )

    def empty_state(mode: str) -> str:
        spec = specs[mode]
        return (
            f'<section class="undecided"><h2>{esc(mode)}: {esc(spec["availability"])}</h2>'
            f"<p>{esc(spec['reason'])}</p></section>"
        )

    def report_panel(mode: str, report: DecisionReport | None) -> str:
        if report is None:
            return empty_state(mode)
        return f'<div class="report">{render_html_fragment(report)}</div>'

    graph = payload["graph"]
    edge_rows = []
    for edge in graph["edges"]:
        ref = f"IFC #{edge['source_ref']}" if edge["source_ref"] is not None else "no source reference"
        edge_rows.append(
            f'<tr data-source="{esc(edge["source"])}" data-target="{esc(edge["target"])}">'
            f"<td>{esc(edge['source'])}</td><td><code>{esc(edge['kind'])}</code></td>"
            f"<td>{esc(edge['target'])}</td><td>{esc(ref)}</td></tr>"
        )
    graph_panel = (
        '<div class="graphwrap"><svg id="graph-svg" role="img" '
        'aria-label="relationship graph"></svg></div>'
        '<p class="note">Layout is a reading order by IFC class rank, top to bottom; '
        "position and distance on this canvas are not evidence. Edges are typed and "
        "carry their IFC source reference. Click a node to select it in every mode."
        "</p>"
        '<div id="graph-legend"></div>'
        '<section><h2>edges</h2><div class="tablewrap"><table><thead><tr>'
        "<th>source</th><th>kind</th><th>target</th><th>provenance</th></tr></thead>"
        f"<tbody>{''.join(edge_rows)}</tbody></table></div></section>"
    )

    state = payload["state"]
    meta_rows = "".join(
        f"<dt>{esc(key)}</dt><dd>{esc(value)}</dd>" for key, value in state["meta"].items()
    )
    state_panel = (
        '<div class="statewrap"><nav id="entity-list" aria-label="entities"></nav>'
        '<div id="entity-card"></div></div>'
        f'<section><h2>world</h2><dl><dt>digest</dt><dd>{_digest(state["world_digest"])}</dd>'
        f"<dt>entities</dt><dd>{len(state['entities'])}</dd>"
        f"<dt>quantities</dt><dd>{state['raw']} raw, {state['derived']} derived</dd>"
        f"<dt>relationships</dt><dd>{state['relationships']}</dd>"
        f"<dt>constraints</dt><dd>{state['constraints']}</dd>{meta_rows}</dl></section>"
    )

    structure_panel = (
        '<iframe id="structure" title="STRUCTURE: as-built viewer" '
        'sandbox="allow-scripts" srcdoc="' + esc(viewer_document, quote=True) + '"></iframe>'
    )

    panels = {
        "MAP": empty_state("MAP"),
        "GLOBE": empty_state("GLOBE"),
        "STRUCTURE": structure_panel,
        "GRAPH": graph_panel,
        "STATE": state_panel,
        "TIME": report_panel("TIME", ledger),
        "EVIDENCE": report_panel("EVIDENCE", decision_report),
        "COMPLEXITY": report_panel("COMPLEXITY", audit),
    }
    mode_buttons = "".join(
        f'<button role="tab" data-mode="{mode}" class="{esc(specs[mode]["availability"])}" '
        f'aria-selected="false" title="{esc(specs[mode]["question"])}">'
        f'<span class="key">{index + 1}</span>{mode}'
        f'<span class="avail">{esc(specs[mode]["availability"])}</span></button>'
        for index, mode in enumerate(MODES)
    )
    panel_markup = "".join(
        f'<section class="panel" data-mode="{mode}" role="tabpanel" hidden>'
        f"{spec_strip(mode)}{panels[mode]}</section>"
        for mode in MODES
    )
    decision = payload["decision"]
    frame = payload["state"]["frame"]
    decision_badge = (
        f'<span class="badge" style="background:{disposition_hex(decision["disposition"])}">'
        f"{esc(decision['disposition'])}</span> {esc(decision['headline'].split(': ', 1)[-1])}"
        if decision
        else '<span class="muted">no decision bound</span>'
    )
    rules = " · ".join(esc(rule) for rule in payload["rules"])
    specs_json = esc(json.dumps(payload["modes"], indent=1))
    title = esc(payload["model"] or "workbench")
    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<meta name="color-scheme" content="light dark">\n'
        f"<title>Notation Workbench: {title}</title>\n"
        f"<style>{_HTML_STYLE}{_WORKBENCH_STYLE}</style></head><body>\n"
        '<div id="workbench">\n'
        '<header id="bar">\n'
        '<div id="brand"><strong>Notation Workbench</strong><span class="muted">GAT instrument</span></div>\n'
        f'<nav id="modes" role="tablist" aria-label="modes">{mode_buttons}</nav>\n'
        '<div id="identity">'
        f'<span class="cell">{title}</span>'
        f'<span class="cell">world {_digest(payload["world_digest"])}</span>'
        f'<span class="cell" title="{esc(frame["convention"])}">frame {esc(frame["id"])} '
        f'({esc(frame["units"])}, {esc(frame["up"])} up, '
        f'{"no CRS" if frame["crs"] is None else esc(str(frame["crs"]))})</span>'
        f'<span class="cell" id="decision-cell">{decision_badge}</span>'
        '<span class="cell" id="selection-cell"><span class="muted">nothing selected</span></span>'
        "</div>\n"
        "</header>\n"
        f'<div id="panels">{panel_markup}</div>\n'
        '<footer id="foot">'
        f"<p>{esc(NON_AUTHORIZING_FOOTER)} {esc(READ_ONLY_FOOTER)}</p>"
        f"<p>{rules}</p>"
        f"<details><summary>ProjectionSpec ({esc(PROJECTION_SPEC_VERSION)}) for every mode"
        f"</summary><pre>{specs_json}</pre></details>"
        "</footer>\n"
        "</div>\n"
        f'<script id="workbench-data" type="application/json">{encoded}</script>\n'
        f"<script>{_WORKBENCH_SCRIPT}</script>\n"
        "</body></html>\n"
    )


def export_workbench_html(
    world: World,
    path: str | Path,
    *,
    model_name: str = "",
    n: int = 8,
    seed: int = 0,
    spacing: float = 0.75,
    decision: Mapping[str, object] | None = None,
    decision_report: DecisionReport | None = None,
    ledger: DecisionReport | None = None,
    audit: DecisionReport | None = None,
    audit_reason: str = "",
    audit_statuses: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Write the instrument; returns each mode's availability."""
    payload = workbench_payload(
        world,
        model_name=model_name,
        n=n,
        seed=seed,
        spacing=spacing,
        decision=decision,
        decision_report=decision_report,
        ledger=ledger,
        audit=audit,
        audit_reason=audit_reason,
        audit_statuses=audit_statuses,
    )
    document = render_workbench_html(
        payload, decision_report=decision_report, ledger=ledger, audit=audit
    )
    Path(path).write_text(document, encoding="utf-8")
    return {spec["mode"]: spec["availability"] for spec in payload["modes"]}


def _digest(value: str) -> str:
    esc = html_mod.escape
    return (
        '<details class="digest">'
        f"<summary><code>{esc(format_digest(value))}</code></summary>"
        f'<code class="full">{esc(value)}</code></details>'
    )


_WORKBENCH_STYLE = """
html, body { height: 100%; }
body { overflow: hidden; }
#workbench { height: 100vh; display: grid; grid-template-rows: auto 1fr auto; }
#bar { background: var(--card); border-bottom: 1px solid var(--rule); padding: 0.5rem 1rem 0;
  display: grid; grid-template-columns: auto 1fr; gap: 0.2rem 1.5rem; align-items: center; }
#brand strong { font-size: 0.95rem; letter-spacing: 0.02em; margin-right: 0.5rem; }
.muted { color: var(--muted); }
#modes { display: flex; flex-wrap: wrap; gap: 0.15rem; }
#modes button { border: 0; border-bottom: 3px solid transparent; background: none; color: var(--ink);
  font: inherit; font-size: 0.78rem; letter-spacing: 0.08em; padding: 0.45rem 0.7rem 0.35rem;
  cursor: pointer; display: inline-flex; align-items: baseline; gap: 0.4rem; }
#modes button .key { color: var(--muted); font-size: 0.7rem; font-variant-numeric: tabular-nums; }
#modes button .avail { font-size: 0.62rem; letter-spacing: 0.04em; text-transform: none;
  color: var(--muted); border: 1px solid var(--rule); border-radius: 999px; padding: 0 0.4em; }
#modes button.available .avail { display: none; }
#modes button.unavailable { color: var(--muted); }
#modes button.unavailable .avail { border-style: dashed; }
#modes button[aria-selected="true"] { border-bottom-color: var(--ink); }
#modes button:focus-visible { outline: 2px solid var(--ink); outline-offset: -2px; }
#identity { grid-column: 1 / -1; display: flex; flex-wrap: wrap; gap: 0 1.2rem; padding: 0.25rem 0 0.5rem;
  font-size: 0.8rem; color: var(--muted); border-top: 1px solid var(--rule-soft); }
#identity .cell code { font-size: 0.9em; }
#identity .badge { font-weight: 600; }
#selection-cell button { border: 1px solid var(--rule); background: none; color: var(--muted);
  border-radius: 999px; font: inherit; font-size: 0.75rem; padding: 0 0.5em; margin-left: 0.4rem; cursor: pointer; }
#panels { overflow: hidden; position: relative; }
.panel { position: absolute; inset: 0; overflow-y: auto; padding: 1rem; box-sizing: border-box; }
.panel[data-mode="STRUCTURE"] { padding: 0; display: grid; grid-template-rows: auto 1fr; }
.panel[data-mode="STRUCTURE"] details.spec { margin: 0.6rem 1rem 0.4rem; }
#structure { width: 100%; height: 100%; border: 0; background: #f5f4f1; }
details.spec { font-size: 0.8rem; color: var(--muted); margin-bottom: 0.8rem; }
details.spec summary { cursor: pointer; letter-spacing: 0.04em; }
details.spec dl { margin: 0.4rem 0 0; background: var(--card); padding: 0.6rem 0.8rem; border-radius: 8px; }
details.spec.unavailable summary { color: var(--muted); }
.report { max-width: 46rem; }
.panel section { max-width: 46rem; }
.panel section:first-of-type { margin-top: 0; }
.graphwrap { background: var(--card); border-radius: 10px; box-shadow: var(--shadow); padding: 0.6rem;
  max-width: 64rem; overflow-x: auto; }
#graph-svg { width: 100%; height: auto; display: block; font: 12px system-ui, sans-serif; }
#graph-svg .edge { stroke: var(--muted); fill: none; stroke-width: 1.4; }
#graph-svg .edge.kind-aggregates { stroke-width: 2.6; }
#graph-svg .edge.kind-contains { stroke-width: 1.1; }
#graph-svg .edge.kind-bounds { stroke-dasharray: 6 4; }
#graph-svg .edge.kind-voids { stroke-dasharray: 2 3; }
#graph-svg .edge.kind-fills { stroke-dasharray: 9 3 2 3; }
#graph-svg .edge.dim { opacity: 0.22; }
#graph-svg .edge.lit { stroke: var(--ink); }
#graph-svg .node rect { fill: var(--card); stroke: var(--rule); stroke-width: 1.2; rx: 6; }
#graph-svg .node { cursor: pointer; }
#graph-svg .node:hover rect { stroke: var(--muted); }
#graph-svg .node.selected rect { stroke: var(--ink); stroke-width: 2.5; }
#graph-svg .node text { fill: var(--ink); }
#graph-svg .node .cls { fill: var(--muted); font-size: 10px; }
#graph-svg .rowlabel { fill: var(--muted); font-size: 10px; letter-spacing: 0.06em; }
#graph-svg .arrow { fill: var(--muted); }
#graph-legend { display: flex; flex-wrap: wrap; gap: 0.4rem 1.2rem; font-size: 0.8rem; color: var(--muted);
  margin: 0.6rem 0; }
#graph-legend svg { width: 3rem; height: 0.8rem; vertical-align: middle; margin-right: 0.3rem; }
#graph-legend svg line { stroke: var(--muted); stroke-width: 1.4; }
.statewrap { display: grid; grid-template-columns: minmax(12rem, 16rem) minmax(0, 1fr); gap: 1rem; max-width: 64rem; }
#entity-list { background: var(--card); border-radius: 10px; box-shadow: var(--shadow); padding: 0.6rem 0.4rem; }
#entity-list h3 { margin: 0.5rem 0.6rem 0.2rem; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }
#entity-list button { display: block; width: 100%; text-align: left; border: 0; background: none; color: var(--ink);
  font: inherit; font-size: 0.88rem; padding: 0.25rem 0.6rem; border-radius: 6px; cursor: pointer; }
#entity-list button:hover { background: var(--rule-soft); }
#entity-list button.selected { background: var(--ink); color: var(--bg); }
#entity-card section { margin-top: 0; }
#entity-card section + section { margin-top: 1rem; }
#entity-card td.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.hit { outline: 2px solid var(--ink); outline-offset: 1px; border-radius: 3px; }
tr.hit td { background: var(--rule-soft); }
#foot { border-top: 1px solid var(--rule); padding: 0.4rem 1rem; margin: 0; font-size: 0.75rem; color: var(--muted); }
#foot p { margin: 0.1rem 0; }
#foot pre { font-size: 0.72rem; max-height: 14rem; overflow: auto; background: var(--card); padding: 0.6rem; border-radius: 8px; }
@media (max-width: 48rem) {
  #bar { grid-template-columns: 1fr; }
  .statewrap { grid-template-columns: 1fr; }
}
"""


_WORKBENCH_SCRIPT = r"""
"use strict";
const DATA = JSON.parse(document.getElementById("workbench-data").textContent);
const MESSAGE_FORMAT = DATA.message_format;
const MODES = DATA.modes.map((spec) => spec.mode);
const SPECS = Object.fromEntries(DATA.modes.map((spec) => [spec.mode, spec]));
const ENTITIES = Object.fromEntries(DATA.state.entities.map((entity) => [entity.entity, entity]));
const state = { mode: null, selection: null };
const text = (value) => document.createTextNode(String(value));
const structure = document.getElementById("structure");
let structureReady = false;

// -- modes ------------------------------------------------------------------
function showMode(mode) {
  if (!MODES.includes(mode)) return;
  state.mode = mode;
  for (const button of document.querySelectorAll("#modes button"))
    button.setAttribute("aria-selected", String(button.dataset.mode === mode));
  for (const panel of document.querySelectorAll(".panel"))
    panel.hidden = panel.dataset.mode !== mode;
  writeHash();
}
for (const button of document.querySelectorAll("#modes button"))
  button.addEventListener("click", () => showMode(button.dataset.mode));
document.addEventListener("keydown", (event) => {
  if (event.target.closest("input, textarea, [contenteditable]")) return;
  const index = "12345678".indexOf(event.key);
  if (index >= 0 && !event.metaKey && !event.ctrlKey && !event.altKey) showMode(MODES[index]);
  if (event.key === "Escape") select(null, "keyboard");
});

// -- identity -----------------------------------------------------------------
// One EntityId names the selected element in every mode; the frames exchange
// only ids and the world digest, never positions.
function select(entity, origin) {
  if (entity !== null && !(entity in ENTITIES)) return;
  state.selection = entity;
  renderSelectionCell();
  renderGraphSelection();
  renderEntityCard();
  markMentions();
  if (origin !== "structure" && structureReady && structure)
    structure.contentWindow.postMessage(
      { format: MESSAGE_FORMAT, kind: "select", world_digest: DATA.world_digest, entity }, "*");
  writeHash();
}
function renderSelectionCell() {
  const cell = document.getElementById("selection-cell");
  cell.replaceChildren();
  if (state.selection === null) {
    const span = document.createElement("span"); span.className = "muted";
    span.append(text("nothing selected")); cell.append(span); return;
  }
  const entity = ENTITIES[state.selection];
  const strong = document.createElement("strong"); strong.append(text(entity.name));
  const code = document.createElement("code"); code.append(text(entity.entity));
  const clear = document.createElement("button"); clear.append(text("clear"));
  clear.title = "clear selection (esc)";
  clear.addEventListener("click", () => select(null, "identity"));
  cell.append(strong, text(" "), code, clear);
  if (!nameIsUnique(entity.name)) {
    const note = document.createElement("span"); note.className = "muted";
    note.append(text(" (name shared by several entities: report mentions are not marked)"));
    cell.append(note);
  }
}
function writeHash() {
  const hash = "#" + (state.mode || "") + (state.selection ? "/" + encodeURIComponent(state.selection) : "");
  if (location.hash !== hash) history.replaceState(null, "", hash);
}
function readHash() {
  const raw = location.hash.replace(/^#/, "");
  if (!raw) return { mode: null, entity: null };
  const [mode, entity] = raw.split("/", 2);
  return { mode: mode || null, entity: entity ? decodeURIComponent(entity) : null };
}

// -- STRUCTURE messages --------------------------------------------------------
window.addEventListener("message", (event) => {
  if (!structure || event.source !== structure.contentWindow) return;
  const message = event.data;
  if (!message || message.format !== MESSAGE_FORMAT) return;
  if (message.world_digest !== DATA.world_digest) return;  // a different world is not ours
  if (message.kind === "ready") {
    structureReady = true;
    if (state.selection !== null)
      structure.contentWindow.postMessage(
        { format: MESSAGE_FORMAT, kind: "select", world_digest: DATA.world_digest, entity: state.selection }, "*");
  } else if (message.kind === "selection") {
    select(message.entity, "structure");
  }
});

// -- GRAPH ----------------------------------------------------------------------
// The SVG namespace comes from the element itself: the page names no URL.
const svgNS = document.getElementById("graph-svg").namespaceURI;
const graphPositions = {};
function renderGraph() {
  const graph = DATA.graph;
  const svg = document.getElementById("graph-svg");
  const width = 1000, rowHeight = 120, top = 56;
  const height = top + Math.max(graph.rows, 1) * rowHeight;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  const el = (name, attrs, cls) => {
    const node = document.createElementNS(svgNS, name);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
    if (cls) node.setAttribute("class", cls);
    return node;
  };
  const defs = el("defs", {});
  const marker = el("marker", { id: "arrow", viewBox: "0 0 10 10", refX: 9, refY: 5,
    markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" });
  marker.append(el("path", { d: "M 0 0 L 10 5 L 0 10 z" }, "arrow"));
  defs.append(marker); svg.append(defs);
  for (const node of graph.nodes) {
    const w = Math.max(92, node.name.length * 7.5 + 28), h = 40;
    const x = 80 + (width - 100) * (node.column + 0.5) / node.columns;
    const y = top + node.row * rowHeight;
    graphPositions[node.entity] = { x, y, w, h };
  }
  graph.row_classes.forEach((classes, row) => {
    const label = el("text", { x: 8, y: top + row * rowHeight + 4 }, "rowlabel");
    label.append(text(classes.join(" / ")));
    svg.append(label);
  });
  for (const edge of graph.edges) {
    const a = graphPositions[edge.source], b = graphPositions[edge.target];
    if (!a || !b) continue;
    const down = b.y > a.y, same = b.y === a.y;
    const y1 = same ? a.y : (down ? a.y + a.h / 2 : a.y - a.h / 2);
    const y2 = same ? b.y : (down ? b.y - b.h / 2 : b.y + b.h / 2);
    const x1 = same ? (b.x > a.x ? a.x + a.w / 2 : a.x - a.w / 2) : a.x;
    const x2 = same ? (b.x > a.x ? b.x - b.w / 2 : b.x + b.w / 2) : b.x;
    const bend = same ? 0 : (y2 - y1) / 2;
    const path = el("path", { d: `M ${x1} ${y1} C ${x1} ${y1 + bend}, ${x2} ${y2 - bend}, ${x2} ${y2}`,
      "marker-end": "url(#arrow)", "data-source": edge.source, "data-target": edge.target },
      "edge kind-" + edge.kind);
    const title = el("title", {});
    title.append(text(`${edge.kind}: ${edge.source} -> ${edge.target}` +
      (edge.source_ref === null ? " (no source reference)" : ` (IFC #${edge.source_ref})`)));
    path.append(title);
    svg.append(path);
  }
  for (const node of graph.nodes) {
    const p = graphPositions[node.entity];
    const g = el("g", { "data-entity": node.entity, tabindex: 0, role: "button" }, "node");
    g.append(el("rect", { x: p.x - p.w / 2, y: p.y - p.h / 2, width: p.w, height: p.h }));
    const name = el("text", { x: p.x, y: p.y - 3, "text-anchor": "middle" });
    name.append(text(node.name));
    const cls = el("text", { x: p.x, y: p.y + 12, "text-anchor": "middle" }, "cls");
    cls.append(text(node.class + (node.source_ref === null ? "" : " #" + node.source_ref)));
    const title = el("title", {}); title.append(text(node.entity));
    g.append(name, cls, title);
    g.addEventListener("click", () => select(node.entity, "graph"));
    g.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") { event.preventDefault(); select(node.entity, "graph"); }
    });
    svg.append(g);
  }
  const legend = document.getElementById("graph-legend");
  for (const entry of graph.kinds) {
    const item = document.createElement("span");
    const sample = el("svg", { viewBox: "0 0 48 12" });
    sample.append(el("line", { x1: 1, y1: 6, x2: 47, y2: 6 }, "edge kind-" + entry.kind));
    item.append(sample, text(`${entry.kind} (${entry.count})`));
    legend.append(item);
  }
}
function renderGraphSelection() {
  const selected = state.selection;
  for (const node of document.querySelectorAll("#graph-svg .node"))
    node.classList.toggle("selected", node.dataset.entity === selected);
  for (const edge of document.querySelectorAll("#graph-svg .edge")) {
    const touches = selected !== null && (edge.dataset.source === selected || edge.dataset.target === selected);
    edge.classList.toggle("lit", touches);
    edge.classList.toggle("dim", selected !== null && !touches);
  }
  for (const row of document.querySelectorAll('.panel[data-mode="GRAPH"] tbody tr'))
    row.classList.toggle("hit", selected !== null &&
      (row.dataset.source === selected || row.dataset.target === selected));
}

// -- STATE ------------------------------------------------------------------------
function renderEntityList() {
  const list = document.getElementById("entity-list");
  const byClass = new Map();
  for (const entity of DATA.state.entities) {
    if (!byClass.has(entity.class)) byClass.set(entity.class, []);
    byClass.get(entity.class).push(entity);
  }
  for (const [cls, entities] of byClass) {
    const heading = document.createElement("h3"); heading.append(text(cls)); list.append(heading);
    for (const entity of entities) {
      const button = document.createElement("button");
      button.dataset.entity = entity.entity;
      button.append(text(entity.name));
      button.title = entity.entity;
      button.addEventListener("click", () => select(entity.entity, "state"));
      list.append(button);
    }
  }
}
function renderEntityCard() {
  for (const button of document.querySelectorAll("#entity-list button"))
    button.classList.toggle("selected", button.dataset.entity === state.selection);
  const card = document.getElementById("entity-card");
  card.replaceChildren();
  const section = document.createElement("section");
  const h2 = document.createElement("h2");
  if (state.selection === null) {
    h2.append(text("belief"));
    const p = document.createElement("p");
    p.append(text("Select an entity in any mode to read its quantities: believed mean and sigma, " +
      "raw (canonical belief) or derived (pushforward), with the IFC record each came from."));
    section.append(h2, p); card.append(section); return;
  }
  const entity = ENTITIES[state.selection];
  h2.append(text(entity.name + " · " + entity.class));
  section.append(h2);
  const dl = document.createElement("dl");
  const rows = [["entity", entity.entity],
    ["provenance", entity.source_ref === null ? "no source reference" : "IFC #" + entity.source_ref]];
  for (const [key, value] of Object.entries(entity.attrs)) rows.push([key, String(value)]);
  for (const [key, value] of rows) {
    const dt = document.createElement("dt"); dt.append(text(key));
    const dd = document.createElement("dd"); dd.append(text(value)); dl.append(dt, dd);
  }
  section.append(dl);
  card.append(section);
  const quantities = document.createElement("section");
  const qh = document.createElement("h2"); qh.append(text("quantities: mean ± sigma")); quantities.append(qh);
  const wrap = document.createElement("div"); wrap.className = "tablewrap";
  const table = document.createElement("table");
  const thead = document.createElement("thead"); const hr = document.createElement("tr");
  for (const label of ["quantity", "role", "mean", "sigma", "unit", "provenance"]) {
    const th = document.createElement("th"); th.append(text(label)); hr.append(th);
  }
  thead.append(hr); table.append(thead);
  const tbody = document.createElement("tbody");
  for (const q of entity.quantities) {
    const tr = document.createElement("tr");
    const cells = [[q.name, ""], [q.role, ""], [String(q.mean), "num"], [String(q.sigma), "num"], [q.unit, ""],
      [q.source_ref === null ? (q.role === "derived" ? "derived by expression" : "no source reference") : "IFC #" + q.source_ref, ""]];
    for (const [value, cls] of cells) {
      const td = document.createElement("td"); td.append(text(value)); if (cls) td.className = cls; tr.append(td);
    }
    tbody.append(tr);
  }
  table.append(tbody); wrap.append(table); quantities.append(wrap); card.append(quantities);
}

// -- mentions in report panels -----------------------------------------------------
// The knowledge-debugger view: where does the selected identity appear in the
// evidence, the timeline, the audit?  Exact-name matches only; never inferred.
function nameIsUnique(name) {
  return DATA.state.entities.filter((entity) => entity.name === name).length === 1;
}
function markMentions() {
  for (const hit of document.querySelectorAll(".report .hit")) hit.classList.remove("hit");
  if (state.selection === null) return;
  const name = ENTITIES[state.selection].name;
  // Reports name subjects by entity name, not EntityId: a shared name is an
  // ambiguous identity, so nothing is marked and the strip says so.
  if (!nameIsUnique(name)) return;
  for (const cell of document.querySelectorAll(".report td, .report dd"))
    if (cell.textContent.trim() === name) cell.classList.add("hit");
}

// -- boot ---------------------------------------------------------------------------
renderGraph();
renderEntityList();
renderEntityCard();
renderSelectionCell();
function applyHash(fallback) {
  const wanted = readHash();
  showMode(MODES.includes(wanted.mode) ? wanted.mode : fallback || state.mode);
  const entity = wanted.entity && wanted.entity in ENTITIES ? wanted.entity : null;
  if (entity !== state.selection) select(entity, "hash");
}
window.addEventListener("hashchange", () => applyHash(null));
applyHash("STRUCTURE");
"""


__all__ = [
    "AVAILABLE",
    "EMPTY",
    "MESSAGE_FORMAT",
    "MODES",
    "PROJECTION_SPEC_VERSION",
    "ProjectionSpec",
    "UNAVAILABLE",
    "WORKBENCH_FORMAT",
    "export_workbench_html",
    "graph_payload",
    "projection_specs",
    "render_workbench_html",
    "state_payload",
    "workbench_payload",
]
