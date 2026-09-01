"""Human rendering of ``gat-headless`` decision responses.

The headless boundary emits closed JSON for machines; this module gives the
same responses a human surface: a terminal rendering and a self-contained
HTML report.  It is a viewer, not a gate — it renders decisions the engine
already made, re-checks their internal identities, and refuses anything it
cannot vouch for.

The design language is shared with the Blender assurance panel
(``integrations/blender/gat_assurance``), which remains the canonical
palette source:

* one signal palette — green is a decided "proceed" (ACCEPT, SATISFIED,
  ADMISSIBLE, PASS), red a decided "stop" (REJECT, VIOLATED, BLOCKED,
  FAIL), amber "more evidence needed" (REQUEST_EVIDENCE, UNRESOLVED,
  WARN); malfunctions are grey, never red;
* terminal output carries the signal in words, ASCII only; colour is a
  GUI/HTML reinforcement, never the sole carrier;
* digests appear as their first 12 hex characters with the full value
  preserved (HTML: native disclosure, no scripts);
* moment capacities render in kN*m at one decimal, probabilities at five;
* every report ends with the read-only footer and none authorizes action.

See ``docs/design-language-v1.md`` for the full contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import html as html_mod
import re
from typing import Mapping, Sequence

RESPONSE_FORMAT = "gat-headless-response-v1"

#: Signal classes — what a rendered term asks of the reader.
PROCEED = "proceed"
STOP = "stop"
ATTENTION = "attention"
UNDECIDED = "undecided"

#: Decision vocabulary -> signal class, across every response family:
#: acceptance dispositions, engineering verdicts, change dispositions,
#: and invariant statuses.
SIGNAL_CLASSES: dict[str, str] = {
    "ACCEPT": PROCEED,
    "SATISFIED": PROCEED,
    "ADMISSIBLE": PROCEED,
    "PASS": PROCEED,
    "REJECT": STOP,
    "VIOLATED": STOP,
    "BLOCKED": STOP,
    "FAIL": STOP,
    "REQUEST_EVIDENCE": ATTENTION,
    "UNRESOLVED": ATTENTION,
    "WARN": ATTENTION,
    "ERROR": UNDECIDED,
}

#: Signal class -> linear RGBA.  The six decision terms shared with the
#: Blender panel must stay bit-identical to its palette (locked by test).
SIGNAL_COLORS: dict[str, tuple[float, float, float, float]] = {
    PROCEED: (0.10, 0.70, 0.20, 1.0),
    STOP: (0.85, 0.08, 0.08, 1.0),
    ATTENTION: (0.95, 0.55, 0.05, 1.0),
    UNDECIDED: (0.35, 0.35, 0.35, 1.0),
}

ACCEPTANCE_DISPOSITIONS = frozenset({"ACCEPT", "REJECT", "REQUEST_EVIDENCE"})
BEAM_DISPOSITIONS = frozenset({"SATISFIED", "VIOLATED", "UNRESOLVED"})
CHANGE_DISPOSITIONS = frozenset({"ADMISSIBLE", "BLOCKED"})

READ_ONLY_FOOTER = "Read-only: no BIM state was changed."
NON_AUTHORIZING_FOOTER = "This report does not authorize any action."
RECOMMENDATION_FOOTER = (
    "Recommendation only; professional approval is still required."
)
PREVIEW_FOOTER = "Preview only; the candidate state was not committed."

_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")


def disposition_color(term: str) -> tuple[float, float, float, float]:
    """Canonical RGBA for a decision term; refuses unknown vocabulary."""
    try:
        return SIGNAL_COLORS[SIGNAL_CLASSES[term]]
    except KeyError as exc:
        raise ValueError(f"unsupported disposition {term!r}") from exc


def disposition_hex(term: str) -> str:
    red, green, blue, _ = disposition_color(term)
    return "#%02x%02x%02x" % tuple(
        round(255 * channel) for channel in (red, green, blue)
    )


# -- shared value formatting ------------------------------------------------


def format_digest(value: str) -> str:
    return value[:12] + "..."


def format_probability(value: float) -> str:
    return f"{value:.5f}"


def format_moment(mean_n_m: float, sigma_n_m: float | None = None) -> str:
    if sigma_n_m is None:
        return f"{mean_n_m / 1000.0:.1f} kN*m"
    return f"{mean_n_m / 1000.0:.1f} +- {sigma_n_m / 1000.0:.1f} kN*m"


def format_number(value: float) -> str:
    return f"{value:.6g}"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


# -- report structure -------------------------------------------------------


@dataclass(frozen=True)
class Card:
    """A titled block of label/value rows."""

    title: str
    fields: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Table:
    """A titled column table; ``accents`` carries one vocabulary term or
    empty string per row so renderers can reinforce it with colour."""

    title: str
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    accents: tuple[str, ...]
    overflow: str = ""


@dataclass(frozen=True)
class DecisionReport:
    """One response, normalized for rendering on any surface."""

    operation: str
    request_id: str
    world_digest: str
    disposition: str
    subject: str
    subline: str
    notes: tuple[str, ...]
    blocks: tuple[Card | Table, ...]
    footers: tuple[str, ...]

    @property
    def headline(self) -> str:
        return f"{self.disposition}: {self.subject}"

    @property
    def color(self) -> tuple[float, float, float, float]:
        return disposition_color(self.disposition)


# -- strict decoding --------------------------------------------------------


def decode_response(value: object) -> DecisionReport:
    """Normalize one headless response, refusing anything inconsistent."""
    response = _object(value, "response")
    if response.get("format") != RESPONSE_FORMAT:
        raise ValueError(f"unsupported response format {response.get('format')!r}")
    if set(response) == {"format", "error"}:
        return _error_report(_object(response["error"], "error"))
    required = {"format", "request_id", "operation", "world_digest", "result"}
    if set(response) != required:
        raise ValueError("response fields differ from gat-headless-response-v1")
    operation = _string(response["operation"], "operation")
    request_id = _string(response["request_id"], "request_id")
    world_digest = _string(response["world_digest"], "world_digest")
    result = _object(response["result"], "result")
    builders = {
        "summary": _summary_report,
        "acceptance": _acceptance_report,
        "beam_assurance": _beam_report,
        "change_impact": _change_report,
    }
    if operation not in builders:
        raise ValueError(f"unsupported headless operation {operation!r}")
    return builders[operation](request_id, world_digest, result)


def _error_report(error: Mapping[str, object]) -> DecisionReport:
    return DecisionReport(
        operation="error",
        request_id="",
        world_digest="",
        disposition="ERROR",
        subject=_string(error.get("type"), "error.type"),
        subline="gat-headless refused the request; no state was evaluated",
        notes=(_string(error.get("message"), "error.message"),),
        blocks=(),
        footers=(READ_ONLY_FOOTER,),
    )


def _summary_report(
    request_id: str, world_digest: str, result: Mapping[str, object]
) -> DecisionReport:
    verification = _object(result.get("verification"), "verification")
    passed = verification.get("passed")
    if not isinstance(passed, bool):
        raise ValueError("verification.passed must be boolean")
    counts = tuple(
        _count(verification.get(key), f"verification.{key}")
        for key in ("pass_count", "warning_count", "failure_count")
    )
    if passed and counts[2] != 0:
        raise ValueError("summary claims verified with failures present")
    trust = _object(result.get("carrier_trust"), "carrier_trust")
    signed = trust.get("signature_verified") is True
    key_id = trust.get("key_id")
    carrier = (
        f"signed, key {key_id}" if signed and isinstance(key_id, str) else "unsigned"
    )
    state_card = Card(
        "state",
        (
            ("world", world_digest),
            ("verification", f"{counts[0]} pass / {counts[1]} warn / {counts[2]} fail"),
            ("carrier", carrier),
        ),
    )
    content_card = Card(
        "content",
        (
            (
                "module",
                f"{_count(result.get('entities'), 'entities')} entities, "
                f"{_count(result.get('relationships'), 'relationships')} relationships, "
                f"{_count(result.get('constraints'), 'constraints')} constraints",
            ),
            (
                "belief",
                f"{_count(result.get('raw_variables'), 'raw_variables')} raw + "
                f"{_count(result.get('derived_variables'), 'derived_variables')} "
                "derived variables",
            ),
        ),
    )
    return DecisionReport(
        operation="summary",
        request_id=request_id,
        world_digest=world_digest,
        disposition="PASS" if passed else "FAIL",
        subject=_string(result.get("source"), "source"),
        subline="authoritative state summary",
        notes=(),
        blocks=(state_card, content_card),
        footers=(NON_AUTHORIZING_FOOTER, READ_ONLY_FOOTER),
    )


def _acceptance_report(
    request_id: str, world_digest: str, result: Mapping[str, object]
) -> DecisionReport:
    disposition = _string(result.get("disposition"), "disposition")
    if disposition not in ACCEPTANCE_DISPOSITIONS:
        raise ValueError(f"unsupported acceptance disposition {disposition!r}")
    if result.get("world_digest") != world_digest:
        raise ValueError("acceptance world identities differ")
    may_authorize = result.get("may_authorize")
    if not isinstance(may_authorize, bool) or may_authorize != (
        disposition == "ACCEPT"
    ):
        raise ValueError("acceptance authorization claim is inconsistent")
    reasons = tuple(
        _string(item, "reason") for item in _array(result.get("reasons"), "reasons")
    )

    check_rows: list[tuple[str, ...]] = []
    accents: list[str] = []
    for item in _array(result.get("checks"), "checks"):
        check = _object(item, "check")
        verdict = _string(check.get("verdict"), "check.verdict")
        if verdict not in BEAM_DISPOSITIONS:
            raise ValueError(f"unsupported check verdict {verdict!r}")
        lower = _number(check.get("p_satisfies_lower"), "p_satisfies_lower")
        upper = _number(check.get("p_satisfies_upper"), "p_satisfies_upper")
        p_range = (
            format_probability(lower)
            if abs(upper - lower) < 5e-6
            else f"{format_probability(lower)}..{format_probability(upper)}"
        )
        confidence = _number(check.get("confidence"), "check.confidence")
        check_rows.append(
            (
                _string(check.get("check_id"), "check.check_id"),
                _string(check.get("kind"), "check.kind"),
                verdict,
                p_range,
                f"{confidence:.0%}",
            )
        )
        accents.append(verdict)

    request_rows: list[tuple[str, ...]] = []
    for item in _array(result.get("evidence_requests"), "evidence_requests"):
        request = _object(item, "evidence request")
        request_rows.append(
            (
                _string(request.get("check_id"), "request.check_id"),
                _string(request.get("action"), "request.action"),
                _string(request.get("target"), "request.target"),
                _string(request.get("reason"), "request.reason"),
            )
        )

    receipt_ids = [
        _string(item, "receipt id")
        for item in _array(result.get("evidence_receipt_ids"), "evidence_receipt_ids")
    ]
    identity_card = Card(
        "identity",
        (
            ("case", _string(result.get("case_digest"), "case_digest")),
            ("world", world_digest),
            ("policy", _string(result.get("policy_id"), "policy_id")),
            ("evidence receipts", ", ".join(receipt_ids) or "none"),
        ),
    )
    blocks: list[Card | Table] = [
        Table(
            "checks",
            ("check", "kind", "verdict", "P(satisfies)", "confidence"),
            tuple(check_rows),
            tuple(accents),
        ),
    ]
    if request_rows:
        blocks.append(
            Table(
                "next evidence",
                ("check", "action", "target", "reason"),
                tuple(request_rows),
                tuple("REQUEST_EVIDENCE" for _ in request_rows),
            )
        )
    blocks.append(identity_card)
    footers = (
        RECOMMENDATION_FOOTER if may_authorize else NON_AUTHORIZING_FOOTER,
        READ_ONLY_FOOTER,
    )
    workflow = _string(result.get("workflow"), "workflow")
    case_id = _string(result.get("case_id"), "case_id")
    return DecisionReport(
        operation="acceptance",
        request_id=request_id,
        world_digest=world_digest,
        disposition=disposition,
        subject=_string(result.get("subject"), "subject"),
        subline=f"{workflow} case {case_id}",
        notes=reasons,
        blocks=tuple(blocks),
        footers=footers,
    )


def _beam_report(
    request_id: str, world_digest: str, result: Mapping[str, object]
) -> DecisionReport:
    disposition = _string(result.get("disposition"), "disposition")
    if disposition not in BEAM_DISPOSITIONS:
        raise ValueError(f"unsupported beam disposition {disposition!r}")
    verification = _object(result.get("verification"), "verification")
    if verification.get("passed") is not True:
        raise ValueError("report refuses an unverified beam response")
    assurance = _object(result.get("assurance"), "assurance")
    if assurance.get("may_authorize") is not False:
        raise ValueError("beam response must remain non-authorizing")
    prior = _object(result.get("prior"), "prior")
    revised = _object(result.get("revised"), "revised")
    change = _object(result.get("decision_change"), "decision_change")
    transition = _object(result.get("transition"), "transition")
    computation = _object(revised.get("computation"), "revised.computation")

    prior_digest = _string(
        transition.get("prior_world_digest"), "transition.prior_world_digest"
    )
    result_digest = _string(
        transition.get("result_world_digest"), "transition.result_world_digest"
    )
    if result_digest != world_digest:
        raise ValueError("beam response world identities differ")
    if _string(prior.get("world_digest"), "prior.world_digest") != prior_digest:
        raise ValueError("beam prior world identities differ")
    if _string(revised.get("world_digest"), "revised.world_digest") != result_digest:
        raise ValueError("beam revised world identities differ")
    prior_verdict = _string(prior.get("verdict"), "prior.verdict")
    revised_verdict = _string(revised.get("verdict"), "revised.verdict")
    if prior_verdict not in BEAM_DISPOSITIONS or revised_verdict != disposition:
        raise ValueError("beam verdict identities differ")
    verdict_changed = change.get("verdict_changed")
    if not isinstance(verdict_changed, bool) or verdict_changed != (
        prior_verdict != revised_verdict
    ):
        raise ValueError("beam verdict-change claim is inconsistent")

    decision_card = Card(
        "decision",
        (
            ("verdict", f"prior {prior_verdict} -> revised {revised_verdict}"),
            (
                "design capacity",
                f"{format_moment(_number(prior.get('target_mean_n_m'), 'prior mean'), _number(prior.get('target_sigma_n_m'), 'prior sigma'))}"
                f" -> "
                f"{format_moment(_number(revised.get('target_mean_n_m'), 'revised mean'), _number(revised.get('target_sigma_n_m'), 'revised sigma'))}",
            ),
            (
                "P(capacity >= demand)",
                f"{format_probability(_number(prior.get('p_satisfies'), 'prior p'))}"
                f" -> "
                f"{format_probability(_number(revised.get('p_satisfies'), 'revised p'))}",
            ),
            ("method", _string(computation.get("method"), "computation.method")),
            (
                "oracle",
                _string(
                    computation.get("independent_oracle_id"),
                    "computation.independent_oracle_id",
                ),
            ),
        ),
    )
    targets = _quantity_names(transition.get("targets"), "transition.targets")
    affected = _quantity_names(transition.get("affected"), "transition.affected")
    transition_card = Card(
        "transition",
        (
            ("prior world", prior_digest),
            ("result world", result_digest),
            ("targets", ", ".join(targets) or "none"),
            ("affected", ", ".join(affected) or "none"),
            (
                "ledger head",
                _string(transition.get("ledger_head_hash"), "ledger_head_hash"),
            ),
        ),
    )
    assurance_card = Card(
        "assurance", tuple(_scalar_fields(assurance))
    )
    evidence_card = Card(
        "evidence", tuple(_scalar_fields(_object(result.get("evidence"), "evidence")))
    )
    return DecisionReport(
        operation="beam_assurance",
        request_id=request_id,
        world_digest=world_digest,
        disposition=disposition,
        subject=_string(result.get("subject"), "subject"),
        subline=f"beam case {_string(result.get('case_id'), 'case_id')}",
        notes=(_string(change.get("reason"), "decision_change.reason"),),
        blocks=(decision_card, transition_card, evidence_card, assurance_card),
        footers=(NON_AUTHORIZING_FOOTER, READ_ONLY_FOOTER),
    )


def _change_report(
    request_id: str, world_digest: str, result: Mapping[str, object]
) -> DecisionReport:
    disposition = _string(result.get("disposition"), "disposition")
    if disposition not in CHANGE_DISPOSITIONS:
        raise ValueError(f"unsupported change disposition {disposition!r}")
    admissible = result.get("admissible")
    if not isinstance(admissible, bool) or admissible != (
        disposition == "ADMISSIBLE"
    ):
        raise ValueError("change admissibility claim is inconsistent")
    if result.get("prior_world_digest") != world_digest:
        raise ValueError("change preview world identities differ")
    verification = _object(result.get("verification"), "verification")
    failures = _array(verification.get("failures"), "verification.failures")
    warnings = _array(verification.get("warnings"), "verification.warnings")
    if admissible and failures:
        raise ValueError("admissible change preview reports failures")
    if not admissible and not failures:
        raise ValueError("blocked change preview reports no failure")

    impact_rows: list[tuple[str, ...]] = []
    impact_accents: list[str] = []
    impacts = _array(result.get("impacts"), "impacts")
    for item in impacts[:20]:
        impact = _object(item, "impact")
        role = (
            "target"
            if impact.get("target") is True
            else "affected"
            if impact.get("affected") is True
            else ""
        )
        impact_rows.append(
            (
                _string(impact.get("variable"), "impact.variable"),
                role,
                f"{format_number(_number(impact.get('mean_before'), 'mean_before'))}"
                f" -> "
                f"{format_number(_number(impact.get('mean_after'), 'mean_after'))}",
                f"{format_number(_number(impact.get('sigma_before'), 'sigma_before'))}"
                f" -> "
                f"{format_number(_number(impact.get('sigma_after'), 'sigma_after'))}",
                _string(impact.get("unit"), "impact.unit"),
            )
        )
        impact_accents.append("")
    overflow = (
        f"and {len(impacts) - 20} more impacted variables"
        if len(impacts) > 20
        else ""
    )

    finding_rows: list[tuple[str, ...]] = []
    finding_accents: list[str] = []
    for status, items in (("FAIL", failures), ("WARN", warnings)):
        for item in items:
            finding = _object(item, "finding")
            finding_rows.append(
                (
                    _string(finding.get("invariant_id"), "invariant_id"),
                    status,
                    _string(finding.get("subject"), "finding.subject"),
                    _string(finding.get("detail"), "finding.detail"),
                )
            )
            finding_accents.append(status)

    identity_card = Card(
        "identity",
        (
            ("scope", _string(result.get("scope_digest"), "scope_digest")),
            ("prior world", world_digest),
            (
                "candidate world",
                _string(result.get("candidate_world_digest"), "candidate_world_digest"),
            ),
            (
                "impacted entities",
                ", ".join(
                    _string(item, "impacted entity")
                    for item in _array(
                        result.get("impacted_entities"), "impacted_entities"
                    )
                )
                or "none",
            ),
        ),
    )
    blocks: list[Card | Table] = [
        Table(
            "impacts",
            ("variable", "role", "mean", "sigma", "unit"),
            tuple(impact_rows),
            tuple(impact_accents),
            overflow=overflow,
        ),
    ]
    if finding_rows:
        blocks.append(
            Table(
                "verification findings",
                ("invariant", "status", "subject", "detail"),
                tuple(finding_rows),
                tuple(finding_accents),
            )
        )
    else:
        blocks.append(Card("verification", (("findings", "none"),)))
    blocks.append(identity_card)
    return DecisionReport(
        operation="change_impact",
        request_id=request_id,
        world_digest=world_digest,
        disposition=disposition,
        subject=_describe_transformation(
            _object(result.get("transformation"), "transformation")
        ),
        subline="design-change preview",
        notes=(),
        blocks=tuple(blocks),
        footers=(PREVIEW_FOOTER, NON_AUTHORIZING_FOOTER, READ_ONLY_FOOTER),
    )


def _describe_transformation(payload: Mapping[str, object]) -> str:
    op = _string(payload.get("op"), "transformation.op")
    var = payload.get("var")
    described = op
    if isinstance(var, Mapping):
        entity = var.get("entity")
        quantity = var.get("quantity")
        if isinstance(entity, Mapping) and isinstance(quantity, str):
            described = f"{op} {entity.get('ifc_class')}.{quantity}"
    if "value" in payload:
        described += f" = {format_number(_number(payload['value'], 'value'))}"
    elif "delta" in payload:
        described += f" + {format_number(_number(payload['delta'], 'delta'))}"
    elif "factor" in payload:
        described += f" x {format_number(_number(payload['factor'], 'factor'))}"
    return described


def _quantity_names(value: object, label: str) -> list[str]:
    names = []
    for item in _array(value, label):
        record = _object(item, f"{label} entry")
        names.append(_string(record.get("quantity"), f"{label}.quantity"))
    return names


def _scalar_fields(
    mapping: Mapping[str, object], prefix: str = ""
) -> list[tuple[str, str]]:
    """Flatten a provenance-style mapping into displayable label/value rows,
    preserving key order; nested lists are skipped rather than guessed at."""
    fields: list[tuple[str, str]] = []
    for key, value in mapping.items():
        label = f"{prefix}{key}"
        if isinstance(value, Mapping):
            fields.extend(_scalar_fields(value, prefix=f"{label}."))
        elif isinstance(value, bool):
            fields.append((label, _yes_no(value)))
        elif isinstance(value, (int, float)):
            fields.append((label, format_number(float(value))))
        elif isinstance(value, str):
            fields.append((label, value))
    return fields


# -- terminal rendering -----------------------------------------------------


def render_text(report: DecisionReport) -> str:
    lines = [report.headline, f"  {report.subline}"]
    for note in report.notes:
        lines.append(f"  note: {note}")
    for block in report.blocks:
        lines.append("")
        if isinstance(block, Card):
            lines.append(block.title)
            width = max((len(label) for label, _ in block.fields), default=0)
            for label, value in block.fields:
                shown = format_digest(value) if _DIGEST_RE.fullmatch(value) else value
                lines.append(f"  {label.ljust(width)}  {shown}")
        else:
            lines.append(block.title)
            widths = [len(column) for column in block.columns]
            for row in block.rows:
                for index, cell in enumerate(row):
                    widths[index] = max(widths[index], len(cell))
            header = "  ".join(
                column.ljust(widths[index])
                for index, column in enumerate(block.columns)
            )
            lines.append(f"  {header}")
            for row in block.rows:
                rendered = "  ".join(
                    cell.ljust(widths[index]) for index, cell in enumerate(row)
                )
                lines.append(f"  {rendered.rstrip()}")
            if block.overflow:
                lines.append(f"  ({block.overflow})")
    lines.append("")
    lines.extend(report.footers)
    if report.request_id:
        lines.append(
            f"request {report.request_id} | world {format_digest(report.world_digest)}"
        )
    return "\n".join(lines)


# -- HTML rendering ---------------------------------------------------------

_HTML_STYLE = """
:root { color-scheme: light; }
body { margin: 0; background: #f5f4f1; color: #1c1c1a;
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 46rem; margin: 2rem auto; padding: 0 1rem; }
header.banner { color: #fff; border-radius: 10px; padding: 1.1rem 1.3rem; }
header.banner h1 { margin: 0; font-size: 1.35rem; letter-spacing: 0.02em; }
header.banner p { margin: 0.2rem 0 0; opacity: 0.92; }
p.note { background: #fff; border-left: 4px solid #c9c6bf;
  padding: 0.6rem 0.9rem; border-radius: 0 8px 8px 0; }
section { background: #fff; border-radius: 10px; padding: 0.9rem 1.2rem;
  margin-top: 1rem; box-shadow: 0 1px 2px rgba(0,0,0,0.06); }
section h2 { margin: 0 0 0.5rem; font-size: 0.8rem; text-transform: uppercase;
  letter-spacing: 0.08em; color: #6b6a66; }
dl { margin: 0; display: grid; grid-template-columns: max-content 1fr;
  gap: 0.25rem 1rem; }
dt { color: #6b6a66; } dd { margin: 0; overflow-wrap: anywhere; }
div.tablewrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 0.92rem; }
th { text-align: left; color: #6b6a66; font-weight: 600;
  border-bottom: 1px solid #e4e2dc; padding: 0.3rem 0.75rem 0.3rem 0; }
td { border-bottom: 1px solid #efede8; padding: 0.35rem 0.75rem 0.35rem 0;
  vertical-align: top; }
tr.proceed td:first-child { box-shadow: inset 3px 0 0 #1ab233; }
tr.stop td:first-child { box-shadow: inset 3px 0 0 #d91414; }
tr.attention td:first-child { box-shadow: inset 3px 0 0 #f28c0d; }
span.badge { display: inline-block; color: #fff; border-radius: 999px;
  padding: 0 0.55em; font-size: 0.85em; }
details.digest { display: inline; }
details.digest summary { cursor: pointer; list-style: none; display: inline; }
details.digest code.full { display: block; margin-top: 0.2rem; }
code { font-size: 0.92em; background: #f0efeb; border-radius: 4px;
  padding: 0.05em 0.3em; }
footer { margin: 1.2rem 0 2rem; color: #6b6a66; font-size: 0.9rem; }
footer p { margin: 0.15rem 0; }
"""


def render_html(report: DecisionReport) -> str:
    esc = html_mod.escape
    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>GAT decision report: {esc(report.subject)}</title>",
        f"<style>{_HTML_STYLE}</style></head><body><main>",
        f'<header class="banner" style="background:{disposition_hex(report.disposition)}">',
        f"<h1>{esc(report.disposition)}</h1>",
        f"<p>{esc(report.subject)}</p>",
        f'<p class="sub">{esc(report.subline)}</p></header>',
    ]
    for note in report.notes:
        parts.append(f'<p class="note">{esc(note)}</p>')
    for block in report.blocks:
        parts.append(f"<section><h2>{esc(block.title)}</h2>")
        if isinstance(block, Card):
            parts.append("<dl>")
            for label, value in block.fields:
                parts.append(f"<dt>{esc(label)}</dt><dd>{_html_value(value)}</dd>")
            parts.append("</dl>")
        else:
            parts.append('<div class="tablewrap"><table><thead><tr>')
            parts.extend(f"<th>{esc(column)}</th>" for column in block.columns)
            parts.append("</tr></thead><tbody>")
            for row, accent in zip(block.rows, block.accents):
                signal = SIGNAL_CLASSES.get(accent, "")
                parts.append(f'<tr class="{signal}">' if signal else "<tr>")
                for index, cell in enumerate(row):
                    if accent and cell == accent:
                        badge = (
                            f'<span class="badge" style="background:'
                            f'{disposition_hex(accent)}">{esc(cell)}</span>'
                        )
                        parts.append(f"<td>{badge}</td>")
                    else:
                        parts.append(f"<td>{esc(cell)}</td>")
                parts.append("</tr>")
            parts.append("</tbody></table></div>")
            if block.overflow:
                parts.append(f'<p class="note">{esc(block.overflow)}</p>')
        parts.append("</section>")
    parts.append("<footer>")
    for footer in report.footers:
        parts.append(f"<p>{esc(footer)}</p>")
    if report.request_id:
        parts.append(
            f"<p>request <code>{esc(report.request_id)}</code> · "
            f"world {_html_value(report.world_digest)}</p>"
        )
    parts.append("</footer></main></body></html>")
    return "\n".join(parts) + "\n"


def _html_value(value: str) -> str:
    """Digest-looking values collapse to 12 chars with native disclosure."""
    esc = html_mod.escape
    if _DIGEST_RE.fullmatch(value):
        return (
            '<details class="digest">'
            f"<summary><code>{esc(format_digest(value))}</code></summary>"
            f'<code class="full">{esc(value)}</code></details>'
        )
    return esc(value)


# -- strict JSON helpers ----------------------------------------------------


def _object(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _array(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not float("-inf") < result < float("inf"):
        raise ValueError(f"{label} must be finite")
    return result


def _count(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


__all__ = [
    "Card",
    "DecisionReport",
    "NON_AUTHORIZING_FOOTER",
    "READ_ONLY_FOOTER",
    "RECOMMENDATION_FOOTER",
    "PREVIEW_FOOTER",
    "SIGNAL_CLASSES",
    "SIGNAL_COLORS",
    "Table",
    "decode_response",
    "disposition_color",
    "disposition_hex",
    "format_digest",
    "format_moment",
    "format_number",
    "format_probability",
    "render_html",
    "render_text",
]
