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
AUDIT_FORMAT = "gat-ifc-audit-v1"

#: Signal classes — what a rendered term asks of the reader.
PROCEED = "proceed"
STOP = "stop"
ATTENTION = "attention"
UNDECIDED = "undecided"

#: Decision vocabulary -> signal class, across every response family:
#: acceptance dispositions, engineering verdicts, change dispositions,
#: invariant statuses, and audit entity/stage statuses.
SIGNAL_CLASSES: dict[str, str] = {
    "ACCEPT": PROCEED,
    "SATISFIED": PROCEED,
    "ADMISSIBLE": PROCEED,
    "PASS": PROCEED,
    "READY": PROCEED,
    "REJECT": STOP,
    "VIOLATED": STOP,
    "BLOCKED": STOP,
    "FAIL": STOP,
    "REQUEST_EVIDENCE": ATTENTION,
    "UNRESOLVED": ATTENTION,
    "WARN": ATTENTION,
    "NEEDS_GEOMETRY_DERIVATION": ATTENTION,
    "MISSING_SOURCE_DATA": ATTENTION,
    "ERROR": UNDECIDED,
    "NOT_RUN": UNDECIDED,
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
    """A titled block of label/value rows; ``accent`` optionally carries one
    vocabulary term so GUI surfaces can reinforce it with colour."""

    title: str
    fields: tuple[tuple[str, str], ...]
    accent: str = ""


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


def decode_ledger(path: str) -> DecisionReport:
    """Read a hash-chained execution ledger and normalize it as a timeline.

    ``read_ledger`` validates the complete chain before anything renders, so
    a tampered or broken ledger is refused with its reason, never drawn.
    """
    from gat.ledger import read_ledger

    return ledger_report(read_ledger(path), source=path)


def ledger_report(ledger, source: str) -> DecisionReport:
    """Render-ready timeline of an already-validated ExecutionLedger."""
    from gat.ledger import LEDGER_FORMAT, LEDGER_SCHEMA_VERSION

    events = list(ledger.events)
    latest = next(
        (event for event in reversed(events) if event.verification is not None),
        None,
    )
    passed = bool(latest.verification.get("passed")) if latest else True

    blocks: list[Card | Table] = []
    shown = events
    elided = 0
    if len(events) > 50:
        shown = events[:5] + events[-45:]
        elided = len(events) - 50
    for position, event in enumerate(shown):
        if elided and position == 5:
            blocks.append(
                Card("elided", ((f"{elided} events", "not shown"),))
            )
        blocks.append(_event_card(event))
    blocks.append(
        Card(
            "chain",
            (
                ("format", f"{LEDGER_FORMAT} v{LEDGER_SCHEMA_VERSION}"),
                ("events", str(len(events))),
                ("head", ledger.head),
                ("integrity", "hash chain verified"),
            ),
        )
    )
    name = source.replace("\\", "/").rsplit("/", 1)[-1]
    return DecisionReport(
        operation="ledger",
        request_id="",
        world_digest=ledger.head,
        disposition="PASS" if passed else "FAIL",
        subject=name,
        subline=f"execution ledger timeline, {len(events)} events, chain verified",
        notes=(),
        blocks=tuple(blocks),
        footers=(NON_AUTHORIZING_FOOTER, READ_ONLY_FOOTER),
    )


_ACCENT_KEYS = frozenset({"verdict", "disposition", "status", "decision"})


def _event_card(event) -> Card:
    operation = event.operation
    title = f"{event.seq} - {event.kind}"
    op_name = operation.get("op")
    if isinstance(op_name, str):
        title = f"{event.seq} - {event.kind}: {op_name}"

    accent = "FAIL" if event.kind == "rejection" else ""
    fields: list[tuple[str, str]] = []
    if event.error_type:
        message = f"{event.error_type}: {event.error_message}".rstrip(": ")
        fields.append(("error", message))
    scalars = _scalar_fields(
        {key: value for key, value in operation.items() if key != "record_type"}
    )
    for label, value in scalars:
        if (
            not accent
            and label.rsplit(".", 1)[-1] in _ACCENT_KEYS
            and value in SIGNAL_CLASSES
        ):
            accent = value
    fields.extend(scalars)
    fields.extend(
        (f"provenance.{label}", value)
        for label, value in _scalar_fields(event.provenance)
    )
    if event.prior_world_digest == event.result_world_digest:
        fields.append(("world", event.result_world_digest))
    else:
        fields.append(("prior world", event.prior_world_digest))
        fields.append(("result world", event.result_world_digest))
    verification = event.verification
    if verification is not None:
        statuses = [
            result.get("status")
            for result in verification.get("results", [])
            if isinstance(result, Mapping)
        ]
        fields.append(
            (
                "verification",
                f"{statuses.count('PASS')} pass / {statuses.count('WARN')} warn"
                f" / {statuses.count('FAIL')} fail",
            )
        )
        if verification.get("passed") is not True:
            accent = "FAIL"
    return Card(title, tuple(fields), accent=accent)


def decode_response(value: object) -> DecisionReport:
    """Normalize one headless response or audit document, refusing anything
    inconsistent."""
    response = _object(value, "response")
    if response.get("format") == AUDIT_FORMAT:
        return _audit_report(response)
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
    risk_rows: list[tuple[str, ...]] = []
    risk_accents: list[str] = []
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
        # Per-element clearance risks, when the check carries them.  An
        # element the case could not clear at its confidence takes the
        # check's own verdict as its accent — the same rule the viewer uses
        # to paint decision subjects.
        details = check.get("details")
        risks = details.get("risks") if isinstance(details, Mapping) else None
        for risk in _array(risks, "check.details.risks") if risks is not None else ():
            risk = _object(risk, "risk")
            p_violates = _number(risk.get("p_violates"), "risk.p_violates")
            mean = _number(risk.get("clearance_mean"), "risk.clearance_mean")
            sigma = _number(risk.get("clearance_sigma"), "risk.clearance_sigma")
            risk_rows.append(
                (
                    _string(risk.get("element"), "risk.element"),
                    _string(check.get("check_id"), "check.check_id"),
                    f"{mean:.3f} +- {sigma:.3f} m",
                    format_probability(p_violates),
                )
            )
            risk_accents.append(verdict if p_violates > 1.0 - confidence else "")

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
    if risk_rows:
        blocks.append(
            Table(
                "element risks",
                ("element", "check", "clearance", "P(violates)"),
                tuple(risk_rows),
                tuple(risk_accents),
            )
        )
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


_AUDIT_STAGE_STATUSES = frozenset({"PASS", "WARN", "BLOCKED", "NOT_RUN"})
_AUDIT_ENTITY_STATUSES = frozenset(
    {"READY", "NEEDS_GEOMETRY_DERIVATION", "MISSING_SOURCE_DATA", "BLOCKED"}
)


def _audit_report(document: Mapping[str, object]) -> DecisionReport:
    """Normalize a ``gat audit`` JSON document into the report grammar."""
    source = _object(document.get("source"), "source")
    pipeline = _object(document.get("pipeline"), "pipeline")
    ready = pipeline.get("pipeline_ready")
    if not isinstance(ready, bool):
        raise ValueError("pipeline.pipeline_ready must be boolean")

    stage_rows: list[tuple[str, ...]] = []
    stage_accents: list[str] = []
    stages = [("parse", _object(document.get("parse"), "parse"))]
    stages += [
        (name, _object(pipeline.get(name), f"pipeline.{name}"))
        for name in ("lowering", "compilation", "verification")
    ]
    statuses: dict[str, str] = {}
    for name, stage in stages:
        status = _string(stage.get("status"), f"{name}.status")
        if status not in _AUDIT_STAGE_STATUSES:
            raise ValueError(f"unsupported audit stage status {status!r}")
        statuses[name] = status
        message = stage.get("message")
        error_type = stage.get("error_type")
        detail = ""
        if isinstance(error_type, str) and error_type:
            detail = error_type
        if isinstance(message, str) and message:
            detail = f"{detail}: {message}" if detail else message
        stage_rows.append((name, status, detail))
        stage_accents.append(status)
    expected_ready = (
        statuses["parse"] == "PASS"
        and statuses["lowering"] == "PASS"
        and statuses["compilation"] == "PASS"
        and statuses["verification"] in ("PASS", "WARN")
    )
    if ready != expected_ready:
        raise ValueError("audit readiness claim contradicts its stage statuses")

    entity_rows: list[tuple[str, ...]] = []
    entity_accents: list[str] = []
    entities = _array(document.get("entities"), "entities")
    for item in entities[:20]:
        entity = _object(item, "entity")
        status = _string(entity.get("status"), "entity.status")
        if status not in _AUDIT_ENTITY_STATUSES:
            raise ValueError(f"unsupported audit entity status {status!r}")
        name = entity.get("name")
        global_id = entity.get("global_id")
        shown = name if isinstance(name, str) and name else (
            global_id if isinstance(global_id, str) else "<unnamed>"
        )
        missing = entity.get("missing_quantities")
        missing_text = (
            ", ".join(str(item) for item in missing)
            if isinstance(missing, list) and missing
            else ""
        )
        entity_rows.append(
            (
                shown,
                _string(entity.get("canonical_class"), "entity.canonical_class"),
                status,
                missing_text,
            )
        )
        entity_accents.append(status)
    entity_overflow = (
        f"and {len(entities) - 20} more supported products"
        if len(entities) > 20
        else ""
    )

    inventory = _object(document.get("inventory"), "inventory")
    opaque = _object(inventory.get("opaque_type_counts"), "opaque_type_counts")
    inventory_card = Card(
        "inventory",
        (
            (
                "instances",
                f"{_count(inventory.get('instance_count'), 'instance_count')} total, "
                f"{_count(inventory.get('supported_product_count'), 'supported_product_count')} "
                "supported products",
            ),
            (
                "opaque types",
                f"{len(opaque)} (preserved verbatim, never silently dropped)",
            ),
        ),
    )

    issue_counts = _object(document.get("issue_counts"), "issue_counts")
    blocks: list[Card | Table] = [
        Table(
            "pipeline stages",
            ("stage", "status", "detail"),
            tuple(stage_rows),
            tuple(stage_accents),
        ),
        inventory_card,
    ]
    if entity_rows:
        blocks.append(
            Table(
                "supported products",
                ("entity", "class", "status", "missing quantities"),
                tuple(entity_rows),
                tuple(entity_accents),
                overflow=entity_overflow,
            )
        )
    if issue_counts:
        blocks.append(
            Table(
                "issues",
                ("code", "count"),
                tuple(
                    (code, str(_count(count, f"issue_counts.{code}")))
                    for code, count in sorted(issue_counts.items())
                ),
                tuple("" for _ in issue_counts),
            )
        )
    identity_fields: list[tuple[str, str]] = [
        ("sha256", _string(source.get("sha256"), "source.sha256")),
        (
            "size",
            f"{_count(source.get('size_bytes'), 'source.size_bytes')} bytes",
        ),
        ("schema", str(document.get("schema") or "<unparsed>")),
    ]
    world_digest = pipeline.get("world_digest")
    if isinstance(world_digest, str) and world_digest:
        identity_fields.append(("world", world_digest))
    blocks.append(Card("identity", tuple(identity_fields)))
    blocks.append(
        Card(
            "assurance",
            tuple(_scalar_fields(_object(document.get("assurance"), "assurance"))),
        )
    )
    path = _string(source.get("path"), "source.path")
    name = path.replace("\\", "/").rsplit("/", 1)[-1]
    return DecisionReport(
        operation="audit",
        request_id="",
        world_digest=world_digest if isinstance(world_digest, str) else "",
        disposition="PASS" if ready else "BLOCKED",
        subject=name,
        subline="fail-closed IFC compatibility audit",
        notes=(),
        blocks=tuple(blocks),
        footers=(NON_AUTHORIZING_FOOTER, READ_ONLY_FOOTER),
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
:root { color-scheme: light dark;
  --bg: #f5f4f1; --card: #ffffff; --ink: #1c1c1a; --muted: #6b6a66;
  --rule: #e4e2dc; --rule-soft: #efede8; --code-bg: #f0efeb; --note-rule: #c9c6bf;
  --shadow: 0 1px 2px rgba(0,0,0,0.06); }
@media (prefers-color-scheme: dark) { :root {
  --bg: #17171a; --card: #222226; --ink: #ecebe6; --muted: #a09e97;
  --rule: #3a3a40; --rule-soft: #2e2e33; --code-bg: #2b2b30; --note-rule: #4a4a50;
  --shadow: 0 1px 2px rgba(0,0,0,0.4); } }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 46rem; margin: 2rem auto; padding: 0 1rem; }
header.banner { color: #fff; border-radius: 10px; padding: 1.1rem 1.3rem; }
header.banner h1 { margin: 0; font-size: 1.35rem; letter-spacing: 0.02em; }
header.banner p { margin: 0.2rem 0 0; opacity: 0.92; }
p.note { background: var(--card); border-left: 4px solid var(--note-rule);
  padding: 0.6rem 0.9rem; border-radius: 0 8px 8px 0; }
section { background: var(--card); border-radius: 10px; padding: 0.9rem 1.2rem;
  margin-top: 1rem; box-shadow: var(--shadow); }
section.proceed { border-left: 4px solid #1ab233; }
section.stop { border-left: 4px solid #d91414; }
section.attention { border-left: 4px solid #f28c0d; }
section.undecided { border-left: 4px solid #595959; }
section h2 { margin: 0 0 0.5rem; font-size: 0.8rem; text-transform: uppercase;
  letter-spacing: 0.08em; color: var(--muted); }
dl { margin: 0; display: grid; grid-template-columns: max-content 1fr;
  gap: 0.25rem 1rem; }
dt { color: var(--muted); } dd { margin: 0; overflow-wrap: anywhere; }
div.tablewrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 0.92rem; }
th { text-align: left; color: var(--muted); font-weight: 600;
  border-bottom: 1px solid var(--rule); padding: 0.3rem 0.75rem 0.3rem 0; }
td { border-bottom: 1px solid var(--rule-soft); padding: 0.35rem 0.75rem 0.35rem 0;
  vertical-align: top; }
tr.proceed td:first-child { box-shadow: inset 3px 0 0 #1ab233; }
tr.stop td:first-child { box-shadow: inset 3px 0 0 #d91414; }
tr.attention td:first-child { box-shadow: inset 3px 0 0 #f28c0d; }
span.badge { display: inline-block; color: #fff; border-radius: 999px;
  padding: 0 0.55em; font-size: 0.85em; }
details.digest { display: inline; }
details.digest summary { cursor: pointer; list-style: none; display: inline; }
details.digest code.full { display: block; margin-top: 0.2rem; }
span.print-digest { display: none; }
code { font-size: 0.92em; background: var(--code-bg); border-radius: 4px;
  padding: 0.05em 0.3em; }
footer { margin: 1.2rem 0 2rem; color: var(--muted); font-size: 0.9rem; }
footer p { margin: 0.15rem 0; }
@media print {
  :root { --bg: #fff; --card: #fff; --shadow: none; --ink: #000; --muted: #444;
    --rule: #bbb; --rule-soft: #ddd; --code-bg: transparent; }
  body { font-size: 11pt; }
  main { max-width: none; margin: 0; padding: 0; }
  header.banner, section, p.note, table { break-inside: avoid; }
  section { border: 1px solid var(--rule); }
  header.banner, span.badge { print-color-adjust: exact; -webkit-print-color-adjust: exact; }
  details.digest { display: none; }
  span.print-digest { display: inline; font-family: ui-monospace, Menlo, monospace;
    font-size: 0.85em; overflow-wrap: anywhere; }
}
"""


def render_html(report: DecisionReport) -> str:
    esc = html_mod.escape
    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<meta name="color-scheme" content="light dark">',
        f"<title>GAT decision report: {esc(report.subject)}</title>",
        f"<style>{_HTML_STYLE}</style></head><body><main>",
    ]
    parts.extend(_html_body_parts(report))
    parts[-1] += "</main></body></html>"
    return "\n".join(parts) + "\n"


def render_html_fragment(report: DecisionReport) -> str:
    """The report's body markup alone — banner, notes, blocks, footer.

    For hosts that compose several reports into one page (the Workbench)
    and carry the report stylesheet themselves.  Byte-identical to the body
    of :func:`render_html`, so a composed report reads exactly like the
    standalone page.
    """
    return "\n".join(_html_body_parts(report))


def _html_body_parts(report: DecisionReport) -> list[str]:
    esc = html_mod.escape
    parts = [
        f'<header class="banner" style="background:{disposition_hex(report.disposition)}">',
        f"<h1>{esc(report.disposition)}</h1>",
        f"<p>{esc(report.subject)}</p>",
        f'<p class="sub">{esc(report.subline)}</p></header>',
    ]
    for note in report.notes:
        parts.append(f'<p class="note">{esc(note)}</p>')
    for block in report.blocks:
        accent = block.accent if isinstance(block, Card) else ""
        signal = SIGNAL_CLASSES.get(accent, "") if accent else ""
        badge = (
            f' <span class="badge" style="background:'
            f'{disposition_hex(accent)}">{esc(accent)}</span>'
            if accent
            else ""
        )
        opening = f'<section class="{signal}">' if signal else "<section>"
        parts.append(f"{opening}<h2>{esc(block.title)}{badge}</h2>")
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
    parts.append("</footer>")
    return parts


def _html_value(value: str) -> str:
    """Digest-looking values collapse to 12 chars with native disclosure."""
    esc = html_mod.escape
    if _DIGEST_RE.fullmatch(value):
        # Closed <details> do not print, so a print-only twin carries the
        # full digest onto paper.
        return (
            '<details class="digest">'
            f"<summary><code>{esc(format_digest(value))}</code></summary>"
            f'<code class="full">{esc(value)}</code></details>'
            f'<span class="print-digest">{esc(value)}</span>'
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
    "decode_ledger",
    "decode_response",
    "disposition_color",
    "disposition_hex",
    "format_digest",
    "format_moment",
    "format_number",
    "format_probability",
    "ledger_report",
    "render_html",
    "render_text",
]
