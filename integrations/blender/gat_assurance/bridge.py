"""Blender-independent decoder for GAT headless workflow responses."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping


RESPONSE_FORMAT = "gat-headless-response-v1"
ACCEPTANCE_DISPOSITIONS = frozenset({"ACCEPT", "REJECT", "REQUEST_EVIDENCE"})
BEAM_DISPOSITIONS = frozenset({"SATISFIED", "VIOLATED", "UNRESOLVED"})
BEAM_METHOD = "ansi-aisc-360-22-f2-1-lrfd-v1"
BEAM_ORACLE_ID = "aisc-v16-example-f1-1b-lrfd-v1"


@dataclass(frozen=True)
class EvidenceRequestView:
    check_id: str
    action: str
    target: str
    reason: str


@dataclass(frozen=True)
class WorkflowView:
    request_id: str
    world_digest: str
    case_id: str
    subject: str
    disposition: str
    reasons: tuple[str, ...]
    requests: tuple[EvidenceRequestView, ...]
    overlay_subjects: tuple[str, ...]

    @property
    def color(self) -> tuple[float, float, float, float]:
        return disposition_color(self.disposition)

    @property
    def headline(self) -> str:
        return f"{self.disposition}: {self.subject}"


@dataclass(frozen=True)
class BeamAssuranceView:
    request_id: str
    world_digest: str
    case_id: str
    subject: str
    disposition: str
    reasons: tuple[str, ...]
    overlay_subjects: tuple[str, ...]
    prior_verdict: str
    prior_capacity_n_m: float
    revised_capacity_n_m: float
    method: str
    oracle_id: str
    may_authorize: bool
    #: (flag name, "yes"/"no") pairs from the response's assurance record,
    #: shown verbatim even when unflattering.
    assurance_flags: tuple[tuple[str, str], ...] = ()
    #: Short human lines describing the conditioning evidence: certificate,
    #: issuer (with its trust status), and the observed value.
    evidence_lines: tuple[str, ...] = ()

    @property
    def requests(self) -> tuple[EvidenceRequestView, ...]:
        return ()

    @property
    def color(self) -> tuple[float, float, float, float]:
        return disposition_color(self.disposition)

    @property
    def headline(self) -> str:
        return f"{self.disposition}: {self.subject}"


def load_response(path: str | Path) -> WorkflowView | BeamAssuranceView:
    return parse_response(json.loads(Path(path).read_text(encoding="utf-8")))


def parse_response(value: object) -> WorkflowView | BeamAssuranceView:
    response = _object(value, "response")
    required = {"format", "request_id", "operation", "world_digest", "result"}
    if set(response) != required:
        raise ValueError("response fields differ from gat-headless-response-v1")
    if response["format"] != RESPONSE_FORMAT:
        raise ValueError(f"unsupported response format {response['format']!r}")
    operation = response["operation"]
    if operation == "beam_assurance":
        return _parse_beam_assurance(response)
    if operation != "acceptance":
        raise ValueError(
            "Blender assurance view requires an acceptance or beam_assurance response"
        )
    result = _object(response["result"], "result")
    disposition = _string(result.get("disposition"), "disposition")
    if disposition not in ACCEPTANCE_DISPOSITIONS:
        raise ValueError(f"unsupported acceptance disposition {disposition!r}")

    reasons_raw = result.get("reasons")
    if not isinstance(reasons_raw, list):
        raise ValueError("reasons must be an array")
    reasons = tuple(_string(item, "reason") for item in reasons_raw)

    requests_raw = result.get("evidence_requests")
    if not isinstance(requests_raw, list):
        raise ValueError("evidence_requests must be an array")
    requests: list[EvidenceRequestView] = []
    for item in requests_raw:
        request = _object(item, "evidence request")
        requests.append(
            EvidenceRequestView(
                _string(request.get("check_id"), "check_id"),
                _string(request.get("action"), "action"),
                _string(request.get("target"), "target"),
                _string(request.get("reason"), "reason"),
            )
        )

    subjects: set[str] = set()
    checks = result.get("checks")
    if not isinstance(checks, list):
        raise ValueError("checks must be an array")
    for item in checks:
        check = _object(item, "check")
        details = _object(check.get("details"), "check.details")
        risks = details.get("risks", [])
        if not isinstance(risks, list):
            raise ValueError("check risk details must be an array")
        for raw_risk in risks:
            risk = _object(raw_risk, "risk")
            subjects.add(_string(risk.get("element"), "risk.element"))

    return WorkflowView(
        request_id=_string(response["request_id"], "request_id"),
        world_digest=_string(response["world_digest"], "world_digest"),
        case_id=_string(result.get("case_id"), "case_id"),
        subject=_string(result.get("subject"), "subject"),
        disposition=disposition,
        reasons=reasons,
        requests=tuple(requests),
        overlay_subjects=tuple(sorted(subjects)),
    )


def _parse_beam_assurance(
    response: Mapping[str, object],
) -> BeamAssuranceView:
    result = _object(response["result"], "result")
    disposition = _string(result.get("disposition"), "disposition")
    if disposition not in BEAM_DISPOSITIONS:
        raise ValueError(f"unsupported beam disposition {disposition!r}")
    prior = _object(result.get("prior"), "prior")
    revised = _object(result.get("revised"), "revised")
    change = _object(result.get("decision_change"), "decision_change")
    computation = _object(revised.get("computation"), "revised.computation")
    assurance = _object(result.get("assurance"), "assurance")
    verification = _object(result.get("verification"), "verification")
    transition = _object(result.get("transition"), "transition")
    beam = _object(result.get("beam"), "beam")
    subject = _string(result.get("subject"), "subject")
    beam_name = _string(beam.get("name"), "beam.name")
    if beam_name != subject:
        raise ValueError("beam name and response subject differ")
    if verification.get("passed") is not True:
        raise ValueError("Blender refuses an unverified beam response")
    may_authorize = assurance.get("may_authorize")
    if may_authorize is not False:
        raise ValueError("beam response must remain non-authorizing")
    world_digest = _string(response["world_digest"], "world_digest")
    prior_digest = _string(
        transition.get("prior_world_digest"),
        "transition.prior_world_digest",
    )
    result_digest = _string(
        transition.get("result_world_digest"),
        "transition.result_world_digest",
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
    method = _string(computation.get("method"), "computation.method")
    oracle_id = _string(
        computation.get("independent_oracle_id"),
        "computation.independent_oracle_id",
    )
    if method != BEAM_METHOD or oracle_id != BEAM_ORACLE_ID:
        raise ValueError("unsupported beam method or validation oracle")

    assurance_flags = tuple(
        (key, "yes" if value else "no")
        for key, value in assurance.items()
        if isinstance(value, bool)
    )
    evidence = _object(result.get("evidence"), "evidence")
    observation = _object(evidence.get("evidence"), "evidence.evidence")
    certificate = _object(
        evidence.get("material_certificate"), "evidence.material_certificate"
    )
    issuer = _object(certificate.get("issuer"), "certificate.issuer")
    trust = "trusted" if issuer.get("trust_verified") is True else "trust not verified"
    evidence_lines = (
        (
            f"{_string(certificate.get('certificate_id'), 'certificate_id')} "
            f"({_string(certificate.get('issued_at'), 'issued_at')})"
        ),
        f"{_string(issuer.get('name'), 'issuer.name')} ({trust})",
        (
            f"{_number(observation.get('observed_value'), 'observed_value'):g} +- "
            f"{_number(observation.get('noise_sigma'), 'noise_sigma'):g} "
            f"{_string(observation.get('unit'), 'unit')} "
            f"({_string(observation.get('kind'), 'kind')})"
        ),
    )

    return BeamAssuranceView(
        request_id=_string(response["request_id"], "request_id"),
        world_digest=world_digest,
        case_id=_string(result.get("case_id"), "case_id"),
        subject=subject,
        disposition=disposition,
        reasons=(_string(change.get("reason"), "decision_change.reason"),),
        overlay_subjects=(subject,),
        prior_verdict=prior_verdict,
        prior_capacity_n_m=_number(
            prior.get("target_mean_n_m"),
            "prior.target_mean_n_m",
        ),
        revised_capacity_n_m=_number(
            revised.get("target_mean_n_m"),
            "revised.target_mean_n_m",
        ),
        method=method,
        oracle_id=oracle_id,
        may_authorize=False,
        assurance_flags=assurance_flags,
        evidence_lines=evidence_lines,
    )


def disposition_color(disposition: str) -> tuple[float, float, float, float]:
    colors = {
        "ACCEPT": (0.10, 0.70, 0.20, 1.0),
        "REJECT": (0.85, 0.08, 0.08, 1.0),
        "REQUEST_EVIDENCE": (0.95, 0.55, 0.05, 1.0),
        "SATISFIED": (0.10, 0.70, 0.20, 1.0),
        "VIOLATED": (0.85, 0.08, 0.08, 1.0),
        "UNRESOLVED": (0.95, 0.55, 0.05, 1.0),
    }
    try:
        return colors[disposition]
    except KeyError as exc:
        raise ValueError(f"unsupported disposition {disposition!r}") from exc


def _object(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a JSON object")
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


__all__ = [
    "EvidenceRequestView",
    "BeamAssuranceView",
    "WorkflowView",
    "disposition_color",
    "load_response",
    "parse_response",
]
