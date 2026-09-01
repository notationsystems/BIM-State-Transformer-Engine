"""Blender-independent decoder for GAT headless workflow responses."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping


RESPONSE_FORMAT = "gat-headless-response-v1"
ACCEPTANCE_DISPOSITIONS = frozenset({"ACCEPT", "REJECT", "REQUEST_EVIDENCE"})


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


def load_response(path: str | Path) -> WorkflowView:
    return parse_response(json.loads(Path(path).read_text(encoding="utf-8")))


def parse_response(value: object) -> WorkflowView:
    response = _object(value, "response")
    required = {"format", "request_id", "operation", "world_digest", "result"}
    if set(response) != required:
        raise ValueError("response fields differ from gat-headless-response-v1")
    if response["format"] != RESPONSE_FORMAT:
        raise ValueError(f"unsupported response format {response['format']!r}")
    if response["operation"] != "acceptance":
        raise ValueError("Blender assurance view requires an acceptance response")
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


def disposition_color(disposition: str) -> tuple[float, float, float, float]:
    colors = {
        "ACCEPT": (0.10, 0.70, 0.20, 1.0),
        "REJECT": (0.85, 0.08, 0.08, 1.0),
        "REQUEST_EVIDENCE": (0.95, 0.55, 0.05, 1.0),
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


__all__ = [
    "EvidenceRequestView",
    "WorkflowView",
    "disposition_color",
    "load_response",
    "parse_response",
]
