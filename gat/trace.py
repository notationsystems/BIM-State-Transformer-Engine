"""Execution trace: what ran, what it touched, and proof it was verified.

Every pipeline stage appends a :class:`TraceEvent` carrying the state
digest after the stage, so an execution is inspectable and reproducible
end to end — the final digest is the determinism witness for README §17.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TraceEvent:
    seq: int
    stage: str          # "compile" | "transform" | "observe" | "reject" | "export"
    name: str
    detail: str
    verify: str         # "PASS" | "WARN" | "FAIL" | "-"
    digest: str

    def render(self) -> str:
        return (
            f"{self.seq:>3}  {self.stage:<9} {self.name:<44} "
            f"{self.verify:<5} {self.digest[:12]}  {self.detail}"
        )


@dataclass
class ExecutionTrace:
    events: list[TraceEvent] = field(default_factory=list)

    def add(self, stage: str, name: str, detail: str, verify: str, digest: str) -> TraceEvent:
        event = TraceEvent(len(self.events), stage, name, detail, verify, digest)
        self.events.append(event)
        return event

    def render(self) -> str:
        header = f"{'#':>3}  {'stage':<9} {'operation':<44} {'ver':<5} {'digest':<12}  detail"
        return "\n".join([header] + [e.render() for e in self.events])
