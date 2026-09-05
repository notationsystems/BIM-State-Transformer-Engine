"""GatSession: the user-facing facade over the compile/transform/verify loop.

    session = GatSession.load_ifc("model.ifc")
    result = session.run(SetParameter(var, 3.4, design_sigma=0.01))
    session.export_ifc("out.ifc")

The session owns the current :class:`~gat.engine.executor.World`, the
human-readable execution trace, the authoritative hash-chained ledger, and
the source AST needed for export.  All numerical work happens in the engine;
the session sequences it and records evidence.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

from gat.adapters.ifc.lower import lower_ifc
from gat.adapters.ifc.parser import IfcFile, parse_ifc, parse_ifc_file
from gat.adapters.ifc.writer import export_ifc
from gat.adapters.json_io import export_json
from gat.adapters.openusd import (
    DEFAULT_OPENUSD_READ_LIMITS,
    OpenUsdKeyPair,
    OpenUsdReadLimits,
    read_openusd,
    write_openusd,
)
from gat.causal import (
    ApprovalRecord,
    AssessmentRecord,
    ExternalActionRecord,
    PolicyRecord,
)
from gat.engine.executor import ExecutionResult, World, execute
from gat.engine.transform import Transformation
from gat.engine.verify import VerificationReport, run_invariants
from gat.errors import GatError, VerificationError
from gat.ids import EntityId, VarId
from gat.ledger import ExecutionLedger, LedgerEvent, write_ledger
from gat.state_snapshot import read_snapshot, write_snapshot
from gat.trace import ExecutionTrace
from gat.source_identity import bind_source_content, validate_identity_version


class GatSession:
    def __init__(
        self, world: World, source_file: IfcFile | None = None,
        *, source_locator: str | None = None,
    ):
        self.world = world
        self.source_file = source_file
        self.source_locator = source_locator
        self.trace = ExecutionTrace()
        self.imported_trace: list = []
        self.ledger = ExecutionLedger.genesis(world)
        self.carrier_signature_verified = False
        self.carrier_signing_key_id: str | None = None
        report = run_invariants(world)
        self.trace.add(
            "compile",
            source_locator if source_locator is not None
            else world.module.meta.get("source", "<module>"),
            f"{len(world.module.entities)} entities, "
            f"{world.binding.n_raw} raw + {world.binding.n_full - world.binding.n_raw} derived vars",
            _verdict(report),
            world.digest(),
        )
        self.initial_report = report

    # -- constructors ------------------------------------------------------

    @classmethod
    def load_ifc(cls, path: str, *, identity_version: int = 2) -> "GatSession":
        """Import exact IFC bytes; use version 1 only for legacy path binding."""
        validate_identity_version(identity_version)
        if identity_version == 1:
            file = parse_ifc_file(path)
            return cls(
                World.compile(lower_ifc(file, source=path)), file,
                source_locator=str(path),
            )
        raw = Path(path).read_bytes()
        return cls._from_ifc_bytes(raw, str(path), identity_version)

    @classmethod
    def from_text(
        cls, text: str, source: str = "<memory>", *, identity_version: int = 2,
    ) -> "GatSession":
        """Import a string, binding its exact UTF-8 encoding as source bytes."""
        validate_identity_version(identity_version)
        return cls._from_ifc_bytes(text.encode("utf-8"), source, identity_version)

    @classmethod
    def _from_ifc_bytes(cls, raw: bytes, source: str, identity_version: int) -> "GatSession":
        file = parse_ifc(raw.decode("utf-8"))
        module = lower_ifc(file, source=source)
        if identity_version == 2:
            module = bind_source_content(module, hashlib.sha256(raw).hexdigest())
        return cls(World.compile(module), file, source_locator=source)

    @classmethod
    def load_snapshot(cls, path: str) -> "GatSession":
        """Resume a verified computational world from a state snapshot."""
        loaded = read_snapshot(path)
        session = cls(loaded.world)
        session.ledger = ExecutionLedger.genesis(
            loaded.world,
            {
                "checkpoint": "gat-state-snapshot",
                "snapshot_digest": loaded.snapshot_digest,
            },
        )
        session.trace = ExecutionTrace(list(loaded.trace_events))
        session.trace.add(
            "resume",
            path,
            f"snapshot {loaded.snapshot_digest[:12]}",
            "PASS",
            loaded.world.digest(),
        )
        return session

    @classmethod
    def load_openusd(
        cls,
        path: str,
        *,
        limits: OpenUsdReadLimits = DEFAULT_OPENUSD_READ_LIMITS,
        trusted_public_keys: Mapping[str, bytes] | None = None,
        require_signature: bool = False,
    ) -> "GatSession":
        """Resume a verified computational world from an OpenUSD carrier."""
        loaded = read_openusd(
            path,
            limits=limits,
            trusted_public_keys=trusted_public_keys,
            require_signature=require_signature,
        )
        session = cls(loaded.world)
        session.ledger = (
            loaded.ledger
            if loaded.ledger is not None
            else ExecutionLedger.genesis(
                loaded.world,
                {
                    "checkpoint": "legacy-openusd-state-carrier",
                    "snapshot_digest": loaded.snapshot_digest,
                    "carrier_version": loaded.carrier_version,
                    "signature_verified": loaded.signature.verified,
                },
            )
        )
        session.carrier_signature_verified = loaded.signature.verified
        session.carrier_signing_key_id = loaded.signature.key_id
        session.trace = ExecutionTrace(list(loaded.trace_events))
        trust = (
            f", signature verified by {loaded.signature.key_id}"
            if loaded.signature.verified
            else ", unsigned or unverified"
        )
        history = (
            f", ledger {loaded.ledger.head[:12]}"
            if loaded.ledger is not None
            else ", new ledger genesis"
        )
        session.trace.add(
            "resume",
            path,
            f"OpenUSD v{loaded.carrier_version} snapshot "
            f"{loaded.snapshot_digest[:12]}{trust}{history}",
            "PASS",
            loaded.world.digest(),
        )
        return session

    # -- identity helpers --------------------------------------------------

    def entity_by_name(self, name: str) -> EntityId:
        matches = [
            eid
            for eid, entity in self.world.module.entities.items()
            if entity.name == name
        ]
        if len(matches) != 1:
            raise KeyError(f"{len(matches)} entities named {name!r}")
        return matches[0]

    def var(self, entity_name: str, quantity: str) -> VarId:
        return VarId(self.entity_by_name(entity_name), quantity)

    # -- execution ---------------------------------------------------------

    def run(
        self,
        t: Transformation,
        strict: bool = True,
        *,
        provenance: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        """Execute and authoritatively record an accepted or rejected operation.

        ``provenance`` is caller-supplied JSON evidence (sensor identity,
        calibration source, external time, approval reference, and so on).
        It is hash-bound to the event but does not alter state semantics.
        """
        before = self.world
        try:
            result = execute(before, t, strict=strict)
        except GatError as exc:
            self.ledger.record_rejection(before, t, exc, provenance)
            report = exc.report if isinstance(exc, VerificationError) else None
            self.trace.add(
                "reject",
                t.describe()[:44],
                (
                    report.failures[0].detail
                    if report is not None and report.failures
                    else str(exc)
                ),
                "FAIL",
                before.digest(),
            )
            raise
        if not result.committed:
            rejection = VerificationError(result.report)
            self.ledger.record_rejection(before, t, rejection, provenance)
            self.trace.add(
                "reject",
                t.describe()[:44],
                result.report.failures[0].detail if result.report.failures else "",
                "FAIL",
                before.digest(),
            )
            return result
        self.ledger.record_transition(before, result, provenance)
        stage = "observe" if "observe" in t.name else "transform"
        self.world = result.world
        propagation_detail = ""
        if result.propagation is not None:
            propagation_detail = (
                f"; {result.propagation.mode} propagation recomputed "
                f"{result.propagation.derived_value_rows_recomputed} derived and "
                f"{result.propagation.full_covariance_rows_recomputed} covariance rows"
            )
        self.trace.add(
            stage,
            t.describe()[:44],
            f"{len(result.affected)} derived affected{propagation_detail}",
            _verdict(result.report),
            self.world.digest(),
        )
        return result

    def verify(self) -> VerificationReport:
        return run_invariants(self.world)

    @classmethod
    def load_usd(cls, path: str) -> "GatSession":
        """Load the legacy NumPy-only USDA interchange fallback.

        Prefer :meth:`load_snapshot` or :meth:`load_openusd` for canonical
        restart and signed-carrier workflows.
        """
        from gat.adapters.usd_io import load_usd

        world, imported_trace = load_usd(path)
        session = cls(world, None)
        session.imported_trace = imported_trace
        if imported_trace:
            session.trace.add(
                "import",
                path,
                f"{len(imported_trace)} provenance events carried over",
                "-",
                world.digest(),
            )
        return session

    # -- non-mutating causal events ---------------------------------------

    def record_assessment(
        self,
        record: AssessmentRecord,
        *,
        provenance: Mapping[str, object] | None = None,
    ) -> LedgerEvent:
        event = self.ledger.record_causal(self.world, record, provenance)
        self.trace.add(
            "assessment", record.subject[:44], record.verdict, "-", self.world.digest()
        )
        return event

    def record_policy(
        self,
        record: PolicyRecord,
        *,
        provenance: Mapping[str, object] | None = None,
    ) -> LedgerEvent:
        event = self.ledger.record_causal(self.world, record, provenance)
        self.trace.add(
            "policy",
            record.policy_type[:44],
            record.disposition,
            "-",
            self.world.digest(),
        )
        return event

    def record_approval(
        self,
        record: ApprovalRecord,
        *,
        provenance: Mapping[str, object] | None = None,
    ) -> LedgerEvent:
        event = self.ledger.record_causal(self.world, record, provenance)
        self.trace.add(
            "approval",
            record.approval_id[:44],
            f"{record.decision.value} by {record.authority}",
            "-",
            self.world.digest(),
        )
        return event

    def record_external_action(
        self,
        record: ExternalActionRecord,
        *,
        provenance: Mapping[str, object] | None = None,
    ) -> LedgerEvent:
        event = self.ledger.record_causal(self.world, record, provenance)
        self.trace.add(
            "action",
            record.action_id[:44],
            f"{record.action_type}: {record.status.value}",
            "-",
            self.world.digest(),
        )
        return event

    # -- export ------------------------------------------------------------

    def export_usd(self, path: str) -> int:
        """Write the legacy NumPy-only USDA interchange fallback.

        Prefer :meth:`export_snapshot` or :meth:`export_openusd` for canonical
        restart and signed-carrier workflows.
        """
        from gat.adapters.usd_io import export_usd

        events = [
            {
                "seq": e.seq,
                "stage": e.stage,
                "name": e.name,
                "detail": e.detail,
                "verify": e.verify,
                "digest": e.digest,
            }
            for e in self.trace.events
        ]
        count = export_usd(self.world, path, events)
        self.trace.add("export", path, f"usd stage, {count} entities", "-", self.world.digest())
        return count

    def export_ifc(self, path: str) -> tuple[int, int]:
        if self.source_file is None:
            raise ValueError("session has no source IFC AST to export against")
        patched, appended = export_ifc(self.source_file, self.world, path)
        self.trace.add(
            "export", path, f"{patched} patched, {appended} appended", "-", self.world.digest()
        )
        return patched, appended

    def export_json(self, path: str) -> None:
        export_json(self.world, path)
        self.trace.add("export", path, "canonical state json", "-", self.world.digest())

    def export_snapshot(self, path: str) -> str:
        """Write a restartable, integrity-bound computational-state snapshot."""
        digest = write_snapshot(self.world, path, tuple(self.trace.events))
        self.trace.add(
            "export",
            path,
            f"restartable snapshot {digest[:12]}",
            "-",
            self.world.digest(),
        )
        return digest

    def export_ledger(self, path: str) -> str:
        """Write the authoritative transition history and return its chain head."""
        head = write_ledger(self.ledger, path)
        self.trace.add(
            "export",
            path,
            f"execution ledger {head[:12]}",
            "-",
            self.world.digest(),
        )
        return head

    def export_openusd(
        self,
        path: str,
        *,
        include_geometry: bool = True,
        signing_key: OpenUsdKeyPair | None = None,
    ) -> str:
        """Write restartable state plus a disposable derived OpenUSD view."""
        digest = write_openusd(
            self.world,
            path,
            tuple(self.trace.events),
            include_geometry=include_geometry,
            signing_key=signing_key,
            ledger=self.ledger,
        )
        self.trace.add(
            "export",
            path,
            f"OpenUSD state carrier {digest[:12]}"
            + ("" if signing_key is None else f", signed by {signing_key.key_id}"),
            "-",
            self.world.digest(),
        )
        return digest


def _verdict(report: VerificationReport) -> str:
    if not report.passed:
        return "FAIL"
    if report.warnings:
        return "WARN"
    return "PASS"
