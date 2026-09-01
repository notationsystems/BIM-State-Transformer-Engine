"""GatSession: the user-facing facade over the compile/transform/verify loop.

    session = GatSession.load_ifc("model.ifc")
    result = session.run(SetParameter(var, 3.4, design_sigma=0.01))
    session.export_ifc("out.ifc")

The session owns the current :class:`~gat.engine.executor.World`, the
execution trace, and the source AST needed for export.  All numerical work
happens in the engine; the session sequences it and records evidence.
"""

from __future__ import annotations

from gat.adapters.ifc.lower import lower_ifc
from gat.adapters.ifc.parser import IfcFile, parse_ifc, parse_ifc_file
from gat.adapters.ifc.writer import export_ifc
from gat.adapters.json_io import export_json
from gat.engine.executor import ExecutionResult, World, execute
from gat.engine.transform import Transformation
from gat.engine.verify import VerificationReport, run_invariants
from gat.errors import VerificationError
from gat.ids import EntityId, VarId
from gat.trace import ExecutionTrace


class GatSession:
    def __init__(self, world: World, source_file: IfcFile | None = None):
        self.world = world
        self.source_file = source_file
        self.trace = ExecutionTrace()
        report = run_invariants(world)
        self.trace.add(
            "compile",
            world.module.meta.get("source", "<module>"),
            f"{len(world.module.entities)} entities, "
            f"{world.binding.n_raw} raw + {world.binding.n_full - world.binding.n_raw} derived vars",
            _verdict(report),
            world.digest(),
        )
        self.initial_report = report

    # -- constructors ------------------------------------------------------

    @classmethod
    def load_ifc(cls, path: str) -> "GatSession":
        file = parse_ifc_file(path)
        module = lower_ifc(file, source=path)
        return cls(World.compile(module), file)

    @classmethod
    def from_text(cls, text: str, source: str = "<memory>") -> "GatSession":
        file = parse_ifc(text)
        module = lower_ifc(file, source=source)
        return cls(World.compile(module), file)

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

    def run(self, t: Transformation, strict: bool = True) -> ExecutionResult:
        try:
            result = execute(self.world, t, strict=strict)
        except VerificationError as exc:
            self.trace.add(
                "reject",
                t.describe()[:44],
                exc.report.failures[0].detail if exc.report.failures else "",
                "FAIL",
                self.world.digest(),
            )
            raise
        stage = "observe" if "observe" in t.name else "transform"
        self.world = result.world
        self.trace.add(
            stage,
            t.describe()[:44],
            f"{len(result.affected)} derived affected",
            _verdict(result.report),
            self.world.digest(),
        )
        return result

    def verify(self) -> VerificationReport:
        return run_invariants(self.world)

    # -- export ------------------------------------------------------------

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


def _verdict(report: VerificationReport) -> str:
    if not report.passed:
        return "FAIL"
    if report.warnings:
        return "WARN"
    return "PASS"
