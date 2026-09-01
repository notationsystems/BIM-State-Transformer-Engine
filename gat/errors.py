"""Exception hierarchy for GAT.

Every error raised by the engine derives from :class:`GatError`, so callers
can catch one type at the boundary.  Parse errors carry source locations;
verification errors carry the full report that rejected the transformation.
"""

from __future__ import annotations


class GatError(Exception):
    """Base class for all GAT errors."""


class SpfParseError(GatError):
    """Lexical or syntactic error in an SPF (ISO 10303-21) file."""

    def __init__(self, message: str, line: int = 0, col: int = 0):
        super().__init__(f"{message} (line {line}, col {col})" if line else message)
        self.line = line
        self.col = col


class LoweringError(GatError):
    """The IFC AST could not be lowered to the Architectural IR."""


class BindingError(GatError):
    """The IR could not be bound onto the Gaussian backend."""


class NumericalError(GatError):
    """A numerical operation left the certified regime (e.g. non-PSD Sigma)."""


class ConditioningError(GatError):
    """An observation could not be conditioned on (degenerate innovation)."""


class VerificationError(GatError):
    """A transformed state violated a hard invariant; the state was rolled back.

    Attributes:
        report: the :class:`~gat.passes.verify.VerificationReport` that failed.
    """

    def __init__(self, report):
        self.report = report
        failed = ", ".join(
            f"{r.invariant_id}[{r.subject}]" for r in report.results if r.status.name == "FAIL"
        )
        super().__init__(f"verification failed: {failed}")


class RegistrationError(GatError):
    """Scan-to-BIM registration failed to converge or was ill-posed."""


class LikelihoodCalibrationError(GatError):
    """Scan evidence failed a provenance, quality, or calibration gate."""


class SnapshotError(GatError):
    """A computational-state snapshot is corrupt, incompatible, or invalid."""


class OpenUsdError(GatError):
    """An OpenUSD state carrier is unavailable, malformed, or incompatible."""


class ScanArtifactError(GatError):
    """An external scan or reconstructed-mesh artifact is unsupported or invalid."""


class DecisionError(GatError):
    """Decision evidence or provenance is inconsistent with the assessed state."""


class LedgerError(GatError):
    """An execution ledger is malformed, has been tampered with, or cannot replay."""


class ProofManifestError(GatError):
    """A computation-proof manifest is malformed, unbound, or inconsistent."""


class BeamGeometryError(GatError):
    """IFC beam geometry could not be derived within the declared adapter scope."""


class CertificateIngestionError(GatError):
    """A material certificate failed its closed ingestion contract."""
