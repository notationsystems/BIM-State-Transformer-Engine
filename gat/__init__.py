"""GAT — Gaussian Architectural Transformer for BIM.

A computational compiler for transforming, propagating, and reasoning over
BIM state using Gaussian representations:

    BIM -> Architectural State -> Gaussian Representation
        -> Transformation -> Verified State
"""

from gat.engine.executor import ExecutionResult, World, execute
from gat.engine.transform import (
    CompositeTransformation,
    Measurement,
    ObserveQuantity,
    ScaleParameter,
    SetParameter,
    ShiftParameter,
    Transformation,
)
from gat.engine.verify import VerificationReport, run_invariants
from gat.errors import (
    GatError,
    LoweringError,
    NumericalError,
    SpfParseError,
    VerificationError,
)
from gat.ids import EntityId, VarId
from gat.session import GatSession

__version__ = "0.1.0"

__all__ = [
    "CompositeTransformation",
    "EntityId",
    "ExecutionResult",
    "GatError",
    "GatSession",
    "LoweringError",
    "Measurement",
    "NumericalError",
    "ObserveQuantity",
    "ScaleParameter",
    "SetParameter",
    "ShiftParameter",
    "SpfParseError",
    "Transformation",
    "VarId",
    "VerificationError",
    "VerificationReport",
    "World",
    "execute",
    "run_invariants",
    "__version__",
]
