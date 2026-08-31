"""The geometric Gaussian layer: continuous, differentiable 3D representation.

Derived from the canonical architectural state (never a second source of
truth), the scene represents building elements as moment-matched 3D
Gaussian primitive sets and supports probabilistic clash detection,
structural attention propagation, scan-to-BIM registration, multi-scale
fusion, compliance under uncertainty, and differentiable design
objectives.
"""

from gat.geometry.attention import AttentionConfig, laplacian_baseline, propagate
from gat.geometry.clash import ClashReport, detect, score_proposed_box
from gat.geometry.compliance import check_compliance
from gat.geometry.fusion import building_level, element_level, moment_match
from gat.geometry.gaussianize import OrientedBox
from gat.geometry.primitives import GaussianCloud
from gat.geometry.registration import (
    RegistrationResult,
    RigidTransformZ,
    ScanRegistrar,
    synthesize_scan,
)
from gat.geometry.splat_io import export_splat_ply
from gat.geometry.stateio import GeometryScene, derive_scene, relative_covariance

__all__ = [
    "AttentionConfig",
    "ClashReport",
    "GaussianCloud",
    "GeometryScene",
    "OrientedBox",
    "RegistrationResult",
    "RigidTransformZ",
    "ScanRegistrar",
    "building_level",
    "check_compliance",
    "derive_scene",
    "detect",
    "element_level",
    "export_splat_ply",
    "laplacian_baseline",
    "moment_match",
    "propagate",
    "relative_covariance",
    "score_proposed_box",
    "synthesize_scan",
]
