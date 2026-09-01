"""Fail-closed IFC beam geometry derivation with identity-bound provenance.

The adapter derives the member axis length from an IFC ``Axis`` polyline for
every supported beam.  For a ``SweptSolid`` body it also discretizes a closed
polyline/composite-curve profile, computes centroidal area moments, and reports
conservative elastic section moduli about the profile axes.  Unsupported body
representations remain explicit ``LENGTH_ONLY`` results; they are never
silently replaced by a bounding box or a name-table lookup.

Reported numerical error is a discretization estimate, not construction or
as-built uncertainty.  A caller must apply a separate, declared uncertainty
policy before these nominal quantities become Gaussian priors.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
import math
import re

from gat.adapters.ifc.parser import EnumVal, IfcFile, RawInstance, Ref, Typed
from gat.adapters.ifc.reader import attr, global_id, name_of, numeric, refs
from gat.adapters.ifc.units import assigned_unit_ids, length_unit_context
from gat.errors import BeamGeometryError, GatError


BEAM_GEOMETRY_FORMAT = "gat-ifc-beam-geometry-v1"
BEAM_GEOMETRY_METHOD = "ifc-axis-composite-profile-moments-v1"
_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")


class BeamGeometryStatus(StrEnum):
    COMPLETE = "COMPLETE"
    LENGTH_ONLY = "LENGTH_ONLY"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class DerivedGeometryQuantity:
    value: float
    unit: str
    numerical_error_bound: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.value) or self.value <= 0.0:
            raise ValueError("derived geometry value must be finite and positive")
        if not math.isfinite(self.numerical_error_bound) or self.numerical_error_bound < 0.0:
            raise ValueError("numerical error bound must be finite and non-negative")
        if not self.unit:
            raise ValueError("derived geometry unit must be non-empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "value": self.value,
            "unit": self.unit,
            "numerical_error_bound": self.numerical_error_bound,
        }


@dataclass(frozen=True)
class BeamGeometryResult:
    beam_step_id: int
    beam_global_id: str
    beam_name: str
    status: BeamGeometryStatus
    axis_length: DerivedGeometryQuantity | None
    cross_section_area: DerivedGeometryQuantity | None
    section_modulus_major: DerivedGeometryQuantity | None
    section_modulus_minor: DerivedGeometryQuantity | None
    source_ifc_sha256: str
    source_refs: tuple[tuple[str, int], ...]
    length_scale_to_metres: float
    max_arc_angle_radians: float
    issues: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.beam_step_id <= 0 or not self.beam_global_id:
            raise ValueError("beam geometry requires a positive STEP id and GlobalId")
        if _DIGEST_RE.fullmatch(self.source_ifc_sha256) is None:
            raise ValueError("source_ifc_sha256 must be a lowercase SHA-256 digest")
        if self.status is BeamGeometryStatus.COMPLETE and any(
            value is None
            for value in (
                self.axis_length,
                self.cross_section_area,
                self.section_modulus_major,
                self.section_modulus_minor,
            )
        ):
            raise ValueError("complete beam geometry must contain all quantities")
        if self.status is BeamGeometryStatus.LENGTH_ONLY and self.axis_length is None:
            raise ValueError("length-only beam geometry requires an axis length")

    def to_dict(self) -> dict[str, object]:
        quantities = {
            "Length": self.axis_length.to_dict() if self.axis_length else None,
            "CrossSectionArea": (
                self.cross_section_area.to_dict() if self.cross_section_area else None
            ),
            "SectionModulusMajorM3": (
                self.section_modulus_major.to_dict()
                if self.section_modulus_major
                else None
            ),
            "SectionModulusMinorM3": (
                self.section_modulus_minor.to_dict()
                if self.section_modulus_minor
                else None
            ),
        }
        return {
            "format": BEAM_GEOMETRY_FORMAT,
            "method": BEAM_GEOMETRY_METHOD,
            "subject": {
                "ifc_class": "IfcBeam",
                "global_id": self.beam_global_id,
                "step_id": self.beam_step_id,
                "name": self.beam_name,
            },
            "status": self.status.value,
            "quantities": quantities,
            "provenance": {
                "source_ifc_sha256": self.source_ifc_sha256,
                "source_refs": [
                    {"role": key, "step_id": value}
                    for key, value in self.source_refs
                ],
                "length_scale_to_metres": self.length_scale_to_metres,
                "max_arc_angle_radians": self.max_arc_angle_radians,
                "uncertainty_scope": (
                    "numerical-discretization-only; no as-built or material tolerance"
                ),
            },
            "issues": list(self.issues),
        }

    def digest(self) -> str:
        encoded = json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class _ProfileProperties:
    area: float
    section_modulus_major: float
    section_modulus_minor: float
    source_refs: tuple[tuple[str, int], ...]


def _require_ref(value: object, label: str) -> Ref:
    if not isinstance(value, Ref):
        raise BeamGeometryError(f"{label} must be an IFC reference")
    return value


def _enum_bool(value: object, label: str) -> bool:
    if isinstance(value, EnumVal) and value.name in {"T", "F"}:
        return value.name == "T"
    raise BeamGeometryError(f"{label} must be .T. or .F.")


def _point(file: IfcFile, reference: Ref, dimensions: int) -> tuple[float, ...]:
    point = file.deref(reference)
    if point.type_name != "IFCCARTESIANPOINT":
        raise BeamGeometryError(f"#{point.step_id} is not IfcCartesianPoint")
    coordinates = attr(point, "Coordinates")
    if not isinstance(coordinates, tuple) or len(coordinates) < dimensions:
        raise BeamGeometryError(f"#{point.step_id} has insufficient coordinates")
    values = tuple(numeric(value) for value in coordinates[:dimensions])
    if not all(math.isfinite(value) for value in values):
        raise BeamGeometryError(f"#{point.step_id} contains non-finite coordinates")
    return values


def _angle_scale_to_radians(file: IfcFile) -> float:
    assigned = assigned_unit_ids(file)
    candidates: list[float] = []
    for unit in file.by_type("IFCSIUNIT"):
        if assigned is not None and unit.step_id not in assigned:
            continue
        unit_type = attr(unit, "UnitType")
        unit_name = attr(unit, "Name")
        if isinstance(unit_type, EnumVal) and unit_type.name == "PLANEANGLEUNIT":
            if not isinstance(unit_name, EnumVal) or unit_name.name != "RADIAN":
                raise BeamGeometryError(f"#{unit.step_id} is not an SI radian unit")
            candidates.append(1.0)
    for unit in file.by_type("IFCCONVERSIONBASEDUNIT"):
        if assigned is not None and unit.step_id not in assigned:
            continue
        unit_type = attr(unit, "UnitType")
        if not isinstance(unit_type, EnumVal) or unit_type.name != "PLANEANGLEUNIT":
            continue
        factor_ref = _require_ref(attr(unit, "ConversionFactor"), "conversion factor")
        factor = file.deref(factor_ref)
        if factor.type_name != "IFCMEASUREWITHUNIT":
            raise BeamGeometryError(
                f"#{factor.step_id} plane-angle factor is not IfcMeasureWithUnit"
            )
        value = numeric(attr(factor, "ValueComponent"))
        base_ref = _require_ref(attr(factor, "UnitComponent"), "angle base unit")
        base = file.deref(base_ref)
        if base.type_name != "IFCSIUNIT":
            raise BeamGeometryError("plane-angle conversion base must be an SI unit")
        base_type = attr(base, "UnitType")
        base_name = attr(base, "Name")
        if not (
            isinstance(base_type, EnumVal)
            and base_type.name == "PLANEANGLEUNIT"
            and isinstance(base_name, EnumVal)
            and base_name.name == "RADIAN"
        ):
            raise BeamGeometryError("plane-angle conversion base must be radians")
        candidates.append(value)
    if not candidates:
        return 1.0
    first = candidates[0]
    if not math.isfinite(first) or first <= 0.0:
        raise BeamGeometryError("plane-angle scale must be finite and positive")
    if any(not math.isclose(value, first, rel_tol=0.0, abs_tol=1e-15) for value in candidates[1:]):
        raise BeamGeometryError("ambiguous project plane-angle units")
    return first


def _shape_representations(file: IfcFile, beam: RawInstance) -> tuple[RawInstance, ...]:
    if len(beam.args) <= 6:
        raise BeamGeometryError(f"#{beam.step_id} IfcBeam has no Representation")
    definition = file.deref(_require_ref(beam.args[6], "beam Representation"))
    if definition.type_name != "IFCPRODUCTDEFINITIONSHAPE":
        raise BeamGeometryError("beam Representation is not IfcProductDefinitionShape")
    return tuple(
        file.deref(reference)
        for reference in refs(attr(definition, "Representations"))
    )


def _polyline_points(file: IfcFile, curve: RawInstance) -> list[tuple[float, float]]:
    if curve.type_name != "IFCPOLYLINE":
        raise BeamGeometryError(f"#{curve.step_id} is not an IfcPolyline")
    points = [
        _point(file, reference, 2)
        for reference in refs(attr(curve, "Points"))
    ]
    if len(points) < 2:
        raise BeamGeometryError(f"#{curve.step_id} polyline has fewer than two points")
    return [(point[0], point[1]) for point in points]


def _axis_length(
    file: IfcFile,
    representations: tuple[RawInstance, ...],
    scale: float,
) -> tuple[float, tuple[tuple[str, int], ...]]:
    axes = [
        representation
        for representation in representations
        if attr(representation, "RepresentationIdentifier") == "Axis"
    ]
    if len(axes) != 1:
        raise BeamGeometryError(f"expected one Axis representation, found {len(axes)}")
    axis = axes[0]
    items = refs(attr(axis, "Items"))
    if len(items) != 1:
        raise BeamGeometryError("Axis representation must contain exactly one item")
    curve = file.deref(items[0])
    points = _polyline_points(file, curve)
    length = sum(
        math.hypot(b[0] - a[0], b[1] - a[1])
        for a, b in zip(points, points[1:])
    ) * scale
    if not math.isfinite(length) or length <= 0.0:
        raise BeamGeometryError("derived beam axis length is not positive")
    return length, (("axis_representation", axis.step_id), ("axis_curve", curve.step_id))


def _trim_parameter(value: object, label: str) -> float:
    if not isinstance(value, tuple) or len(value) != 1:
        raise BeamGeometryError(f"{label} must contain one parameter value")
    item = value[0]
    if not isinstance(item, Typed) or item.name != "IFCPARAMETERVALUE":
        raise BeamGeometryError(f"{label} must use IfcParameterValue")
    result = numeric(item)
    if not math.isfinite(result):
        raise BeamGeometryError(f"{label} is non-finite")
    return result


def _circle_segment_points(
    file: IfcFile,
    trimmed: RawInstance,
    *,
    same_sense: bool,
    scale: float,
    angle_scale: float,
    max_arc_angle: float,
) -> tuple[list[tuple[float, float]], tuple[tuple[str, int], ...]]:
    circle = file.deref(_require_ref(attr(trimmed, "BasisCurve"), "trimmed basis curve"))
    if circle.type_name != "IFCCIRCLE":
        raise BeamGeometryError(f"#{trimmed.step_id} basis curve is not IfcCircle")
    placement = file.deref(_require_ref(attr(circle, "Position"), "circle Position"))
    if placement.type_name != "IFCAXIS2PLACEMENT2D":
        raise BeamGeometryError("circle Position must be IfcAxis2Placement2D")
    center = _point(file, _require_ref(attr(placement, "Location"), "circle Location"), 2)
    direction_value = attr(placement, "RefDirection")
    rotation = 0.0
    direction_ref_id: int | None = None
    if isinstance(direction_value, Ref):
        direction_ref_id = direction_value.step_id
        direction = file.deref(direction_value)
        ratios = attr(direction, "DirectionRatios")
        if not isinstance(ratios, tuple) or len(ratios) < 2:
            raise BeamGeometryError("2D circle direction has insufficient ratios")
        dx, dy = numeric(ratios[0]), numeric(ratios[1])
        rotation = math.atan2(dy, dx)
    radius = numeric(attr(circle, "Radius")) * scale
    if not math.isfinite(radius) or radius <= 0.0:
        raise BeamGeometryError("circle radius must be finite and positive")
    start = _trim_parameter(attr(trimmed, "Trim1"), "Trim1") * angle_scale
    end = _trim_parameter(attr(trimmed, "Trim2"), "Trim2") * angle_scale
    sense = _enum_bool(attr(trimmed, "SenseAgreement"), "SenseAgreement")
    if sense:
        span = (end - start) % math.tau
    else:
        span = -((start - end) % math.tau)
    if math.isclose(span, 0.0, abs_tol=1e-14):
        span = math.tau if sense else -math.tau
    count = max(1, math.ceil(abs(span) / max_arc_angle))
    angles = [start + span * index / count for index in range(count + 1)]
    points = [
        (
            center[0] * scale + radius * math.cos(angle + rotation),
            center[1] * scale + radius * math.sin(angle + rotation),
        )
        for angle in angles
    ]
    if not same_sense:
        points.reverse()
    source_refs = [
        ("trimmed_curve", trimmed.step_id),
        ("circle", circle.step_id),
        ("circle_placement", placement.step_id),
    ]
    if direction_ref_id is not None:
        source_refs.append(("circle_direction", direction_ref_id))
    return points, tuple(source_refs)


def _curve_points(
    file: IfcFile,
    curve: RawInstance,
    *,
    scale: float,
    angle_scale: float,
    max_arc_angle: float,
) -> tuple[list[tuple[float, float]], tuple[tuple[str, int], ...]]:
    if curve.type_name == "IFCPOLYLINE":
        return (
            [(x * scale, y * scale) for x, y in _polyline_points(file, curve)],
            (("profile_polyline", curve.step_id),),
        )
    if curve.type_name != "IFCCOMPOSITECURVE":
        raise BeamGeometryError(f"unsupported profile curve {curve.type_name}")
    joined: list[tuple[float, float]] = []
    source_refs: list[tuple[str, int]] = [("profile_curve", curve.step_id)]
    tolerance = max(scale * 1e-7, 1e-10)
    for segment_ref in refs(attr(curve, "Segments")):
        segment = file.deref(segment_ref)
        if segment.type_name != "IFCCOMPOSITECURVESEGMENT":
            raise BeamGeometryError(f"unsupported composite segment {segment.type_name}")
        same_sense = _enum_bool(attr(segment, "SameSense"), "SameSense")
        parent = file.deref(_require_ref(attr(segment, "ParentCurve"), "ParentCurve"))
        source_refs.append(("profile_segment", segment.step_id))
        if parent.type_name == "IFCPOLYLINE":
            points = [(x * scale, y * scale) for x, y in _polyline_points(file, parent)]
            refs_used = (("profile_polyline", parent.step_id),)
            if not same_sense:
                points.reverse()
        elif parent.type_name == "IFCTRIMMEDCURVE":
            points, refs_used = _circle_segment_points(
                file,
                parent,
                same_sense=same_sense,
                scale=scale,
                angle_scale=angle_scale,
                max_arc_angle=max_arc_angle,
            )
        else:
            raise BeamGeometryError(f"unsupported composite parent {parent.type_name}")
        source_refs.extend(refs_used)
        if joined:
            gap = math.hypot(joined[-1][0] - points[0][0], joined[-1][1] - points[0][1])
            if gap > tolerance:
                raise BeamGeometryError(
                    f"profile segment #{segment.step_id} is discontinuous by {gap:.6g} m"
                )
            joined.extend(points[1:])
        else:
            joined.extend(points)
    if len(joined) < 4:
        raise BeamGeometryError("closed profile has too few sampled points")
    closing_gap = math.hypot(joined[-1][0] - joined[0][0], joined[-1][1] - joined[0][1])
    if closing_gap > tolerance:
        raise BeamGeometryError(f"profile curve is not closed; gap={closing_gap:.6g} m")
    joined[-1] = joined[0]
    return joined, tuple(source_refs)


def _polygon_properties(points: list[tuple[float, float]]) -> tuple[float, float, float]:
    if points[0] != points[-1]:
        points = [*points, points[0]]
    crosses: list[float] = []
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        crosses.append(x0 * y1 - x1 * y0)
    area = 0.5 * sum(crosses)
    if math.isclose(area, 0.0, abs_tol=1e-18):
        raise BeamGeometryError("profile polygon has zero area")
    if area < 0.0:
        return _polygon_properties(list(reversed(points)))
    cx = sum(
        (x0 + x1) * cross
        for ((x0, _), (x1, _)), cross in zip(zip(points, points[1:]), crosses)
    ) / (6.0 * area)
    cy = sum(
        (y0 + y1) * cross
        for ((_, y0), (_, y1)), cross in zip(zip(points, points[1:]), crosses)
    ) / (6.0 * area)
    ix_origin = sum(
        (y0 * y0 + y0 * y1 + y1 * y1) * cross
        for ((_, y0), (_, y1)), cross in zip(zip(points, points[1:]), crosses)
    ) / 12.0
    iy_origin = sum(
        (x0 * x0 + x0 * x1 + x1 * x1) * cross
        for ((x0, _), (x1, _)), cross in zip(zip(points, points[1:]), crosses)
    ) / 12.0
    ix = ix_origin - area * cy * cy
    iy = iy_origin - area * cx * cx
    xs = [point[0] for point in points[:-1]]
    ys = [point[1] for point in points[:-1]]
    distances_x = (max(ys) - cy, cy - min(ys))
    distances_y = (max(xs) - cx, cx - min(xs))
    if min(*distances_x, *distances_y, ix, iy) <= 0.0:
        raise BeamGeometryError("profile has invalid centroidal section properties")
    zx = min(ix / distances_x[0], ix / distances_x[1])
    zy = min(iy / distances_y[0], iy / distances_y[1])
    return area, max(zx, zy), min(zx, zy)


def _profile_properties(
    file: IfcFile,
    profile: RawInstance,
    *,
    scale: float,
    angle_scale: float,
    max_arc_angle: float,
) -> _ProfileProperties:
    if profile.type_name != "IFCARBITRARYCLOSEDPROFILEDEF":
        raise BeamGeometryError(f"unsupported swept profile {profile.type_name}")
    outer = file.deref(_require_ref(attr(profile, "OuterCurve"), "profile OuterCurve"))
    points, source_refs = _curve_points(
        file,
        outer,
        scale=scale,
        angle_scale=angle_scale,
        max_arc_angle=max_arc_angle,
    )
    area, major, minor = _polygon_properties(points)
    return _ProfileProperties(
        area,
        major,
        minor,
        (("profile", profile.step_id), ("outer_curve", outer.step_id), *source_refs),
    )


def _body_profile(
    file: IfcFile,
    representations: tuple[RawInstance, ...],
    *,
    scale: float,
    angle_scale: float,
    max_arc_angle: float,
) -> tuple[_ProfileProperties, _ProfileProperties, tuple[tuple[str, int], ...]]:
    bodies = [
        representation
        for representation in representations
        if attr(representation, "RepresentationIdentifier") == "Body"
    ]
    if len(bodies) != 1:
        raise BeamGeometryError(f"expected one Body representation, found {len(bodies)}")
    body = bodies[0]
    representation_type = attr(body, "RepresentationType")
    if representation_type != "SweptSolid":
        raise BeamGeometryError(f"unsupported Body representation {representation_type!r}")
    items = refs(attr(body, "Items"))
    if len(items) != 1:
        raise BeamGeometryError("SweptSolid Body must contain exactly one item")
    solid = file.deref(items[0])
    if solid.type_name != "IFCEXTRUDEDAREASOLID":
        raise BeamGeometryError(f"unsupported swept solid {solid.type_name}")
    profile = file.deref(_require_ref(attr(solid, "SweptArea"), "SweptArea"))
    fine = _profile_properties(
        file,
        profile,
        scale=scale,
        angle_scale=angle_scale,
        max_arc_angle=max_arc_angle,
    )
    coarse = _profile_properties(
        file,
        profile,
        scale=scale,
        angle_scale=angle_scale,
        max_arc_angle=max_arc_angle * 2.0,
    )
    source_refs = (
        ("body_representation", body.step_id),
        ("swept_solid", solid.step_id),
        *fine.source_refs,
    )
    return fine, coarse, source_refs


def derive_beam_geometry(
    file: IfcFile,
    beam: RawInstance,
    *,
    source_ifc_sha256: str,
    max_arc_angle_radians: float = math.radians(1.0),
) -> BeamGeometryResult:
    """Derive one beam's nominal geometry without making it authoritative state."""
    if beam.type_name != "IFCBEAM":
        raise BeamGeometryError("beam geometry derivation requires IfcBeam")
    if _DIGEST_RE.fullmatch(source_ifc_sha256) is None:
        raise BeamGeometryError("source IFC digest must be lowercase SHA-256")
    if (
        not math.isfinite(max_arc_angle_radians)
        or not 0.0 < max_arc_angle_radians <= math.pi / 4.0
    ):
        raise BeamGeometryError("max arc angle must be in (0, pi/4]")
    units = length_unit_context(file)
    return _derive_beam_geometry_with_context(
        file,
        beam,
        source_ifc_sha256=source_ifc_sha256,
        max_arc_angle_radians=max_arc_angle_radians,
        length_scale_to_metres=units.scale_to_metres,
        angle_scale_to_radians=_angle_scale_to_radians(file),
    )


def _derive_beam_geometry_with_context(
    file: IfcFile,
    beam: RawInstance,
    *,
    source_ifc_sha256: str,
    max_arc_angle_radians: float,
    length_scale_to_metres: float,
    angle_scale_to_radians: float,
) -> BeamGeometryResult:
    try:
        representations = _shape_representations(file, beam)
        axis_length, axis_refs = _axis_length(
            file,
            representations,
            length_scale_to_metres,
        )
    except GatError as exc:
        return BeamGeometryResult(
            beam.step_id,
            global_id(beam),
            name_of(beam),
            BeamGeometryStatus.BLOCKED,
            None,
            None,
            None,
            None,
            source_ifc_sha256,
            (),
            length_scale_to_metres,
            max_arc_angle_radians,
            (str(exc),),
        )
    length_quantity = DerivedGeometryQuantity(axis_length, "m", 0.0)
    try:
        fine, coarse, profile_refs = _body_profile(
            file,
            representations,
            scale=length_scale_to_metres,
            angle_scale=angle_scale_to_radians,
            max_arc_angle=max_arc_angle_radians,
        )
    except GatError as exc:
        return BeamGeometryResult(
            beam.step_id,
            global_id(beam),
            name_of(beam),
            BeamGeometryStatus.LENGTH_ONLY,
            length_quantity,
            None,
            None,
            None,
            source_ifc_sha256,
            tuple(axis_refs),
            length_scale_to_metres,
            max_arc_angle_radians,
            (str(exc),),
        )
    area_error = max(abs(fine.area - coarse.area), fine.area * 1e-12)
    major_error = max(
        abs(fine.section_modulus_major - coarse.section_modulus_major),
        fine.section_modulus_major * 1e-12,
    )
    minor_error = max(
        abs(fine.section_modulus_minor - coarse.section_modulus_minor),
        fine.section_modulus_minor * 1e-12,
    )
    return BeamGeometryResult(
        beam.step_id,
        global_id(beam),
        name_of(beam),
        BeamGeometryStatus.COMPLETE,
        length_quantity,
        DerivedGeometryQuantity(fine.area, "m2", area_error),
        DerivedGeometryQuantity(fine.section_modulus_major, "m3", major_error),
        DerivedGeometryQuantity(fine.section_modulus_minor, "m3", minor_error),
        source_ifc_sha256,
        tuple(sorted({*axis_refs, *profile_refs})),
        length_scale_to_metres,
        max_arc_angle_radians,
    )


def derive_all_beam_geometry(
    file: IfcFile,
    *,
    source_ifc_sha256: str,
    max_arc_angle_radians: float = math.radians(1.0),
) -> tuple[BeamGeometryResult, ...]:
    """Derive every beam independently so one unsupported body cannot hide others."""
    if _DIGEST_RE.fullmatch(source_ifc_sha256) is None:
        raise BeamGeometryError("source IFC digest must be lowercase SHA-256")
    if (
        not math.isfinite(max_arc_angle_radians)
        or not 0.0 < max_arc_angle_radians <= math.pi / 4.0
    ):
        raise BeamGeometryError("max arc angle must be in (0, pi/4]")
    units = length_unit_context(file)
    angle_scale = _angle_scale_to_radians(file)
    return tuple(
        _derive_beam_geometry_with_context(
            file,
            beam,
            source_ifc_sha256=source_ifc_sha256,
            max_arc_angle_radians=max_arc_angle_radians,
            length_scale_to_metres=units.scale_to_metres,
            angle_scale_to_radians=angle_scale,
        )
        for beam in file.by_type("IFCBEAM")
    )


__all__ = [
    "BEAM_GEOMETRY_FORMAT",
    "BEAM_GEOMETRY_METHOD",
    "BeamGeometryResult",
    "BeamGeometryStatus",
    "DerivedGeometryQuantity",
    "derive_all_beam_geometry",
    "derive_beam_geometry",
]
