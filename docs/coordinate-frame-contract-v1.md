# Coordinate-frame contract v1

Status: mathematical primitives and synthetic qualification tests. These APIs
do not yet replace the IR's yaw-only `Placement`, lower general IFC placements,
or generate an acceptance verdict or calibrated measurement update.

## Representation and composition

`gat.geometry.frames` provides `CoordinateFrame`, `FrameGraph`, and
`RigidTransform` under `gat-coordinate-frame-v1`.

- Each frame has an identity, one parent (or no parent for the root), a
  right-handed orthonormal basis, an origin, and an explicit `m` or `mm` unit.
- Rotation columns are child axes in parent coordinates. All stored origins
  (`translation_m`) are in metres in the parent's axes, regardless of either
  frame's coordinate unit. Point inputs/outputs use their declared frame units.
- A transform maps column vectors by `p_parent_m = R @ p_child_m + t_m`.
  Composition `a.compose(b)` means `a @ b`; the child transform acts first.
- The graph requires unique IDs, one root, known parents, and no cycles. The
  root has the identity transform. Unknown frames, reflections, shear, scale
  embedded in rotation, nonfinite inputs and unsupported units are rejected.
- `FrameGraph.covariance` rotates and scales a point covariance using exact
  frame transforms. Metre-to-millimetre conversion multiplies covariance by
  1,000,000. Translation does not change point covariance.

The convention follows the parent/child composition rule described in
[Modern Robotics, homogeneous transformation matrices](https://modernrobotics.northwestern.edu/nu-gm-book-resource/3-3-1-homogeneous-transformation-matrices/).
The implementation uses proper rotation matrices, not independent unrestricted
Euler-angle Gaussians. It adds no SciPy dependency.

```python
import numpy as np
from gat.geometry.frames import CoordinateFrame, FrameGraph, RigidTransform

frames = FrameGraph([
    CoordinateFrame("building", None, RigidTransform.identity()),
    CoordinateFrame("storey", "building", RigidTransform(np.eye(3), [0, 0, 3])),
    CoordinateFrame("opening", "storey", RigidTransform(np.eye(3), [2, 1, 0]), "mm"),
])
point_m = frames.point([1000, 0, 0], "opening", "building")  # [3, 1, 3]
```

`representation_digest()` binds the graph's declared frame identities, units,
parents, rotations and origins. Changing the frame representation changes that
digest; it does not imply new physical evidence or a changed building. This
digest is not a world digest and cannot be substituted for an evidence receipt.
The graph does not infer containment, connectivity, material or evidence links.

## Joint uncertainty convention

`RigidTransform.propagate_points(points_m, joint_covariance)` propagates N points
and **one shared uncertain pose** to first order. The nominal pose is a full
3D rigid transform. The perturbation convention is explicitly:

```text
T_actual = T_nominal @ Exp(delta)
delta = (tx, ty, tz, rx, ry, rz) in the local/child tangent frame
translation errors: metres; rotation errors: radians
joint variable order: (p1_xyz, ..., pN_xyz, delta)
```

For a local point p, the first-order pose Jacobian is `R @ [I, -skew(p)]`.
The function stacks these Jacobians for all points and propagates the **full**
joint covariance. Point/point, point/pose, and translation/rotation correlations
are supported. Zero cross-blocks declare independence; the function does not
invent independent pose copies for separate objects.

Consequently common translation cancels from relative position, and shared
orientation can induce correlated relative uncertainty. The output is nominal
transformed points and a full joint covariance, not the exact expectation or
distribution of nonlinear rotations. Covariances must be finite, symmetric,
and positive semidefinite within declared floating-point roundoff.

This is a local small-error approximation. Large-angle distributions,
multimodal association, uncertain nested frame-chain covariance composition,
and nonlinear confidence coverage require separate methods and qualification.
Exact nested placements are supported; do not treat several uncertain
ancestors as independent without an explicit joint model.

## Qualification evidence and remaining gates

The synthetic tests check:

1. Nested rotation/translation composition against direct application, including
   noncommuting order and inverse transforms.
2. Point and covariance round trips between metres and millimetres.
3. Opening-minus-assembly fit margin, variance and Gaussian probability under
   global translation and full 3D rotation, with geometry and covariance
   transformed together.
4. Exact cancellation of shared translation from relative position, and the
   relative uncertainty caused by shared rotation.
5. Full joint propagation against an independently evaluated finite-difference
   Jacobian, including point/pose cross-correlations.
6. Rejection of invalid topology, bases, units and covariance matrices.

Metre-scale transform checks use absolute tolerances down to 1e-12; the
opening-fit probability check uses 1e-10. The finite-difference probe uses
1e-6 perturbations with covariance tolerance 1e-12 absolute / 1e-8 relative.
These tolerances qualify the supplied synthetic cases, not arbitrary scene
scales, sensor accuracy, or a cross-platform bitwise execution contract.

Calibration is a separate evidence boundary. No calibration claim or
measurement update is produced by this module. The existing scan adapter's
independent-pose requirement is unchanged; registration to the BIM remains an
association step, not independent confirmation that the BIM is correct.

Next gates: bind these frame records to the bounded IFC opening-fit workflow;
map independently measured dimensions and calibrated pose into the joint model;
carry calibration version/source/control identities and residuals separately;
check predictions on held-out measurements; expose frame, uncertainty and
residual metadata in Claude's report/viewer surfaces. No field measurements
or independently established external-model outcome are available yet.

### Bounded external compatibility target

The existing pinned buildingSMART wall/opening/window sample is a small next
integration target, not field evidence. Its exact content hash is
`73b0e45d931d5dc13bfee5fdc7bd80f796526445458b2de74c4168d209097832`
(12,492 bytes; IFC4; CC-BY-4.0). Source, commit and licensing provenance are in
[`validation/ifc-corpus-v1.json`](../validation/ifc-corpus-v1.json).

Local audit confirms three supported products, three missing required
quantities, one `MISSING_SOURCE_DATA` product and two
`NEEDS_GEOMETRY_DERIVATION` products. The pipeline remains `BLOCKED`.
Implement only the geometry, placement and dependency coverage needed for an
explicit opening-fit scope; do not invent quantities to make the model pass.
Independently measured dimensions and a reference outcome remain missing.
