"""The geometric Gaussian layer demonstration.

    python -m gat.demo.geometry

Seven acts over the shipped two-office model:

  G0. GAUSSIANIZE — boxes -> moment-matched primitive sets; splat export.
  G1. CLASH       — probabilistic clash scoring: corner joints vs a
                    proposed duct; correlated-cancellation of shared
                    parameters in relative uncertainty.
  G2. ATTENTION   — structural attention propagation vs the content-blind
                    Laplacian ablation.
  G3. REGISTER    — scan-to-BIM: recover a withheld rigid transform from a
                    synthetic laser scan by GMM alignment.
  G4. FUSE        — level-of-detail moment merging and an exact GIS frame
                    transport.
  G5. COMPLY      — design compliance margins under the joint belief.
  G6. OPTIMIZE    — differentiable layout: tune the party-wall opening for
                    a daylight target under chance constraints, verify the
                    dual-number gradient witness, commit via interventions.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

from gat.geometry import (
    AttentionConfig,
    OrientedBox,
    ScanRegistrar,
    RigidTransformZ,
    building_level,
    check_compliance,
    derive_scene,
    detect,
    element_level,
    export_splat_ply,
    laplacian_baseline,
    propagate,
    score_proposed_box,
    synthesize_scan,
)
from gat.geometry.attention import element_payload_means
from gat.geometry.fusion import FrameTransform
from gat.geometry.objectives import (
    LayoutObjective,
    OptimizationResult,
    as_interventions,
    optimize_layout,
    ray_transmittance,
    scalar_ray_depth_dual,
)
from gat.geometry.stateio import relative_covariance
from gat.ids import VarId
from gat.ir.core import LessEqual
from gat.session import GatSession

MODEL = os.path.join(os.path.dirname(__file__), "model.ifc")


def _hr(title: str) -> None:
    print()
    print(f"=== {title} " + "=" * max(0, 66 - len(title)))


def main() -> int:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "out"
    os.makedirs(out_dir, exist_ok=True)

    session = GatSession.load_ifc(MODEL)
    world = session.world

    # ---- G0 --------------------------------------------------------------
    _hr("G0  GAUSSIANIZE: boxes -> moment-matched Gaussian primitives")
    scene = derive_scene(world)
    print(f"{len(scene.elements)} elements -> {len(scene.cloud)} primitives")
    mean, cov = scene.cloud.of_element(
        scene.element_by_name("Office-A").row
    ).mixture_moments()
    print(
        f"Office-A mixture moments: center ({mean[0]:.3f}, {mean[1]:.3f}, {mean[2]:.3f}), "
        f"cov diag ({cov[0,0]:.4f}, {cov[1,1]:.4f}, {cov[2,2]:.4f})  [exact box moments]"
    )
    assert abs(cov[0, 0] - 5.0**2 / 12.0) < 1e-12
    ply = os.path.join(out_dir, "building_splats.ply")
    n = export_splat_ply(scene.cloud, ply)
    print(f"wrote {ply}: {n} splats (3DGS viewer compatible)")

    # ---- G1 --------------------------------------------------------------
    _hr("G1  CLASH: probabilistic scoring replaces boolean intersection")
    report = detect(scene)
    print(report.render())
    duct = OrientedBox(origin=(4.0, 1.8, 2.6), angle=0.0, extents=(3.0, 0.4, 0.4))
    print("\nproposed duct at z 2.6-3.0 crossing the party wall:")
    duct_report = score_proposed_box(scene, duct, position_sigma=0.02)
    print(duct_report.render())
    worst = duct_report.worst()
    assert worst is not None and worst.p_clash > 0.999
    duct_hi = OrientedBox(origin=(4.0, 1.8, 3.55), angle=0.0, extents=(3.0, 0.4, 0.4))
    clear_report = score_proposed_box(scene, duct_hi, position_sigma=0.02)
    print(f"\nsame duct rerouted above the storey: {len(clear_report.items)} clashes")
    assert not clear_report.items

    south = scene.element_by_name("Wall-South")
    party = scene.element_by_name("Wall-Party")
    rel = relative_covariance(scene, south, party)
    var_a = float(
        (scene.center_jacobian_wrt_raw(south) @ world.belief.sigma
         @ scene.center_jacobian_wrt_raw(south).T)[2, 2]
    )
    print(
        f"\ncorrelated cancellation: each wall top is +-{math.sqrt(var_a)*1000:.1f} mm in z, "
        f"but their RELATIVE z offset is +-{math.sqrt(max(rel[2,2],0))*1000:.1f} mm "
        f"(shared storey height cancels in (J_a - J_b) Sigma (J_a - J_b)^T)"
    )
    assert rel[2, 2] < 1e-12 and var_a > 1e-6

    # ---- G2 --------------------------------------------------------------
    _hr("G2  ATTENTION: semantic propagation, with its ablation")
    config = AttentionConfig()
    att_cloud, alpha = propagate(scene, config)
    lap_cloud = laplacian_baseline(scene, config)
    base = element_payload_means(scene, scene.cloud)
    att = element_payload_means(scene, att_cloud)
    lap = element_payload_means(scene, lap_cloud)
    print("thermal U-value payload  (base -> attention | laplacian ablation):")
    for name in ("Wall-South", "Wall-Party", "Door-1", "Office-A", "Office-B"):
        print(
            f"  {name:<12} {base[name][0]:.3f} -> {att[name][0]:.3f} | {lap[name][0]:.3f}"
        )
    door_drift_att = abs(att["Door-1"][0] - base["Door-1"][0])
    door_drift_lap = abs(lap["Door-1"][0] - base["Door-1"][0])
    print(
        f"content-dependence: attention preserves the door's thermal identity "
        f"({door_drift_att:.3f} drift) where the content-blind baseline blurs it "
        f"({door_drift_lap:.3f} drift)"
    )
    assert door_drift_att < door_drift_lap
    payload = scene.cloud.features[:, 6]
    for cloud in (att_cloud,):
        assert cloud.features[:, 6].min() >= payload.min() - 1e-9
        assert cloud.features[:, 6].max() <= payload.max() + 1e-9  # maximum principle

    # ---- G3 --------------------------------------------------------------
    _hr("G3  REGISTER: scan-to-BIM by Gaussian mixture alignment")
    truth = RigidTransformZ(theta=math.radians(15.0), t=(0.30, -0.20, 0.05))
    scan = synthesize_scan(
        scene, n_points=4000, noise_sigma=0.01, outlier_frac=0.02,
        transform=truth, seed=7,
    )
    print(
        f"synthetic scan: {scan.shape[0]} points, 10 mm sensor noise, 2% outliers, "
        f"withheld pose: yaw 15.000 deg, t (0.300, -0.200, 0.050)"
    )
    registrar = ScanRegistrar(scene)
    result = registrar.register(scan)
    yaw_err, trans_err = result.transform.compose_error(truth)
    print(
        f"recovered: yaw {math.degrees(result.transform.theta):.3f} deg, "
        f"t ({result.transform.t[0]:.3f}, {result.transform.t[1]:.3f}, {result.transform.t[2]:.3f})"
    )
    print(
        f"errors: yaw {math.degrees(yaw_err)*60:.2f} arcmin, translation {trans_err*1000:.1f} mm; "
        f"fit accepted: {result.accepted}"
    )
    sig = result.pose_sigma()
    print(
        f"pose information (Gauss-Newton): sigma_yaw {math.degrees(sig[0])*60:.2f} arcmin, "
        f"sigma_t ({sig[1]*1000:.1f}, {sig[2]*1000:.1f}, {sig[3]*1000:.1f}) mm"
    )
    for stage in (result.coarse_trace, result.nll_trace):
        assert (np.diff(np.asarray(stage)) <= 1e-9).all(), (
            "EM outer iteration must be monotone within each annealing stage"
        )
    assert yaw_err < math.radians(0.2) and trans_err < 0.02 and result.accepted

    # ---- G4 --------------------------------------------------------------
    _hr("G4  FUSE: level-of-detail merging and GIS transport")
    l1 = element_level(scene)
    worst_l1 = max(l1, key=lambda node: node.merge_error)
    print(
        f"L1 (per element): {len(l1)} Gaussians; largest merge error "
        f"{worst_l1.merge_error:.3f} nats ({worst_l1.label})"
    )
    l3 = building_level(scene)
    print(
        f"L3 (building): 1 Gaussian, volume {l3.weight:.2f} m3, "
        f"merge error {l3.merge_error:.3f} nats — the macro-scale token"
    )
    gis = FrameTransform(
        A=np.array(
            [[math.cos(0.35), -math.sin(0.35), 0.0],
             [math.sin(0.35), math.cos(0.35), 0.0],
             [0.0, 0.0, 1.0]]
        ),
        b=np.array([683214.0, 5337890.0, 42.0]),  # a UTM-like offset
    )
    moved = gis.apply_cloud(scene.cloud)
    m0, c0 = scene.cloud.mixture_moments()
    m1, c1 = moved.mixture_moments()
    expected = gis.A @ m0 + gis.b
    assert np.abs(m1 - expected).max() < 1e-6
    assert np.abs(c1 - gis.A @ c0 @ gis.A.T).max() < 1e-9
    print(
        "GIS frame transport is exact: mixture moments transform affinely "
        f"(centroid -> ({m1[0]:.1f}, {m1[1]:.1f}, {m1[2]:.1f}))"
    )

    # ---- G5 --------------------------------------------------------------
    _hr("G5  COMPLY: rule margins under the joint belief")
    compliance = check_compliance(world)
    print(compliance.render())
    assert compliance.passed

    # ---- G6 --------------------------------------------------------------
    _hr("G6  OPTIMIZE: differentiable layout under chance constraints")
    opening = session.entity_by_name("Opening-1")
    door = session.entity_by_name("Door-1")
    storey = session.entity_by_name("Level 1")
    office_b = session.entity_by_name("Office-B")
    wall_party = session.entity_by_name("Wall-Party")

    # Field-query gradient witness: transmittance of a ray through the
    # party wall, differentiated w.r.t. wall thickness by dual numbers.
    prims = scene.cloud.of_element(scene.element_by_name("Wall-Party").row)
    origin = np.array([4.5, 2.0, 1.5])
    direction = np.array([1.0, 0.0, 0.0])
    T = ray_transmittance(
        prims.means, prims.covs, prims.weights, origin, direction, 2.0, kappa=8.0
    )
    from gat.geometry.dual import Dual

    # Differentiate w.r.t. the transverse spread Sigma_zz — the informative
    # direction (spread along a fully-crossing ray cancels analytically).
    k0 = int(np.argmin(np.linalg.norm(prims.means - np.array([5.1, 2.0, 1.5]), axis=1)))
    eps_cov = np.zeros((3, 3)); eps_cov[2, 2] = 1.0
    dual_out = scalar_ray_depth_dual(
        prims.means[k0], Dual(prims.covs[k0], eps_cov), prims.weights[k0],
        origin, direction, 2.0, kappa=8.0,
    )
    h = 1e-7
    cov_p = prims.covs[k0].copy(); cov_p[2, 2] += h
    cov_m = prims.covs[k0].copy(); cov_m[2, 2] -= h
    from gat.geometry.objectives import ray_optical_depth

    fd = (
        ray_optical_depth(prims.means[[k0]], cov_p[None], prims.weights[[k0]], origin, direction, 2.0, 8.0)
        - ray_optical_depth(prims.means[[k0]], cov_m[None], prims.weights[[k0]], origin, direction, 2.0, 8.0)
    ) / (2 * h)
    dual_grad = float(np.asarray(dual_out.eps).reshape(()))
    print(
        f"ray through party wall: transmittance {T:.4f}; gradient witness "
        f"d(depth)/d(Sigma_zz): dual {dual_grad:+.6f} vs finite-diff {fd:+.6f}"
    )
    assert abs(dual_grad - fd) < 1e-4 * max(1.0, abs(fd)) and abs(dual_grad) > 1e-3

    objective = LayoutObjective(
        cost_var=VarId(storey, "TotalWallCost"),
        daylight_area_var=VarId(opening, "Area"),
        daylight_floor_var=VarId(office_b, "FloorArea"),
        energy_terms=(
            (VarId(session.entity_by_name("Wall-South"), "GrossSideArea"), 0.25),
            (VarId(session.entity_by_name("Wall-North"), "GrossSideArea"), 0.25),
            (VarId(session.entity_by_name("Wall-West"), "GrossSideArea"), 0.25),
            (VarId(session.entity_by_name("Wall-East"), "GrossSideArea"), 0.25),
        ),
        daylight_target=0.10,
        constraints=(
            LessEqual(VarId(door, "Width"), VarId(opening, "Width")),
            LessEqual(VarId(door, "Height"), VarId(opening, "Height")),
            LessEqual(VarId(opening, "Height"), VarId(wall_party, "Height")),
            LessEqual(VarId(opening, "Width"), VarId(wall_party, "Length")),
        ),
    )
    params = (VarId(opening, "Width"), VarId(opening, "Height"))
    result: OptimizationResult = optimize_layout(world, params, objective)
    ratio_before = world.full.mean(VarId(opening, "Area")) / world.full.mean(
        VarId(office_b, "FloorArea")
    )
    print(
        f"optimize opening (W, H) for daylight ratio 0.100 (currently {ratio_before:.4f}) "
        f"under 2-sigma chance constraints:"
    )
    print(
        f"  ({result.initial[0]:.3f}, {result.initial[1]:.3f}) -> "
        f"({result.optimized[0]:.3f}, {result.optimized[1]:.3f}) "
        f"in {len(result.trajectory)} accepted steps; objective "
        f"{result.objective_initial:.3f} -> {result.objective_final:.3f}"
    )
    commit = session.run(as_interventions(result, design_sigma=0.003))
    new_world = session.world
    new_ratio = new_world.full.mean(VarId(opening, "Area")) / new_world.full.mean(
        VarId(office_b, "FloorArea")
    )
    door_w = new_world.full.mean(VarId(door, "Width"))
    door_h = new_world.full.mean(VarId(door, "Height"))
    open_w = new_world.full.mean(VarId(opening, "Width"))
    open_h = new_world.full.mean(VarId(opening, "Height"))
    # The 0.100 target is NOT reachable: the door dictates a minimum
    # aperture (0.9 x 2.0 plus 2-sigma margins ~ ratio 0.115), so the
    # optimizer settles at the constrained optimum — which is the point.
    floor_ratio = (door_w * door_h) / new_world.full.mean(VarId(office_b, "FloorArea"))
    print(
        f"  committed under verification ({'PASS' if commit.report.passed else 'FAIL'}); "
        f"daylight ratio {ratio_before:.4f} -> {new_ratio:.4f} "
        f"(target 0.100 is blocked by the door's minimum aperture, floor {floor_ratio:.4f});"
    )
    print(
        f"  clearances held with margin: door W {door_w:.3f} <= opening W {open_w:.3f}, "
        f"door H {door_h:.3f} <= opening H {open_h:.3f}"
    )
    assert commit.committed
    assert new_ratio < ratio_before - 0.005, "optimizer must move toward the target"
    assert new_ratio > floor_ratio - 1e-9, "cannot beat the constraint floor"
    assert open_w - door_w > 0.02 and open_h - door_h > 0.02, "2-sigma margins held"

    _hr("GEOMETRY VERDICT")
    print("One canonical state; a continuous Gaussian geometric view; clash,")
    print("attention, registration, fusion, compliance and optimization all")
    print("computed on it — and every write-back went through verification.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
