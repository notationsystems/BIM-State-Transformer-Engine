"""The GAT v0 milestone demonstration (README §17).

    python -m gat.demo

Five acts on the shipped two-office model:

  0. COMPILE   — IFC -> architectural state -> Gaussian belief, verified.
  1. OBSERVE   — a laser-scan volume measurement conditions the belief;
                 the correction flows through shared parameters into the
                 *other* room (covariance as architectural coupling).
  2. TRANSFORM — one design intervention (raise the storey clear height)
                 cascades deterministically through walls, spaces, costs.
  3. REJECT    — an infeasible change (opening taller than its wall) is
                 caught by verification and rolled back, digest-proof.
  4. EMIT      — export to IFC (posterior sigmas round-trip through
                 property sets) and JSON; reload and re-verify.
  5. DETERMINISM — the whole pipeline reruns from scratch to an identical
                 state digest: same input + same operations = same state.

Every number printed is computed by the engine; the script asserts its own
goldens (hand-derivable from the model) as it narrates.
"""

from __future__ import annotations

import os
import sys

from gat.errors import VerificationError
from gat.engine.transform import ObserveQuantity, SetParameter
from gat.session import GatSession

MODEL = os.path.join(os.path.dirname(__file__), "model.ifc")
TOL = 1e-9


def _check(label: str, actual: float, expected: float, tol: float = TOL) -> None:
    if abs(actual - expected) > tol:
        raise AssertionError(f"golden failed: {label}: {actual!r} != {expected!r}")


def _hr(title: str) -> None:
    print()
    print(f"=== {title} " + "=" * max(0, 66 - len(title)))


def run_pipeline(out_dir: str, quiet: bool = False) -> str:
    """Execute the full demo pipeline; returns the final state digest."""
    say = (lambda *a, **k: None) if quiet else print

    # ---- ACT 0: COMPILE --------------------------------------------------
    _hr("ACT 0  COMPILE: BIM -> architectural state -> Gaussian belief") if not quiet else None
    session = GatSession.load_ifc(MODEL)
    world = session.world

    census: dict[str, int] = {}
    for eid in world.module.entities:
        census[eid.ifc_class] = census.get(eid.ifc_class, 0) + 1
    say(
        "entities:",
        ", ".join(f"{n} {cls}" for cls, n in sorted(census.items())),
    )
    n_raw = world.binding.n_raw
    n_derived = world.binding.n_full - n_raw
    say(f"variables: {n_raw} raw + {n_derived} derived = {world.binding.n_full}")
    say(f"relationships: {len(world.module.rels)} edges; "
        f"constraints: {len(world.module.constraints)}")

    clear_height = session.var("Level 1", "ClearHeight")
    vol_a = session.var("Office-A", "Volume")
    vol_b = session.var("Office-B", "Volume")
    len_a = session.var("Office-A", "Length")
    party_net = session.var("Wall-Party", "NetSideArea")
    total_cost = session.var("Level 1", "TotalWallCost")

    say(f"prior Office-A.Volume    = {world.full.mean(vol_a):.6f} +- {world.full.std(vol_a):.6f} m3")
    say(f"prior Office-B.Volume    = {world.full.mean(vol_b):.6f} +- {world.full.std(vol_b):.6f} m3")
    corr_ab = world.full.corr(vol_a, vol_b)
    say(f"prior corr(V_A, V_B)     = {corr_ab:+.6f}   (coupled through the shared storey height)")
    say(f"prior TotalWallCost      = {world.full.mean(total_cost):.2f} +- {world.full.std(total_cost):.2f}")
    report = session.verify()
    p, w, f = report.counts()
    say(f"verification: {p} pass, {w} warn, {f} fail")

    _check("prior V_A", world.full.mean(vol_a), 60.0)
    _check("prior V_B", world.full.mean(vol_b), 48.0)
    assert corr_ab > 0.75, "shared storey height must couple the room volumes"
    assert report.passed and not report.warnings

    # ---- ACT 1: OBSERVE --------------------------------------------------
    if not quiet:
        _hr("ACT 1  OBSERVE: as-built laser scan of Office-A volume")
    before_vb = world.full.mean(vol_b)
    before_ch = world.full.mean(clear_height)
    say(f"measurement: Office-A.Volume = 59.400000 +- 0.050000 m3  (design {world.full.mean(vol_a):.6f})")
    result = session.run(ObserveQuantity.single(vol_a, 59.4, 0.05))
    world = session.world
    obs = result.transformation
    assert isinstance(obs, ObserveQuantity) and obs.record is not None
    say(f"innovation: {obs.record.innovations[0]:+.6f} m3; jitter used: {obs.record.jitter:.1e}")
    say(f"posterior Office-A.Volume = {world.full.mean(vol_a):.6f} +- {world.full.std(vol_a):.6f}")
    say(f"posterior ClearHeight     = {world.full.mean(clear_height):.6f} +- {world.full.std(clear_height):.6f}"
        f"   (was {before_ch:.6f})")
    say(f"posterior Office-B.Volume = {world.full.mean(vol_b):.6f}   (was {before_vb:.6f})")
    say("-> measuring room A moved room B: the shared storey height carried the correction across.")
    assert world.full.mean(vol_b) < before_vb, "V_B must absorb part of the correction"
    assert world.full.mean(clear_height) < before_ch

    # ---- ACT 2: TRANSFORM ------------------------------------------------
    if not quiet:
        _hr("ACT 2  TRANSFORM: raise the storey clear height to 3.40 m")
    corr_len_ch = world.full.corr(clear_height, len_a)
    say(f"pre-intervention corr(ClearHeight, Office-A.Length) = {corr_len_ch:+.6f}  (induced by ACT 1)")
    t = SetParameter(clear_height, 3.4, design_sigma=0.01)
    say(f"apply: {t.describe()}   [do-intervention: overrides belief, severs correlations]")
    result = session.run(t)
    world = session.world
    say(f"affected derived variables: {len(result.affected)} of {n_derived}")
    say(f"  ClearHeight              -> {world.full.mean(clear_height):.6f}")
    say(f"  Wall-Party.NetSideArea   -> {world.full.mean(party_net):.6f} m2   (= 4.0*3.4 - 1.0*2.1)")
    say(f"  Office-A.Volume          -> {world.full.mean(vol_a):.6f} m3")
    say(f"  Office-B.Volume          -> {world.full.mean(vol_b):.6f} m3")
    say(f"  TotalWallCost            -> {world.full.mean(total_cost):.2f} +- {world.full.std(total_cost):.2f}")
    say(f"post-intervention corr(ClearHeight, Office-A.Length) = {world.full.corr(clear_height, len_a):+.6f}  (severed)")

    _check("party net area", world.full.mean(party_net), 4.0 * 3.4 - 1.0 * 2.1)
    _check("severed corr", world.full.corr(clear_height, len_a), 0.0)
    wall_height = session.var("Wall-Party", "Height")
    _check("wall height follows storey", world.full.mean(wall_height), 3.4)
    # Descendants of ClearHeight: 5 walls x 6 quantities, 2 space volumes,
    # 2 storey rollups = 34; untouched: door/opening areas, floor areas,
    # TotalFloorArea (5 of 39).
    assert len(result.affected) == 34, f"affected set was {len(result.affected)}"

    # ---- ACT 3: REJECT ---------------------------------------------------
    if not quiet:
        _hr("ACT 3  REJECT: an infeasible change is caught and rolled back")
    opening_h = session.var("Opening-1", "Height")
    digest_before = world.digest()
    bad = SetParameter(opening_h, 3.6, design_sigma=0.005)
    say(f"attempt: {bad.describe()}  (opening would exceed the 3.40 m wall)")
    try:
        session.run(bad)
        raise AssertionError("infeasible transformation was not rejected")
    except VerificationError as exc:
        for failure in exc.report.failures:
            say(f"  FAIL {failure.invariant_id} [{failure.subject}] residual={failure.residual:+.6f}")
    digest_after = session.world.digest()
    say(f"state digest before: {digest_before[:16]}...  after: {digest_after[:16]}...")
    say("-> identical: the rejected transformation left no trace on the state.")
    assert digest_before == digest_after

    # ---- ACT 4: EMIT -----------------------------------------------------
    if not quiet:
        _hr("ACT 4  EMIT: IFC + JSON export, reload, re-verify")
    os.makedirs(out_dir, exist_ok=True)
    out_ifc = os.path.join(out_dir, "model_transformed.ifc")
    out_json = os.path.join(out_dir, "state.json")
    patched, appended = session.export_ifc(out_ifc)
    session.export_json(out_json)
    say(f"wrote {out_ifc}: {patched} quantity records patched, {appended} instances appended")
    say(f"wrote {out_json}: canonical state snapshot")

    reloaded = GatSession.load_ifc(out_ifc)
    max_err = 0.0
    for var in world.binding.raw_index.vars:
        err = abs(reloaded.world.belief.mean(var) - world.belief.mean(var))
        max_err = max(max_err, err)
    sigma_err = abs(
        reloaded.world.belief.std(clear_height) - world.belief.std(clear_height)
    )
    say(f"round-trip: max |mean drift| = {max_err:.2e}; ClearHeight sigma drift = {sigma_err:.2e}")
    rep = reloaded.verify()
    say(f"reloaded state verification: {'PASS' if rep.passed else 'FAIL'}")
    assert max_err < 1e-9 and sigma_err < 1e-9 and rep.passed

    # ---- final digest ----------------------------------------------------
    return session.world.digest()


def main() -> int:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "out"
    digest_1 = run_pipeline(out_dir)

    _hr("ACT 5  DETERMINISM: rerun the entire pipeline from scratch")
    digest_2 = run_pipeline(out_dir, quiet=True)
    print(f"run 1 final digest: {digest_1}")
    print(f"run 2 final digest: {digest_2}")
    verdict = "IDENTICAL" if digest_1 == digest_2 else "MISMATCH"
    print(f"-> {verdict}: same input + same operations = byte-identical state.")
    assert digest_1 == digest_2

    _hr("VERDICT")
    print("A BIM parameter changed once; the consequences propagated through")
    print("every dependent variable, with uncertainty, under verification,")
    print("deterministically, and round-tripped back to IFC.  (README §17)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
