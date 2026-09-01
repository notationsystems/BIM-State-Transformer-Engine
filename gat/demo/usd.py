"""The state-space interchange demonstration (the "killer test").

    python -m gat.demo.usd

Runs the same two-transformation program twice:

  CONTINUOUS   runtime A applies T1 then T2, uninterrupted.
  TRANSFERRED  runtime A applies T1, serializes its ENTIRE computational
               world to an OpenUSD stage, and dies (a real subprocess
               exit).  Runtime B — a fresh process-independent session —
               reconstructs the world from the stage, passes the formal
               invariant suite, and applies T2.

Success is S2_transferred ~= S2_continuous with *bitwise* equality of the
posterior mean and covariance, identical configuration digests, and a
passing verification report — i.e. OpenUSD carried a computational world,
not a picture of one:

    Computational World -> OpenUSD -> Computational World

The eight transfer levels are reported explicitly, so what survived the
boundary is a measured claim.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

from gat.adapters.usd_io import state_equivalence
from gat.engine.transform import ObserveQuantity, SetParameter
from gat.session import GatSession

MODEL = os.path.join(os.path.dirname(__file__), "model.ifc")


def _t1(session: GatSession):
    vol_a = session.var("Office-A", "Volume")
    return session.run(ObserveQuantity.single(vol_a, 59.4, 0.05))


def _t2(session: GatSession):
    ch = session.var("Level 1", "ClearHeight")
    return session.run(SetParameter(ch, 3.4, design_sigma=0.01))


def runtime_a(usd_path: str) -> int:
    """Load, apply T1, serialize the world, and exit — this process dies."""
    session = GatSession.load_ifc(MODEL)
    _t1(session)
    session.export_usd(usd_path)
    print(f"[runtime A, pid {os.getpid()}] T1 applied; state -> {usd_path}; exiting.")
    return 0


def main() -> int:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "out"
    os.makedirs(out_dir, exist_ok=True)
    usd_path = os.path.join(out_dir, "state.usda")

    print("=== CONTINUOUS: T1 then T2 in one uninterrupted runtime " + "=" * 12)
    continuous = GatSession.load_ifc(MODEL)
    _t1(continuous)
    s1_continuous = continuous.world
    _t2(continuous)
    s2_continuous = continuous.world
    print(f"S2 (continuous) digest: {s2_continuous.digest()[:16]}...")

    print()
    print("=== TRANSFERRED: T1 in runtime A -> USD -> T2 in runtime B " + "=" * 9)
    proc = subprocess.run(
        [sys.executable, "-m", "gat.demo.usd", "--runtime-a", usd_path],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    )
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise AssertionError("runtime A failed")
    print(f"[runtime B, pid {os.getpid()}] reconstructing from {usd_path}")

    runtime_b = GatSession.load_usd(usd_path)
    print(f"provenance carried over: {len(runtime_b.imported_trace)} trace events")

    print("\ninvariant suite  D(E(S1)) ~= S1:")
    report = state_equivalence(s1_continuous, runtime_b.world)
    print(report.render())
    assert report.equivalent

    rep = runtime_b.verify()
    print(f"reconstructed state verification: "
          f"{'PASS' if rep.passed and not rep.warnings else 'ISSUES'}")
    assert rep.passed

    _t2(runtime_b)
    s2_transferred = runtime_b.world

    print("\ncontinuation  S2_transferred vs S2_continuous:")
    mu_ok = s2_transferred.full.mu.tobytes() == s2_continuous.full.mu.tobytes()
    sigma_ok = s2_transferred.full.sigma.tobytes() == s2_continuous.full.sigma.tobytes()
    final = state_equivalence(s2_continuous, s2_transferred)
    print(final.render())
    print(f"  posterior mean bitwise equal:       {mu_ok}")
    print(f"  posterior covariance bitwise equal: {sigma_ok}")
    assert mu_ok and sigma_ok and final.equivalent

    levels = [
        ("1 geometry", "placements and display boxes survive (I_geometry)"),
        ("2 semantics", "classes, names, attributes survive (I_identity, I_semantics)"),
        ("3 topology", "relationship graph survives (I_topology)"),
        ("4 gaussian", "mu, Sigma survive bitwise (I_gaussian)"),
        ("5 computation", "constraints + dependency expressions survive (I_constraints)"),
        ("6 transformation", "runtime B recompiled Jacobians and continued (T2 applied)"),
        ("7 provenance", "metadata and execution trace survive (I_provenance)"),
        ("8 equivalence", "S2 restart == S2 continuous, bitwise + configuration"),
    ]
    print("\ntransfer levels achieved:")
    for label, meaning in levels:
        print(f"  PASS  level {label:<16} {meaning}")

    print("\n=== VERDICT " + "=" * 58)
    print("The USD stage carried a computational world across a process")
    print("death: identity, uncertainty, dependencies, transformation")
    print("semantics and provenance survived, and computation CONTINUED to a")
    print("bitwise-identical state.  Computational World -> OpenUSD ->")
    print("Computational World.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--runtime-a":
        sys.exit(runtime_a(sys.argv[2]))
    sys.exit(main())
