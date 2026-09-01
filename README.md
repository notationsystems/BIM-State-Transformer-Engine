# BIM State Transformer — GAT Computational Core 

**BIM State Transformer** is a portable, decision-focused computational state
runtime for Building Information Modeling (BIM). Its **GAT (Gaussian
Architectural Transformer)** core turns IFC design intent into an auditable
architectural belief that can absorb physical evidence, propagate
consequences, and verify the resulting state.

The project investigates whether BIM can be treated not merely as a digital description of a building, but as a **computational state space** in which architectural geometry, relationships, constraints, uncertainty, and derived properties can be transformed and propagated systematically.

> **Design intent + evidence + criteria → belief → decision or next evidence → verified state**

BIM State Transformer is an experimental research engine. It is not intended
to replace BIM authoring software, IFC, OpenUSD, CAD, or architectural
simulation systems. Its purpose is to investigate a computational layer that
can operate **between BIM representations and downstream analysis, inference,
simulation, and optimization.** GAT remains the Gaussian belief/conditioning
engine; OpenUSD remains an optional portable state carrier.

## Current direction — decision-focused active BIM

The project's north star is now explicit:

> **Given design intent, uncertain physical evidence, and an engineering
> criterion, maintain an auditable belief about the asset and determine
> whether the criterion is resolved—or which permissible evidence should be
> acquired next—before propagating and verifying any change.**

Gaussians, splats, and active inference are mechanisms in service of that
goal; none is the product by itself.  The flagship workflow is as-built MEP
assurance: determine whether a proposed route clears the uncertain existing
condition and, if the answer is unresolved, identify the next worthwhile
scan or field measurement.  Hard invariants remain a separate verification
shield, and no probabilistic assessment authorizes a physical intervention.

The executable loop is:

```text
IFC design intent + physical evidence -> posterior architectural belief
    -> SATISFIED / VIOLATED / UNRESOLVED
    -> stop, or select worthwhile evidence
    -> condition belief -> propagate consequences -> verify -> export
```

### First deployment slice — construction acceptance

GAT now exposes a case-level workflow contract for the three initial
high-value decisions: as-built clearance acceptance, prefabrication/opening
fit, and design-change/RFI impact. `AcceptanceCase` aggregates multiple
probabilistic checks into exactly one fail-closed disposition:
`ACCEPT`, `REJECT`, or `REQUEST_EVIDENCE`. The default policy will not accept
an as-built case merely because its BIM prior looks safe; every satisfied
check must be covered by calibrated evidence bound to the same assessed
world.

`gat-headless` provides a closed, read-only JSON boundary for CI and host
applications. Evidence receipts supplied over that boundary must resolve to
a verified transition in the state's hash-chained ledger and, before they can
close a case, arrive in an OpenUSD carrier whose signature verifies against a
deployment-configured trusted key. A response is still a recommendation rather than
an approval or external action.

The first host adapter is in `integrations/blender/gat_assurance`: a
self-contained Blender 4.2+ sidebar that reads headless acceptance responses,
shows the next evidence request, and colors matching Blender/Bonsai objects
without modifying IFC state. See
[`docs/workflow-deployment-v1.md`](docs/workflow-deployment-v1.md) for the
protocol, trust boundary, extension packaging, and current limits.

---

## GAT v0 — the implemented engine

The first engine exists and answers the §17 milestone question executably. The
core runtime depends on **numpy only**; OpenUSD support is an optional extra.
Tests use stdlib `unittest`.

```bash
pip install numpy
pip install ".[openusd]"       # optional Pixar OpenUSD carrier
gat audit path/to/model.ifc --text  # non-mutating IFC compatibility audit
python -m gat.demo            # the state-propagation milestone (README §17)
python -m gat.demo.geometry   # the geometric Gaussian layer
python -m gat.demo.active_inference  # choose, then assimilate, the next observation
python -m gat.demo.portability       # resume the same belief in a new process
python -m gat.demo.openusd_portability  # do the same through a composed USD stage
python -m gat.demo.ledger_replay execution-ledger.json  # replay accepted + rejected history
python -m gat.demo.temporal_process  # explicit process prediction -> evidence update -> replay
python -m gat.demo.workflow          # opening acceptance + non-mutating RFI preview
python -m gat.demo.beam_assurance out/beam  # complete evidence-to-verification chain
gat-headless request.json -o response.json  # read-only workflow boundary
python -m unittest discover   # the test suite
```

### Real-IFC validation

`gat audit` inventories an unfamiliar IFC without weakening the authoritative
loader or silently skipping unsupported entities. It continues past the first
preflight problem, reports the active project unit context, missing quantities,
available geometry fallbacks, placement limitations, and the real
lower/compile/verify outcome, and binds the report to the input bytes. An audit
never authorizes a decision.

The loader now normalizes recognized SI-prefixed IFC length quantities,
placement translations, and authored sigmas into GAT's canonical metre state.
IFC export reverses that boundary so posterior means and sigmas retain the
source project's unit convention. Conversion-based length units remain
fail-closed pending complete conversion-chain support.

The commit-pinned public corpus now runs in CI. It includes the 19 MB
buildingSMART Medical–Dental Clinic structural model: 317,671 IFC instances,
4 storeys, and 738 beams. GAT parses and inventories the whole file, identifies
all beams as structural candidates, and fails closed at the current
multi-storey and source-evidence boundaries. Measured results also include the
PCERT architecture scene and the 65 MB Schependomlaan model. See
[`docs/real-ifc-validation-v1.md`](docs/real-ifc-validation-v1.md) for the
reproducible corpus, exact failure taxonomy, and adapter-hardening order.

```python
from gat import GatSession, SetParameter, ObserveQuantity

session = GatSession.load_ifc("gat/demo/model.ifc")     # IFC -> IR -> Gaussian belief, verified
ch = session.var("Level 1", "ClearHeight")

# Observation: a laser-scan measurement conditions the joint belief;
# the correction propagates into every correlated quantity.
session.run(ObserveQuantity.single(session.var("Office-A", "Volume"), 59.4, 0.05))

# Intervention: one design change cascades through walls, spaces, costs —
# verified, or rolled back with digest proof.
session.run(SetParameter(ch, 3.4, design_sigma=0.01))

session.export_ifc("out/model_transformed.ifc")          # posterior sigmas round-trip through IFC
```

### The layered state-space substrate

GAT v0 treats one object — **the evolving architectural state** — and implements each mathematical layer as a separate, inspectable module:

| Layer | Question it answers | Concept | Module | v0 status |
|---|---|---|---|---|
| Boundary | What can enter/leave the system? | Explicit evidence/action adapters | `gat/adapters/`, `gat/geometry/scan_io.py`, `gat/geometry/scan_likelihood.py` | implemented (IFC, JSON, scan artifacts, calibrated clearance likelihood) |
| Epistemic identity | What kind of claim entered state, and from which source? | Typed calibrated evidence | `gat/evidence.py` | implemented (`MEASURED`, `ESTIMATED`, `INFERRED`, `ASSUMED`, `SIMULATED`, `DERIVED`) |
| Portability | Can computation resume across a runtime boundary? | Restartable state snapshot + operational equivalence | `gat/state_snapshot.py`, `gat/adapters/openusd.py` | implemented (JSON and OpenUSD carriers) |
| History | Can the evolution be independently reproduced? | Closed, hash-chained accepted/rejected event ledger | `gat/ledger.py` | implemented (schema v1 + exact replay) |
| Computational integrity | Can an external proof system attest one exact accepted transition? | Proof-carrying state-transition manifest | `gat/proof_manifest.py` | implemented (backend-neutral binding; cryptographic verifier optional) |
| Causality | What was assessed, selected, approved, or done without changing belief? | Typed state-bound causal events | `gat/causal.py` | implemented (closed records + lifecycles) |
| Inference | What state explains the evidence? | Bayesian conditioning | `gat/gaussian/condition.py` | implemented (linearized observations) |
| State | What do we currently believe? | Gaussian state μ, Σ | `gat/gaussian/` | implemented |
| Decision | Is the stated criterion resolved? | Posterior decision confidence | `gat/engine/decision.py` | implemented (minimum scalar criterion) |
| Policy | What evidence is worth acquiring next? | Decision-relevant information − burden | `gat/engine/active_inference.py` | implemented (one-step scalar observations) |
| Configuration | Which states are the *same architecture*? | Moduli / configuration quotient | `gat/engine/configuration.py` | implemented (relabeling × rigid motion × re-encoding quotient) |
| Transformation | How do we change configuration? | Operators / maps | `gat/engine/transform.py` | implemented |
| Differential | How does a small perturbation propagate? | Jacobian | `gat/ir/exprs.py`, `gat/engine/sensitivity.py` | implemented (analytic, FD-witnessed) |
| Probabilistic | How does uncertainty propagate? | Σ′ = JΣJᵀ, conditioning | `gat/engine/propagate.py`, `gat/gaussian/condition.py` | implemented (Joseph form, raw-space solves) |
| Dynamical | How does belief evolve through time? | Calibrated linear-Gaussian process + stability | `gat/engine/dynamics.py`, `gat/engine/stability.py` | implemented (exact process transition + rollout) |
| Stability | Do perturbations grow or contract? | Lyapunov analysis | `gat/engine/stability.py` | implemented (product spectrum + uncertainty energy) |
| Verification | Is the resulting state valid? | Constraints / invariants | `gat/engine/verify.py` | implemented (mandatory, with rollback) |

Four pillars, four different questions — none of them redundant:

| Concept | Question | Role |
|---|---|---|
| Moduli space | What configurations exist, and which are equivalent? | Global configuration geometry |
| Gaussian state | Where is the state, and how uncertain? | Local probabilistic state |
| Jacobians | How does a transformation locally change the state? | Differential propagation |
| Lyapunov dynamics | What happens to perturbations over repeated transformations? | Stability of evolution |

### The state-propagation core

`python -m gat.demo` runs the §17 milestone on a shipped two-office model (24 raw + 39 derived variables) in five acts: **compile** (IFC → IR → belief), **observe** (a volume measurement of one room shifts the *other* room through the shared storey-height parameter — covariance as architectural coupling), **transform** (one `SetParameter` on the storey clear height cascades through 34 derived variables), **reject** (an opening taller than its wall fails verification and rolls back, proven by digest equality), **emit** (IFC + JSON export, reload, zero-drift round-trip including posterior sigmas), and a **determinism finale**: the whole pipeline reruns to a bitwise-identical state digest.

Key semantics, fixed by design review:

* The **raw belief is the only canonical Gaussian**; every derived quantity is an exact-mean, first-order-covariance pushforward through analytic Jacobians (`Σ_full = [I;G] Σ_raw [I;G]ᵀ`). Derived state is recomputed, never mutated — raw/derived inconsistency is impossible by construction and machine-checked anyway (invariant GAUSS-03).
* **Interventions ≠ observations.** `SetParameter` is a do-intervention (severs correlations into the overridden variable); `ObserveQuantity` is Gaussian conditioning (sharpens belief *through* correlations). Both are first-class, inspectable, composable operators.
* All solves run in **full-rank raw space** (batch Joseph form, Cholesky solves, no explicit inverses, no eigendecompositions in the execution path); the rank-deficient full joint is a read-only view.
* **Verification is part of execution**: every transformation ends in the full invariant registry; strict failures roll back.
* Every session owns an **authoritative execution ledger**. Accepted and rejected
  transformations, exact prior/result state digests, caller evidence provenance,
  complete invariant results, and rejection reasons are hash-chained and can be
  independently replayed from the genesis checkpoint.
* A **restartable state snapshot** preserves the closed IR and exact indexed
  joint raw belief `(μ, Σ)`, then recompiles expressions, derived state,
  Jacobians, and invariants in the receiving runtime. It is deliberately
  stronger than the compact JSON projection used by dashboards and renderers.

### The geometric Gaussian layer

`python -m gat.demo.geometry` demonstrates the representational shift — elements as continuous, differentiable 3D Gaussian primitive sets, **derived from** (never replacing) the canonical state:

* **Gaussianization** — oriented boxes split into grids of primitives that exactly moment-match the uniform measure per sub-box (a provable invariant, tested at 1e-12), with analytic Jacobians back to state parameters.
* **Probabilistic clash detection** — separating-axis clearance with a delta-method sigma computed under the *joint* belief via the relative Jacobian `(J_a − J_b) Σ (J_a − J_b)ᵀ`, so shared parameters cancel (two walls driven by one storey height do not jitter relative to each other). Reported: `P(clash) = Φ(−c/σ)` (a probability of a real event), soft overlap mass, and a χ²₃ *separation significance* — deliberately not labeled a probability.
* **Structural attention** — analytic scaled-dot-product weights over primitive tokens (content × Gaussian overlap kernel × relationship-graph affinity), a diffusion update obeying the maximum principle, semantic identity channels frozen. **No learned weights exist**, and the module says so: this is deterministic kernel message passing wearing the attention API. A content-blind Laplacian ablation ships alongside so the content-dependence is demonstrated, not asserted.
* **Scan-to-BIM registration** — robust GMM alignment (uniform outlier component) with monotone EM (closed-form GLS translation + Armijo-guarded Gauss-Newton yaw), 8 deterministic starts, coarse-to-fine annealing; recovers a withheld pose to arcminutes/millimetres and reports the Gauss-Newton pose information matrix.
* **External scan-artifact boundary** — `load_ply_points` and `ScanRegistrar.register_ply` accept standard ASCII or binary-little-endian PLY vertices, including a triangle mesh's vertices.  This allows an external reconstruction system to supply evidence without becoming a GAT runtime dependency.  The first integration target is `Geometry-Grounded-Gaussian-Splatting`'s post-processed `recon_post.ply` mesh artifact; its vertices enter GAT's existing registration fit gate unchanged.
* **Evidence quarantine** — `ScanRegistrar.evidence` is available only after an accepted fit and binds its report to the exact scan SHA-256 and canonical scene version.  It conserves Gaussian-mixture responsibility mass while aggregating effective point support, normalized fit, primitive-support diversity, and cross-element assignment confidence.  These are auditable evidence diagnostics, not physical surface coverage and not an automatic BIM dimension update.
* **As-built clearance assurance** — `assess_clearance` evaluates a proposed MEP box against every solid element under the joint BIM belief.  Because obstruction events can share uncertain parameters, it reports dependence-safe bounds `max(Pᵢ) ≤ P(any violation) ≤ min(1, ΣPᵢ)` rather than assuming independence.  An unresolved assessment can be paired with accepted scan evidence through `plan_clearance_evidence`, which recommends either extracting a calibrated element measurement or performing a targeted rescan; both paths leave canonical state unchanged.
* **Calibrated scan-to-clearance likelihood** — `adapt_clearance_likelihood` measures the responsibility-weighted support face controlling the selected clearance, never an element centroid.  It requires survey-control or independently calibrated SLAM pose covariance, while scan-to-BIM registration remains only a fit/association gate.  Sampling, independent-pose, and systematic calibration variance are combined explicitly; provenance, element/face support, assignment, face alignment, spatial coverage, pose agreement, and innovation gates must all pass before a belief-bound `ObserveLinearized` can enter the usual condition → propagate → verify pipeline.
* **Multi-scale fusion** — exact moment-matched level-of-detail merging (element → building) with KL merge error, and exact affine transport into geo-referenced frames.
* **Compliance under uncertainty** — every rule is a margin with mean and sigma under the joint belief; `P(satisfied) = Φ(μ/σ)` with PASS/MARGINAL/FAIL.
* **Differentiable layout** — cost/daylight/energy objectives with exact DAG gradients, chance-constraint penalties sharing the compliance margins, closed-form Gaussian ray transmittance with forward-mode dual-number gradient witnesses; results commit through ordinary verified interventions.
* **Splat interoperability** — the scene exports to the standard 3D Gaussian Splatting PLY layout (positions, quaternions, log-scales, SH DC color by semantic class), loadable in stock 3DGS viewers.

### Zero-dependency USDA interchange proof

`python -m gat.demo.usd` runs the original interchange "killer test" through a
hand-written USDA subset: not merely *USD export*, but

> **Computational World → OpenUSD → Computational World**

Runtime A applies a transformation, serializes its **entire computational state** to a USD stage — real `Xform`/`Cube` prims any USD tool can open, with identity, semantics, topology, the joint Gaussian belief (floats via shortest round-tripping repr → bitwise reconstruction), typed constraints, the *defining expressions* of every derived quantity, and the execution trace carried in a `gat` custom-data namespace — and **dies** (a real subprocess exit). Runtime B reconstructs the world from the stage, passes a formal invariant suite (`I_identity ∧ I_geometry ∧ I_topology ∧ I_semantics ∧ I_gaussian ∧ I_constraints ∧ I_provenance ∧ I_configuration`), recompiles the dependency DAG and Jacobians *from the transferred definitions*, and continues the computation. Success criterion, demonstrated and tested:

```
S2_transferred ≃ S2_continuous     (posterior mean and covariance BITWISE equal)
```

All eight transfer levels — geometry, BIM semantics, topology, Gaussian state, computational state, transformation semantics, provenance, and operational state-space equivalence — pass in the shipped demo and test suite (`tests/test_usd_interchange.py`).

This adapter is the frozen, NumPy-only fallback and executable proof. It is not
the canonical production carrier and will not receive new carrier features.
`GatStateSnapshot` plus the native OpenUSD carrier described below are the two
canonical restart paths; native OpenUSD is the scene-graph and signed-ledger
bridge for Blender, Omniverse, and other USD hosts.

### The `gat` command line

The engine without writing Python — for BIM coordinators, GIS pipelines, and artists:

```bash
gat audit   model.ifc --text                   # fail-closed IFC compatibility inventory
gat verify  model.ifc                          # invariants + compliance under uncertainty
gat check   model.ifc --proposed duct.json     # probabilistic clash report; exit 1 on a likely clash
gat inspect model.ifc --var "Level 1.TotalWallCost"   # mean ± sigma, sensitivities, variance attribution
gat splats  model.ifc out/ --variations 25     # 3DGS splat PLYs: nominal + 25 sampled as-builts
gat sample  model.ifc --n 500                  # invariant checking over belief realizations
```

Every command is deterministic and read-only; `--json` switches to machine output. `gat audit` (also available as `gat-ifc-audit`) inventories a real-world IFC file's compatibility **before** any ingestion is attempted — unsupported entities are reported explicitly, never silently skipped. `gat splats --variations` is the **belief-driven variation generator**: each PLY is a sampled realization of `N(mu, Sigma)` — a physically consistent plausible as-built whose imperfections are correlated exactly as the model says (walls sharing a storey height move together), with a manifest recording each sample's dimensions and verification status. Grounded procedural variation for art and previz pipelines, instead of noise functions.

Sampling also closes a scientific loop: `gat.engine.sampling.empirical_pair_clearance` Monte-Carlo-estimates clash probabilities from realizations and the test suite asserts they agree with the analytic delta-method scores within Monte-Carlo error — the uncertainty machinery is *measured* to be calibrated, not assumed.

### Analysis APIs

```python
from gat.engine.sensitivity import sensitivities_of, variance_attribution
from gat.engine.stability import analyze
from gat.engine.configuration import configuration_digest

sensitivities_of(world, cost_var)       # exact d(cost)/d(every raw parameter)
variance_attribution(world, volume_var) # whose uncertainty is this? (sums to 1)
analyze(world, [obs, change])           # contracting / marginal / amplifying + energy trace
configuration_digest(world)             # identity modulo relabeling, rigid motion, re-encoding
```

### Explicit temporal process dynamics

`EvolveLinearGaussian` provides the first real state-space evolution primitive:
selected raw variables follow `x′ = A x + b + w`, with calibrated
`w ~ N(0,Q)` over a declared interval. The same embedded transition matrix
transports every cross-covariance; process noise enters only the selected raw
block; derived state is rebuilt and verified normally. `forecast_process`
rolls forward without mutating a session, while `session.run(process)` is an
explicit committed and replayable transition.

```python
from hashlib import sha256
import numpy as np
from gat import EvolveLinearGaussian, forecast_process

process = EvolveLinearGaussian(
    (clear_height,), np.array([[1.0]]), np.array([-0.0005]),
    np.array([[0.0002**2]]), elapsed_seconds=86400.0,
    model_id="daily-settlement-monitor-v1",
    calibration_digest=sha256(calibration_bytes).hexdigest(),
)
seven_day = forecast_process(session.world, process, steps=7)  # immutable forecast
session.run(process, provenance={"clock": "controller-A", "interval": "day-1"})
```

A subsequent measurement remains a distinct likelihood update, giving an
inspectable `predict → observe → verify` loop rather than silently treating
time as another observation. See
[`docs/temporal-dynamics-v1.md`](docs/temporal-dynamics-v1.md) for covariance
semantics, validation, and current limits.

### Decision-focused evidence planning (FEP-style)

GAT closes the loop around an explicit question.  A `MinimumDecision` is
SATISFIED or VIOLATED only when the corresponding posterior conclusion meets
the required confidence; otherwise it is UNRESOLVED.  Resolved decisions stop
without collecting more data.  For an unresolved decision, scalar observation
candidates are ranked by a one-step linear-Gaussian expected-free-energy proxy:
pragmatic risk plus calibrated action cost, minus target-relevant mutual
information.  Cost is declared in nats, so time, money, access, and safety
burden must first be mapped onto that common prior-surprise scale.

```python
from gat import MinimumDecision, ObservationCandidate, plan_decision_evidence

volume = session.var("Office-A", "Volume")
decision = MinimumDecision(volume, minimum=60.0, confidence=0.95)
plan = plan_decision_evidence(
    session.world,
    decision,
    [
        ObservationCandidate(
            session.var("Level 1", "ClearHeight"), 0.01, cost_nats=0.05
        ),
        ObservationCandidate(volume, 0.05, cost_nats=0.10),
    ],
)
if plan.selected is not None:
    result = session.run(plan.selected.candidate.observe(60.2))
```

This is intentionally a restricted active-inference policy, not a claim to
implement a general Free Energy Principle agent.  Passive observations do not
alter the building, so their expected pragmatic risk is shared before the
reading is known; they differ through epistemic value and declared burden.
Forecast variances are first-order linearizations at the current belief.  A
real reading alone enters the ordinary `ObserveQuantity → propagate → verify`
path; selecting an action never mutates state.

Registered scan artifacts already follow that separation.  An accepted pose
can produce a per-element evidence report, while a rejected fit, a different
scan, or a changed canonical scene is refused:

```python
from gat.geometry import ScanRegistrar, derive_scene, load_ply_points

points = load_ply_points("recon_post.ply")
registrar = ScanRegistrar(derive_scene(session.world))
registration = registrar.register(points)
evidence = registrar.evidence(points, registration)
print(evidence.render())
```

This report exposes where the reconstruction supports the model and where
assignment is ambiguous.  It deliberately stops before conditioning BIM
dimensions: partially visible surfaces do not provide unbiased element-center
measurements.

The first flagship as-built decision composes that quarantined evidence with
probabilistic MEP clearance:

```python
import numpy as np

from gat.geometry import (
    ClearanceDecision,
    ClearanceLikelihoodCalibration,
    IndependentPoseCalibration,
    OrientedBox,
    adapt_clearance_likelihood,
    assess_clearance,
    derive_scene,
    plan_clearance_evidence,
)

route = OrientedBox((4.0, 1.8, 3.06), 0.0, (3.0, 0.4, 0.4))
clearance = assess_clearance(
    scene,
    ClearanceDecision(
        route, required_clearance=0.05, confidence=0.95, position_sigma=0.002
    ),
)
inspection = plan_clearance_evidence(clearance, evidence)
if inspection.selected is not None:
    survey_pose = IndependentPoseCalibration(
        transform=model_from_scan_survey_pose,
        covariance=np.diag([yaw_var, x_var, y_var, z_var]),
        scan_digest=registration.scan_digest,
        source_id="survey-control-network-42",
    )
    likelihood = adapt_clearance_likelihood(
        scene, registrar, points, registration, evidence, inspection,
        survey_pose, ClearanceLikelihoodCalibration(),
    )
    session.run(likelihood.observation)  # condition -> propagate -> verify
    resolved = assess_clearance(derive_scene(session.world), clearance.decision)
```

The adapter is deliberately stricter than registration.  A fitted BIM pose
cannot be recycled as independent dimensional evidence; callers must declare
an external pose source and its full yaw/translation covariance.  The emitted
likelihood is bound to the scan digest, scene version, evidence configuration,
pose source, exact prior belief, element, and support direction, so it cannot
be reused after any intervening state change.  The triage priority remains an
inspection heuristic, not a claimed expected information gain.

### Authoritative execution history

`GatSession.run` records every accepted transition and rejected attempt in a
`gat-execution-ledger` v1 chain. The operation vocabulary is closed and
JSON-safe; it includes calibrated `ObserveLinearized` evidence with its exact
prior and evidence bindings. Each event contains its prior/result world
digests, full verification record or rejection reason, optional caller
provenance, predecessor hash, and event hash. Replay must regenerate the same
commit or the same rejection and reconstruct the exact dense joint belief.

```python
from gat import read_ledger, replay_ledger

checkpoint = session.world
session.run(
    likelihood.observation,
    provenance={
        "scan_digest": likelihood.scan_digest,
        "pose_source": likelihood.pose_source_id,
        "calibration": "clearance-likelihood-v1",
    },
)
head = session.export_ledger("execution-ledger.json")
replayed = replay_ledger(checkpoint, read_ledger("execution-ledger.json"))
assert replayed.world.digest() == session.world.digest()
```

Hash chaining is tamper-evident relative to a trusted checkpoint or head; it
does not by itself authenticate the publisher. See
[`docs/execution-ledger-v1.md`](docs/execution-ledger-v1.md) for the schema,
closed algebra, replay rules, and trust boundary.

Assessments, policies, approvals, and field actions are recorded separately
from transformations. Their ledger events are bound to the exact current world
but must preserve its digest. Approval and external-action ids follow explicit
lifecycles, while stale assessments and impossible transitions fail closed:

```python
from gat import decision_assessment_record, decision_policy_record

assessment = assess_decision(session.world, decision)
plan = plan_decision_evidence(session.world, decision, candidates)
session.record_assessment(decision_assessment_record(assessment))
session.record_policy(decision_policy_record(plan))
assert session.world.digest() == assessment.world_digest  # no hidden mutation
```

Only a later acquired measurement conditions the belief. See
[`docs/causal-events-v1.md`](docs/causal-events-v1.md) for typed records,
lifecycle rules, and the authority/signature boundary.

### Reference experiment — one complete beam chain

`python -m gat.demo.beam_assurance out/beam` executes the project's first
complete identity-preserving engineering chain:

```text
IfcBeam -> canonical raw/derived state -> typed material certificate
  -> Gaussian conditioning -> deterministic bending capacity
  -> SATISFIED/VIOLATED -> state-bound assessment -> replay/snapshot
  -> optional SP1 request (BACKEND_REQUIRED; no proof claimed)
```

The shipped beam starts at `fy = 350 +/- 8 MPa`, section modulus
`Z = 0.001 +/- 0.00001 m3`, and resistance factor `phi = 0.9`. Its design
capacity is `315000 +/- 7858.9 N*m`, satisfying a `301000 N*m` demand at
95% confidence. A `MEASURED` certificate reports `325 +/- 2 MPa`; Bayesian
conditioning produces the posterior `326.471 +/- 1.940 MPa`, not a false
exact assignment to 325 MPa. The two identified descendants—nominal and
design moment capacity—are recomputed, and the revised
`293823.5 +/- 3418.0 N*m` capacity is `VIOLATED` at the same confidence.

The verification record contains the beam/variable identities, evidence and
source digests, prior/result world identities, changed belief, covariance
change, affected variables, model/validation/dependency/computation digests,
probabilities, verdicts, and a human-readable causal reason. The emitted IFC,
state snapshot, ledger, summary, and proof request let another runtime replay,
verify, and continue the exact belief. See
[`docs/beam-assurance-reference-chain.md`](docs/beam-assurance-reference-chain.md)
for the contracts and explicit limitations.

### Proof-carrying computation claims

An accepted ledger transition can now be packaged as a
`gat-computation-proof-manifest` v1. The manifest binds the exact prior and
result worlds, closed operation, verification report, event and ledger head,
proof-program and verifying-key digests, an explicit numerical contract, and
the external proof bytes. Engineering-model and validation-profile digests
travel beside the claim so computational integrity cannot silently masquerade
as engineering validity. A manifest may additionally bind a computation-result
digest, but only when a later state-bound assessment in the same ledger records
that exact digest.

```python
from hashlib import sha256
from gat import (
    NumericContract,
    create_computation_proof_manifest,
    verify_computation_proof_manifest,
)

numeric = NumericContract(
    "clearance-micrometre-v1",
    sha256(numeric_profile_bytes).hexdigest(),
    "signed-fixed-point",
    "nearest-ties-to-even",
    "checked",
)
manifest = create_computation_proof_manifest(
    session.ledger,
    event_seq=len(session.ledger.events) - 1,
    numeric_contract=numeric,
    model_contract_digest=sha256(engineering_contract_bytes).hexdigest(),
    validation_profile_digest=sha256(validation_profile_bytes).hexdigest(),
    computation_result_digest=engineering_result.computation_digest,  # optional
    proof_system="sp1",
    proof_type="groth16",
    program_digest=sha256(guest_elf).hexdigest(),
    verifying_key_digest=sha256(verifying_key).hexdigest(),
    proof_artifact=proof_bytes,
)

# Binding alone is not proof verification. A host must supply the backend.
report = verify_computation_proof_manifest(
    manifest, session.ledger, proof_bytes, verifier=sp1_verifier
)
assert report.proof_verified
```

GAT does not include an SP1 runtime, fetch proof locators, infer privacy from a
proof-type label, or authorize a building action because a proof verifies.
See [`docs/proof-carrying-state-v1.md`](docs/proof-carrying-state-v1.md) for
the statement, numerical and trust contracts.

### Portable computational-state snapshots

`GatSession.export_snapshot` writes `GatStateSnapshot v1`: a versioned,
integrity-bound restart record containing entity and variable identities,
topology, constraints, closed expression trees, provenance, exact raw-variable
order, and the dense joint covariance. Derived values, dependency graphs,
Jacobians, geometry views, and verification reports are not serialized; they
are deterministically rebuilt and checked by the receiving runtime.

```python
from gat import GatSession, ShiftParameter, computational_equivalence

before = session.world
session.export_snapshot("checkpoint.gat.json")
resumed = GatSession.load_snapshot("checkpoint.gat.json")

assert computational_equivalence(before, resumed.world).passed
resumed.run(ShiftParameter(resumed.var("Level 1", "ClearHeight"), 0.10))
```

`python -m gat.demo.portability` proves the stronger continuation condition in
a separate Python process:

```text
T₂(decode(encode(T₁(S₀)))) == T₂(T₁(S₀))
```

The comparison covers IR semantics, raw and derived variable order, full mean
and covariance, invariant results, and the weaker architectural configuration
quotient. The first OpenUSD carrier now transports this contract and connects
it to a scene graph, while GAT's IR and belief remain canonical executable
state:

```python
session.export_openusd("checkpoint.usdc")
resumed = GatSession.load_openusd("checkpoint.usdc")
assert computational_equivalence(session.world, resumed.world).passed
```

Carrier v3 can authenticate the complete authoritative belief, trace, and
execution-ledger head with an optional Ed25519 key:

```python
from gat import generate_openusd_keypair

publisher = generate_openusd_keypair("survey-authority")
session.export_openusd("signed.usdc", signing_key=publisher)
trusted = GatSession.load_openusd(
    "signed.usdc",
    trusted_public_keys={publisher.key_id: publisher.public_key},
    require_signature=True,
)
```

The USD stage exposes entities, quantities, topology, raw-variable indexing,
mean, and complete covariance as inspectable prims, relationships, and native
arrays. `/GAT/State` is authoritative; `/GAT/View` is optional derived box
geometry. USD references and namespace renames preserve state because GAT
identity is carried by attributes and relationships rather than prim paths.
Bounded read policies constrain composed prims, dense state data, and ledger
events; `migrate_openusd` rewrites supported v1/v2 carriers into current v3 and refuses to
strip or re-bless an unverified signature. Derived overrides and variants may
compose freely, while authoritative opinions remain digest- and
signature-checked.
See [`docs/openusd-carrier-v0.md`](docs/openusd-carrier-v0.md) for the contract.

### Honesty notes

* Covariance propagation is first-order (means are exact); the validity regime (mm-scale sigmas on m-scale dimensions) is stated where it matters.
* Determinism is guaranteed as same-platform byte identity; cross-platform agreement is tolerance-level (BLAS reduction order).
* The v0 IFC adapter reads dimensional quantities and placements, not solid-model geometry — an explicit adapter-boundary decision (§12), swappable without touching the engine.
* The hand-written USDA interchange adapter is a zero-dependency fallback. JSON snapshots and the native, optionally signed OpenUSD carrier are the canonical restart formats.
* Gaussian overlap is a proxy for boolean geometry; clash scores are calibrated probabilities of the *modeled* clearance event, and the void-blindness of Gaussianized walls (openings are not subtracted) is a known v0 limit.
* The scan likelihood assumes the selected BIM support is locally a planar box face and freezes responsibility assignments for its first-order update. Edge/corner witnesses are rejected; richer mesh/plane latent variables and nonlinear posterior checks remain future work.
* Snapshot v1 and its OpenUSD carrier require a compatible `gat-world-v1` runtime and a closed,
  declarative expression algebra. It does not serialize arbitrary executable
  code, derived caches, or the source IFC syntax tree. The OpenUSD adapter uses
  custom `gat:` properties; a generated typed schema plugin, external key
  management/revocation, and a cross-implementation conformance suite are not
  implemented yet.
* Ledger v1 has bounded parsing and detects edits, deletion, and reordering,
  but an unsigned chain does not establish publisher identity. JSON snapshot
  resume currently begins a new ledger genesis; OpenUSD carrier v3 preserves
  and optionally signs exact ledger continuation.
* Causal approval and external-action records enforce ordering and bind claims
  to state, but names such as `authority` are not independent identity proofs.
  A carrier signature authenticates its publisher, not every named participant.

---

## 1. Premise

Contemporary BIM provides a rich representation of architectural assets:

* geometry
* topology
* materials
* spaces
* building elements
* assemblies
* relationships
* properties
* classifications
* constraints

However, BIM models are generally treated primarily as **descriptive representations**.

GAT explores a different interpretation:

> **A BIM model can be compiled into a structured architectural state representation and subjected to formally defined state transformations.**

This creates a distinction between:

```text
BIM Representation
        │
        ▼
Architectural State
        │
        ▼
Computational Representation
        │
        ▼
Transformation
        │
        ▼
Derived Architectural State
```

The Gaussian representation provides one mathematical mechanism for representing continuous state, uncertainty, and statistical coupling within that computational system.

---

# 2. What GAT Is

GAT is organized around three fundamental concepts:

### Architectural State

A structured representation of a BIM asset containing:

* geometry
* topology
* semantic identity
* relationships
* physical properties
* constraints
* observations
* derived quantities

### Gaussian State

A representation of continuous variables and their uncertainty using Gaussian state structures such as:

$$
\mathcal{N}(\mu,\Sigma)
$$

where:

* \(\mu\) represents the estimated state
* \(\Sigma\) represents covariance and coupling between variables

### Transformation

A formally defined operation that maps one architectural state into another:

$$
S_{t+1}=T(S_t,u_t)
$$

The objective is not simply to store Gaussian distributions, but to make transformations over architectural state **explicit, composable, inspectable, and verifiable**.

---

# 3. Why a Compiler?

GAT uses the term **compiler** deliberately.

A compiler transforms one formal representation into another through an intermediate representation and a defined transformation system.

GAT investigates the analogous pipeline:

```text
             BIM / IFC
                 │
                 ▼
              Parsing
                 │
                 ▼
      Architectural Intermediate
             Representation
                 │
                 ▼
        Gaussian State Layer
                 │
                 ▼
       GAT Transformation System
                 │
       ┌─────────┼─────────┐
       ▼         ▼         ▼
    Project   Propagate  Condition
       │         │         │
       └─────────┼─────────┘
                 ▼
          Derived State
                 │
                 ▼
            Verification
                 │
                 ▼
        BIM / Analysis Output
```

The compiler abstraction provides a place to define:

* representations
* operators
* transformation rules
* invariants
* intermediate representations
* validation
* deterministic execution
* serialization
* interoperability

---

# 4. Gaussian Does Not Mean Everything Is Gaussian

GAT does **not** assume that an entire BIM model is a Gaussian distribution.

BIM contains multiple classes of information:

$$
X =
X_{\mathrm{semantic}}
\cup
X_{\mathrm{topological}}
\cup
X_{\mathrm{geometric}}
\cup
X_{\mathrm{physical}}
\cup
X_{\mathrm{temporal}}
\cup
X_{\mathrm{uncertain}}
$$

Some quantities are naturally discrete:

```text
IfcWall
IfcDoor
IfcSpace
contains
supports
adjacent_to
```

Others are continuous:

```text
length
width
temperature
load
position
material property
energy demand
sensor measurement
```

GAT therefore treats Gaussian representations as a **computational layer for appropriate continuous and uncertain state**, rather than as a universal representation for BIM semantics.

The architectural state remains heterogeneous.

---

# 5. Core Architecture

```text
┌─────────────────────────────────────────────┐
│                  BIM MODEL                  │
│                                             │
│  Geometry · Semantics · Topology · Objects  │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│          ARCHITECTURAL IR / STATE           │
│                                             │
│  Identity · Relationships · Constraints     │
│  Geometry · Properties · Dependencies       │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│             GAUSSIAN STATE LAYER            │
│                                             │
│       μ · Σ · Precision · Factors           │
│       Continuous State · Uncertainty        │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│             GAT TRANSFORMER                 │
│                                             │
│ Transform · Propagate · Condition · Infer   │
│ Compose · Project · Optimize                │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│              VERIFIED STATE                 │
│                                             │
│ Geometry · Topology · Constraints · State   │
└─────────────────────────────────────────────┘
```

---

# 6. Architectural State

A fundamental GAT state may be represented conceptually as:

$$
S =
(X,\mu,\Sigma,G,C,R)
$$

where:

* \(X\) — state variables
* \(\mu\) — estimated continuous state
* \(\Sigma\) — covariance
* \(G\) — architectural relationship/topology graph
* \(C\) — constraints
* \(R\) — representation metadata

This is a conceptual model rather than a requirement that every implementation use exactly this structure.

The architecture is intentionally extensible.

---

# 7. Transformation

The fundamental operation of GAT is a transformation:

$$
T:S\rightarrow S'
$$

For a state change:

$$
S' = T(S,u)
$$

where \(u\) represents an operation, observation, design modification, constraint, or external input.

For locally differentiable transformations, uncertainty may be propagated through the Jacobian:

$$
\Sigma'
\approx
J\Sigma J^T
$$

This provides a mathematically explicit mechanism for investigating how uncertainty and coupling propagate through architectural state.

---

# 8. BIM Change Propagation

Consider a simple architectural modification:

```text
Wall height
    ↓
Room geometry
    ↓
Opening relationships
    ↓
Quantities
    ↓
Connected assemblies
    ↓
Derived analysis
```

Instead of treating each update independently, GAT investigates whether the dependency structure can be represented explicitly:

```text
Wall
 │
 ├── bounds → Room
 │
 ├── contains → Openings
 │
 ├── material → Assembly
 │
 └── geometry → Quantity
```

A transformation can therefore produce a new state while identifying affected downstream state.

The long-term objective is:

> **Change the state once; propagate the consequences through the computational representation.**

---

# 9. BIM as a Computational Language

GAT treats BIM as more than a file format.

An IFC or BIM model can be viewed as a structured language describing an architectural system.

Under this interpretation:

```text
BIM
 │
 ├── Syntax
 │
 ├── Semantics
 │
 ├── Relationships
 │
 ├── Constraints
 │
 └── State
```

GAT investigates the possibility of compiling this representation into a computational intermediate representation.

This creates a potential analogy:

| Conventional compiler | GAT                         |
| --------------------- | --------------------------- |
| Source language       | BIM / IFC                   |
| Parser                | BIM parser                  |
| AST / IR              | Architectural IR            |
| Type system           | BIM semantics / constraints |
| Runtime state         | Architectural state         |
| Transformation        | GAT operator                |
| Optimization          | State optimization          |
| Verification          | BIM/state invariants        |
| Target                | BIM / simulation / analysis |

The analogy is useful because it provides a mature conceptual framework for thinking about representation and transformation.

---

# 10. Relationship to Existing Technologies

GAT is deliberately constructed from established areas rather than claiming that its primitive mathematics is entirely new.

Its conceptual lineage includes:

* Building Information Modeling
* IFC
* computational geometry
* graph representations
* probabilistic graphical models
* Gaussian inference
* covariance-based uncertainty propagation
* state-space methods
* Bayesian inference
* numerical optimization
* compiler intermediate representations
* digital twins
* simulation
* machine learning

The research question is therefore not:

> "Has anyone ever used Gaussian mathematics?"

Nor:

> "Has anyone ever used transformers with BIM?"

Instead:

> **Can these concepts be composed into a coherent compiler architecture for transforming BIM architectural state?**

GAT is an investigation into that architectural boundary.

---

# 11. What GAT Is Not

GAT is not intended initially to be:

* a replacement for Revit
* a replacement for Archicad
* a CAD system
* a BIM authoring environment
* a rendering engine
* a generic LLM
* a generic Gaussian statistics package
* a replacement for FEM
* a replacement for IFC
* a digital twin platform

Instead, GAT is intended to investigate a **computational transformation layer** that can interface with these systems.

---

# 12. Interoperability

The initial external representation should be treated as an adapter boundary.

Conceptually:

```text
IFC
 │
 ▼
IFC Adapter
 │
 ▼
Canonical Architectural State
 │
 ▼
GAT
 │
 ├── Analysis
 ├── Simulation
 ├── Optimization
 └── Inference
 │
 ▼
Derived State
 │
 ▼
IFC / BIM / Other Systems
```

This prevents the internal computational model from becoming unnecessarily coupled to one BIM application or file format.

Potential future adapters include:

```text
IFC
Revit
Archicad
Rhino
Tekla
GIS
FEM
Point Clouds
Sensors
Digital Twins
```

These are potential integration targets, not initial implementation requirements.
OpenUSD is now an implemented optional restart carrier and scene-graph bridge.

---

# 13. Potential Applications

If the underlying architecture proves useful, GAT could eventually support research into:

### Design change propagation

Determine which architectural states are affected by a design modification.

### Uncertainty-aware BIM

Represent uncertainty in dimensions, materials, loads, measurements, and derived properties.

### BIM-to-simulation compilation

Compile BIM state into representations suitable for simulation.

### Structural reasoning

Represent coupled architectural and structural variables and propagate changes.

### Building performance

Propagate uncertain architectural parameters into energy or environmental models.

### Construction planning

Represent dependencies between architectural state and construction operations.

### Digital twins

Use observations to update an existing architectural state.

### Design optimization

Search architectural state spaces subject to constraints and objectives.

These applications are deliberately downstream of the core engine.

---

# 14. Design Principles

GAT follows several architectural principles.

### 1. State before interface

The canonical architectural state is more important than any particular UI.

### 2. Representation is not execution

A BIM representation describes state.

GAT executes transformations over that state.

### 3. Semantics are not Gaussianized

Discrete identity, topology, and semantic relationships remain explicit.

### 4. Gaussian state is explicit

Continuous uncertainty and covariance should be represented directly rather than hidden inside opaque models.

### 5. Transformations are first-class

A transformation should be inspectable as an operation on state.

### 6. Verification is part of execution

A transformed state should be checked against relevant invariants.

### 7. Determinism where possible

Equivalent inputs and operations should produce reproducible computational results.

### 8. Interoperability at the boundary

External BIM systems should connect through explicit adapters.

### 9. Extensions are supersets

New representations, operators, inference systems, and backends should extend the architecture rather than invalidate verified foundations.

---

# 15. Development Philosophy

GAT is developed experimentally:

```text
Build
  ↓
Run
  ↓
Observe
  ↓
Measure
  ↓
Fix
  ↓
Audit
  ↓
Extend
```

Architectural claims should be separated from implementation claims.

A feature is not considered validated merely because it can execute.

The project aims to distinguish:

```text
Representation
Operation
Execution
Observation
Verification
Evidence
```

This is particularly important for a system intended to operate on engineering and architectural state.

---

# 16. Initial Research Questions

The project begins with several concrete questions:

1. Can BIM geometry and semantics be represented as a canonical architectural state?
2. Can continuous BIM variables be mapped into Gaussian state representations?
3. Can covariance represent meaningful architectural dependencies?
4. Can architectural transformations be expressed as composable operators?
5. Can uncertainty be propagated through those transformations?
6. Can transformed states be verified against BIM invariants?
7. Can the resulting representation compile back into interoperable BIM representations?
8. Does this architecture provide capabilities that are difficult to obtain from conventional BIM workflows?

The purpose of the project is to **test these questions computationally**, not assume their answers.

---

# 17. Initial Milestone

The first implementation should remain deliberately small.

### GAT v0

```text
IFC
 ↓
Architectural State
 ↓
Gaussian State
 ↓
Simple Transformation
 ↓
Propagation
 ↓
Verification
```

The first working demonstration should preferably answer one concrete question:

> **When a BIM architectural parameter changes, can GAT deterministically transform the corresponding state and correctly propagate the consequences through dependent variables?**

If that works reliably, additional capabilities can be added incrementally.

---

# 18. Long-Term Architecture

The eventual system may evolve toward:

```text
                         GAT
                          │
          ┌───────────────┼────────────────┐
          │               │                │
       BIM IR        Gaussian IR       Structural IR
          │               │                │
          └───────────────┼────────────────┘
                          │
                   Transformation
                          │
             ┌────────────┼────────────┐
             ▼            ▼            ▼
          Inference   Simulation   Optimization
             │            │            │
             └────────────┼────────────┘
                          ▼
                    Verified State
                          │
             ┌────────────┼────────────┐
             ▼            ▼            ▼
            BIM          IFC      Digital Twin
```

The Gaussian representation is therefore one component of a larger computational architecture.

---

# 19. Status

**Research / Experimental — v0 implemented**

GAT is an exploratory open-source project. The v0 engine (see *GAT v0 — the implemented engine* above) implements the §17 milestone end to end: the state-propagation core, the geometric Gaussian layer, explicit temporal process dynamics, decision-focused active inference, calibrated scan evidence, a deterministic causal execution ledger, restartable computational state through JSON and OpenUSD, backend-neutral proof-carrying transition commitments, and self-asserting demonstrations backed by a full test suite.

The architecture, mathematical assumptions, representations, and implementation are expected to evolve as they are tested against real BIM data and engineering workflows.

## Ecosystem

GAT's adapter boundary (§12) is designed to meet the surrounding toolchain rather than replace it. Reference points and intended integration targets in and around the [notationsystems](https://github.com/notationsystems) organization:

* [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [Geometry-Grounded-Gaussian-Splatting](https://github.com/notationsystems/Geometry-Grounded-Gaussian-Splatting) — GAT's geometric layer follows the 3DGS covariance factorization (`Σ = R·S·SᵀRᵀ`) and exports viewer-compatible splat PLYs.  Geometry-Grounded's `recon_post.ply` is also a clean external evidence input to `ScanRegistrar.register_ply`; GAT consumes the standard artifact and does not bundle its reconstruction runtime.
* [arch-render-ai-toolkit](https://github.com/notationsystems/arch-render-ai-toolkit) — BIM-metadata-to-render-prompt middleware; GAT's canonical JSON state export (`GatSession.export_json`) is shaped for exactly this kind of consumer.
* [OpenMEP](https://github.com/notationsystems/OpenMEP) — MEP components for Revit/Dynamo; its ducts, pipes, and fittings are the intended proposal source for GAT's now-executable as-built clearance decision and scan-guided inspection loop.
* [bridge-pipeline](https://github.com/notationsystems/bridge-pipeline) — generative structural design with BIM compliance checking; a downstream candidate for GAT's verified-state and compliance reports.
* [AI-CAD-BIM-Parametric-Assets](https://github.com/notationsystems/AI-CAD-BIM-Parametric-Assets) — parametric IFC/STEP datasets; a source of test models beyond the shipped demo.
* [OpenUSD](https://openusd.org/) — GAT's first optional scene-graph carrier
  for the restartable state contract and derived geometry. It is an
  interoperability layer, not a replacement for the canonical runtime.
* [SP1](https://github.com/succinctlabs/sp1) — the first intended external
  verifier backend for proof-carrying GAT transitions. The core currently
  defines and verifies the portable binding manifest only; it does not bundle
  a zkVM, proving service, blockchain, or implicit confidentiality claim.

No claim is made that Gaussian representations are universally optimal for BIM.

The purpose of the project is to determine **where they are useful, where they fail, and what computational architecture is required around them.**

---

# 20. Core Proposition

The central proposition of GAT can be stated simply:

$$
\boxed{
\text{Design Intent} + \text{Physical Evidence} + \text{Criteria}
\rightarrow
\text{Auditable Belief}
\rightarrow
\text{Decision or Next Evidence}
\rightarrow
\text{Verified State}
\rightarrow
\text{Restartable Continuation}
}
$$

Or, in words:

> **GAT investigates how an IFC-grounded architectural belief can be kept
> synchronized with uncertain physical evidence, used to resolve explicit
> engineering decisions, and directed toward the next worthwhile observation
> when the evidence is insufficient.**

Stated as the layered research hypothesis the v0 engine now tests:

> **GAT investigates an architectural state-space substrate in which
> heterogeneous observations update provenance-bound probabilistic state,
> decision confidence supplies a stopping rule, evidence actions trade
> epistemic value against burden, and every accepted transformation propagates
> through mandatory validity checks.**

None of this is assumed correct. The v0 implementation exists to discover **which of these layers genuinely compose, which require modification, and which turn out to be unnecessary** — that empirical question is the project.

---

## License

MIT — see [LICENSE](LICENSE).

## Project Status

Early-stage research and development.
