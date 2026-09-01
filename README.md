# GAT — Gaussian Architectural Transformer for BIM

**GAT (Gaussian Architectural Transformer)** is a computational compiler for transforming, propagating, and reasoning over Building Information Modeling (BIM) state using Gaussian representations.

GAT investigates whether BIM can be treated not merely as a digital description of a building, but as a **computational state space** in which architectural geometry, relationships, constraints, uncertainty, and derived properties can be transformed and propagated systematically.

> **BIM → Architectural State → Gaussian Representation → Transformation → Verified State**

GAT is an experimental research engine. It is not intended to replace BIM authoring software, IFC, CAD, or architectural simulation systems. Its purpose is to investigate a computational layer that can operate **between BIM representations and downstream analysis, inference, simulation, and optimization.**

---

## GAT v0 — the implemented engine

The first engine exists and answers the §17 milestone question executably. Runtime dependency: **numpy only**. Tests: stdlib `unittest`.

```bash
pip install numpy
python -m gat.demo            # the state-propagation milestone (README §17)
python -m gat.demo.geometry   # the geometric Gaussian layer
python -m unittest discover   # the test suite
```

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
| Boundary | What can enter/leave the system? | Markov blanket / adapter boundary | `gat/adapters/` | implemented (IFC in/out, JSON out) |
| Inference | What state explains the information? | LLM / Bayesian hypothesis | first-class `Transformation` objects as the operation interface | interface only (deliberate) |
| State | What do we currently believe? | Gaussian state μ, Σ | `gat/gaussian/` | implemented |
| Configuration | Which states are the *same architecture*? | Moduli / configuration quotient | `gat/engine/configuration.py` | implemented (relabeling × rigid motion × re-encoding quotient) |
| Transformation | How do we change configuration? | Operators / maps | `gat/engine/transform.py` | implemented |
| Differential | How does a small perturbation propagate? | Jacobian | `gat/ir/exprs.py`, `gat/engine/sensitivity.py` | implemented (analytic, FD-witnessed) |
| Probabilistic | How does uncertainty propagate? | Σ′ = JΣJᵀ, conditioning | `gat/engine/propagate.py`, `gat/gaussian/condition.py` | implemented (Joseph form, raw-space solves) |
| Dynamical | What happens over repeated transformations? | State-space dynamics | `gat/engine/stability.py` | implemented (per-operator perturbation maps) |
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

### The geometric Gaussian layer

`python -m gat.demo.geometry` demonstrates the representational shift — elements as continuous, differentiable 3D Gaussian primitive sets, **derived from** (never replacing) the canonical state:

* **Gaussianization** — oriented boxes split into grids of primitives that exactly moment-match the uniform measure per sub-box (a provable invariant, tested at 1e-12), with analytic Jacobians back to state parameters.
* **Probabilistic clash detection** — separating-axis clearance with a delta-method sigma computed under the *joint* belief via the relative Jacobian `(J_a − J_b) Σ (J_a − J_b)ᵀ`, so shared parameters cancel (two walls driven by one storey height do not jitter relative to each other). Reported: `P(clash) = Φ(−c/σ)` (a probability of a real event), soft overlap mass, and a χ²₃ *separation significance* — deliberately not labeled a probability.
* **Structural attention** — analytic scaled-dot-product weights over primitive tokens (content × Gaussian overlap kernel × relationship-graph affinity), a diffusion update obeying the maximum principle, semantic identity channels frozen. **No learned weights exist**, and the module says so: this is deterministic kernel message passing wearing the attention API. A content-blind Laplacian ablation ships alongside so the content-dependence is demonstrated, not asserted.
* **Scan-to-BIM registration** — robust GMM alignment (uniform outlier component) with monotone EM (closed-form GLS translation + Armijo-guarded Gauss-Newton yaw), 8 deterministic starts, coarse-to-fine annealing; recovers a withheld pose to arcminutes/millimetres and reports the Gauss-Newton pose information matrix.
* **Multi-scale fusion** — exact moment-matched level-of-detail merging (element → building) with KL merge error, and exact affine transport into geo-referenced frames.
* **Compliance under uncertainty** — every rule is a margin with mean and sigma under the joint belief; `P(satisfied) = Φ(μ/σ)` with PASS/MARGINAL/FAIL.
* **Differentiable layout** — cost/daylight/energy objectives with exact DAG gradients, chance-constraint penalties sharing the compliance margins, closed-form Gaussian ray transmittance with forward-mode dual-number gradient witnesses; results commit through ordinary verified interventions.
* **Splat interoperability** — the scene exports to the standard 3D Gaussian Splatting PLY layout (positions, quaternions, log-scales, SH DC color by semantic class), loadable in stock 3DGS viewers.

### OpenUSD state-space interchange

`python -m gat.demo.usd` runs the interchange "killer test": not *USD export*, but

> **Computational World → OpenUSD → Computational World**

Runtime A applies a transformation, serializes its **entire computational state** to a USD stage — real `Xform`/`Cube` prims any USD tool can open, with identity, semantics, topology, the joint Gaussian belief (floats via shortest round-tripping repr → bitwise reconstruction), typed constraints, the *defining expressions* of every derived quantity, and the execution trace carried in a `gat` custom-data namespace — and **dies** (a real subprocess exit). Runtime B reconstructs the world from the stage, passes a formal invariant suite (`I_identity ∧ I_geometry ∧ I_topology ∧ I_semantics ∧ I_gaussian ∧ I_constraints ∧ I_provenance ∧ I_configuration`), recompiles the dependency DAG and Jacobians *from the transferred definitions*, and continues the computation. Success criterion, demonstrated and tested:

```
S2_transferred ≃ S2_continuous     (posterior mean and covariance BITWISE equal)
```

All eight transfer levels — geometry, BIM semantics, topology, Gaussian state, computational state, transformation semantics, provenance, and operational state-space equivalence — pass in the shipped demo and test suite (`tests/test_usd_interchange.py`).

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

### Honesty notes

* Covariance propagation is first-order (means are exact); the validity regime (mm-scale sigmas on m-scale dimensions) is stated where it matters.
* Determinism is guaranteed as same-platform byte identity; cross-platform agreement is tolerance-level (BLAS reduction order).
* The v0 IFC adapter reads dimensional quantities and placements, not solid-model geometry — an explicit adapter-boundary decision (§12), swappable without touching the engine.
* Gaussian overlap is a proxy for boolean geometry; clash scores are calibrated probabilities of the *modeled* clearance event, and the void-blindness of Gaussianized walls (openings are not subtracted) is a known v0 limit.

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

GAT is an exploratory open-source project. The v0 engine (see *GAT v0 — the implemented engine* above) implements the §17 milestone end to end: the state-propagation core, the geometric Gaussian layer, the analysis layers (sensitivity, stability, configuration identity), and two self-asserting demonstrations with a full test suite.

The architecture, mathematical assumptions, representations, and implementation are expected to evolve as they are tested against real BIM data and engineering workflows.

## Ecosystem

GAT's adapter boundary (§12) is designed to meet the surrounding toolchain rather than replace it. Reference points and intended integration targets in and around the [notationsystems](https://github.com/notationsystems) organization:

* [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [Geometry-Grounded-Gaussian-Splatting](https://github.com/notationsystems/Geometry-Grounded-Gaussian-Splatting) — GAT's geometric layer follows the 3DGS covariance factorization (`Σ = R·S·SᵀRᵀ`) and exports viewer-compatible splat PLYs.
* [arch-render-ai-toolkit](https://github.com/notationsystems/arch-render-ai-toolkit) — BIM-metadata-to-render-prompt middleware; GAT's canonical JSON state export (`GatSession.export_json`) is shaped for exactly this kind of consumer.
* [OpenMEP](https://github.com/notationsystems/OpenMEP) — MEP components for Revit/Dynamo; MEP element classes (ducts, pipes, fittings) are the natural next entity family for the clash and propagation layers (the proposed-duct scoring in the geometry demo is that use case in miniature).
* [bridge-pipeline](https://github.com/notationsystems/bridge-pipeline) — generative structural design with BIM compliance checking; a downstream candidate for GAT's verified-state and compliance reports.
* [AI-CAD-BIM-Parametric-Assets](https://github.com/notationsystems/AI-CAD-BIM-Parametric-Assets) — parametric IFC/STEP datasets; a source of test models beyond the shipped demo.

No claim is made that Gaussian representations are universally optimal for BIM.

The purpose of the project is to determine **where they are useful, where they fail, and what computational architecture is required around them.**

---

# 20. Core Proposition

The central proposition of GAT can be stated simply:

$$
\boxed{
\text{BIM}
\rightarrow
\text{Architectural State}
\rightarrow
\text{Gaussian Representation}
\rightarrow
\text{Transformation}
\rightarrow
\text{Verified State}
}
$$

Or, in words:

> **GAT investigates BIM as a computational state space and Gaussian representations as a mechanism for transforming, propagating, and reasoning over continuous architectural state.**

Stated as the layered research hypothesis the v0 engine now tests:

> **GAT investigates an architectural state-space substrate in which heterogeneous observations are converted into probabilistic state hypotheses, architectural configurations are represented within a structured configuration space, transformations are propagated differentially and probabilistically, and repeated state evolution can be evaluated for stability and validity.**

None of this is assumed correct. The v0 implementation exists to discover **which of these layers genuinely compose, which require modification, and which turn out to be unnecessary** — that empirical question is the project.

---

## License

TBD.

## Project Status

Early-stage research and development.
