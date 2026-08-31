# GAT — Gaussian Architectural Transformer for BIM

**GAT (Gaussian Architectural Transformer)** is a computational compiler for transforming, propagating, and reasoning over Building Information Modeling (BIM) state using Gaussian representations.

GAT investigates whether BIM can be treated not merely as a digital description of a building, but as a **computational state space** in which architectural geometry, relationships, constraints, uncertainty, and derived properties can be transformed and propagated systematically.

> **BIM → Architectural State → Gaussian Representation → Transformation → Verified State**

GAT is an experimental research engine. It is not intended to replace BIM authoring software, IFC, CAD, or architectural simulation systems. Its purpose is to investigate a computational layer that can operate **between BIM representations and downstream analysis, inference, simulation, and optimization.**

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

**Research / Experimental**

GAT is an exploratory open-source project.

The architecture, mathematical assumptions, representations, and implementation are expected to evolve as they are tested against real BIM data and engineering workflows.

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

---

## License

TBD.

## Project Status

Early-stage research and development.
