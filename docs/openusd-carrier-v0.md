# GAT OpenUSD State Carrier v0

Status: experimental carrier family v0, current contract version 3. Readers
also accept versions 1 and 2 for explicit migration.

This document defines how GAT transports a restartable computational belief
through OpenUSD. It does not make USD geometry or composed scene opinions the
canonical GAT state.

## Required invariants

An accepted carrier must reconstruct the same:

- closed architectural IR and source identities;
- entity, quantity, relationship, and constraint semantics;
- raw and derived variable order;
- indexed raw mean and complete dense covariance;
- recomputed derived mean and covariance;
- invariant results and architectural configuration digest.
- hash-chained execution history and its exact final-world binding (v3).

The receiving runtime calls GAT's ordinary snapshot decoder and invariant
registry. A stage is not accepted merely because USD can parse it.

## Carrier versus observation

A verified GAT `State` branch is a representation of an already-formed
computational belief. It is not a general rule that incoming USD attributes are
true or are measurements. Non-GAT scene data, images, point clouds, Gaussian
splats, and sensor values must remain quarantined artifacts until an adapter
classifies their source, binds semantic identity, supplies a measurement model
and uncertainty, and emits accepted evidence through the normal conditioning
path. USD composition cannot bypass that boundary.

Likewise, external geometry generators and engineering solvers are projections
or operators over canonical state; they do not own or mutate `World` directly.
Their results must re-enter through an explicit transformation and invariant
check. This preserves the distinction between designed, observed, inferred,
generated, and predicted representations of the asset.

## Stage topology

```text
<defaultPrim>
│   carrier version and optional Ed25519 signature
├── State                         authoritative
│   ├── Entities
│   │   └── E_*/Quantities/Q_*
│   ├── Relationships
│   ├── Constraints
│   ├── Belief
│   ├── Provenance
│   └── Ledger                       authoritative in v3
│       └── Events/E_*               one inspectable event prim per record
└── View                          optional, derived and disposable
    └── Entities/*/Bounds
```

The default prim may be renamed or introduced through a USD reference. Readers
locate `State` relative to the composed default prim rather than assuming the
literal path `/GAT`.

## Authoritative mapping

| GAT meaning | USD representation |
|---|---|
| Entity identity | Entity prim with `gat:ifcClass` and `gat:globalId` |
| Quantity identity | Quantity prim plus `gat:owner` relationship |
| Topology | Relationship prim with `gat:source` and `gat:target` relationships |
| Raw variable index | Ordered targets of `Belief.gat:rawVariables` |
| Raw mean | Native `double[] gat:rawMean` |
| Full raw covariance | Native row-major `double[] gat:rawCovariance` plus dimension |
| Placement | Native `double3` origin and `double` yaw |
| Closed expression AST | Canonical JSON value on the quantity prim |
| Constraint | Canonical JSON value on its constraint prim |
| Trace provenance | Canonical JSON value on the provenance prim |
| Execution ledger | Ledger metadata plus ordered event prims with operation, provenance, verification/error evidence, state digests, and hash links |
| Restart compatibility | Snapshot format, schema version, and runtime contract attributes |
| Corruption detection | Snapshot SHA-256 algorithm and digest attributes |
| Publisher authentication | Optional Ed25519 signature metadata on the default prim |

Relationships are used for entity and variable references so USD namespace
editing and composition can translate paths without changing GAT identity.
Identity never depends on a prim name or path.

## Derived view

`View` contains box geometry derived from the accepted world. It carries the
source world digest and relationships back to authoritative entity prims. It
may be recolored, hidden, replaced, layered over, or omitted entirely without
changing reconstruction. No reader may infer belief values from this branch.

Future Gaussian-splat, mesh, material, and simulation views belong under this
same non-authoritative boundary unless a later carrier version explicitly
promotes their semantics into GAT IR.

## Decode and acceptance

The reader:

1. rejects root files and composed stages that exceed the configured budgets;
2. validates carrier metadata and its supported version;
3. traverses the authoritative prims in explicit ordinal order;
4. resolves USD relationships in the composed namespace;
5. reconstructs and validates the closed, bounded ledger for v3 carriers;
6. verifies a claimed signature—including the ledger head in v3—when its public key is trusted, and fails
   closed when trusted provenance is required;
7. reconstructs a `GatStateSnapshot v1` value and verifies its integrity digest;
8. recompiles the closed IR and rebuilds all derived state;
9. runs the complete GAT invariant registry;
10. requires module, world, and configuration digests to agree;
11. requires the ledger head's result world to equal the restored world.

Unknown executable opcodes are rejected. Derived caches, Jacobians, geometry,
and reports are recomputed rather than trusted.

## Composition contract

The authoritative subtree may be referenced beneath a different default prim.
Namespace-aware relationship targets must continue to resolve after prim
renames or reparenting. Stronger opinions may alter the derived view freely.
Any stronger opinion that changes authoritative decoded content must also
produce a valid new snapshot digest and pass GAT verification.

The conformance suite validates reference composition, stronger presentation
opinions, namespace renaming, and variant selections. Derived variants are
free to change; a variant or stronger layer that changes an authoritative mean
without updating the bound snapshot is rejected.

## Resource policy

`OpenUsdReadLimits` bounds the root-layer byte size, composed prim count,
entities, quantities, relationships, constraints, raw variables, dense
covariance values, cumulative embedded JSON characters, trace events, and
ledger events.
Limits are checked before GAT reconstruction and before large USD arrays are
copied into Python where the format permits. The byte limit applies to the root
layer; composed references are additionally constrained by prim and state-data
budgets.

## Versioning and trust

- Carrier format: `gat-openusd-state-carrier`
- Current carrier version: `3`
- Accepted legacy carrier versions: `1`, `2`
- Embedded snapshot contract: `gat-state-snapshot`, schema `1`
- Runtime contract: `gat-world-v1`

SHA-256 binds content for corruption detection; it is not by itself a statement
of publisher trust. Carrier v2 introduced a domain-separated Ed25519 signature
over the carrier contract, snapshot contract, runtime contract, and snapshot
digest. Carrier v3 uses a new signature domain and additionally signs the
ledger format, schema version, and chain head. Because that head commits every
event, replacing an internally valid history that happens to reach the same
state still invalidates a trusted signature. The snapshot digest covers trace
provenance and every authoritative state value. The derived `View`, prim
namespace, and composition location remain intentionally unsigned.

Trust is caller supplied. `require_signature=True` requires a signature whose
key identifier resolves in the provided public-key map. Unknown keys and
unsigned carriers fail closed. A present signature that fails under a supplied
trusted key is always rejected.

`migrate_openusd` canonicalizes any supported carrier into current version 3.
A signed source must first verify under a trusted key and must be signed anew;
migration cannot silently strip or re-bless its provenance. A v3 source keeps
its exact ledger. Versions 1 and 2 have no embedded ledger contract, so their
migration creates an explicit genesis event at the restored checkpoint with
the source carrier version and snapshot digest in hash-bound provenance.

`GatSession.export_openusd` always supplies the current session ledger, and
`GatSession.load_openusd` restores it so a later transformation extends the
same chain. Direct `write_openusd` calls that omit a ledger create a deterministic
one-event genesis at the exported world.

The beam-specific conformance experiment is
`python -m gat.demo.beam_openusd_portability`. It requires a separately launched
runtime to authenticate a signed checkpoint containing the material evidence
transition and beam assessment, reproduce the transported computation digest
and verdict, apply a second certificate, and produce exactly the same world and
ledger as uninterrupted execution. Carrier authentication remains distinct
from certificate trust and engineering authorization.

The implementation uses custom `gat:` properties and does not yet ship a
generated OpenUSD typed or applied API schema plugin. This keeps the carrier
inspectable and portable while its semantics are still evolving. External key
management, certificate chains, revocation, and schema-registry identifiers
remain outside the current implementation.
