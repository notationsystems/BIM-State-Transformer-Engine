# Beam assurance reference chain

## Acceptance target

This experiment exists to test one proposition, not to add another isolated
subsystem:

```text
evidence -> state -> computation -> decision -> verification
```

The same `IfcBeam` and `VarId` identities must survive every transition. A
claim is useful only when a receiver can determine what was believed, why it
was believed, which calculation ran, why the verdict changed, and whether the
state can be replayed or continued.

Run it with:

```bash
python -m gat.demo.beam_assurance out/beam
```

## Closed chain

| Stage | Authoritative identity or record |
|---|---|
| IFC beam | `IfcBeam:GATBEAMELEMENT00000100` / `Beam-B1` |
| Raw state | `YieldStrengthMPa`, `PlasticSectionModulusMajorM3`, and `Length` `VarId`s |
| Derived state | `NominalMomentCapacity` and `DesignMomentCapacity` `VarId`s |
| Evidence | Strict `gat-material-certificate-observation-v1` ingestion produces a `CalibratedObservation` while preserving certificate, issuer, specimen, subject, likelihood, calibration, and source identities. |
| State transition | `ObserveQuantity` in the closed ledger transformation algebra |
| Engineering computation | `ansi-aisc-360-22-f2-1-lrfd-v1` plus model, dependency, oracle-validation, and result digests |
| Decision | Gaussian minimum-capacity assessment at a declared demand and confidence |
| Verification | Mandatory invariant report plus a non-mutating, state-bound assessment event |
| Transport | Posterior IFC, exact state snapshot, and hash-chained execution ledger |
| Optional proof | SP1 request with exact commitments; status remains `BACKEND_REQUIRED` until a real guest, proof, key, and verifier exist |

The IFC adapter lowers an `IfcBeam` only when it carries the complete
`GAT_Structural` opt-in property set. Unannotated beams remain opaque, so
public IFC scope does not silently expand. The contract contains:

```text
YieldStrengthMPa
YieldStrengthMPaSigma
PlasticSectionModulusMajorM3
PlasticSectionModulusMajorM3Sigma
```

It also requires a separate `GAT_StructuralScope` assertion matching the
implemented profile exactly: doubly symmetric W-shape, compact section,
continuously braced, major-axis bending, and plastic section modulus `Zx`.
The LRFD resistance factor is fixed by the implementation at `phi_b = 0.90`;
it is not an editable IFC input. Missing, invalid, or out-of-scope fields fail
closed.

The certificate reader rejects unknown and duplicate JSON fields, unsupported
property/unit pairs, non-finite values, non-positive uncertainty, and subject
mismatch. It preserves the exact source-byte digest and records that issuer
trust, signature validation, revocation checking, and decision authorization
have not yet been established. Ingestion makes the claim computable; it does
not make the issuer trusted or the resulting decision professionally approved.

## Calculation and decision

The reference calculation is deliberately inspectable:

```text
nominal moment capacity = 1e6 * fy[MPa] * Zx[m3]
design moment capacity  = 0.90 * nominal moment capacity
decision                = P(design capacity >= demand) at confidence c
```

For the shipped fixture:

| Value | Prior | After material evidence |
|---|---:|---:|
| Yield strength belief | `350 +/- 8 MPa` | `326.471 +/- 1.940 MPa` |
| Certificate observation | — | `325 +/- 2 MPa` |
| Design capacity | `315000 +/- 7858.9 N*m` | `293823.5 +/- 3418.0 N*m` |
| `P(capacity >= 301000 N*m)` | `0.96258` | `0.01788` |
| Verdict at 95% | `SATISFIED` | `VIOLATED` |

The observation and posterior are intentionally distinct. Treating the
certificate's 325 MPa as exact state would erase its declared uncertainty and
would misrepresent Bayesian conditioning as assignment.

## Selective recomputation

`BeamBendingEvaluator` derives its dependency slice from the canonical
Jacobian. Its cache key commits the model contract, validation profile, raw
dependency means, and dependency covariance submatrix. A `Length`
observation changes the world identity but reuses the beam computation; a
yield-strength observation changes the dependency digest and reruns it. The
result records `recomputed`, changed inputs, and affected descendants.

This claim is intentionally narrow. The generic GAT pushforward currently
rebuilds the dense derived Gaussian view after a transition. The reference
experiment demonstrates selective scheduling of the structural check, not a
global incremental sparse propagation engine.

## Epistemic boundary

Evidence kind is part of content identity:

```text
MEASURED | ESTIMATED | INFERRED | ASSUMED | SIMULATED | DERIVED
```

Changing `MEASURED` to `INFERRED` changes the evidence digest. Unit mismatch,
non-finite values, non-positive uncertainty, an unknown subject, or a derived
subject are rejected before conditioning. An LLM may propose or extract a
claim, but it cannot bypass this calibrated evidence boundary or perform the
engineering calculation.

The calculation is validated against AISC's published V16.0 Companion Example
F.1-1B for ANSI/AISC 360-22 Equation F2-1. The pinned oracle reproduces the
published `421 kip-ft` nominal and `379 kip-ft` LRFD available strength for a
compact, continuously braced W18x50. The validation profile and oracle identity
are committed into every beam computation record. See
[`aisc360-22-beam-validation-v1.md`](aisc360-22-beam-validation-v1.md).

The public IFC geometry provider is likewise non-authorizing. It derives beam
axis length from `Curve2D` polylines and area plus centroidal elastic section
moduli from `IfcExtrudedAreaSolid` arbitrary closed composite profiles. Every
quantity records source IFC digest, STEP representation ids, unit scale,
algorithm version, arc-discretization policy, and a numerical discretization
error. Unsupported bodies remain explicit partial results rather than guessed
properties. Its section moduli are elastic values and therefore cannot supply
the AISC check's required plastic `Zx` input.

## Emitted artifacts

| File | Purpose |
|---|---|
| `beam_posterior.ifc` | Interoperable posterior marginal means/sigmas for the source-backed beam variables |
| `beam_state.gat.json` | Exact restartable IR, variable order, mean, and dense covariance |
| `beam_ledger.json` | Replayable evidence transition and prior/revised assessment history |
| `beam_assurance_summary.json` | Human- and tool-readable result/change summary |
| `beam_sp1_request.json` | Proof-ready public commitments with an explicit unverified status |

The snapshot is the continuation artifact. IFC preserves source-backed
marginals but is not a complete joint-covariance carrier. OpenUSD can carry
the canonical snapshot and ledger, but OpenUSD does not define GAT's
computational semantics; the core must remain operable without it.

## Proof boundary

No SP1 proof is generated by this experiment. The request commits the
transition, invariant report, evidence, model, validation profile, numerical
profile, computation result, assessment, and final ledger head. A future
adapter must supply an allow-listed guest program, verifying key, proof bytes,
and real backend verification. Digest binding alone must continue to report
`proof_verified = false`.

## Limits

This is not a code-complete structural design check. The shipped certificate
is schema-realistic but is not a field-issued or independently authenticated
document. The public clinic IFC contains no certificate that authorizes a
structural decision. The implemented design-code scope relies on an explicit
compact/continuously-braced classification; it does not derive or independently
verify that classification from geometry. The calculation does not model load
combinations, local or lateral-torsional buckling,
shear, deflection, connections, fire, fatigue, code resistance rules, model
form error, or professional approval. First-order Gaussian propagation may be
unsuitable for strongly nonlinear or non-Gaussian cases.

Those limits are part of the result. The artifact demonstrates trustworthy
identity and transformation semantics for one bounded calculation; it does
not claim that the bounded calculation is sufficient to declare a real beam
safe.
