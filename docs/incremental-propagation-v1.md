# Incremental propagation and measured dense-state scale v1

Status: implemented experimental execution path and reproducible measurement
harness. This document is not a cross-platform performance guarantee.

## What changed

Compilation still computes the complete derived mean, total Jacobian, and full
covariance. A later transformation now compares the preceding and resulting
raw beliefs and invalidates only the dependency-closed rows that can change:

```text
changed raw means
    -> affected derived values
    -> affected Jacobian rows
    -> affected cached (J Sigma) rows
    -> affected full-covariance rows/columns
```

Raw covariance row changes additionally invalidate every derived descendant of
those raw variables. Observation updates are handled from the rows that
actually changed, rather than trusting the observation's declared target. If a
change invalidates the whole state or the preceding verified world has no
Jacobian cache, execution falls back to complete pushforward.

The cache is derived data. It is not serialized, signed, or included in world
identity. Snapshot and OpenUSD receivers recompile it from the transported IR
and belief. Every candidate still runs the complete invariant registry before
commit.

`ExecutionResult.propagation` reports the mode and exact row counts. This makes
selectivity observable without changing the causal transformation contract.

## Equivalence contract

Incremental and complete paths produce bitwise-identical means and covariance
on the same runtime. For each invalidated covariance block, the incremental
path evaluates both matrix directions before symmetrization, matching the
canonical complete pushforward's floating-point order. This stronger condition
is required because even a sub-ULP difference would change a world digest and
break exact snapshot/OpenUSD continuation.

Ordinary execution always uses the incremental algorithm, so replay and
OpenUSD continuation remain deterministic within the stated runtime numerical
contract. A diagnostic complete pushforward is an oracle, not an alternative
commit path.

## Reproduce the scale probe

```bash
python -m gat.demo.incremental_scale \
  --sizes 16 32 64 128 256 512 1024 1536 2048 \
  --repeats 3 \
  --time-cliff-seconds 1.0 \
  --output incremental-scale.json
```

The synthetic module gives every storey one raw variable and two nonlinear
derived descendants. One edit therefore has a constant two-row dependency
scope while total state grows. The harness measures complete pushforward,
incremental pushforward, full verification, resident dense-array size, numeric
agreement, and actual invalidated rows. It has no timing assertion in tests.

## First measured reference

The checked-in reference was measured once on Windows 11, Python 3.12.13, and
NumPy 2.3.5. CPU identity was not captured, so these values are evidence about
that run only:

| Storeys | Full variables | Complete pushforward | Incremental pushforward | Verification | Cached-product / covariance rows |
|---:|---:|---:|---:|---:|---:|
| 512 | 1,536 | 0.082 s | 0.038 s | 0.033 s | 2 / 2 |
| 1,024 | 3,072 | 0.412 s | 0.175 s | 0.147 s | 2 / 2 |
| 1,536 | 4,608 | 1.064 s | 0.398 s | 0.341 s | 2 / 2 |
| 2,048 | 6,144 | 2.172 s | 0.814 s | 0.682 s | 2 / 2 |

Under the explicit one-second definition, complete pushforward crossed the
cliff at 1,536 synthetic storeys; incremental propagation plus verification
crossed it at 2,048. The source record is
[`validation/incremental-scale-reference-v1.json`](../validation/incremental-scale-reference-v1.json).

## What remains dense

Incremental recomputation does not make the state sparse. The canonical raw
covariance and disposable full covariance remain dense float64 arrays, and the
cached total Jacobian and cached `J Sigma_raw` product are dense. In the
synthetic 1-raw/2-derived model, those resident arrays alone reach
approximately 1 GiB at 2,896 storeys and 4 GiB at 5,792; temporary products
and Python/IR overhead make practical limits lower. Global or highly correlated
observations can invalidate most rows. Any raw covariance change currently
refreshes every cached-product row, and raw-belief PSD verification still
performs a dense Cholesky factorization.

The measurement therefore supports a specific next decision: retain the dense
engine for bounded assurance slices, while designing a sparse precision/factor
backend only when a deployment model exceeds the measured local-update or
memory budget. It does not justify replacing the verified dense reference
engine speculatively.
