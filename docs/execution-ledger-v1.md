# GAT execution ledger v1

## Purpose

`gat-execution-ledger` is the authoritative history of computational changes
to one GAT world. It answers a stricter question than the human-readable trace:

> Starting from this exact checkpoint, can another compatible runtime reproduce
> every accepted transition, every rejected attempt, every invariant result,
> and the complete final joint belief exactly?

Ledger v1 records operations in GAT's closed transformation algebra plus typed
state-preserving causal records. It does not deserialize or execute arbitrary
code.

## Causal structure

The first event is `genesis`. It binds the chain to the exact world, module,
raw belief, architectural-configuration quotient, runtime contract, and full
initial verification report. Every later event is one of:

* `transition` — a transformation passed verification and committed. The event
  binds the prior and result world digests and the complete invariant report.
* `rejection` — the attempt did not commit. Prior and result world digests must
  be equal. The event binds the error type, message, digest, and, for invariant
  failures, the complete failed verification report.
* `assessment`, `policy`, `approval`, or `external_action` — a typed event
  about the exact current world. These events must preserve the world digest
  and cannot carry transformation verification or error fields.

Each event includes the preceding event hash. Its own SHA-256 is calculated
over all event material except `event_hash`, including operation, caller
provenance, verification evidence, errors, state digests, and previous hash.
Deleting, reordering, or editing any event therefore breaks the chain.

There is no implicit wall-clock timestamp. Sequence and state causality are
authoritative. When time is supplied by a trusted sensor, survey system, or
approval service, the caller can include it in `provenance`; it then becomes
hash-bound evidence rather than an unearned claim by the runtime.

## Closed operation vocabulary

Schema v1 admits exactly these recursive operation records:

| Opcode | Meaning |
|---|---|
| `set_parameter` | Do-intervention with fresh design uncertainty |
| `shift_parameter` | Exact affine shift |
| `scale_parameter` | Exact affine scale |
| `observe_quantity` | One or more Gaussian quantity measurements |
| `observe_linearized` | Prior- and evidence-bound scalar adapter likelihood |
| `evolve_linear_gaussian` | Calibrated temporal prediction with exact covariance transport |
| `composite` | Atomic ordered composition of v1 operations |

The complementary causal vocabulary records assessments, policy selections,
human approval claims, and external-action lifecycles. See
[`causal-events-v1.md`](causal-events-v1.md). These events are replayed and
validated but never applied to the belief.

Variable identity is structural (`ifc_class`, `global_id`, `quantity`), not a
display string. Linearized scan evidence carries its complete row, raw target
set and ordering, exact prior belief/world digests, evidence digest, noise,
prediction, observation, and label. Unknown opcodes, extra fields, non-finite
numbers, non-positive uncertainty, and malformed digests fail closed.

## Root document

```json
{
  "format": "gat-execution-ledger",
  "schema_version": 1,
  "runtime_contract": "gat-world-v1",
  "events": ["genesis", "transition or rejection", "..."],
  "integrity": {
    "algorithm": "sha256",
    "head": "hash of the last event"
  }
}
```

Readers enforce a 16 MiB encoded-size limit and a 100,000-event limit before
reconstruction. The canonical writer sorts object keys, rejects NaN and
infinities, emits UTF-8, and produces byte-identical output for identical
same-platform executions and provenance.

## Session API

Every `GatSession` begins a ledger at its exact initial world. `run` records
both strict and non-strict rejections as well as successful transitions:

```python
from gat import GatSession, ObserveQuantity, read_ledger, replay_ledger

session = GatSession.load_ifc("gat/demo/model.ifc")
checkpoint = session.world
session.run(
    ObserveQuantity.single(session.var("Office-A", "Volume"), 59.4, 0.05),
    provenance={
        "sensor": "volume-laser-A",
        "calibration": "cal-2026-08",
        "external_time": "2026-09-01T04:00:00Z"
    },
)
head = session.export_ledger("execution-ledger.json")

replayed = replay_ledger(checkpoint, read_ledger("execution-ledger.json"))
assert replayed.world.digest() == session.world.digest()
```

Replay first validates the complete chain. It then regenerates each operation,
runs the normal provenance checks, conditioning, propagation, and invariant
registry, and compares the resulting error or committed world plus the full
verification record. Rejected events must reject for the same reason. The
reconstructed mean and dense covariance are exact, not merely marginally
equivalent.

Run the independent-process demonstration with:

```bash
python -m gat.demo.ledger_replay execution-ledger.json
```

## Trust boundary and current limits

Hash chaining provides tamper evidence relative to a trusted checkpoint or
trusted ledger head. It does not authenticate who created the chain: an
attacker able to replace the entire unsigned file and trusted head can create a
new internally consistent chain. Provenance fields such as `approval` or
`pose_source` are bound claims, not independently verified identities.

Exports remain trace-only, while policy proposals, decision assessments,
human approvals, and external actions are distinct typed causal events. A
resumed JSON snapshot currently starts a new ledger genesis at the restored
state. OpenUSD carrier v3 embeds the inspectable
event chain, binds its head into the optional Ed25519 signature, restores it,
and extends the same chain after continuation.

The next chronological milestone is explicit temporal/process dynamics for
evidence that describes change through time rather than a static correction.
