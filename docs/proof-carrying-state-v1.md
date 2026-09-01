# GAT proof-carrying state-transition manifest v1

## Purpose

`gat-computation-proof-manifest` is a portable commitment to one accepted GAT
state transition and its external proof artifact. It answers:

> Does this proof claim concern this exact accepted operation, prior state,
> result state, invariant report, numerical contract, and ledger history?

It does not answer whether the observations were truthful, the calibration was
representative, the engineering model was appropriate, or the building is
safe. Those are evidence, calibration, validation, and professional-authority
questions outside the proof claim.

The permitted claim scope is fixed in schema v1:

```text
computational-integrity-only
```

A document that changes or broadens that scope fails closed.

## Relationship to the execution ledger

The ledger and proof manifest have different jobs:

| Artifact | Establishes |
|---|---|
| Execution ledger | Hash-chained history that a compatible GAT runtime can replay |
| OpenUSD signature | Identity-bound publication of an exact snapshot and ledger head |
| Proof manifest | Exact public statement and proof-artifact commitment for one accepted transition |
| Backend verifier | Whether the external cryptographic proof verifies |
| Engineering validation | Whether the proved program and assumptions were suitable |

The manifest binds a transition event by sequence and event hash, its exact
ledger head, prior/result world digests, operation digest, and invariant-report
digest. It may also bind an optional deterministic computation-result digest.
When declared, that exact digest must occur in a later state-bound assessment
under the transition's result world and within the committed ledger head.
Rejected attempts and state-preserving causal events cannot receive a
state-transition proof manifest themselves.

The ledger head is intentionally exact. If more events are appended, the old
manifest remains a statement about the earlier ledger version and will not
verify against the extended chain. A host must retain the exact referenced
ledger or produce a new proof/aggregation statement.

## Public values

`ComputationProofManifest.public_values()` returns the canonical public
statement the proof program must expose:

```json
{
  "claim_scope": "computational-integrity-only",
  "runtime_contract": "gat-world-v1",
  "ledger_head": "sha256",
  "event_seq": 1,
  "event_hash": "sha256",
  "prior_world_digest": "sha256",
  "result_world_digest": "sha256",
  "operation_digest": "sha256",
  "verification_digest": "sha256",
  "computation_result_digest": "sha256-or-null",
  "numeric_contract_digest": "sha256",
  "model_contract_digest": "sha256",
  "validation_profile_digest": "sha256",
  "evidence_commitments": ["sha256"]
}
```

The manifest stores a SHA-256 digest of these canonical values beside the
program, verifying-key, and proof-artifact digests. A backend verifier must
verify the proof against this public-values commitment. Merely comparing
digests is not cryptographic proof verification.

## Numerical contract

Proof programs require deterministic, reviewable numerical semantics. Every
manifest therefore binds an external numerical-profile document plus an
inspectable summary:

* `profile_id` and `profile_digest` identify the complete specification,
  including units and per-field scaling;
* `arithmetic` is one of `checked-integer`, `signed-fixed-point`, or
  `ieee754-binary64`;
* `rounding` names the exact rounding policy;
* integer and fixed-point arithmetic must use checked overflow;
* binary64 arithmetic must reject non-finite results.

The first implemented SP1 experiment is the bounded AISC F2-1 beam yielding
calculation in [`sp1-beam-guest-v1.md`](sp1-beam-guest-v1.md). It uses
milli-MPa, mm3, and milli-N*mm checked integers and keeps the Gaussian update
outside the proof boundary. A future fixed-point `Sigma' = J Sigma J^T`
experiment would require separately reviewed scaling and quantization-error
bounds.

## Creation and verification

```python
from gat import (
    NumericContract,
    create_computation_proof_manifest,
    verify_computation_proof_manifest,
    write_computation_proof_manifest,
)

manifest = create_computation_proof_manifest(
    ledger,
    transition_event_seq,
    numeric_contract=NumericContract(
        "beam-milli-mpa-mm3-milli-nmm-v1",
        numeric_profile_digest,
        "checked-integer",
        "nearest-ties-to-even",
        "checked",
    ),
    model_contract_digest=model_contract_digest,
    validation_profile_digest=validation_profile_digest,
    computation_result_digest=engineering_result_digest,  # optional
    evidence_commitments=(certificate_digest, certificate_source_digest),
    proof_system="sp1",
    proof_type="core-v6.5.0",
    program_digest=guest_elf_digest,
    verifying_key_digest=verifying_key_digest,
    proof_artifact=proof_bytes,
    locator="proofs/transition-0001.bin",
)
write_computation_proof_manifest(manifest, "transition-0001.proof.json")

binding_only = verify_computation_proof_manifest(manifest, ledger, proof_bytes)
assert binding_only.bound
assert not binding_only.proof_verified

verified = verify_computation_proof_manifest(
    manifest, ledger, proof_bytes, verifier=deployment_sp1_verifier
)
assert verified.proof_verified
```

Verification has separate checks for manifest integrity, ledger integrity,
accepted-event identity, exact ledger head, transition statement, public
values, the optional assessment-bound computation result, proof bytes, and the
backend cryptographic proof. When no verifier is
supplied, the final check is `NOT_CHECKED`; it can never be reported as passed.
If an earlier binding check fails, the backend is not invoked.

Proof locators are inert metadata. GAT never dereferences them automatically;
the host must obtain and supply the exact proof bytes under its own transport
and size policy.

## SP1 and confidentiality

The schema remains backend-neutral. The repository now contains one pinned SP1
v6.5.0 Rust guest and host for the bounded beam claim; the Python core still
does not bundle or silently install a zkVM. Proof generation is an explicit
Linux/macOS deployment step, and GAT accepts a verified claim only through an
explicit backend-verifier callback.

Deployments must enforce an allowed tuple of proof system, proof type, program
digest, verifying-key digest, and numerical profile. They must also evaluate
the proof system's actual privacy properties. GAT deliberately does not infer
zero knowledge from a label in the manifest.

## OpenUSD carrier direction

The proof artifact remains content-addressed and external. A future OpenUSD
carrier revision may embed the small manifest or a manifest reference beneath
authoritative state and bind its digest into the carrier signature. It should
not embed large proofs by default, and it should commit to GAT's canonical
semantic state rather than raw `.usdc` file bytes.

That carrier change remains deferred until the beam guest has passed
adversarial and performance testing with retained proof artifacts.
