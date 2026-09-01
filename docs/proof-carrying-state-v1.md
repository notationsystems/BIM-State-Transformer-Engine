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
digest. Rejected attempts and state-preserving causal events cannot receive a
state-transition proof manifest.

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

The first recommended SP1 experiment is a micrometre-scaled clearance policy,
not a dense Gaussian update or finite-element solve. A later fixed-point
`Sigma' = J Sigma J^T` experiment should have separately reviewed scaling and
quantization-error bounds.

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
        "clearance-micrometre-v1",
        numeric_profile_digest,
        "signed-fixed-point",
        "nearest-ties-to-even",
        "checked",
    ),
    model_contract_digest=model_contract_digest,
    validation_profile_digest=validation_profile_digest,
    evidence_commitments=(scan_digest, calibration_digest),
    proof_system="sp1",
    proof_type="groth16",
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
values, proof bytes, and the backend cryptographic proof. When no verifier is
supplied, the final check is `NOT_CHECKED`; it can never be reported as passed.
If an earlier binding check fails, the backend is not invoked.

Proof locators are inert metadata. GAT never dereferences them automatically;
the host must obtain and supply the exact proof bytes under its own transport
and size policy.

## SP1 and confidentiality

The schema is backend-neutral, with SP1 as the first intended adapter. The
core package does not compile a Rust guest, generate a proof, select a proving
service, or verify SP1 internally.

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

That carrier change is deferred until a real backend verifier and fixed-point
clearance guest have passed adversarial and performance testing.
