# GAT causal events v1

## Why these are not transformations

Four important things can happen around an architectural belief without
changing that belief:

| Event | What it means | Why state stays unchanged |
|---|---|---|
| Assessment | A criterion was evaluated against this belief | Evaluation does not alter the asset or posterior |
| Policy | A next evidence/action option was selected or declined | Selection precedes observation or intervention |
| Approval | An authority approved, rejected, deferred, or revoked a scope | Authorization is not physical execution |
| External action | Field work was proposed, authorized, started, or ended | Only returned evidence may later condition belief |

Treating any of these as a state transformation would collapse the causal
distinctions that GAT relies on. They are therefore typed, hash-chained ledger
events whose prior and result world digests must be identical.

## Closed records

All records carry the exact `world_digest` they describe and finite JSON
`details`. Their fixed fields are:

* `AssessmentRecord`: id, type, subject, verdict, method, optional evidence
  digest, details.
* `PolicyRecord`: id, type, disposition, optional selected action and evidence
  digest, details.
* `ApprovalRecord`: id, authority, decision, scope digest, reason, details.
* `ExternalActionRecord`: id, action type, lifecycle status, optional
  authorization reference and result-evidence digest, details.

Unknown record types, missing or extra fields, malformed digests, stale world
bindings, and NaN/infinite detail values fail closed.

The built-in adapters turn `DecisionAssessment`, `DecisionEvidencePlan`,
`ClearanceAssessment`, and `ClearanceEvidencePlan` into these records without
discarding the scores and criteria that explain the outcome.

## Lifecycle rules

Approval ids obey:

```text
NEW -> APPROVED -> REVOKED
NEW -> REJECTED
NEW -> DEFERRED -> DEFERRED | APPROVED | REJECTED
```

External action ids obey:

```text
NEW -> PROPOSED -> AUTHORIZED -> STARTED -> COMPLETED | FAILED | CANCELLED
                \-> CANCELLED
      \-> CANCELLED
```

Authorized, started, completed, and failed records require an authorization
reference. Terminal states cannot be reopened under the same action id. These
rules establish ledger chronology; they do not prove that the external party
or work is genuine.

## Example

```python
from gat import (
    ApprovalDecision,
    ApprovalRecord,
    ExternalActionRecord,
    ExternalActionStatus,
    decision_assessment_record,
    decision_policy_record,
)

assessment = assess_decision(session.world, decision)
plan = plan_decision_evidence(session.world, decision, candidates)
session.record_assessment(decision_assessment_record(assessment))
session.record_policy(decision_policy_record(plan))

scope = sha256(proposal_bytes).hexdigest()
session.record_approval(
    ApprovalRecord(
        session.world.digest(), "approval-17", "licensed-engineer:CA-1234",
        ApprovalDecision.APPROVED, scope,
    )
)
session.record_external_action(
    ExternalActionRecord(
        session.world.digest(), "scan-17", "targeted-clearance-scan",
        ExternalActionStatus.PROPOSED,
    )
)
```

Ledger replay validates the records, their world binding, and lifecycle while
leaving mean and covariance bit-identical. If a completed scan produces a
calibrated measurement, that evidence enters later as an ordinary
`ObserveQuantity` or `ObserveLinearized` transition. OpenUSD carrier v3
preserves and optionally signs the combined state-changing and non-mutating
history.

## Trust boundary

The record says what the publisher claims occurred. An `authority` string or
`authorization_ref` is not an identity proof. A trusted OpenUSD carrier
signature authenticates the carrier publisher and the complete ledger head,
not necessarily the named approver. Independent approver signatures,
certificates, revocation, and external action receipts remain future trust
adapters.
