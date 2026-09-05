# Decision evidence planning outcomes

`plan_decision_evidence` assesses a `MinimumDecision` before considering
available observations. It returns a read-only `DecisionEvidencePlan`.

| Disposition | Meaning | Selected observation |
|---|---|---|
| `DECISION_RESOLVED` | Current belief resolves the criterion at its confidence threshold. | None |
| `NO_AVAILABLE_EVIDENCE` | The criterion is unresolved and the candidate iterable is empty. | None |
| `NO_WORTHWHILE_EVIDENCE` | Candidates exist, but none has information gain exceeding its cost. | None |
| `OBSERVE` | A worthwhile candidate was selected. | The highest-ranked option |

An empty candidate set is an ordinary operational outcome. It does not
resolve the criterion, approve an intervention, or change the world. The
returned assessment retains its world digest, `options` is empty, and
`should_observe` is false. Lists, tuples, and one-shot iterators are supported.
Resolved decisions still stop before consuming candidates.

`NO_AVAILABLE_EVIDENCE` is an additive Python API enum value. Clients that
exhaustively match `EvidenceDisposition` should handle it as an unresolved
condition requiring an external source of candidate evidence. The lower-level
`plan_observations` function continues to require at least one candidate.
This change does not add a new headless operation or change clearance-planner
behavior, response schemas, or historical artifact verification.

The existing planner ranks target-relevant information gain against cost in
nats. This outcome contract does not claim that ranking minimizes expected
decision error, monetary cost, or time to resolution; those require separate
qualification against simple measurement-policy baselines.
