# Construction workflow deployment contract v1

This milestone turns GAT's mathematical assessments into a host-neutral
workflow boundary for three initial use cases:

1. as-built clearance acceptance;
2. prefabrication and opening verification; and
3. design-change / RFI impact review.

The deployment model is deliberately asymmetric:

```text
authoritative GAT state + evidence ledger
                 |
        gat-headless (read only)
                 |
      closed JSON response v1
        /        |         \
 Blender      CI gate    future Kit extension
```

Blender, CI, and future Omniverse clients may display decisions and evidence
requests. They do not receive permission to reinterpret likelihoods, commit
transformations, record approvals, or perform external actions.

## Acceptance semantics

An `AcceptanceCase` contains one or more probabilistic checks bound to the
same exact world digest. Supported checks are:

- `CLEARANCE`: dependence-safe geometric clearance bounds;
- `MINIMUM`: one quantity must exceed a threshold; and
- `DIFFERENCE`: `lhs - rhs` must exceed a margin, used for door/opening and
  prefabricated-assembly fit.

The default `gat-safe-acceptance-v1` policy is fail-closed:

| Condition | Disposition |
|---|---|
| any check is `VIOLATED` | `REJECT` |
| any check is `UNRESOLVED` | `REQUEST_EVIDENCE` |
| all checks are satisfied but evidence is missing | `REQUEST_EVIDENCE` |
| all checks are satisfied and covered by verified evidence | `ACCEPT` |

`ACCEPT` means the case may be presented for authorization. It is not an
`ApprovalRecord`, does not identify an authority, and does not mutate BIM
state. Approval and external action remain separate causal event classes.

## Evidence trust

A headless acceptance receipt is not trusted merely because it is JSON. The
handler requires every receipt to resolve to a hash-chained `transition`
event in the loaded execution ledger. The event must:

- contain a verified `ObserveLinearized` transition;
- carry the same evidence digest and prior/result world digests;
- carry matching evidence kind, calibration id, and check scope in its
  hash-bound provenance; and
- be transported in an OpenUSD carrier whose Ed25519 signature verifies
  against a deployment-configured trusted public key.

This authenticates the carrier publisher and ledger history. It does not by
itself prove that a calibration is scientifically valid or that an approver
has the claimed professional authority.

Trust roots are deliberately outside request JSON. The command line accepts
`--trusted-key KEY_ID=PATH`, where the file contains a raw or base64 Ed25519
public key. A future network service must source these keys from its own
deployment configuration rather than from an API caller.

## Headless interface

Install the package and invoke either the console entry point or module:

```text
gat-headless request.json -o response.json
python -m gat.headless request.json -o response.json
gat-headless request.json --trusted-key survey-authority=survey-authority.pub
```

Every request has exactly five top-level fields:

```json
{
  "format": "gat-headless-request-v1",
  "request_id": "opening-review-17",
  "operation": "acceptance",
  "state": {
    "kind": "ifc",
    "path": "model.ifc"
  },
  "payload": {
    "case_id": "opening-17",
    "workflow": "OPENING_VERIFICATION",
    "subject": "Door D-204 into Opening O-204",
    "checks": [
      {
        "kind": "difference",
        "check_id": "width",
        "lhs": {"entity_name": "Opening O-204", "quantity": "Width"},
        "rhs": {"entity_name": "Door D-204", "quantity": "Width"},
        "minimum_margin": 0.05,
        "confidence": 0.95,
        "label": "installed opening width fit"
      }
    ]
  }
}
```

With no ledger-bound as-built evidence, a satisfied numerical check still
returns `REQUEST_EVIDENCE`. Design-only review may explicitly select a policy
with `require_verified_evidence_for_accept: false`; callers must not present
that result as as-built acceptance.

The other read-only operations are:

- `summary`: state and invariant counts; and
- `change_impact`: `set_parameter`, `shift_parameter`, or `scale_parameter`
  previewed through the exact apply -> propagate -> verify pipeline.

A failed change preview exposes the candidate impacts and invariant failures
but leaves the authoritative world unchanged.

## Blender/Bonsai extension

`integrations/blender/gat_assurance` is a self-contained Blender 4.2+
extension. It requests local file access only. The sidebar loads an
`acceptance` response, displays its disposition and next evidence request,
and colors relevant objects. Objects are matched by their Blender name or a
`gat_entity_name` custom property.

The extension is deliberately read-only and does not depend on Bonsai. This
lets it coexist with native IFC authoring while avoiding direct manipulation
of Bonsai's IFC state. Build it with:

```text
blender --command extension build --source-dir integrations/blender/gat_assurance
```

## Current limits

- The interface is a deterministic command handler, not an HTTP server.
- Scan registration and likelihood construction are invoked through the
  Python API; a raw-scan headless request will be a later adapter.
- Evidence age is represented by exact causal state, not wall-clock expiry.
- One receipt covers only the check ids declared in its ledger provenance.
- The Blender extension imports completed responses; live service calls and
  point-cloud interaction are not yet implemented.
