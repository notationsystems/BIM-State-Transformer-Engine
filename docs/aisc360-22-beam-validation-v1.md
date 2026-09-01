# ANSI/AISC 360-22 beam validation v1

## Implemented claim

`ansi-aisc-360-22-f2-1-lrfd-v1` implements one design-code claim only:

```text
ANSI/AISC 360-22, Section F2, Equation F2-1
major-axis flexural yielding
doubly symmetric W-shape
compact section
continuously braced
LRFD phi_b = 0.90
Mn = Mp = Fy * Zx
```

The input is plastic section modulus `Zx`. The elastic major/minor section
moduli produced by the current IFC profile-moment adapter are different
properties and are rejected by the check's scope contract.

The implementation does not cover lateral-torsional buckling, local buckling,
noncompact or slender sections, unbraced members, minor-axis bending, shear,
deflection, combined forces, load combinations, or classification derivation.

## Independent oracle

The pinned record at `validation/aisc360-22-f1-1b-v1.json` transcribes AISC's
official *V16.0 Companion, Vol. 1: Design Examples*, Example F.1-1B, printed
page F-8 (PDF page 65). AISC identifies the example as reflecting the 2022
Specification and the 16th Edition Steel Construction Manual.

The example verifies a continuously braced compact ASTM A992/A992M W18x50:

| Published value | Oracle value |
| --- | ---: |
| `Fy` | `50 ksi` |
| `Zx` | `101 in3` |
| Factored demand | `266 kip-ft` |
| `Mn = Fy Zx` | `421 kip-ft` |
| `phi_b Mn`, `phi_b = 0.90` | `379 kip-ft` |
| LRFD result | `PASS` |

The test converts only the published inputs to SI, runs the production
calculation, converts the outputs back to kip-ft, and requires agreement with
the published whole-kip-ft results within `0.5 kip-ft`. It does not copy a
production result into the expected values.

The oracle record pins the official landing/download URLs, example/page,
document update date, and exact downloaded PDF SHA-256. The source PDF is not
vendored.

## State-chain integration

The opt-in IFC contract now separates numerical belief variables from scope
assertions:

- `GAT_Structural`: `YieldStrengthMPa` and
  `PlasticSectionModulusMajorM3`, each with sigma;
- `GAT_StructuralScope`: exact code method, shape family, section
  classification, bracing, bending axis, and section-property kind.

The lowering adapter admits the beam only when both contracts are complete and
match the validated profile. The resistance factor comes from the versioned
method, not an editable source field. The Gaussian wrapper propagates uncertainty
through the same F2-1 expression and binds both the code-validation profile
digest and independent oracle id into its validation/computation records.

The AISC oracle validates the deterministic nominal and LRFD available-strength
calculation only. GAT's later probability threshold and three-valued
`SATISFIED` / `VIOLATED` / `UNRESOLVED` assessment are an uncertainty-aware
decision overlay; they are not presented as an AISC-prescribed acceptance
procedure.

This validates arithmetic and implementation fidelity for the stated case. It
does not authenticate source engineering data, prove the beam satisfies the
scope assertions, or replace professional structural review.
