# Real-IFC validation v1

Status: implemented audit boundary, measured baseline, SI length-unit
normalization, bounded beam geometry derivation, and strict material-certificate
ingestion. The next phase is independent design-code validation.

## Why this exists

GAT's authoritative IFC loader is fail-closed. That protects a decision from
being computed over an incomplete state, but it also means an unfamiliar file
used to stop at its first incompatibility. `gat audit` is the non-mutating
discovery boundary: it parses the file, inventories every product in the
adapter's declared scope, and then attempts the unchanged
lower -> compile -> verify pipeline.

An audit is not a partial importer. In particular:

- it never changes the IFC source;
- it never synthesizes missing quantities;
- it never commits a partial world;
- it never authorizes a decision; and
- skipped or unsupported entities remain explicit.

Use it with either installed entry point:

```console
gat audit model.ifc --text
gat audit model.ifc --output audit.json
gat-ifc-audit model.ifc --compact
```

Exit code `0` means the current supported scope completed the full pipeline,
`2` means the report was produced but the pipeline is blocked, and `3` means
the source or output could not be read or written.

## Report contract

`gat-ifc-audit-v1` binds its result to the exact source SHA-256 and byte size.
It includes:

- IFC schema and exact entity-type counts;
- every currently supported product and its STEP id, GlobalId, class, and name;
- required, available, and missing quantities;
- whether a geometry representation exists from which a future adapter could
  derive missing quantities;
- active project length-unit kind, prefix, scale to metres, and whether
  normalization is required;
- placement incompatibilities without stopping the rest of the inventory;
- a non-authorizing beam-geometry summary with complete, partial, and blocked
  counts plus an ordered result digest;
- separate lowering, compilation, and verification stage results; and
- an explicit assurance statement that the audit cannot authorize decisions.

Product statuses have narrow meanings:

| Status | Meaning |
| --- | --- |
| `READY` | The product satisfies the current preflight contract. |
| `NEEDS_GEOMETRY_DERIVATION` | Required quantities are missing, but an IFC geometry representation exists. |
| `MISSING_SOURCE_DATA` | Required quantities are missing and no current fallback source exists. |
| `BLOCKED` | Metadata, quantity graphs, or placements cannot be interpreted safely. |

`pipeline_ready` is still only about `supported-product-scope-only`. A later
acceptance request must name a decision scope and prove that every relevant
entity and dependency is covered. Partial ingestion may never produce
`ACCEPT`.

## Measured public-model baseline

The manifest at `validation/ifc-corpus-v1.json` pins repository commits,
artifact SHA-256 digests, sizes, URLs, expected audit results, and CC-BY-4.0
licensing. The two small buildingSMART fixtures and the 19 MB Medical–Dental
Clinic structural model run in CI. Schependomlaan is kept out of routine CI
because it is a 65 MB Git LFS artifact, but its measured baseline is pinned and
can be reproduced with `--include-large`.

| Model | Exact supported products | Measured result |
| --- | ---: | --- |
| Shipped two-office model | 10 | 10 `READY`; full pipeline passes. |
| buildingSMART wall/opening/window Reference View | 3 | 2 need geometry derivation; 1 lacks a fallback. |
| buildingSMART PCERT Building Architecture | 7 | 4 `READY`; 2 need geometry derivation; 1 lacks a fallback. |
| buildingSMART Medical–Dental Clinic Structural | 37 in architectural scope; 738 beam candidates | 738 lengths; 277 swept-solid sections complete; 461 surface-model beams explicitly length-only; multi-storey lowering blocked. |
| Schependomlaan as-planned IFC2x3 | 1,086 | 1,022 need geometry derivation; 63 also have unsupported placements; 1 lacks a fallback. |

The two small IFC4 fixtures and Schependomlaan declare millimetres; the clinic
structural model declares metres. The authoritative loader resolves the active
`IfcProject.UnitsInContext`, converts SI-prefixed source lengths into canonical
metres, and carries the source scale into export. Their audits therefore
progress beyond the former unit gate. Schependomlaan contains one storey, 880
walls, and 205 doors in GAT's current class scope. Every one of those 1,086
products still lacks at least one quantity required by the v0 state contract;
1,022 expose geometry that can be used by a future derivation adapter.

The normalization boundary covers source-backed length quantities, local
placement translations, authored length sigmas, and posterior means and
sigmas written back to IFC. Default uncertainty policy is already expressed
in canonical metres and is not scaled twice. SI metre prefixes are accepted;
missing declarations retain the existing explicit assumed-metre warning.
Conversion-based units remain fail-closed until their full
`IfcMeasureWithUnit` chain is implemented and verified.

The clinic model is the first measured structural boundary. It contains
317,671 IFC instances, 4 storeys, 738 beams, 195 columns, and 13 slabs. GAT
parses the complete file, inventories all 738 beams as candidates for explicit
`GAT_Structural` opt-in, and refuses to claim a structural verdict:
the source has no GAT evidence marker, the current architectural lowering
contract requires one storey, and the storeys do not provide `ClearHeight`.
That fail-closed result is now a CI baseline rather than an anecdotal manual
run.

The v1 provider derives `Length` from each beam's axis polyline. For the 277
`IfcExtrudedAreaSolid` bodies, it samples the arbitrary closed composite profile
(line and trimmed-circle segments), computes centroidal polygon properties,
and emits `CrossSectionArea`, `SectionModulusMajorM3`, and
`SectionModulusMinorM3`. Fine/coarse sampling disagreement supplies a bounded
numerical-discretization error; every result binds the source IFC digest, STEP
representation ids, unit scale, method version, and sampling policy. The 461
`IfcFaceBasedSurfaceModel` bodies are not approximated from names or bounding
boxes and therefore remain `LENGTH_ONLY`. The audit summary explicitly states
that these values do not authorize structural decisions.

Material evidence now enters through a strict versioned JSON certificate
contract. It preserves certificate, issuer, beam, batch, specimen, property,
unit, method, calibration, and source-byte identities and binds the observation
to the exact canonical `VarId`. Unknown or duplicate fields, invalid numerical
values, unsupported units/properties, and subject mismatches fail closed. The
contract also states that issuer trust, signatures, revocation, and decision
authorization are not yet verified. The reference beam experiment uses this
reader, but its shipped certificate is a fixture—not real authenticated field
evidence.

## Reproducing the corpus

```console
python validation/fetch_ifc_corpus.py validation-corpus
GAT_IFC_VALIDATION_ROOT=validation-corpus \
  python -m unittest tests.test_public_ifc_corpus -v

# Optional 65 MB Schependomlaan run
python validation/fetch_ifc_corpus.py validation-corpus --include-large
gat audit validation-corpus/Schependomlaan-planning.ifc --text
```

The downloader accepts only fixed destination basenames and refuses content
whose byte size or SHA-256 differs from the manifest. Routine tests do not
silently download external data.

## Adapter-hardening order derived from the measurements

1. **Completed:** resolve the active project SI length unit; prove equivalent
   metre-normalized state, placement, covariance, and source-unit round trips.
2. **Completed (bounded v1):** derive axis length for all 738 clinic beams and
   section properties for the 277 swept/extruded arbitrary profiles; report the
   remaining 461 surface models as explicit partial results.
3. **Completed (schema v1):** ingest material certificates as typed
   observations, preserving issuer,
   specimen/batch identity, calibration, units, and source digest.
4. **Completed:** attach provenance to every derived quantity: source
   representation ids, algorithm version, unit scale, and uncertainty policy.
5. **Next:** validate one versioned beam design-code calculation against an
   independent oracle and bind its validation profile to the verdict.
6. Then add storey-local ownership and support general rigid 3D placement
   composition; test both the clinic boundary and the 63 measured
   Schependomlaan failures.
7. Only then admit a real-model structural decision scope into beam assurance.

No tolerance mode should bypass these steps by silently dropping entities.
