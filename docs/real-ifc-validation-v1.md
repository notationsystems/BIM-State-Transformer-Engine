# Real-IFC validation v1

Status: implemented audit boundary and measured baseline; adapter hardening is
the next phase.

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
- length-unit kind, prefix, and scale to metres;
- placement incompatibilities without stopping the rest of the inventory;
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
licensing. The two small buildingSMART fixtures run in CI. Schependomlaan is
kept out of routine CI because it is a 65 MB Git LFS artifact, but its measured
baseline is pinned and can be reproduced with `--include-large`.

| Model | Exact supported products | Measured result |
| --- | ---: | --- |
| Shipped two-office model | 10 | 10 `READY`; full pipeline passes. |
| buildingSMART wall/opening/window Reference View | 3 | 2 need geometry derivation; 1 lacks a fallback. |
| buildingSMART PCERT Building Architecture | 7 | 4 `READY`; 2 need geometry derivation; 1 lacks a fallback. |
| Schependomlaan as-planned IFC2x3 | 1,086 | 1,022 need geometry derivation; 63 also have unsupported placements; 1 lacks a fallback. |

All three public models declare millimetres. The current authoritative loader
rejects them before numerical state construction because it accepts only
unprefixed SI metres. Schependomlaan contains one storey, 880 walls, and 205
doors in GAT's current class scope. Every one of those 1,086 products lacks at
least one quantity required by the v0 state contract; 1,022 expose geometry
that can be used by a future derivation adapter.

This is the first empirical adapter result: the dominant next task is not a
new decision subsystem. It is unit-aware geometry derivation, followed by full
3D placement support. Multi-storey ownership remains a known limitation, but
it was not the first blocker in these measured files.

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

1. Introduce an explicit unit-normalization context and prove metre-normalized
   results against current metre-native fixtures.
2. Add a geometry-quantity provider behind the existing lowering boundary,
   initially for the swept/extruded solids seen in the pinned corpus.
3. Attach provenance to every derived quantity: source representation ids,
   algorithm version, unit scale, and uncertainty policy.
4. Support general rigid 3D placement composition and test the 63 measured
   Schependomlaan failures.
5. Add storey-local dependency ownership using separate multi-storey fixtures.
6. Only then admit a real-model decision scope into clearance acceptance.

No tolerance mode should bypass these steps by silently dropping entities.
