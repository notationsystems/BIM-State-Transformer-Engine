# Portable IFC import identity v2

New `GatSession.load_ifc` and `GatSession.from_text` imports use
`gat-ifc-content-v2`. Identical input bytes imported from different paths
produce the same module and world digests within the same numerical runtime.
The IFC audit uses the same binding. Moving a source file no longer invalidates
a newly produced assessment solely because the locator changed.

## Four distinct identities

| Identity | Binding |
|---|---|
| Source locator | `session.source_locator` and the compile trace's `name`; provenance, outside the module/world hash. |
| Exact source bytes | `module.meta.source_content_sha256`; the SHA-256 of the one byte buffer read and parsed. `meta.source` is its `sha256:` URI. |
| Semantic projection | `gat.source_identity.semantic_model_digest(module)`; domain-separated printed computational IR excluding source locator/content metadata. |
| Execution state | Existing `World.digest()` and ledger; the module includes the import contract tag and content hash, and the world binds its numerical state. |

The source contract tag is `module.meta.source_identity_contract`. The runtime
hash algorithm remains `gat-world-v1`: this is a versioned import binding,
not a silent change to the module printer, world hash, snapshot format, or
ledger verification algorithm. Old readers continue to hash serialized module
metadata exactly as before. No serialized historical object is rewritten.

The semantic projection includes EntityIds and names, quantities, units,
placements, relationships, constraints and adapter metadata. It excludes IFC
STEP source references and file metadata that the IR printer does not express.
It is neither geometry equivalence nor permission to reuse evidence. Even a
nonsemantic source-byte edit changes the content-bound execution identity;
only the semantic projection may remain equal. Coordinate-frame equivalence
and tolerance-based numerical agreement are separate qualification results.

`load_ifc` reads bytes once and decodes that exact buffer as UTF-8. `from_text`
hashes the supplied string's UTF-8 encoding. Text obtained through a reader
that normalizes line endings is a different byte source. No line-ending,
whitespace, header, or IFC-instance normalization is claimed.

## Historical verification and rollout

Existing snapshots, ledgers, signatures, and portable carriers retain their
serialized v1 metadata and identity. `load_snapshot`, `load_openusd`, and
ledger replay do not reinterpret their sources as v2. Checked-in historical
snapshot and ledger fixtures pin the pre-change world digest.

For deliberate reproduction of a historical path-bound IFC import:

```python
legacy = GatSession.load_ifc(original_path_string, identity_version=1)
portable = GatSession.load_ifc(relocated_path)  # identity_version=2
```

Headless IFC state requests accept the optional `identity_version` integer
(1 or 2, default 2). Other state kinds reject that option. A legacy headless
request without this field must explicitly choose 1 if it needs to reproduce
its old world digest. Consumers should not assume that re-importing the old
path under the new default will reproduce a historical assessment.

This is fresh import, not authenticated migration. No v1-to-v2 migration
receipt is issued: a historical state without its exact source bytes cannot
establish a new source-content claim. V1 assessments/evidence are not rebound
to v2 worlds, and old signatures do not sign newly imported worlds. Continue
from the verified historical snapshot/carrier when retaining that evidence
chain is required. Unknown import versions fail closed.

Viewer and Workbench load v2 inputs through the supplied model path, without
substituting the request's path. A copied input can render its assessment even
after the original path disappears. For v1, `--request` must explicitly carry
`state.identity_version=1` and its original path must still name the same file;
the accompanying IFC audit uses v1 as well. No fallback silently rebinds a
legacy assessment to a relocated copy. Original locators remain available in
compile-trace provenance. Report bodies and interactive instruments remain
separate surfaces.

## Qualification envelope

The regression tests cover relocation, relative/absolute paths, same-byte text
import, same-path changed bytes, audit/session agreement, relocated assessment
binding, snapshot continuation, ledger replay, and exact historical identities.
They do not establish cross-platform bitwise reproducibility or calibrated
physical accuracy. Qualification records must declare Python, NumPy, numerical
library/build, architecture, engine commit and input content hash. Exact replay
within a supported environment and physical-result agreement under explicit
tolerances are different claims.
