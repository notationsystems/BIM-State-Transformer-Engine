# GAT design language v1

Every human surface of GAT — the terminal CLI, the HTML decision report,
and the Blender assurance panel — renders the same backend vocabulary the
same way. This document is the contract those surfaces share; the code
that enforces it is `gat/report.py` (terminal + HTML) and
`integrations/blender/gat_assurance/bridge.py` (panel, the canonical
palette source). A lockstep test keeps the two palettes bit-identical.

## Principles

1. **Surfaces render decisions; they never make them.** A human surface is
   a viewer over `gat-headless` responses and engine reports. It re-checks
   the response's internal identities and refuses anything inconsistent —
   an unverified beam response, an authorization claim that contradicts the
   disposition, a blocked preview with no failure. Rendering is fail-closed
   like everything upstream of it.
2. **The vocabulary is the backend's, verbatim.** Surfaces show `ACCEPT`,
   `REQUEST_EVIDENCE`, `VIOLATED`, `ADMISSIBLE` exactly as the engine
   emitted them. No surface invents softer or scarier words.
3. **Colour reinforces, never carries.** Terminal output encodes the signal
   in words only (ASCII, no ANSI colour — stable in logs and CI). GUI and
   HTML surfaces add colour as reinforcement; a colour-blind reader loses
   nothing.
4. **Red is a decision, not a malfunction.** A refused or failed request is
   grey (`ERROR`), so red keeps meaning "the engine decided: stop".
5. **Reports are inert artifacts; instruments are self-contained.** Human
   surfaces come in two classes. *Reports* (decision pages, timelines,
   audits) carry no scripts and fetch nothing; interactivity is native
   disclosure (`<details>`) only, so a report attached to an RFI or
   archived cannot change meaning or behaviour. *Instruments* (the 3D
   viewer) may carry their own inline scripts but obey the same isolation:
   one file, no network access, no external resources, and they render
   state without ever mutating it.
6. **Every report says what it is not.** Each rendering ends with the
   read-only footer, and states either that it does not authorize any
   action or (for `ACCEPT`) that it is a recommendation still requiring
   professional approval — the same epistemic humility as
   `may_authorize` in the acceptance layer.

## Signal classes and palette

Vocabulary maps to three decided signal classes plus one undecided class.
RGBA values are linear floats; hex values are their 8-bit form.

| Signal | Vocabulary | RGBA | Hex |
|---|---|---|---|
| proceed | `ACCEPT`, `SATISFIED`, `ADMISSIBLE`, `PASS`, `READY` | (0.10, 0.70, 0.20, 1.0) | `#1ab233` |
| stop | `REJECT`, `VIOLATED`, `BLOCKED`, `FAIL` | (0.85, 0.08, 0.08, 1.0) | `#d91414` |
| attention | `REQUEST_EVIDENCE`, `UNRESOLVED`, `WARN`, `NEEDS_GEOMETRY_DERIVATION`, `MISSING_SOURCE_DATA` | (0.95, 0.55, 0.05, 1.0) | `#f28c0d` |
| undecided | `ERROR`, `NOT_RUN` | (0.35, 0.35, 0.35, 1.0) | `#595959` |

The six terms shared with the Blender panel (`ACCEPT`, `REJECT`,
`REQUEST_EVIDENCE`, `SATISFIED`, `VIOLATED`, `UNRESOLVED`) must stay
bit-identical to `bridge.disposition_color`. Unknown vocabulary raises;
it is never rendered with a guessed colour.

## Value formatting

| Value | Rule | Example |
|---|---|---|
| SHA-256 digest / event hash | first 12 hex chars + `...` (HTML: `…` with the full value behind native disclosure) | `04ab99c21e55...` |
| probability | five decimals | `0.96258` |
| moment capacity | kN\*m at one decimal, `mean +- sigma` | `315.0 +- 7.9 kN*m` |
| belief transition | `prior -> revised`, prior always first | `SATISFIED -> VIOLATED` |
| general quantity | six significant digits | `0.0283` |
| boolean assurance flag | `yes` / `no`, shown even when unflattering | `issuer_trust_verified  no` |

Terminal renderings are pure ASCII (`+-`, `->`, `...`, `kN*m`); GUI and
HTML surfaces may use the typographic forms (`±`, `→`, `…`, `kN·m`).

## Themes and print

Report pages follow the reader's system theme (`prefers-color-scheme`)
through design tokens — background, card, ink, muted, rules — while the
four signal colours stay literally identical in light and dark, because
they are semantic, not decorative. In print, shadows and tinted
backgrounds drop away, cards and tables never break across pages, badges
and the headline banner keep their colour, and every digest prints in
full (the on-screen disclosure collapses; a print-only twin carries the
value). Instruments stay light-only for now; their palette is the HUD's.

## Report anatomy

Every rendering of a headless response has the same skeleton, in order:

1. **headline** — `DISPOSITION: subject` (identical to the panel's
   headline), coloured by signal class on GUI surfaces;
2. **subline** — the case context (`OPENING_VERIFICATION case opening-1`,
   `beam case beam-b1-certificate`, `design-change preview`);
3. **notes** — the engine's own reasons, verbatim;
4. **blocks** — cards (label/value rows) and tables (rows accented by
   their verdict), including an `identity` card carrying the digests that
   bind the report to an exact world;
5. **footers** — authorization status, then the read-only footer, then
   `request <id> | world <digest>`.

Long tables truncate at 20 rows in the terminal with an explicit
`(and N more ...)` note — never silently.

## Ledger timelines

The execution ledger is the system's flight recorder; `gat ledger` renders
it as a timeline in the same grammar. Each event becomes one card titled
`seq - kind` (plus the operation name for transitions), carrying the
event's scalar content, its provenance, its world digests, and its
verification counts. Cards are accented by the vocabulary the event itself
recorded — an assessment's `verdict`, a policy's `disposition`, a failed
verification — via the standard palette, and a rejection event is accented
`stop`. The chain card at the end shows format, event count, head hash,
and integrity.

Two fail-closed rules are specific to ledgers: a tampered or broken hash
chain is refused with its reason, never drawn (`read_ledger` validates the
complete chain before rendering begins); and timelines longer than 50
events elide the middle explicitly — first 5, a marked gap, last 45 —
never silently.

## Projection instruments and Workbench modes

The platform synthesis names a projection triad — kepler.gl for analytical
geography (*where is the pattern?*), CesiumJS for geodetic reality (*where
does it exist?*), Three.js for computational structure (*how is it
constituted?*) — behind Workbench modes: MAP, GLOBE, STRUCTURE, GRAPH,
STATE, TIME, EVIDENCE, COMPLEXITY. The surfaces in this repository are the
first instances of those modes over one engine, and should be read as such:

| Mode | Surface today | Engine role it fills |
|---|---|---|
| STRUCTURE | `gat view` (belief ellipsoids, realizations, decision overlay) | the Three.js seat, currently a self-contained WebGL renderer |
| GRAPH | `gat workbench` GRAPH panel (IR relationship graph, typed edges, IFC provenance) | instrument |
| STATE | `gat workbench` STATE panel (per-entity quantities: mean ± sigma, raw / derived) | instrument |
| EVIDENCE | `gat report` pages (decision, evidence, assurance cards) | report class |
| TIME | `gat ledger` timeline | report class |
| COMPLEXITY | `gat audit --html` (what the corpus can and cannot represent) | report class |
| MAP / GLOBE | reserved; declared *unavailable* with the reason (no coordinate reference system is lowered) | kepler.gl / CesiumJS seats, not yet filled |

`gat workbench` composes all eight behind one mode toolbar in one offline
file, with a single selection identity (`EntityId`) shared by every mode
and a `ProjectionSpec` per mode that states its source, transformation,
meaning, information loss, identity, frame and time. The contract is
`docs/projection-spec-v1.md`.

Three rules carry across every mode: projection never mutates its source;
identity survives representation (the same digest names the same world in
every view); and visual adjacency is never evidence — a layout, a colour,
or a proximity on screen proposes nothing to the corpus. The "knowledge
debugger" intent — make missing provenance, unresolved identity, and
contradiction *visible* rather than hidden — is already the reason
assurance flags render `no` in plain sight and audit statuses like
`NEEDS_GEOMETRY_DERIVATION` are accented, not filtered out.

## Surfaces

* `gat report response.json` — terminal rendering of any of the four
  headless operations (`summary`, `acceptance`, `beam_assurance`,
  `change_impact`) or of a headless error response. Exit codes: 0 decision
  rendered, 1 rendered error response, 2 invalid input, 3 I/O error.
* `gat report response.json --html -o report.html` — the same content as a
  self-contained, script-free HTML page for sharing and archiving, e.g.
  `gat-headless request.json | gat report - --html -o decision.html`.
* `gat ledger ledger.json [--html] [-o PATH]` — the execution-ledger
  timeline described above. Exit codes: 0 rendered timeline, 2 invalid or
  tampered chain, 3 I/O error.
* `gat audit model.ifc --html` — the fail-closed compatibility inventory
  in the report grammar: pipeline stages, supported products, and issues
  accented by their own statuses, plus the audit's honest assurance card.
  A saved audit JSON also renders through `gat report audit.json`, and a
  readiness claim that contradicts its stage statuses is refused.
* Blender panel — loads the same response files; colours Bonsai objects by
  signal class; never mutates IFC state. For beam responses it shows the
  conditioning evidence (certificate, issuer with trust status, observed
  value) and the assurance flags verbatim — `issuer_trust_verified no`
  stays visible, exactly as in the HTML report.
* `gat view model.ifc -o viewer.html [--variations N]` — the offline 3D
  instrument: the nominal belief plus `N` belief-sampled as-built
  realizations, each splat drawn as its k-sigma ellipsoid (slider from
  0.5 to 3 sigma; the default 1.75 ≈ √3 sigma is where moment-matched
  tiles exactly fill their boxes), orbit/zoom/pan, per-class visibility,
  and one chip per realization whose dot carries the signal palette for
  its invariant verdict. Element hues are identity colours (muted
  architectural neutrals), never signal colours; the geometry on screen
  *is* the uncertainty, so no synthetic "error bars" are drawn. Clicking
  an element inspects it — name, class, and each extent as realized in the
  current sample beside its nominal mean ± sigma, with fixed (non-belief)
  axes labelled as such — and outlines its realized box in ink, always
  visible. Under a sampled realization the nominal belief is ghosted in
  grey so drift is legible; preset views (iso, top, front, side, reset)
  keep orientation cheap.
* `gat view … --decision response.json [--request request.json]` — the
  decision overlay. The response is bound to the loaded model fail-closed:
  its world digest (for beam assurance, its prior-world digest) must equal
  the model's, and a request must carry the same request id and
  operation. Elements the case could not clear at its confidence
  (`P(violation) > 1 − confidence`), the assessed beam, or a change's
  impacted entities are painted with the disposition colour — the one
  place signal colour *fills* geometry, because it *is* a verdict (audit
  statuses may *outline* a piece, see below, and never fill it). The
  request's proposed clearance boxes are drawn as always-visible wireframes
  in the same colour, so a REJECT is seen at the exact spot it was
  decided. Because the engine's world digest currently carries the model's
  source path string, `gat view` loads the model through the request's own
  path form when it names the same file; a digest mismatch is refused with
  that hint rather than silently re-bound. The HUD card repeats the report's headline badge, reasons,
  risk table, next evidence, and footers.
* `gat view … [--audit]` — the EXPLODE view and audit outlines. The
  explode slider (or `x`) displaces every piece along the IR hierarchy:
  perimeter elements move away from the plan centroid, an opening travels
  with the wall it voids and a door with the opening it fills, each a step
  further out and higher, and spaces lift. Leader lines tie each piece back
  to its assembled place, picking follows the displaced piece, and the
  inspection card states the displacement as "drawn N m from its place for
  reading; not a position" — the offsets are a reading order derived from
  the relationship graph, never geometry. With `--audit` (or inside the
  workbench when the audit is bound) each piece carries its IFC audit
  status, bound by GlobalId and refused if the vocabulary is unknown; a
  piece the corpus could not fully represent (`NEEDS_GEOMETRY_DERIVATION`,
  `MISSING_SOURCE_DATA`, `BLOCKED`) is *outlined* in its status colour
  while its fill keeps the identity hue, because an audit status describes
  the corpus, not a verdict on the asset. `READY` pieces carry no outline.
  The meta line states the frame the scene is drawn in (id, units, up
  axis, CRS or none, and the engine's placement-uncertainty limit) from
  the stated frame record; see `docs/projection-spec-v1.md`, *Frames*.
* `gat workbench model.ifc -o workbench.html [--decision … --request … --ledger … --no-audit]`
  — the Notation Workbench: the eight modes above behind one toolbar
  (keys 1–8), the viewer embedded in a sandboxed frame as STRUCTURE, the
  relationship graph as GRAPH, the belief per entity as STATE, and the
  report pages as TIME / EVIDENCE / COMPLEXITY. Selecting an element in
  any mode selects it everywhere by `EntityId`; the identity strip shows
  name and id, the URL hash carries `#MODE/EntityId`, and report panels
  mark exact-name mentions of the selection. Modes with nothing bound are
  shown as *empty* with the flag that fills them; MAP and GLOBE are shown
  as *unavailable* with the reason. Nothing is hidden and nothing is
  faked. Frame and page exchange only ids and the world digest
  (`gat-workbench-message-v1`); a message from another world is ignored.
  `python -m gat.demo.workbench out/` builds the worked clearance review
  (crossing duct vs `Wall-Party`) end to end and asserts the page.

## Non-goals

The report layer does not add new judgements (no derived scores, grades,
or traffic-light aggregations beyond the engine's own dispositions), does
not re-run any computation, and does not render responses whose internal
identities disagree — those are refused with the reason, exactly as the
Blender bridge refuses them.
