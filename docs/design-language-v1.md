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
5. **Reports are inert artifacts.** HTML reports are self-contained,
   carry no scripts and fetch nothing; interactivity is native disclosure
   (`<details>`) only. A decision report may be attached to an RFI or
   archived without changing meaning or behaviour.
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
| proceed | `ACCEPT`, `SATISFIED`, `ADMISSIBLE`, `PASS` | (0.10, 0.70, 0.20, 1.0) | `#1ab233` |
| stop | `REJECT`, `VIOLATED`, `BLOCKED`, `FAIL` | (0.85, 0.08, 0.08, 1.0) | `#d91414` |
| attention | `REQUEST_EVIDENCE`, `UNRESOLVED`, `WARN` | (0.95, 0.55, 0.05, 1.0) | `#f28c0d` |
| undecided | `ERROR` | (0.35, 0.35, 0.35, 1.0) | `#595959` |

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

## Surfaces

* `gat report response.json` — terminal rendering of any of the four
  headless operations (`summary`, `acceptance`, `beam_assurance`,
  `change_impact`) or of a headless error response. Exit codes: 0 decision
  rendered, 1 rendered error response, 2 invalid input, 3 I/O error.
* `gat report response.json --html -o report.html` — the same content as a
  self-contained, script-free HTML page for sharing and archiving, e.g.
  `gat-headless request.json | gat report - --html -o decision.html`.
* Blender panel — loads the same response files; colours Bonsai objects by
  signal class; never mutates IFC state.

## Non-goals

The report layer does not add new judgements (no derived scores, grades,
or traffic-light aggregations beyond the engine's own dispositions), does
not re-run any computation, and does not render responses whose internal
identities disagree — those are refused with the reason, exactly as the
Blender bridge refuses them.
