# ProjectionSpec v1 and the Notation Workbench

`gat workbench` composes the human surfaces of this repository into one
offline instrument with eight projection modes. This document is the
contract behind it: what a *projection* is allowed to claim, how a mode
declares itself, how modes share one identity, and what the frontend seats
that are not yet filled (kepler.gl, CesiumJS) will have to read. The code
is `gat/workbench.py`; the design language it inherits is
`docs/design-language-v1.md`.

## The requirement

> Every representation identifies its source, its transformation, its
> supported meaning, and its information loss.

A building element appears as an IFC entity, as scan geometry, as a
computational variable, and as a selected object on a screen. A visually
convincing correspondence between those is not evidence that they describe
the same subject at compatible times and in compatible coordinate frames.
So every Workbench mode carries a `ProjectionSpec` that states, verbatim on
the page, what it projects and what it drops — and the instrument states
which modes it *cannot* fill and why, instead of faking them.

## The triad and the modes

| Mode | Seat | Question | Surface class | Today |
|---|---|---|---|---|
| MAP | kepler.gl — analytical geography | Where is the pattern? | instrument | **unavailable** — no coordinate reference system is lowered into the IR |
| GLOBE | CesiumJS — geodetic reality | Where does it exist? | connected instrument (not yet defined) | **unavailable** — no geodetic datum; tiles would need the network |
| STRUCTURE | Three.js seat — computational structure | How is it constituted? | instrument | `gat view` embedded (self-contained WebGL renderer) |
| GRAPH | IR relationship graph `G` | What relates to what, on whose authority? | instrument | deterministic reading-order layout, typed edges, IFC source references |
| STATE | IR entities `X` over `N(mu, Sigma)` | What is believed, and how surely? | instrument | per-entity quantities: mean ± sigma, raw or derived, provenance |
| TIME | execution ledger | What happened, in what order, did the chain hold? | report | `gat ledger` timeline, chain verified before drawing |
| EVIDENCE | decision report | What was decided, on what evidence, what is missing? | report | `gat report` of the bound `gat-headless` response |
| COMPLEXITY | IFC compatibility audit | What can this corpus represent, and what not? | report | `gat audit` of the source file |

Modes are numbered 1–8 in this order on the toolbar and on the keyboard.

## ProjectionSpec fields (`gat-projection-spec-v1`)

| Field | Meaning |
|---|---|
| `mode` | one of the eight names above |
| `seat` | which projection library or engine artifact fills the mode |
| `question` | the one question the mode answers |
| `surface_class` | `report` (script-free, inert), `instrument` (self-contained, offline, inline scripts), or `connected instrument` (would fetch external resources — **not yet defined** as a class; naming it here is the honest placeholder) |
| `source` | the engine artifact the projection reads |
| `transformation` | what the mode does to its source to draw it |
| `meaning` | what the picture may be read as |
| `loss` | what the projection drops or approximates |
| `identity` | how subjects are named in this mode (`EntityId`, `VarId`, request id, event hash, world digest) |
| `frame` | the coordinate frame or unit system, or `none` |
| `time` | which state in time the mode shows (one world digest, ledger sequence, file version) |
| `availability` | `available`, `empty`, or `unavailable` (below) |
| `reason` | for `empty` and `unavailable`: why, and what would fill the mode |
| `mutates_source` | always `false` |

The specs are embedded in the page twice: as a disclosure strip at the top
of each mode, and in full as JSON in the footer.

## Availability

* **available** — the mode has a source and renders it.
* **empty** — the mode exists for this corpus, but nothing is bound to it in
  this document (no `--ledger`, no `--decision`, `--no-audit`). The panel
  says exactly which flag or artifact fills it. The toolbar keeps the mode
  visible with an `empty` mark; it is never hidden.
* **unavailable** — the corpus cannot fill the mode. MAP and GLOBE are
  unavailable in this release because no IfcSite placement, IfcMapConversion
  or coordinate reference system reaches the IR; an invented position would
  be visual adjacency presented as evidence. Whether and how those
  semantics are lowered is an engine decision; this document only names
  what the seats will read once they exist (`frame` with a CRS, `time` with
  a survey epoch, and for GLOBE a defined connected-instrument class with
  declared tile sources).

## Identity across modes

One selection is shared by every mode. It is an `EntityId`
(`IfcClass:GlobalId`) — never a name, never a position, never an index into
a mode's own arrays. The identity strip shows the name *and* the id.

* GRAPH nodes, STATE list entries and STRUCTURE elements all carry the
  `EntityId`; selecting in any one selects in all.
* The page and the embedded viewer share one world digest, and every
  message between them carries it; a message from a different world is
  ignored, not reconciled.
* Report panels (TIME, EVIDENCE, COMPLEXITY) mark exact-name mentions of
  the selected entity so the reader sees where the identity appears in the
  evidence. Because `gat-headless` responses name subjects by entity *name*,
  a name shared by several entities is an ambiguous identity: nothing is
  marked and the strip says so. Carrying `EntityId`s in responses would
  remove the ambiguity — an engine contract, noted here rather than worked
  around.
* The URL hash carries `#MODE/EntityId`, so a view can be shared and
  restored by identity.

## Message contract (`gat-workbench-message-v1`)

The STRUCTURE viewer runs in a sandboxed `srcdoc` frame (`allow-scripts`
only; opaque origin). Messages are the only channel between it and the
page; there is no DOM access in either direction.

| Direction | `kind` | Fields | Meaning |
|---|---|---|---|
| viewer → page | `ready` | `world_digest` | the frame has booted; the page replays its current selection |
| viewer → page | `selection` | `world_digest`, `entity` or `null`, `name` | the user selected (or cleared) an element in the viewer |
| page → viewer | `select` | `world_digest`, `entity` or `null` | select (or clear) this identity in the viewer, quietly |

Every message carries `format: "gat-workbench-message-v1"`. Receivers check
the format, the source window, and the world digest before acting, and
drop anything else silently. No message mutates state on either side:
selection is a view property, not a model property.

## Rules that hold in every mode

1. **Projection never mutates its source.** The workbench renders and
   re-checks; it never writes. Fail-closed rules from the report layer
   apply unchanged: a decision from another world is refused, a tampered
   ledger is refused before drawing, an audit whose readiness contradicts
   its stages is refused.
2. **Identity survives representation.** The same `EntityId` and the same
   world digest name the same thing in every mode and across the frame
   boundary.
3. **Visual adjacency is never evidence.** The GRAPH layout is a reading
   order by IFC class rank and says so on the panel; distances on the
   canvas carry no information. Nothing on any panel proposes anything to
   the corpus.

## What the frontend contributes to industrial gates

Of the five readiness gates — representation fidelity, computational
validity, uncertainty calibration, operational reliability, workflow
validation — the frontend can only help with the first and the last, and
only partly:

* *Representation fidelity*: the identity contract above is tested
  (`tests/test_workbench.py::StatePayloadTests::test_identity_survives_representation`),
  and every `ProjectionSpec` declares its loss, frame and time so that a
  reader can tell whether two representations are even comparable.
* *Workflow validation*: the instrument lets a practitioner see the
  decision at the spot it was decided (STRUCTURE), the evidence it rests on
  and the evidence still requested (EVIDENCE), the history (TIME) and the
  corpus limits (COMPLEXITY) without leaving one file. Whether people use
  it correctly is a field question this document cannot answer.

## Non-goals

The workbench adds no judgement of its own: no derived scores, no
aggregated traffic lights, no inferred correspondences between modes. It
does not fetch anything. It does not implement the kepler.gl or CesiumJS
seats; it reserves their modes and states what they would need.
