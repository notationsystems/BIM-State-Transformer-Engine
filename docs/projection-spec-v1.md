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
| STRUCTURE | Three.js seat — computational structure | How is it constituted? | instrument | `gat view` embedded (self-contained WebGL renderer), with the EXPLODE reading offset and per-piece audit outlines |
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

## Exploded views are reading offsets

STRUCTURE can pull the asset apart. The displacement of each piece is
derived from the relationship graph (radially from the plan centroid; an
opening with the wall it voids, a door with the opening it fills, one step
further each; spaces lifted), scaled by a slider, and drawn with leader
lines back to the assembled place. It is declared in the mode's
`transformation` and `loss`, stated on the inspection card ("drawn N m from
its place for reading; not a position"), and never written anywhere: the
scene, the world and the carrier are untouched. The same rule would hold
for an OpenUSD expression of the exploded layout — a variant or
time-sampled transforms over the derived view, never over `/GAT/State`.

Audit statuses ride along per piece, bound by GlobalId from a
`gat-ifc-audit-v1` document and refused if the vocabulary is unknown. A
piece the corpus could not fully represent is outlined in its status
colour; its fill keeps the identity hue, because an audit status describes
the corpus, not a verdict on the asset.

## Frames: stated today, read tomorrow

Every projection draws in a frame it states. `frame_record(world)` reads
what the adapter recorded — the IFC length unit and its scale to metres,
the placement convention of the lowering (corner-origin box, yaw about
+Z) — and adds the viewer's own convention (right-handed, Z up, metres,
the same the OpenUSD carrier declares), the absence of a CRS, and the
engine's current limit: **placements are exact metadata, so the belief
carries dimensions only**. The record is embedded in the scene, shown in
the viewer's meta line and the workbench identity strip, and is the source
of the `frame` field of the STRUCTURE and STATE specs. A frame change on
the display side is never evidence: the tests assert that computing frame
records and reading offsets in other frames leaves the world digest
untouched.

Projections must behave consistently under a change of frame, and the
frontend tests that on its own layer: EXPLODE offsets are invariant under
translation, rotate with the scene, and scale with the unit; the box
centre the viewer uses agrees with the engine's. GRAPH is frame-free by
construction. These are the frontend's share of the coordinate-
transformation equivalence tests; the engine's share (nested placements,
pose uncertainty, physical-result equivalence within a declared tolerance)
is the milestone the engine team owns.

### What the viewer will read when the engine carries frames and pose

This is a consumer's statement of the fields the surfaces will draw, not a
design of the engine's representation. It exists so the two can be shaped
together.

| Record | Fields the surfaces read | How it will be drawn |
|---|---|---|
| frame | `id`, `parent`, `transform` (rigid, declared convention), `units`, `up`, `handedness`, `crs`, `epoch` | identity strip and meta line; MAP/GLOBE availability flips only when `crs` is present |
| pose belief per element | position mean and 3×3 covariance in the parent frame; a rotation uncertainty in a representation that respects rotation geometry (e.g. a yaw variance for gravity-aligned cases), with correlations to dimensions where the engine carries them | a position ellipsoid at the box centre and a yaw fan, distinct from the dimensional ellipsoids; the inspection card separates "loose in place" from "loose in size" |
| residual | observation id, `VarId` or pose component, predicted mean ± sigma, measured value ± sigma, standardised residual, calibration id, frame id, independence flag | a CALIBRATION report: coverage per quantity class and frame, standardised-residual table, and markers on the pieces in STRUCTURE; a residual from a scan aligned to the model is drawn as association, never as independent evidence |

Nothing above is rendered until the engine emits it; the surfaces will
refuse a pose or residual record whose frame id they cannot resolve.

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
