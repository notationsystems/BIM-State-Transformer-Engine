"""Self-contained 3D viewer for the architectural belief and its samples.

``export_viewer_html`` writes one offline HTML file containing a small
WebGL renderer plus the derived Gaussian scene of the nominal belief and
``n`` belief-sampled as-built realizations.  Each splat is drawn as its
1-sigma ellipsoid (adjustable to k-sigma), so the geometry a viewer orbits
*is* the uncertainty: walls thicken where the belief is loose and sharpen
where evidence has tightened it.

Surface class: this is an *instrument*, not a decision report.  It carries
its own scripts (inline, self-contained, no network access, no external
resources) — decision reports remain script-free.  The signal palette is
reserved for verification verdicts (each sample's PASS/FAIL chip); element
hues are identity colours only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from gat.engine.executor import World
from gat.engine.sampling import sample_worlds
from gat.engine.verify import Status, run_invariants
from gat.geometry.stateio import derive_scene
from gat.report import decode_response, disposition_hex

VIEWER_SCENE_FORMAT = "gat-viewer-scene-v1"
REQUEST_FORMAT = "gat-headless-request-v1"
_OVERLAY_OPERATIONS = frozenset({"acceptance", "beam_assurance", "change_impact"})

#: Identity hues per IFC class (muted architectural neutrals) and the
#: alpha its splats render with; unknown classes fall back to "other".
CLASS_STYLES: dict[str, tuple[str, float]] = {
    "IfcWall": ("#8a8578", 0.92),
    "IfcDoor": ("#b0763f", 0.95),
    "IfcOpeningElement": ("#c9c2b4", 0.55),
    "IfcSpace": ("#7f9db5", 0.14),
    "IfcBuildingStorey": ("#6b6a66", 0.85),
    "other": ("#9c9285", 0.85),
}


def _sample_entry(world: World, label: str, spacing: float) -> dict[str, object]:
    scene = derive_scene(world, spacing=spacing)
    cloud = scene.cloud
    covs = np.asarray(cloud.covs, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(covs)
    eigenvalues = np.clip(eigenvalues, 1e-12, None)
    # Column i of axes is (unit eigenvector i) * sigma_i: the ellipsoid map.
    axes = eigenvectors * np.sqrt(eigenvalues)[:, None, :]
    report = run_invariants(world)
    failures = [
        f"{result.invariant_id} [{result.subject}]"
        for result in report.results
        if result.status is Status.FAIL
    ]
    return {
        "label": label,
        "passed": report.passed,
        "failures": failures,
        "world_digest": world.digest(),
        "element": [int(index) for index in cloud.element_index],
        "centers": [round(float(value), 4) for value in cloud.means.ravel()],
        "axes": [round(float(value), 4) for value in axes.ravel()],
    }


def decision_overlay(
    world: World,
    response: Mapping[str, object],
    request: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Bind a headless decision to this world for drawing, fail-closed.

    The response must have been evaluated on exactly the loaded world (its
    world digest, or for beam assurance its prior-world digest, must equal
    ``world.digest()``); an optional request must carry the same request id
    and operation, and contributes the proposed geometry the response does
    not echo.  Anything that disagrees is refused, never drawn.
    """
    report = decode_response(response)
    if report.operation not in _OVERLAY_OPERATIONS:
        raise ValueError(
            f"viewer overlays render decisions, not {report.operation!r} documents"
        )
    result = response["result"]
    loaded = world.digest()
    evaluated = {report.world_digest}
    if report.operation == "beam_assurance":
        evaluated.add(result["transition"]["prior_world_digest"])
    if loaded not in evaluated:
        raise ValueError("decision was evaluated on a different world than the model")

    subjects: list[str] = []
    risks: list[dict[str, object]] = []
    requests: list[dict[str, str]] = []
    if report.operation == "acceptance":
        for check in result["checks"]:
            # Only elements the case could not clear at its confidence are
            # painted with the disposition; the rest stay identity-coloured.
            uncleared_above = 1.0 - float(check["confidence"])
            for risk in check.get("details", {}).get("risks", []):
                element = str(risk["element"])
                if float(risk["p_violates"]) > uncleared_above and element not in subjects:
                    subjects.append(element)
                risks.append(
                    {
                        "check_id": check["check_id"],
                        "element": element,
                        "clearance_mean": float(risk["clearance_mean"]),
                        "clearance_sigma": float(risk["clearance_sigma"]),
                        "p_violates": float(risk["p_violates"]),
                    }
                )
        requests = [
            {"action": str(item["action"]), "target": str(item["target"])}
            for item in result["evidence_requests"]
        ]
    elif report.operation == "beam_assurance":
        subjects = [str(result["subject"])]
    else:
        subjects = [str(name) for name in result["impacted_entities"]]

    proposals: list[dict[str, object]] = []
    if request is not None:
        if request.get("format") != REQUEST_FORMAT:
            raise ValueError(f"unsupported request format {request.get('format')!r}")
        if request.get("request_id") != report.request_id:
            raise ValueError("request and response ids differ")
        if request.get("operation") != report.operation:
            raise ValueError("request and response operations differ")
        payload = request.get("payload", {})
        for check in payload.get("checks", []) if isinstance(payload, Mapping) else []:
            if check.get("kind") != "clearance":
                continue
            proposal = check["proposal"]
            proposals.append(
                {
                    "label": str(check.get("label") or check["check_id"]),
                    "origin": [float(v) for v in proposal["origin"]],
                    "angle": float(proposal["angle"]),
                    "extents": [float(v) for v in proposal["extents"]],
                }
            )

    return {
        "headline": report.headline,
        "disposition": report.disposition,
        "color": disposition_hex(report.disposition),
        "subline": report.subline,
        "reasons": list(report.notes),
        "requests": requests,
        "subjects": subjects,
        "risks": risks,
        "proposals": proposals,
        "footers": list(report.footers),
    }


def viewer_payload(
    world: World,
    n: int = 8,
    seed: int = 0,
    spacing: float = 0.75,
    model_name: str = "",
    decision: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the embedded scene document: nominal belief + ``n`` samples,
    optionally carrying one bound decision overlay.

    Deterministic: same world + n + seed + spacing => identical payload.
    """
    if n < 0:
        raise ValueError("variation count must be non-negative")
    scene = derive_scene(world, spacing=spacing)
    classes: list[str] = []
    elements = []
    for element in scene.elements:
        ifc_class = element.entity_id.ifc_class
        if ifc_class not in classes:
            classes.append(ifc_class)
        style = CLASS_STYLES.get(ifc_class, CLASS_STYLES["other"])
        elements.append(
            {
                "name": element.name,
                "class": classes.index(ifc_class),
                "color": style[0],
                "alpha": style[1],
                "solid": element.is_solid,
            }
        )
    samples = [_sample_entry(world, "nominal", spacing)]
    samples.extend(
        _sample_entry(sampled, f"sample {index + 1}", spacing)
        for index, sampled in enumerate(sample_worlds(world, n, seed))
    )
    return {
        "format": VIEWER_SCENE_FORMAT,
        "model": model_name,
        "world_digest": world.digest(),
        "seed": seed,
        "spacing": spacing,
        "classes": classes,
        "elements": elements,
        "samples": samples,
        "decision": dict(decision) if decision is not None else None,
    }


def export_viewer_html(
    world: World,
    path: str | Path,
    n: int = 8,
    seed: int = 0,
    spacing: float = 0.75,
    model_name: str = "",
    decision: Mapping[str, object] | None = None,
) -> int:
    """Write the viewer HTML; returns the number of embedded samples."""
    payload = viewer_payload(
        world,
        n=n,
        seed=seed,
        spacing=spacing,
        model_name=model_name,
        decision=decision,
    )
    encoded = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    document = _TEMPLATE.replace("__GAT_SCENE_JSON__", encoded)
    Path(path).write_text(document, encoding="utf-8")
    return len(payload["samples"])


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GAT as-built viewer</title>
<style>
html, body { margin: 0; height: 100%; overflow: hidden; background: #f5f4f1;
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; color: #1c1c1a; }
canvas { display: block; width: 100vw; height: 100vh; touch-action: none; }
#hud { position: fixed; top: 1rem; left: 1rem; width: 17rem; background: #fff;
  border-radius: 10px; box-shadow: 0 1px 4px rgba(0,0,0,0.12);
  padding: 0.8rem 1rem; user-select: none; }
#hud h1 { margin: 0; font-size: 1rem; }
#hud .meta { color: #6b6a66; font-size: 0.8rem; margin: 0.15rem 0 0.6rem;
  overflow-wrap: anywhere; }
#hud h2 { margin: 0.7rem 0 0.3rem; font-size: 0.7rem; text-transform: uppercase;
  letter-spacing: 0.08em; color: #6b6a66; }
#samples { display: flex; flex-wrap: wrap; gap: 0.3rem; }
#samples button { border: 1px solid #d8d5cd; background: #fff; border-radius: 999px;
  padding: 0.1rem 0.6rem; font: inherit; font-size: 0.8rem; cursor: pointer; }
#samples button .dot { display: inline-block; width: 0.55em; height: 0.55em;
  border-radius: 50%; margin-right: 0.35em; }
#samples button.active { background: #1c1c1a; color: #fff; border-color: #1c1c1a; }
#failures { color: #d91414; font-size: 0.78rem; margin: 0.3rem 0 0; }
label.cls { display: block; font-size: 0.85rem; }
label.cls .swatch { display: inline-block; width: 0.7em; height: 0.7em;
  border-radius: 3px; margin: 0 0.4em 0 0.2em; }
#sigma { width: 100%; }
#hud footer { margin-top: 0.7rem; color: #6b6a66; font-size: 0.72rem; }
#hud footer p { margin: 0.15rem 0; }
#decision { border-left: 4px solid #999; padding: 0.4rem 0.6rem; margin: 0.4rem 0 0.2rem;
  background: #faf9f6; border-radius: 0 8px 8px 0; font-size: 0.82rem; }
#decision .badge { display: inline-block; color: #fff; border-radius: 999px;
  padding: 0 0.55em; font-weight: 600; margin-right: 0.35em; }
#decision p { margin: 0.25rem 0; }
#decision table { border-collapse: collapse; width: 100%; font-size: 0.76rem; margin-top: 0.3rem; }
#decision td { padding: 0.1rem 0.4rem 0.1rem 0; border-bottom: 1px solid #ece9e2; vertical-align: top; }
#decision label { display: block; margin-top: 0.35rem; font-size: 0.78rem; }
</style></head><body>
<canvas id="gl"></canvas>
<div id="hud">
  <h1>GAT as-built viewer</h1>
  <div class="meta" id="meta"></div>
  <div id="decision" hidden></div>
  <h2>Realization</h2>
  <div id="samples"></div>
  <div id="failures"></div>
  <h2>Uncertainty envelope <span id="sigmaValue">1.8&sigma;</span></h2>
  <input id="sigma" type="range" min="0.5" max="3" step="0.25" value="1.75">
  <h2>Elements</h2>
  <div id="classes"></div>
  <footer>
    <p>orbit: drag &middot; zoom: wheel &middot; pan: shift-drag &middot; &larr;/&rarr; realization</p>
    <p>Read-only: no BIM state was changed.</p>
  </footer>
</div>
<script id="scene" type="application/json">__GAT_SCENE_JSON__</script>
<script>
"use strict";
const SCENE = JSON.parse(document.getElementById("scene").textContent);
const PASS_COLOR = "#1ab233", FAIL_COLOR = "#d91414";

// -- tiny matrix math ------------------------------------------------------
function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
  return [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1,
          0, 0, 2 * far * near * nf, 0];
}
function lookAt(eye, target, up) {
  const z = norm3(sub3(eye, target)), x = norm3(cross3(up, z)), y = cross3(z, x);
  return [x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0,
          -dot3(x, eye), -dot3(y, eye), -dot3(z, eye), 1];
}
function sub3(a, b) { return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }
function dot3(a, b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
function cross3(a, b) { return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }
function norm3(a) { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0]/l, a[1]/l, a[2]/l]; }

// -- unit sphere (octahedron subdivided once, non-indexed) -----------------
function unitSphere() {
  const p = [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]];
  const faces = [[0,2,4],[2,1,4],[1,3,4],[3,0,4],[2,0,5],[1,2,5],[3,1,5],[0,3,5]];
  const out = [];
  const mid = (a, b) => norm3([(a[0]+b[0])/2, (a[1]+b[1])/2, (a[2]+b[2])/2]);
  for (const [ia, ib, ic] of faces) {
    const a = p[ia], b = p[ib], c = p[ic];
    const ab = mid(a, b), bc = mid(b, c), ca = mid(c, a);
    for (const tri of [[a,ab,ca],[ab,b,bc],[ca,bc,c],[ab,bc,ca]]) out.push(...tri);
  }
  return out; // 96 unit vertices
}
const SPHERE = unitSphere();

// -- WebGL setup -----------------------------------------------------------
const canvas = document.getElementById("gl");
const gl = canvas.getContext("webgl", { antialias: true, alpha: false });
function shader(type, source) {
  const s = gl.createShader(type);
  gl.shaderSource(s, source); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS))
    throw new Error(gl.getShaderInfoLog(s));
  return s;
}
function program(vs, fs) {
  const p = gl.createProgram();
  gl.attachShader(p, shader(gl.VERTEX_SHADER, vs));
  gl.attachShader(p, shader(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS))
    throw new Error(gl.getProgramInfoLog(p));
  return p;
}
const SPLAT_PROGRAM = program(`
  attribute vec3 aCenter, aOffset, aNormal; attribute vec4 aColor;
  uniform mat4 uProj, uView; uniform float uSigma;
  varying vec3 vNormal; varying vec4 vColor;
  void main() {
    vec3 world = aCenter + uSigma * aOffset;
    gl_Position = uProj * uView * vec4(world, 1.0);
    vNormal = aNormal; vColor = aColor;
  }`, `
  precision mediump float;
  varying vec3 vNormal; varying vec4 vColor;
  void main() {
    vec3 n = normalize(vNormal);
    float key = max(dot(n, normalize(vec3(0.5, 0.35, 0.8))), 0.0);
    float fill = max(dot(n, normalize(vec3(-0.4, -0.2, 0.4))), 0.0);
    vec3 shaded = vColor.rgb * (0.42 + 0.5 * key + 0.18 * fill);
    gl_FragColor = vec4(shaded, vColor.a);
  }`);
const LINE_PROGRAM = program(`
  attribute vec3 aPosition; uniform mat4 uProj, uView;
  void main() { gl_Position = uProj * uView * vec4(aPosition, 1.0); }`, `
  precision mediump float; uniform vec4 uColor;
  void main() { gl_FragColor = uColor; }`);

function hexToRgb(hex) {
  return [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
}

// -- decision overlay (bound server-side to this exact world) --------------
const DECISION = SCENE.decision || null;
const DECISION_SUBJECTS = new Set(DECISION ? DECISION.subjects : []);
const DECISION_RGB = DECISION ? hexToRgb(DECISION.color) : null;

// -- per-sample geometry (built lazily, grouped by element class) ----------
const FLOATS = 13; // center 3 + offset 3 + normal 3 + color 4
const built = new Map();
function buildSample(index) {
  const overlay = DECISION !== null && state.overlay;
  const key = index + (overlay ? ":decision" : "");
  if (built.has(key)) return built.get(key);
  const sample = SCENE.samples[index];
  const count = sample.element.length;
  const order = Array.from({ length: count }, (_, i) => i);
  const classOf = (i) => SCENE.elements[sample.element[i]].class;
  const tinted = (i) => overlay && DECISION_SUBJECTS.has(SCENE.elements[sample.element[i]].name);
  const alphaOf = (i) => tinted(i) ? 0.96 : SCENE.elements[sample.element[i]].alpha;
  order.sort((a, b) => (alphaOf(b) - alphaOf(a)) || (classOf(a) - classOf(b)) || (a - b));
  const data = new Float32Array(count * SPHERE.length * FLOATS);
  const ranges = []; // {class, transparent, start, count} in vertices
  let cursor = 0;
  for (const i of order) {
    const element = SCENE.elements[sample.element[i]];
    const rgb = tinted(i) ? DECISION_RGB : hexToRgb(element.color);
    const alpha = alphaOf(i);
    const transparent = alpha < 0.5;
    const c = sample.centers.slice(i * 3, i * 3 + 3);
    const A = sample.axes.slice(i * 9, i * 9 + 9); // row-major world map
    const s0 = Math.hypot(A[0], A[3], A[6]) || 1e-9;
    const s1 = Math.hypot(A[1], A[4], A[7]) || 1e-9;
    const s2 = Math.hypot(A[2], A[5], A[8]) || 1e-9;
    const last = ranges[ranges.length - 1];
    if (!last || last.class !== element.class || last.transparent !== transparent) {
      ranges.push({ class: element.class, transparent, start: cursor / FLOATS, count: 0 });
    }
    for (const u of SPHERE) {
      const off = [
        A[0]*u[0] + A[1]*u[1] + A[2]*u[2],
        A[3]*u[0] + A[4]*u[1] + A[5]*u[2],
        A[6]*u[0] + A[7]*u[1] + A[8]*u[2]];
      const n = norm3([
        (A[0]*u[0])/(s0*s0) + (A[1]*u[1])/(s1*s1) + (A[2]*u[2])/(s2*s2),
        (A[3]*u[0])/(s0*s0) + (A[4]*u[1])/(s1*s1) + (A[5]*u[2])/(s2*s2),
        (A[6]*u[0])/(s0*s0) + (A[7]*u[1])/(s1*s1) + (A[8]*u[2])/(s2*s2)]);
      data.set([c[0], c[1], c[2], off[0], off[1], off[2], n[0], n[1], n[2],
                rgb[0], rgb[1], rgb[2], alpha], cursor);
      cursor += FLOATS;
    }
    ranges[ranges.length - 1].count += SPHERE.length;
  }
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
  const entry = { buffer, ranges };
  built.set(key, entry);
  return entry;
}

// Proposed clearance geometry: corner-origin yawed boxes as 12 edges each.
const proposals = (() => {
  if (!DECISION || !DECISION.proposals.length) return null;
  const lines = [];
  for (const box of DECISION.proposals) {
    const c = Math.cos(box.angle), s = Math.sin(box.angle);
    const corner = (fx, fy, fz) => {
      const lx = fx * box.extents[0], ly = fy * box.extents[1], lz = fz * box.extents[2];
      return [box.origin[0] + c * lx - s * ly, box.origin[1] + s * lx + c * ly, box.origin[2] + lz];
    };
    const v = [corner(0,0,0), corner(1,0,0), corner(1,1,0), corner(0,1,0),
               corner(0,0,1), corner(1,0,1), corner(1,1,1), corner(0,1,1)];
    for (const [a, b] of [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]])
      lines.push(...v[a], ...v[b]);
  }
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(lines), gl.STATIC_DRAW);
  return { buffer, count: lines.length / 3 };
})();

// -- scene bounds and camera ----------------------------------------------
const nominal = SCENE.samples[0];
const bounds = { lo: [1e9, 1e9, 1e9], hi: [-1e9, -1e9, -1e9] };
for (let i = 0; i < nominal.centers.length; i += 3) {
  for (let axis = 0; axis < 3; axis += 1) {
    bounds.lo[axis] = Math.min(bounds.lo[axis], nominal.centers[i + axis]);
    bounds.hi[axis] = Math.max(bounds.hi[axis], nominal.centers[i + axis]);
  }
}
const target = [0, 1, 2].map((a) => (bounds.lo[a] + bounds.hi[a]) / 2);
const radius = Math.max(1, Math.hypot(
  bounds.hi[0] - bounds.lo[0], bounds.hi[1] - bounds.lo[1], bounds.hi[2] - bounds.lo[2]));
const camera = { yaw: 0.9, pitch: 0.5, dist: radius * 1.6, pan: [0, 0, 0] };

const grid = (() => {
  const lines = [];
  const z = bounds.lo[2] - 0.02;
  const lo = [Math.floor(bounds.lo[0] - 2), Math.floor(bounds.lo[1] - 2)];
  const hi = [Math.ceil(bounds.hi[0] + 2), Math.ceil(bounds.hi[1] + 2)];
  for (let x = lo[0]; x <= hi[0]; x += 1) lines.push(x, lo[1], z, x, hi[1], z);
  for (let y = lo[1]; y <= hi[1]; y += 1) lines.push(lo[0], y, z, hi[0], y, z);
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(lines), gl.STATIC_DRAW);
  return { buffer, count: lines.length / 3 };
})();

// -- state and HUD ---------------------------------------------------------
// sqrt(3) sigma is where moment-matched tiles exactly fill their boxes
// (Var(U[-h,h]) = h^2/3), so the default envelope reproduces the solids.
const state = {
  sample: 0,
  sigma: 1.75,
  overlay: true,
  hidden: new Set(SCENE.classes.map((name, i) => name === "IfcSpace" ? i : -1).filter((i) => i >= 0)),
};
const meta = document.getElementById("meta");
meta.textContent = (SCENE.model ? SCENE.model + " | " : "") +
  "world " + SCENE.world_digest.slice(0, 12) + "... | seed " + SCENE.seed;

if (DECISION) {
  const card = document.getElementById("decision");
  card.hidden = false;
  card.style.borderLeftColor = DECISION.color;
  const text = (value) => document.createTextNode(String(value));
  const headline = document.createElement("p");
  const badge = document.createElement("span");
  badge.className = "badge";
  badge.style.background = DECISION.color;
  badge.append(text(DECISION.disposition));
  headline.append(badge, text(DECISION.headline.replace(DECISION.disposition + ": ", "")));
  card.append(headline);
  const subline = document.createElement("p");
  subline.style.color = "#6b6a66";
  subline.append(text(DECISION.subline));
  card.append(subline);
  for (const reason of DECISION.reasons) {
    const p = document.createElement("p"); p.append(text(reason)); card.append(p);
  }
  if (DECISION.risks.length) {
    const table = document.createElement("table");
    for (const risk of DECISION.risks) {
      const row = document.createElement("tr");
      for (const cell of [risk.element,
                          risk.clearance_mean.toFixed(3) + " +- " + risk.clearance_sigma.toFixed(3) + " m",
                          "P(viol) " + risk.p_violates.toFixed(4)]) {
        const td = document.createElement("td"); td.append(text(cell)); row.append(td);
      }
      table.append(row);
    }
    card.append(table);
  }
  for (const request of DECISION.requests) {
    const p = document.createElement("p");
    p.append(text("next evidence: " + request.action + " " + request.target));
    card.append(p);
  }
  const toggle = document.createElement("label");
  const input = document.createElement("input");
  input.type = "checkbox"; input.checked = true;
  input.addEventListener("change", () => { state.overlay = input.checked; draw(); });
  toggle.append(input, text(" highlight decision subjects" +
    (DECISION.proposals.length ? " and proposed geometry" : "")));
  card.append(toggle);
  for (const footer of DECISION.footers) {
    const p = document.createElement("p"); p.style.color = "#6b6a66"; p.append(text(footer)); card.append(p);
  }
}

const samplesBox = document.getElementById("samples");
SCENE.samples.forEach((sample, index) => {
  const chip = document.createElement("button");
  chip.innerHTML = '<span class="dot"></span>' + sample.label;
  chip.querySelector(".dot").style.background = sample.passed ? PASS_COLOR : FAIL_COLOR;
  chip.title = sample.passed ? "passes invariants" : "fails invariants";
  chip.addEventListener("click", () => selectSample(index));
  samplesBox.appendChild(chip);
});
function selectSample(index) {
  state.sample = index;
  Array.from(samplesBox.children).forEach((chip, i) =>
    chip.classList.toggle("active", i === index));
  const sample = SCENE.samples[index];
  document.getElementById("failures").textContent =
    sample.failures.length ? "FAIL: " + sample.failures.join(", ") : "";
  draw();
}
const classesBox = document.getElementById("classes");
SCENE.classes.forEach((name, index) => {
  const label = document.createElement("label");
  label.className = "cls";
  const input = document.createElement("input");
  input.type = "checkbox";
  input.checked = !state.hidden.has(index);
  input.addEventListener("change", () => {
    if (input.checked) state.hidden.delete(index); else state.hidden.add(index);
    draw();
  });
  const swatch = document.createElement("span");
  swatch.className = "swatch";
  const styled = SCENE.elements.find((element) => element.class === index);
  swatch.style.background = styled ? styled.color : "#999";
  label.append(input, swatch, document.createTextNode(name));
  classesBox.appendChild(label);
});
const sigmaInput = document.getElementById("sigma");
sigmaInput.addEventListener("input", () => {
  state.sigma = parseFloat(sigmaInput.value);
  document.getElementById("sigmaValue").innerHTML = state.sigma.toFixed(1) + "&sigma;";
  draw();
});
window.addEventListener("keydown", (event) => {
  if (event.key === "ArrowRight")
    selectSample((state.sample + 1) % SCENE.samples.length);
  if (event.key === "ArrowLeft")
    selectSample((state.sample + SCENE.samples.length - 1) % SCENE.samples.length);
});

// -- input: orbit / pan / zoom ---------------------------------------------
let dragging = null;
canvas.addEventListener("pointerdown", (event) => {
  dragging = { x: event.clientX, y: event.clientY, pan: event.shiftKey || event.button === 2 };
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener("pointermove", (event) => {
  if (!dragging) return;
  const dx = event.clientX - dragging.x, dy = event.clientY - dragging.y;
  dragging.x = event.clientX; dragging.y = event.clientY;
  if (dragging.pan) {
    const scale = camera.dist * 0.0016;
    const sy = Math.sin(camera.yaw), cy = Math.cos(camera.yaw);
    camera.pan[0] -= (cy * dx) * scale; camera.pan[1] += (sy * dx) * scale;
    camera.pan[2] += dy * scale;
  } else {
    camera.yaw += dx * 0.008;
    camera.pitch = Math.min(1.45, Math.max(-0.2, camera.pitch + dy * 0.006));
  }
  draw();
});
canvas.addEventListener("pointerup", () => { dragging = null; });
canvas.addEventListener("contextmenu", (event) => event.preventDefault());
canvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  camera.dist *= Math.exp(event.deltaY * 0.0012);
  camera.dist = Math.min(radius * 8, Math.max(radius * 0.2, camera.dist));
  draw();
}, { passive: false });

// -- render ----------------------------------------------------------------
function bindSplatAttributes() {
  const stride = FLOATS * 4;
  const locations = ["aCenter", "aOffset", "aNormal", "aColor"].map(
    (name) => gl.getAttribLocation(SPLAT_PROGRAM, name));
  gl.vertexAttribPointer(locations[0], 3, gl.FLOAT, false, stride, 0);
  gl.vertexAttribPointer(locations[1], 3, gl.FLOAT, false, stride, 12);
  gl.vertexAttribPointer(locations[2], 3, gl.FLOAT, false, stride, 24);
  gl.vertexAttribPointer(locations[3], 4, gl.FLOAT, false, stride, 36);
  locations.forEach((location) => gl.enableVertexAttribArray(location));
}
function draw() {
  const width = canvas.clientWidth * (window.devicePixelRatio || 1);
  const height = canvas.clientHeight * (window.devicePixelRatio || 1);
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width; canvas.height = height;
  }
  gl.viewport(0, 0, width, height);
  gl.clearColor(0.961, 0.957, 0.945, 1);
  gl.enable(gl.DEPTH_TEST);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  const eye = [
    target[0] + camera.pan[0] + camera.dist * Math.cos(camera.pitch) * Math.sin(camera.yaw),
    target[1] + camera.pan[1] + camera.dist * Math.cos(camera.pitch) * Math.cos(camera.yaw),
    target[2] + camera.pan[2] + camera.dist * Math.sin(camera.pitch)];
  const focus = [target[0] + camera.pan[0], target[1] + camera.pan[1], target[2] + camera.pan[2]];
  const proj = perspective(0.9, canvas.width / canvas.height, 0.05, radius * 30);
  const view = lookAt(eye, focus, [0, 0, 1]);

  gl.useProgram(LINE_PROGRAM);
  gl.uniformMatrix4fv(gl.getUniformLocation(LINE_PROGRAM, "uProj"), false, proj);
  gl.uniformMatrix4fv(gl.getUniformLocation(LINE_PROGRAM, "uView"), false, view);
  gl.uniform4f(gl.getUniformLocation(LINE_PROGRAM, "uColor"), 0.85, 0.84, 0.81, 1);
  gl.bindBuffer(gl.ARRAY_BUFFER, grid.buffer);
  const linePosition = gl.getAttribLocation(LINE_PROGRAM, "aPosition");
  gl.vertexAttribPointer(linePosition, 3, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(linePosition);
  gl.drawArrays(gl.LINES, 0, grid.count);

  const { buffer, ranges } = buildSample(state.sample);
  gl.useProgram(SPLAT_PROGRAM);
  gl.uniformMatrix4fv(gl.getUniformLocation(SPLAT_PROGRAM, "uProj"), false, proj);
  gl.uniformMatrix4fv(gl.getUniformLocation(SPLAT_PROGRAM, "uView"), false, view);
  gl.uniform1f(gl.getUniformLocation(SPLAT_PROGRAM, "uSigma"), state.sigma);
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  bindSplatAttributes();
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  for (const pass of [false, true]) {
    gl.depthMask(!pass);
    for (const range of ranges) {
      if (range.transparent !== pass || state.hidden.has(range.class)) continue;
      gl.drawArrays(gl.TRIANGLES, range.start, range.count);
    }
  }
  gl.depthMask(true);

  if (proposals && state.overlay) {
    // The decided geometry must stay visible even inside a wall it crosses.
    gl.disable(gl.DEPTH_TEST);
    gl.useProgram(LINE_PROGRAM);
    gl.uniformMatrix4fv(gl.getUniformLocation(LINE_PROGRAM, "uProj"), false, proj);
    gl.uniformMatrix4fv(gl.getUniformLocation(LINE_PROGRAM, "uView"), false, view);
    gl.uniform4f(gl.getUniformLocation(LINE_PROGRAM, "uColor"),
                 DECISION_RGB[0], DECISION_RGB[1], DECISION_RGB[2], 1);
    gl.bindBuffer(gl.ARRAY_BUFFER, proposals.buffer);
    gl.vertexAttribPointer(linePosition, 3, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(linePosition);
    gl.drawArrays(gl.LINES, 0, proposals.count);
    gl.enable(gl.DEPTH_TEST);
  }
}
window.addEventListener("resize", draw);
selectSample(0);
</script>
</body></html>
"""

__all__ = ["CLASS_STYLES", "VIEWER_SCENE_FORMAT", "export_viewer_html", "viewer_payload"]
