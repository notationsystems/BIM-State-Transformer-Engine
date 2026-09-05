# Engine and Workbench integration candidate

This candidate combines the original commits from the following reviews, then
adds explicit v1/v2 viewer and audit integration. It preserves their individual
authorship and can be reviewed as one combined build.

| Review | Contribution | Integrated source commit |
|---|---|---|
| #22 | Workbench CI walkthrough | `ce19143` |
| #23 | No-available-evidence outcome | `8197448` |
| #25 | Portable IFC import identity | `98f3247` |
| #26 (Claude) | Frame-aware projections and display equivariance | `a5f5d35` |
| #27 | Coordinate-frame and shared-pose primitives | `28b2add` |

The common base includes the merged EXPLODE increment (#24). The constituent
reviews remain independently identifiable; this document does not assert that
any draft has been approved or merged into main.

## Combined behavior

New content-bound IFC imports render through the selected model path. They do
not depend on the original request locator continuing to exist. Explicit
legacy requests still load through the original path and receive an audit
using that same import identity version. Changed bytes or a relocated legacy
copy are refused; there is no implicit fallback across identity contracts.

Claude's frame metadata, EXPLODE behavior, selection sharing, audit outlines,
and report/instrument distinction remain intact. The viewer continues to say
that canonical placements are exact metadata: the frame/pose mathematical
helpers have not been wired into the IR or IFC lowering. A display change is
not evidence and does not create a calibration claim.

Six additional integration tests cover both viewer and Workbench with portable
and legacy imports, audit agreement, stale locators, altered bytes, invalid
versions, and nonmutating frame projections.

## Reproducing validation

```console
python -m unittest discover -s tests -v
python validation/fetch_ifc_corpus.py ci-ifc-corpus
```

Set `GAT_IFC_VALIDATION_ROOT` to the fetched corpus directory before running
the suite to include the external-model baseline checks. With the optional
OpenUSD dependencies and the corpus present, this candidate ran **580 tests,
OK, no skips** on Windows / Python 3.12.14 / NumPy 2.3.5 / usd-core 26.8 /
cryptography 50.0.1. Workbench, construction workflow and separate-process
OpenUSD continuation demos also passed.

The browser check requires Node.js with Playwright resolvable and a supported
browser installed. It is an optional development check, not a Python runtime
dependency:

```console
python -m gat.demo.workbench ci-out/workbench
node validation/workbench_browser_smoke.cjs ci-out/workbench
```

Set `GAT_BROWSER_CHANNEL=msedge` to use an installed Edge browser rather than
Playwright's bundled Chromium. The smoke check exercises graph-to-viewer
selection, frame-limit text, EXPLODE, audit controls, evidence verdict, state
selection and a 700px layout. It records a screenshot in the demo directory
and rejects JavaScript/console errors. This check passed with zero errors in
headless Edge (Chromium) on the validation host.

Linux/Python matrix checks and the native SP1 proof CI lane remain separate
from these local results. No practitioner measurements were introduced.

## Next shared step

Bind the frame primitives to the scoped external opening-fit import and
emit actual frame/pose records before changing the viewer's exact-placement
label. Retain unsupported quantity/dependency diagnostics. Stable risk-subject
EntityIds remain a separate engine-to-Workbench follow-up. Calibrated
residuals and confidence coverage need independently measured data; a
successful synthetic or public-model audit is not field validation.
