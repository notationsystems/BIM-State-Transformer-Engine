# GAT temporal dynamics v1

## Contract

GAT's first explicit process model is a calibrated linear-Gaussian transition
over selected raw architectural variables:

```text
x[k+1] = A x[k] + b + w,       w ~ N(0, Q)
```

The operation declares its selected variables, transition matrix `A`, offset
`b`, PSD process covariance `Q`, elapsed seconds, model id, and calibration
SHA-256. All other raw variables follow identity dynamics. GAT constructs one
full transition matrix `F` and applies:

```text
mu'    = F mu + embedded(b)
Sigma' = F Sigma F^T + embedded(Q)
```

This transports target/untargeted cross-covariances correctly. Derived means
and covariances are then rebuilt by the normal pushforward, and the complete
invariant registry decides whether the future state may commit.

## Predict versus commit

`forecast_process` rolls a model forward from an immutable `World` and returns
the verified states/digests without changing a `GatSession`. `session.run`
performs one explicit committed temporal transition and records it in the
authoritative ledger.

```python
from hashlib import sha256
import numpy as np
from gat import EvolveLinearGaussian, forecast_process

process = EvolveLinearGaussian(
    targets=(clear_height,),
    transition=np.array([[1.0]]),
    offset=np.array([-0.0005]),
    process_covariance=np.array([[0.0002**2]]),
    elapsed_seconds=86400.0,
    model_id="daily-settlement-monitor-v1",
    calibration_digest=sha256(calibration_bytes).hexdigest(),
)

forecast = forecast_process(session.world, process, steps=7)  # no mutation
session.run(process, provenance={"clock": "controller-A", "interval": "day-1"})
```

The ledger codec stores the complete matrices, ordered structural variable
identities, elapsed time, and calibration binding. Replay must reconstruct the
same final dense covariance and the same verification results. OpenUSD carrier
v3 preserves the process events and exact continuation.

## Predict-update loop

Process evolution creates the next prior. A later physical reading is still a
separate likelihood update:

```text
verified posterior[k]
    -> calibrated process transition
    -> verified prior[k+1]
    -> provenance-bound observation
    -> verified posterior[k+1]
```

That separation matters for the Free Energy Principle direction: dynamics say
what the model predicts through time; observations say what evidence was
received; policy may choose which evidence to acquire. The current
implementation supplies these inspectable pieces but does not claim a learned
generative model or general active-inference agent.

## Validation and limits

* Targets must be unique raw variables. Derived variables remain deterministic
  functions and cannot own independent process noise.
* Matrices must have exact dimensions and finite values. `Q` must be symmetric
  and PSD within GAT's documented numerical certification tolerance.
* Elapsed time must be positive. It is an interval carried by the operation,
  not an absolute trusted clock; trusted clock claims belong in event
  provenance.
* Dynamics are fixed, local, linear, and time-homogeneous per operation. There
  is no state-dependent drift, switching regime, control input, nonlinear
  posterior, smoothing, or learned parameter estimation yet.
* Gaussian process noise models epistemic/aleatoric uncertainty only to the
  extent justified by the named calibration. A digest binds a calibration
  artifact; it does not prove that calibration scientifically valid.

Run `python -m gat.demo.temporal_process` for an executable forecast,
predict-update, invariant, and replay witness.
