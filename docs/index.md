# romda

**Real-time reduced-order modelling and bias-aware data assimilation.**

`romda` is an open-source Python package for combining low-order forecast models with
experimental data in real time. It provides:

- **Ensemble data assimilation** — the ensemble Kalman filter (EnKF), the ensemble
  square-root Kalman filter (EnSRKF), and the **regularized bias-aware EnKF (r-EnKF)**
  of [Nóvoa, Racca & Magri (2023)](https://doi.org/10.1016/j.cma.2023.116502), plus a
  linear Kalman filter.
- **Forecast models** — physical low-order models (Rijke tube, azimuthal
  thermoacoustics, Van der Pol, Lorenz 63/96, Kuramoto–Sivashinsky) and data-driven
  reduced-order models (Echo State Networks and POD-ESN).
- **Bias estimators** — interchangeable models of the (unknown) model bias: an ESN
  estimator, a constant (persistent) bias, and a zero-bias placeholder.
- **Modal decompositions** — POD and spectral POD (Sieber 2016 and Towne 2018) with an
  sklearn-style `fit` / `encode` / `decode` interface.

<figure markdown>
  ![Real-time bias-aware DA](figs/DA/wBADA.gif){ width="700" }
  <figcaption>Real-time bias-aware data assimilation: forecast, bias correction,
  assimilation, and update.</figcaption>
</figure>

Model choice trades off accuracy against computational cost — from cheap low-order
physical models to expensive high-fidelity simulations, with data-driven reduced-order
models in between:

<figure markdown>
  ![Model accuracy vs. computational cost](figs/DA/models_pyramid.png){ width="600" }
  <figcaption>The source of model bias: physical models trade fidelity for speed.</figcaption>
</figure>

## Quick example

```python
from romda.estimators.ensembles import rBA_EnKF
from romda.models.physical import VdP
from romda.bias_estimators import ConstantBias
from romda.observations import Observations

truth = Observations(model=VdP, t_start=0.6, t_stop=0.8, Nt_obs=30,
                      add_noise=True, manual_bias='linear')

ensemble = rBA_EnKF(parent_model=VdP(dt=truth.dt),
                     parent_bias=ConstantBias,
                     m=10, std_phi=0.1, std_alpha=dict(zeta=(40., 60.)))

for d, t_d in zip(truth.y_obs, truth.t_obs):
    ensemble.forecast_step(t_end=t_d)
    ensemble.analysis_step(d=d, Cdd=...)
```

## Where to go

- [Getting started](getting-started.md) — install the package and run the first example.
- [Experiments](experiments.md) — run physical-model and thermoacoustic cases.
- [Class hierarchy](class-hierarchy.md) — how models, estimators, bias estimators and
  observations relate.
- [API reference](api/index.md) — browse models, estimators, observations, and bias estimators.
- [Tutorials](tutorials.md) — follow the notebooks from core classes to reduced-order models.
- Sibling packages: [`dynamodels`](https://andreanovoa.github.io/dynamodels/) (the
  physical models) and [`ntsa`](https://andreanovoa.github.io/ntsa/) (nonlinear
  time-series analysis), both usable independently of `romda`.

## Citing

If you use `romda` in your research, please cite the relevant publications, in
particular:

> Nóvoa, A., Racca, A., & Magri, L. (2023). Inferring unknown unknowns: Regularized
> bias-aware ensemble Kalman filter. *Computer Methods in Applied Mechanics and
> Engineering*, 418, 116502.
