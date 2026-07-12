# romda

**Real-time reduced-order modelling and bias-aware data assimilation.**

`romda` is an open-source Python package for combining low-order forecast models with
experimental data in real time. It provides:

- **Ensemble data assimilation** — the ensemble Kalman filter (EnKF), the ensemble
  square-root Kalman filter (EnSRKF), and the **regularized bias-aware EnKF (r-EnKF)**
  of [Nóvoa, Racca & Magri (2023)](https://doi.org/10.1016/j.cma.2023.116502).
- **Forecast models** — physical low-order models (Rijke tube, azimuthal
  thermoacoustics, Van der Pol, Lorenz 63/96, Kuramoto–Sivashinsky) and data-driven
  reduced-order models (Echo State Networks and POD-ESN).
- **Bias estimators** — interchangeable models of the (unknown) model bias: an ESN
  estimator, a constant (persistent) bias, a drift-linear bias, and a zero-bias
  placeholder.
- **Modal decompositions** — POD and spectral POD (Sieber 2016 and Towne 2018) with an
  sklearn-style `fit` / `encode` / `decode` interface.

<figure markdown>
  ![Real-time bias-aware DA](figs/DA/wBADA.gif){ width="700" }
  <figcaption>Real-time bias-aware data assimilation: forecast, bias correction,
  assimilation, and update.</figcaption>
</figure>

## Where to start

- [Getting started](getting-started.md) — install the package and run your first
  twin experiment.
- [Architecture](concepts/architecture.md) — how `Model`, `Ensemble`, `Filter` and
  `Bias` fit together.
- [Bias-aware data assimilation](concepts/bias-aware-da.md) — the maths behind the
  r-EnKF.
- [Tutorials](tutorials.md) — Jupyter notebooks from basic concepts to real-data
  digital twins.
- [API reference](api/index.md) — the full documented API.

## Citing

If you use `romda` in your research, please cite the relevant publications listed in
[Publications](publications.md) — in particular:

> Nóvoa, A., Racca, A., & Magri, L. (2023). Inferring unknown unknowns: Regularized
> bias-aware ensemble Kalman filter. *Computer Methods in Applied Mechanics and
> Engineering*, 418, 116502.
