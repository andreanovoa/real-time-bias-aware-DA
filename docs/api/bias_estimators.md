# `romda.bias_estimators`

Interchangeable estimators of the model bias, used by the bias-aware
[`rBA_EnKF`][romda.data_assimilation.rBA_EnKF]. All derive from the
[`Bias`][romda.bias_estimators.bias.Bias] base class: they forecast the bias between
analyses (`time_integrate`), provide its Jacobian (`state_derivative`), and are updated
from the analysis innovation (`update_state_from_innovation`).

## Base class

::: romda.bias_estimators.bias.Bias

## Echo State Network estimator

::: romda.bias_estimators.ESN_bias

## Constant (persistent) bias

::: romda.bias_estimators.ConstantBias

## Zero bias

::: romda.bias_estimators.NoBias

## Drift-linear bias

::: romda.bias_estimators.DriftLinearBias

## Training-data helpers

::: romda.bias_estimators.aux.create_bias_training_dataset

::: romda.bias_estimators.aux.sample_model_states

## References

- Nóvoa, Racca & Magri (2023). Inferring unknown unknowns: Regularized bias-aware
  ensemble Kalman filter. *Comput. Methods Appl. Mech. Eng.*, 418, 116502.
  [DOI: 10.1016/j.cma.2023.116502](https://doi.org/10.1016/j.cma.2023.116502) |
  [Erratum (2024)](../2023_CMAME_Erratum.pdf).
- Nóvoa, Noiray, Dawson & Magri (2024). A real-time digital twin of azimuthal
  thermoacoustic instabilities. *J. Fluid Mech.*, 1001, A49.
  [DOI: 10.1017/jfm.2024.1052](https://doi.org/10.1017/jfm.2024.1052).
