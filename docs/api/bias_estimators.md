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
