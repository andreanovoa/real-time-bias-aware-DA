# Linear model

`LinearModel` advances reduced coordinates with a linear map and optional process noise,
which makes it useful for Kalman-filter examples and baseline forecasts.

$$
\boldsymbol{\psi}_{t+1} = \mathbf{F}\boldsymbol{\psi}_t + \boldsymbol{\eta}_t.
$$

::: romda.models.data_driven.linear_model.LinearModel
