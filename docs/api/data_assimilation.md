# `romda.data_assimilation`

Ensemble filters. All filters are constructed with the observation operator `M` and
applied as callables on the augmented forecast ensemble
`Af = [φ; α; y]` (state, parameters, observables).

::: romda.data_assimilation.Filter

::: romda.data_assimilation.EnKF

::: romda.data_assimilation.EnSRKF

::: romda.data_assimilation.rBA_EnKF
