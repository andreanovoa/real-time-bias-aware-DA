# `romda.ensemble`

The `Ensemble` class wraps a forecast [`Model`][romda.models.model.Model] with an
ensemble of perturbed states (and, optionally, parameters), and orchestrates the
sequential data assimilation loop together with a
[filter](data_assimilation.md) and a [bias estimator](bias_estimators.md).

::: romda.ensemble.Ensemble
