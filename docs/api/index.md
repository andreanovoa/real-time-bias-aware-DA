# API reference

The public API of `romda` is organised in six modules:

| Module | Contents |
| --- | --- |
| [`romda.ensemble`](ensemble.md) | The `Ensemble` wrapper: forecasting, analysis, inflation, parameter bounds |
| [`romda.data_assimilation`](data_assimilation.md) | `EnKF`, `EnSRKF` and the regularized bias-aware `rBA_EnKF` |
| [`romda.bias_estimators`](bias_estimators.md) | `Bias` base class, `ESN_bias`, `ConstantBias`, `DriftLinearBias`, `NoBias` |
| [`romda.models`](models.md) | `Model` base class, integrators, history, physical and data-driven models |
| [`romda.observations`](observations.md) | Truth / observation generation and loading |
| [`romda.tools`](tools.md) | `POD`, `SPOD`, Towne SPOD, and the `EchoStateNetwork` core |

The typical composition is:

```python
from romda import Ensemble, Observations
from romda.models.physical import Rijke
from romda.data_assimilation import rBA_EnKF
from romda.bias_estimators import ESN_bias

truth = Observations(model=Rijke, ...)
ensemble = Ensemble(parent_model=Rijke(dt=truth.dt),
                    parent_bias=ESN_bias,
                    da_method=rBA_EnKF, ...)
```
