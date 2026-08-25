"""
romda — real-time reduced-order modelling and bias-aware data assimilation.

Main entry points
-----------------
- ``romda.models``            : physical and data-driven forecast models
- ``romda.estimators``        : Estimator hierarchy — EnKF, EnSRKF, rBA_EnKF,
                                the deterministic filters and the smoothers
- ``romda.bias_estimators``   : ESN, constant, drift-linear and no-bias estimators
- ``romda.observations``      : truth/observation generation and loading

The ML building blocks (POD / SPOD / autoencoders) live in
``romda.models.data_driven`` (``autoencoders``); the ESN reservoir core comes
from the external ``echostatenetwork`` package.

"""

__version__ = "3.0.0"

from romda import bias_estimators, estimators, models
from romda.observations import Observations

__all__ = [
    "models",
    "bias_estimators",
    "estimators",
    "Observations",
]
