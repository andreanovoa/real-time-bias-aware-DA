"""
romda — real-time reduced-order modelling and bias-aware data assimilation.

Main entry points
-----------------
- ``romda.models``            : physical and data-driven forecast models
- ``romda.ensemble.Ensemble`` : ensemble wrapper for forecasting and assimilation
- ``romda.data_assimilation`` : EnKF, EnSRKF and the regularized bias-aware EnKF
- ``romda.bias_estimators``   : ESN, constant, drift-linear and no-bias estimators
- ``romda.observations``      : truth/observation generation and loading
- ``romda.tools``             : POD / SPOD decompositions and the ESN core
"""

__version__ = "2.3.0"

from romda import models
from romda import bias_estimators
from romda import data_assimilation
from romda import tools

from romda.ensemble import Ensemble
from romda.observations import Observations

__all__ = [
    "models",
    "bias_estimators",
    "data_assimilation",
    "tools",
    "Ensemble",
    "Observations",
]
