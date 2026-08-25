"""
estimators/__init__.py
======================
State/parameter estimators: filters and smoothers.

Class hierarchy
---------------
Estimator                       (abstract base)                   →  base.py
├── EnsembleEstimator           (ensemble Kalman filter family)  →  ensembles.py
│   ├── EnKF
│   ├── EnSRKF
│   └── rBA_EnKF
├── DeterministicEstimator      (owns mean + covariance)         →  deterministic.py
│   └── KalmanFilter

Each intermediate base declares an abstract kernel holding the pure maths
(``EnsembleEstimator._analysis_kernel``, ...) and only leaf classes are instantiable.

Forecast strategy (uniform across all estimators)
--------------------------------------------------
Every estimator must use a Model instance for the forecast step.
The forecast of the model is taken care within the model's time_integrate() method.

Observation operator
--------------------
For all estimators the observation operator M maps state --> observation space:
    y = M @ psi    if M is an ndarray (Nq, N)
    y = M(psi)     if M is a callable
If M is not supplied explicitly, it is read from model.M.

Notation
--------
N      state dimension (Nphi + Na + Nq for augmented ensemble)
Na     number of estimated parameters
Nphi   number of model state variables
Nq     number of observables
m      ensemble size
psi    state vector / ensemble matrix (N,) or (N, m)
M      measurement operator (Nq, N) or callable
d      observation vector (Nq,)
Cdd    observation noise covariance (Nq, Nq)
Cpp    model (prior/forecast) covariance (N, N)
K      Kalman gain matrix (N, Nq)

References
----------
Kalman (1960) KF
Evensen (2009) EnKF, EnSRKF
Nóvoa & Magri (2022) rBA-EnKF
"""

from romda.estimators.base import Estimator
from romda.estimators.deterministic import DeterministicEstimator, KalmanFilter
from romda.estimators.ensembles import (
    EnKF,
    EnsembleEstimator,
    EnSRKF,
    rBA_EnKF,
)
from romda.estimators.inflation import multiplicative_inflation

__all__ = [
    "Estimator",
    "EnsembleEstimator",
    "EnKF",
    "EnSRKF",
    "rBA_EnKF",
    "DeterministicEstimator",
    "KalmanFilter",
    "multiplicative_inflation",
]
