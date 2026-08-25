"""
estimators/inflation.py
=======================
Covariance inflation methods (Evensen 2009, Chap. 15).

The analysis step of an ensemble filter reduces the ensemble variance not only
where the data justify it, but also through *spurious correlations* between the
state and the predicted measurements that a finite ensemble cannot average to
zero (Sect. 15.1). Inflation counteracts that underestimation of the spread.

- `multiplicative_inflation` — fixed multiplicative inflation about the ensemble mean
  (Anderson & Anderson 1999; Evensen 2009, Eq. 15.3, Sect. 15.2), controlled by
  the estimator's ``inflation_factor``.

References
----------
Evensen (2009). *Data Assimilation: The Ensemble Kalman Filter*, 2nd ed.,
Springer, Chap. 15.
"""

from __future__ import annotations

import numpy as np

__all__ = ['multiplicative_inflation']


def multiplicative_inflation(A: np.ndarray, rho: float) -> np.ndarray:
    r"""Multiplicative inflation of an ensemble about its mean.

    $\boldsymbol{\psi}_j \leftarrow \rho\,(\boldsymbol{\psi}_j -
    \bar{\boldsymbol{\psi}}) + \bar{\boldsymbol{\psi}}$ — Evensen (2009),
    Eq. (15.3) — with $\rho$ typically slightly greater than one (e.g. 1.01).

    Parameters
    ----------
    A : (N, m) ndarray
        Ensemble matrix.
    rho : float
        Inflation factor.

    Returns
    -------
    (N, m) ndarray
    """
    A_mean = np.mean(A, axis=-1, keepdims=True)
    return A_mean + rho * (A - A_mean)


