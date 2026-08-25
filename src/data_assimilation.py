"""The sequential data assimilation driver shared by every entry point.

`run_da_loop` alternates ``forecast_step`` / ``analysis_step`` over the
observations of a truth record. It is duck-typed on purpose: it accepts any
`romda.estimators.Estimator` — all it needs is ``copy()``,
``forecast_step(t_end, close=...)``, ``analysis_step(d, Cdd)`` and
``model.t_CR``.
"""

import numpy as np


def _legacy_observation_covariance(truth, std_obs):
    """The historical default ``Cdd`` of the mains (variance ``std_obs * max|y|^2``).

    Kept verbatim from ``scripts/mains/common.observation_covariance`` so that
    calling `run_da_loop` without an explicit ``Cdd`` behaves exactly as the
    published scripts/ablations runs did. New work should pass
    ``Cdd=romda.observations.observation_covariance(truth, std_obs)`` — the
    corrected ``(std_obs * max|y|)^2`` variant — explicitly.
    """
    Nq = truth.y_obs.shape[1]
    return np.diag(std_obs * np.ones(Nq)) * np.max(abs(truth.y_obs), axis=0) ** 2


def run_da_loop(ensemble,
                truth,
                std_obs: float = 0.1,
                Cdd: np.ndarray = None,
                t_extra: float = None,
                close: bool = True):
    """Run sequential data assimilation over all observations in ``truth``.

    Parameters
    ----------
    ensemble : Estimator or romda.legacy.ensemble.Ensemble
        Initialized ensemble (copied, so the input is left untouched).
    truth : Observations
        Provides ``y_obs`` and ``t_obs``.
    std_obs : float
        Fractional observation error used to build ``Cdd`` when not provided
        (legacy variance semantics, see `_legacy_observation_covariance`).
    Cdd : np.ndarray, optional
        Observation error covariance. Overrides ``std_obs`` if given.
    t_extra : float, optional
        Extra forecast time beyond the last observation (default ``10 t_CR``).
    close : bool
        Close the model's multiprocessing pool after the final forecast.

    Returns
    -------
    Estimator or Ensemble
        The filtered ensemble with full model (and bias) histories.
    """
    filter_ens = ensemble.copy()

    if Cdd is None:
        Cdd = _legacy_observation_covariance(truth, std_obs)

    for d, t_d in zip(truth.y_obs, truth.t_obs):
        filter_ens.forecast_step(t_end=t_d)
        filter_ens.analysis_step(d=d, Cdd=Cdd.copy())

    if t_extra is None:
        t_extra = 10. * filter_ens.model.t_CR

    filter_ens.forecast_step(t_end=truth.t_obs[-1] + t_extra, close=close)
    return filter_ens
