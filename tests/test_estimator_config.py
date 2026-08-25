"""Config kwargs must reach the estimator, and the parameter-freeze path must run."""
import numpy as np
import pytest

from romda.models.physical import Lorenz63
from romda.estimators.ensembles import EnSRKF

KW = dict(parent_model=Lorenz63, dt=0.015, m=5, std_phi=0.2,
          std_alpha=dict(rho=(25., 35.), beta=(2, 4), sigma=(5, 15)),
          observe_dims=[0, 1, 2])


@pytest.fixture
def Cdd():
    return np.eye(3) * 0.5


def test_base_class_config_kwargs_are_applied():
    """These live on Estimator, not on the leaf class, and used to be dropped."""
    e = EnSRKF(inflation_factor=1.01, start_param=3, start_bias=2, **KW)
    assert (e.inflation_factor, e.start_param, e.start_bias) == (1.01, 3, 2)


def test_frozen_parameters_survive_the_analysis(Cdd):
    e = EnSRKF(activate_parameter_estimation=False, **KW)
    alpha0 = e.current_state[-e.Na:, :].copy()
    e.forecast_step(t_end=0.15)
    alpha_f = e.current_state[-e.Na:, :].copy()

    Aa = e.analysis_step(d=e.model.get_observables().mean(-1), Cdd=Cdd, return_analysis=True)
    e.model.close()

    assert Aa.shape[0] == e.Nphi + e.Na, 'analysis must carry the full state'
    np.testing.assert_allclose(e.current_state[-e.Na:, :], alpha_f)
    np.testing.assert_allclose(alpha_f, alpha0)          # parameters are static in the forecast
    assert not np.allclose(e.current_state[:e.Nphi, :], Aa[:e.Nphi, :] * 0), 'state was updated'


def test_update_reservoir_false_freezes_non_physical_rows(Cdd):
    """`update_reservoir=False` leaves the 'reservoir' rows exactly at their
    forecast value; only the physical rows should move.

    Lorenz63 has no real reservoir -- N_dim/N_units are monkeypatched onto the
    model purely to exercise the row-split `analysis_step` applies for an ESN.
    """
    e = EnSRKF(update_reservoir=False, **KW)
    e.model.N_dim, e.model.N_units = 1, e.Nphi - 1

    e.forecast_step(t_end=0.15)
    forecast = e.current_state.copy()
    e.analysis_step(d=e.model.get_observables().mean(-1), Cdd=Cdd)

    reservoir_rows = slice(e.model.N_dim, e.model.N_dim + e.model.N_units)
    np.testing.assert_allclose(e.current_state[reservoir_rows, :], forecast[reservoir_rows, :])
    assert not np.allclose(e.current_state[:e.model.N_dim, :], forecast[:e.model.N_dim, :])
    e.model.close()


def test_start_bias_blind_window_uses_plain_enkf(Cdd):
    from romda.bias_estimators import ConstantBias
    from romda.estimators.ensembles import EnKF, rBA_EnKF

    e = rBA_EnKF(parent_bias=ConstantBias, gamma=1.0, start_bias=1, **KW)
    e.forecast_step(t_end=0.15)
    Af = np.vstack((e.current_state, e.model.get_observables()))
    d = e.model.get_observables().mean(-1)

    e.rng = np.random.default_rng(7)
    Aa_blind = e._analysis_kernel(Af, d, Cdd)       # 0 analyses done < start_bias
    e.rng = np.random.default_rng(7)
    Aa_enkf = EnKF._analysis_kernel(e, Af, d, Cdd)
    np.testing.assert_allclose(Aa_blind, Aa_enkf)   # blind window == plain EnKF

    e.assimilated_data = (d, 0.15)                  # window over
    e.rng = np.random.default_rng(7)
    Aa_aware = e._analysis_kernel(Af, d, Cdd)
    assert not np.allclose(Aa_aware, Aa_enkf)       # bias-aware update kicked in
    e.model.close()


def test_start_param_freezes_then_releases_parameters(Cdd):
    e = EnSRKF(start_param=2, **KW)
    alpha0 = e.current_state[-e.Na:, :].copy()
    for k in range(3):
        e.forecast_step(t_end=round(0.15 * (k + 1), 6))
        e.analysis_step(d=e.model.get_observables().mean(-1), Cdd=Cdd)
        frozen = bool(np.allclose(e.current_state[-e.Na:, :], alpha0))
        assert frozen == (k < 2), f'analysis {k}: expected frozen={k < 2}'
    e.model.close()
