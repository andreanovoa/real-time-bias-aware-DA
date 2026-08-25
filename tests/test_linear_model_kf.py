"""Regression tests for the LinearModel / KalmanFilter fixes surfaced by tutorial 15."""
import numpy as np
import pytest

from romda.estimators import KalmanFilter
from romda.models.data_driven import LinearModel

THETA = 2 * np.pi / 40
F = 0.998 * np.array([[np.cos(THETA), -np.sin(THETA)],
                      [np.sin(THETA), np.cos(THETA)]])


def test_process_noise_draw():
    # Q != 0 used to crash: multivariate_normal was called with a 2-D mean
    model = LinearModel(F=F, psi0=np.zeros((2, 5)), Q=0.1 * np.eye(2), seed=0)
    psi, _ = model.time_integrate(Nt=50)
    assert psi.shape == (50, 2, 5)
    assert psi[-1].std() > 0                      # noise actually entered
    assert not np.allclose(psi[:, :, 0], psi[:, :, 1])  # per-member draws differ


def test_get_observables_honours_Nt():
    M_obs = np.array([[1.0, 0.0]])
    model = LinearModel(F=F, M_obs=M_obs, psi0=np.array([1.0, 0.0]))
    model.update_history(*model.time_integrate(Nt=10))
    y = model.get_observable_hist()               # full history, used to return only the last state
    assert y.shape == (model.hist.shape[0], 1, 1)
    np.testing.assert_allclose(y[:, 0, 0], model.hist[:, 0, 0])
    assert model.get_observables().shape == (1, 1)


def test_filename_and_init_ensemble():
    model = LinearModel(F=F, psi0=np.array([1.0, 0.0]))
    assert model.filename                          # crashed: Q_noise had no class default
    model.init_ensemble(m=4, std_phi=0.1)          # crashed: t_transient == 0 -> psi[-1] on empty
    assert model.hist.shape[-1] == 4


def test_kf_covariance_uses_model_F_per_step():
    Nt_obs, Cpp0 = 5, 0.5 * np.eye(2)
    model = LinearModel(F=F, M_obs=np.array([[1.0, 0.0]]), psi0=np.array([1.0, 0.0]), dt=1.0)
    kf = KalmanFilter(N=2, Nq=1, Cdd=np.eye(1), psi0=np.array([1.0, 0.0]), Cpp0=Cpp0,
                      M=np.array([[1.0, 0.0]]), model=model)   # no F_jac: defaults to model.F
    kf.forecast_step(t_end=model.current_time + Nt_obs * model.dt)
    Fk = np.linalg.matrix_power(F, Nt_obs)
    np.testing.assert_allclose(kf.Cpp, Fk @ Cpp0 @ Fk.T, atol=1e-12)  # public accessor, applied per step


def test_kf_matches_hand_rolled_update():
    M = np.array([[1.0, 0.0]])
    Cdd, Cpp0, psi0 = np.array([[0.04]]), 0.5 * np.eye(2), np.array([1.0, 0.0])
    model = LinearModel(F=F, M_obs=M, psi0=psi0)
    kf = KalmanFilter(N=2, Nq=1, Cdd=Cdd, psi0=psi0, Cpp0=Cpp0, M=M, model=model)
    kf.forecast_step(t_end=1.0)
    d = np.array([0.9])
    psi_a = kf.analysis_step(d, Cdd, return_analysis=True)

    Cf = F @ Cpp0 @ F.T
    K = Cf @ M.T @ np.linalg.inv(M @ Cf @ M.T + Cdd)
    expected = F @ psi0 + K @ (d - M @ (F @ psi0))
    np.testing.assert_allclose(psi_a, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
