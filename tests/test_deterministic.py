"""Checks for the DeterministicEstimator / KalmanFilter branch."""
import numpy as np
import pytest

from romda.estimators import KalmanFilter


def test_kf_f_only_converges():
    """Linear KF without a model: track a decaying rotation from noisy obs."""
    rng = np.random.default_rng(0)
    theta = 0.1
    F = 0.99 * np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta),  np.cos(theta)]])
    M = np.array([[1.0, 0.0]])          # observe first component only
    Cdd = np.array([[0.05]])

    kf = KalmanFilter(N=2, Nq=1, Cdd=Cdd, psi0=np.zeros(2),
                      Cpp0=np.eye(2), M=M, F=F, Q=1e-4 * np.eye(2))

    x = np.array([1.0, 0.0])            # truth
    for _ in range(100):
        x = F @ x
        kf.forecast_step()
        d = M @ x + rng.normal(0, np.sqrt(Cdd[0, 0]))
        kf.analysis_step(d=d, Cdd=Cdd)

    assert np.linalg.norm(kf.current_state - x) < 0.5
    assert np.all(np.linalg.eigvalsh(kf._Cpp) > 0)          # covariance stays PD
    assert kf.current_time == 100                            # F-only time counts steps
    assert len(kf.assimilated_data.times) == 100

    with pytest.raises(ValueError):
        kf.forecast_step(t_end=1.0)     # t_end unsupported without a model


def test_kf_with_model_roundtrip():
    """KF wrapping a nonlinear model: analysis is written back to the model."""
    from romda.models.physical import Lorenz63

    model = Lorenz63(dt=0.01)
    N = model.Nphi
    kf = KalmanFilter(N=N, Nq=model.Nq, Cdd=0.1 * np.eye(model.Nq),
                      psi0=np.asarray(model.current_state, float).ravel()[:N],
                      Cpp0=np.eye(N), model=model, Q=1e-3 * np.eye(N))

    kf.forecast_step(Nt=10)
    psi_f = kf.current_state.copy()
    d = psi_f[:model.Nq] + 0.5
    kf.analysis_step(d=d, Cdd=0.1 * np.eye(model.Nq))

    assert not np.allclose(kf.current_state, psi_f)          # update moved the mean
    model_state = np.asarray(model.current_state, float)
    model_state = model_state.mean(axis=-1) if model_state.ndim > 1 else model_state
    np.testing.assert_allclose(model_state, kf.current_state)  # written back to history
