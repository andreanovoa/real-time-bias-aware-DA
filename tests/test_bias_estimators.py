"""Unit tests for the bias estimators (ESN_bias excluded: covered by the integration tests)."""
import numpy as np
import pytest

from romda.bias_estimators import Bias, ConstantBias, NoBias, DriftLinearBias

Nq = 2


class TestConstantBias:

    def test_init_from_innovation(self):
        cb = ConstantBias(innovation=np.array([1., 2.]), t=0.0, dt=0.01)
        assert cb.Nq == Nq
        assert cb.N_dim == Nq  # unbiased observations by default
        np.testing.assert_allclose(cb.current_bias.ravel(), [1., 2.])

    def test_init_with_constant_k(self):
        cb = ConstantBias(innovation=np.zeros(Nq), t=0.0, dt=0.01, k=0.5)
        np.testing.assert_allclose(cb.current_bias.ravel(), 0.5)

    def test_forecast_is_constant(self):
        cb = ConstantBias(innovation=np.array([1., -1.]), t=0.0, dt=0.01, N_ens=3)
        state, t = cb.time_integrate(Nt=10)
        cb.update_history(state, t)
        assert state.shape == (10, cb.N_dim, 3)
        np.testing.assert_allclose(cb.current_bias, cb.get_bias(state, mean=True)[0])  # current_bias is the ensemble mean
        np.testing.assert_allclose(state, np.broadcast_to(state[0], state.shape))  # constant in time
        assert cb.current_time == pytest.approx(0.1)

    def test_state_derivative_is_zero(self):
        cb = ConstantBias(innovation=np.ones(Nq), t=0.0, dt=0.01)
        J = cb.state_derivative()
        assert J.shape == (Nq, Nq)
        np.testing.assert_allclose(J, 0.)

    def test_innovation_update(self):
        cb = ConstantBias(innovation=np.zeros(Nq), t=0.0, dt=0.01)
        new_innovation = np.array([[0.3], [0.7]]) * np.ones((1, 5))  # (Nq, 5) member innovations
        updated = cb.update_state_from_innovation(new_innovation)
        cb.update_history(updated, t=cb.current_time, update_last_state=True)
        np.testing.assert_allclose(cb.current_bias.ravel(), [0.3, 0.7])

    def test_biased_observations_state_is_doubled(self):
        cb = ConstantBias(innovation=np.ones(Nq), t=0.0, dt=0.01, biased_observations=True)
        assert cb.N_dim == 2 * Nq
        assert cb.current_state.shape[0] == 2 * Nq
        assert cb.current_bias.shape[0] == Nq
        assert cb.current_innovations.shape[0] == Nq


class TestNoBias:

    def test_bias_is_always_zero(self):
        nb = NoBias(innovation=np.ones(Nq), t=0.0, dt=0.01)
        np.testing.assert_allclose(nb.current_bias, 0.)

        state, t = nb.time_integrate(Nt=5)
        nb.update_history(state, t)
        np.testing.assert_allclose(nb.current_bias, 0.)

        updated = nb.update_state_from_innovation(np.ones((Nq, 1)))
        nb.update_history(updated, t=nb.current_time, update_last_state=True)
        np.testing.assert_allclose(nb.current_bias, 0.)


class TestDriftLinearBias:

    def test_zero_dynamics_reduces_to_constant(self):
        db = DriftLinearBias(innovation=np.array([1., 2.]), t=0.0, dt=0.01)
        state, t = db.time_integrate(Nt=10)
        db.update_history(state, t)
        np.testing.assert_allclose(db.current_bias.ravel(), [1., 2.])

    def test_exponential_decay(self):
        decay = 2.0
        db = DriftLinearBias(innovation=np.array([1., 1.]), t=0.0, dt=0.001, decay_rate=decay)
        state, t = db.time_integrate(Nt=1000)
        db.update_history(state, t)
        expected = np.exp(-decay * t[-1])
        np.testing.assert_allclose(db.current_bias.ravel(), expected, rtol=1e-2)

    def test_pure_drift(self):
        v = np.array([[0.5], [-0.5]])
        db = DriftLinearBias(innovation=np.zeros(2), t=0.0, dt=0.01, drift_velocity=v)
        state, t = db.time_integrate(Nt=100)
        db.update_history(state, t)
        np.testing.assert_allclose(db.current_bias, v * t[-1], rtol=1e-6)

    def test_state_derivative_is_linear_matrix(self):
        A = np.array([[0., 1.], [-1., 0.]])
        db = DriftLinearBias(innovation=np.zeros(2), t=0.0, dt=0.01, linear_matrix=A)
        np.testing.assert_allclose(db.state_derivative(), A)

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError):
            DriftLinearBias(innovation=np.zeros(2), t=0.0, dt=0.01,
                            drift_velocity=np.zeros(3))
        with pytest.raises(ValueError):
            DriftLinearBias(innovation=np.zeros(2), t=0.0, dt=0.01,
                            linear_matrix=np.zeros((3, 3)))


class TestBiasBase:

    def test_format_state_shapes(self):
        cb = ConstantBias(innovation=np.zeros(Nq), t=0.0, dt=0.01, N_ens=3)
        assert cb._format_state(np.zeros(Nq)).shape == (1, Nq, 3)
        assert cb._format_state(np.zeros((Nq, 3))).shape == (1, Nq, 3)
        assert cb._format_state(np.zeros((5, Nq, 3))).shape == (5, Nq, 3)

    def test_update_history_time_consistency(self):
        cb = ConstantBias(innovation=np.zeros(Nq), t=0.0, dt=0.1)
        state, t = cb.time_integrate(Nt=7)
        cb.update_history(state, t)
        assert len(cb.hist_t) == 8  # initial condition + 7 steps
        assert cb.current_time == pytest.approx(0.7)

    def test_history_grows_beyond_initial_capacity(self):
        cb = ConstantBias(innovation=np.zeros(Nq), t=0.0, dt=0.1, initial_capacity=5)
        for _ in range(4):
            state, t = cb.time_integrate(Nt=3)
            cb.update_history(state, t)
        assert len(cb.hist_t) == 13
        assert cb.current_time == pytest.approx(1.2)

    def test_bayesian_update_runs(self):
        cb = ConstantBias(innovation=np.zeros(Nq), t=0.0, dt=0.1, N_ens=6,
                          bayesian_update=True)
        # spread the ensemble so the covariance is not singular
        spread = np.random.default_rng(1).normal(scale=0.1, size=(cb.N_dim, 6))
        cb.update_history(cb.current_state + spread, t=cb.current_time, update_last_state=True)

        innovation = 0.5 * np.ones((Nq, 6)) + np.random.default_rng(2).normal(scale=0.05, size=(Nq, 6))
        updated = cb.update_state_from_innovation(innovation)
        assert updated.shape == (cb.N_dim, 6)
        assert np.isfinite(updated).all()
