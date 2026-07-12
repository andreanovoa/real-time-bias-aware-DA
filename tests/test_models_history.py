"""Tests for the Model classes, the HistoryTracker and the integrators."""
import numpy as np
import pytest

from romda.models import HistoryTracker
from romda.models.physical import VdP, Lorenz63


class TestHistoryTracker:

    def make_tracker(self, N=3, m=2, capacity=10):
        h = HistoryTracker(initial_capacity=capacity)
        h.update_history(np.zeros((1, N, m)), t=np.array([0.]), reset=True)
        return h

    def test_reset(self):
        h = self.make_tracker()
        assert h.hist.shape == (1, 3, 2)
        assert h.current_time == 0.

    def test_append_and_current_state(self):
        h = self.make_tracker()
        new = np.random.default_rng(0).normal(size=(4, 3, 2))
        h.update_history(new, t=np.arange(1, 5, dtype=float))
        assert h.hist.shape == (5, 3, 2)
        np.testing.assert_allclose(h.current_state, new[-1])
        assert h.current_time == 4.

    def test_update_last_state(self):
        h = self.make_tracker()
        h.update_history(np.ones((1, 3, 2)), t=np.array([0.5]), update_last_state=True)
        assert h.hist.shape == (1, 3, 2)
        np.testing.assert_allclose(h.current_state, 1.)
        assert h.current_time == 0.5

    def test_capacity_grows(self):
        h = self.make_tracker(capacity=3)
        h.update_history(np.zeros((10, 3, 2)), t=np.arange(1, 11, dtype=float))
        assert h.hist.shape[0] == 11
        assert h.capacity >= 11

    def test_time_state_mismatch_raises(self):
        h = self.make_tracker()
        with pytest.raises(AssertionError):
            h.update_history(np.zeros((2, 3, 2)), t=np.array([1.]))


class TestVdP:

    def test_single_forecast(self):
        model = VdP(dt=1e-4)
        psi, t = model.time_integrate(Nt=100)
        assert psi.shape == (100, model.Nphi, 1)
        assert np.isfinite(psi).all()
        model.update_history(psi, t)
        assert model.current_time == pytest.approx(t[-1])

    def test_get_observables(self):
        model = VdP(dt=1e-4)
        psi, t = model.time_integrate(Nt=10)
        model.update_history(psi, t)
        y = model.get_observables()
        assert y.shape == (model.Nq, 1)
        y_hist = model.get_observable_hist()
        assert y_hist.shape == (11, model.Nq, 1)

    def test_observation_operator_shape(self):
        model = VdP(dt=1e-4)
        assert model.M.shape == (model.Nq, model.N)

    def test_parameter_getters(self):
        model = VdP(dt=1e-4, beta=75.)
        assert model.beta == 75.
        assert model.alpha0['beta'] == 75.
        alphas = model.get_alpha()
        assert alphas[0]['beta'] == 75.


class TestLorenz63:

    def test_forecast_is_chaotic_but_finite(self):
        model = Lorenz63(dt=0.02)
        psi, t = model.time_integrate(Nt=500)
        assert np.isfinite(psi).all()
        assert psi.shape[1] == 3
