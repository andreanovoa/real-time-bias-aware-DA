"""Tests for the Observations (truth/data generation) class."""
import numpy as np
import pytest

from romda.observations import Observations
from romda.models.physical import VdP


def make_truth(**kwargs):
    defaults = dict(model=VdP, t_start=0.6, t_stop=0.8, t_max=1.0, Nt_obs=30)
    defaults.update(kwargs)
    return Observations(**defaults)


class TestObservations:

    def test_shapes_and_times(self):
        truth = make_truth()
        assert truth.y_true.ndim == 3
        assert truth.y_raw.shape == truth.y_true.shape
        assert len(truth.t_true) == truth.y_true.shape[0]
        assert truth.y_obs.shape[0] == len(truth.t_obs)
        assert truth.t_obs[0] >= truth.t_start - truth.dt
        assert truth.t_obs[-1] <= truth.t_stop + truth.dt

    def test_no_bias_no_noise_raw_equals_true(self):
        truth = make_truth()
        np.testing.assert_allclose(truth.y_raw, truth.y_true)
        np.testing.assert_allclose(truth.b_true, 0.)

    def test_manual_bias_without_noise(self):
        """Regression test: y_raw must be the biased truth when add_noise=False."""
        truth = make_truth(manual_bias='linear')
        assert truth.y_raw.shape == truth.y_true.shape
        assert np.any(truth.b_true != 0.)
        np.testing.assert_allclose(truth.y_raw, truth.y_true)

    @pytest.mark.parametrize('bias_name', ['linear', 'periodic', 'cosine', 'time'])
    def test_manual_bias_types(self, bias_name):
        truth = make_truth(manual_bias=bias_name)
        assert truth.name_bias == bias_name
        assert np.isfinite(truth.b_true).all()

    def test_callable_bias(self):
        def my_bias(y, t):
            return 0.1 * y, 'my_bias'
        truth = make_truth(manual_bias=my_bias)
        assert truth.name_bias == 'my_bias'
        np.testing.assert_allclose(truth.b_true, (truth.y_true - truth.b_true) * 0.1, rtol=1e-10)

    def test_additive_gaussian_noise(self):
        truth = make_truth(add_noise=True, noise_level=0.05, noise_type='gauss, add')
        noise = truth.y_raw - truth.y_true
        assert np.std(noise) > 0.
        # noise std should be roughly noise_level * max|y|
        expected = 0.05 * np.max(abs(truth.y_true))
        assert np.std(noise) == pytest.approx(expected, rel=0.5)

    def test_frozen_after_init(self):
        truth = make_truth()
        with pytest.raises(AttributeError):
            truth.y_raw = np.zeros(3)

    def test_provided_data(self):
        t = np.linspace(0., 1., 101)
        y = np.sin(2 * np.pi * t)[:, np.newaxis]
        truth = Observations(model=None, y_true=y, t_true=t, Nt_obs=10)
        assert truth.y_true.shape == (101, 1, 1)
        assert len(truth.t_obs) > 0

    def test_update_obs_idx(self):
        truth = make_truth()
        n_obs_before = len(truth.t_obs)
        truth.update_obs_idx(Nt_obs=60)
        assert len(truth.t_obs) < n_obs_before
