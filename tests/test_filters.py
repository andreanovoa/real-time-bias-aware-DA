"""Unit tests for the ensemble filters in data_assimilation.py."""
import numpy as np
import pytest

from romda.data_assimilation import Filter, EnKF, EnSRKF, rBA_EnKF

rng = np.random.default_rng(0)

N, Nq, m = 5, 2, 12


def make_case(bias_free=True):
    """Random forecast ensemble augmented with its observables, plus an observation."""
    M = np.hstack((np.zeros([Nq, N]), np.eye(Nq)))
    Af_state = rng.normal(size=(N, m)) + 2.0
    y = Af_state[:Nq, :] + 0.5           # observables (linear in the state here)
    Af = np.vstack((Af_state, y))
    d = np.mean(y, axis=-1) + 1.0        # observation offset from the forecast mean
    Cdd = 0.01 * np.eye(Nq)
    return M, Af, d, Cdd


class TestFilterBase:

    def test_default_gamma_is_none(self):
        M = np.hstack((np.zeros([Nq, N]), np.eye(Nq)))
        f = Filter(M)
        assert f.gamma is None
        assert not f.is_bias_aware

    def test_observation_operator_truncation(self):
        M, Af, *_ = make_case()
        f = Filter(M)
        # if parameter estimation is deactivated, the state has fewer rows and M is cut
        M_cut = f.observation_operator(Af[:-1])
        assert M_cut.shape == (Nq, Af.shape[0] - 1)


@pytest.mark.parametrize('filter_class', [EnKF, EnSRKF])
class TestBiasUnawareFilters:

    def test_shape_and_finite(self, filter_class):
        M, Af, d, Cdd = make_case()
        Aa = filter_class(M)(Af, d, Cdd)
        assert Aa.shape == Af.shape
        assert np.isfinite(Aa).all()

    def test_not_bias_aware(self, filter_class):
        M = np.hstack((np.zeros([Nq, N]), np.eye(Nq)))
        # even if a gamma is provided, these filters are not bias-aware
        assert not filter_class(M, gamma=2.0).is_bias_aware

    def test_analysis_moves_towards_observation(self, filter_class):
        M, Af, d, Cdd = make_case()
        Aa = filter_class(M)(Af, d, Cdd)
        y_f = np.mean(Af[-Nq:], axis=-1)
        y_a = np.mean(Aa[-Nq:], axis=-1)
        assert np.linalg.norm(y_a - d) < np.linalg.norm(y_f - d)


class TestRegularizedBiasAwareEnKF:

    def test_is_bias_aware(self):
        M = np.hstack((np.zeros([Nq, N]), np.eye(Nq)))
        assert rBA_EnKF(M, gamma=1.).is_bias_aware
        assert rBA_EnKF(M).gamma == 1.0  # default regularization

    @pytest.mark.parametrize('b_shape', ['1d', 'column', 'ensemble'])
    def test_bias_shapes_accepted(self, b_shape):
        M, Af, d, Cdd = make_case()
        b = 0.1 * np.ones(Nq)
        if b_shape == 'column':
            b = b[:, np.newaxis]
        elif b_shape == 'ensemble':
            b = np.tile(b[:, np.newaxis], (1, m))
        J = 0.1 * np.eye(Nq)
        Aa = rBA_EnKF(M, gamma=1.)(Af, d, Cdd, Cdd.copy(), b, J)
        assert Aa.shape == Af.shape
        assert np.isfinite(Aa).all()

    def test_invalid_bias_shape_raises(self):
        M, Af, d, Cdd = make_case()
        b = np.ones((Nq, m + 3))
        J = np.zeros((Nq, Nq))
        with pytest.raises(ValueError):
            rBA_EnKF(M, gamma=1.)(Af, d, Cdd, Cdd.copy(), b, J)

    def test_unbiased_limit_moves_towards_observation(self):
        """With b = 0 and J = 0 the r-EnKF is a (stochastic) EnKF."""
        M, Af, d, Cdd = make_case()
        b = np.zeros(Nq)
        J = np.zeros((Nq, Nq))
        Aa = rBA_EnKF(M, gamma=5.)(Af, d, Cdd, Cdd.copy(), b, J)
        y_f = np.mean(Af[-Nq:], axis=-1)
        y_a = np.mean(Aa[-Nq:], axis=-1)
        assert np.linalg.norm(y_a - d) < np.linalg.norm(y_f - d)

    def test_bias_corrected_analysis(self):
        """With a known bias, the bias-corrected analysis observable Y = q + b should
        approach the observation, i.e., the biased observable approaches d - b."""
        M, Af, d, Cdd = make_case()
        b = 0.5 * np.ones(Nq)
        J = np.zeros((Nq, Nq))  # bias insensitive to the state
        Aa = rBA_EnKF(M, gamma=0.)(Af, d, Cdd, Cdd.copy(), b, J)
        q_a = np.mean(Aa[-Nq:], axis=-1)
        q_f = np.mean(Af[-Nq:], axis=-1)
        assert np.linalg.norm(q_a + b - d) < np.linalg.norm(q_f + b - d)
