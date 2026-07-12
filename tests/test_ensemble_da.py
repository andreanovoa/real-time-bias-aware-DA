"""
Integration tests for the Ensemble class and the (bias-aware) data assimilation loop.

These are twin experiments on the Van der Pol model: the truth is generated with the
same model (optionally with a manually-added bias), and the ensemble assimilates the
observations sequentially.
"""
import numpy as np
import pytest

from romda.observations import Observations
from romda.models.physical import VdP
from romda.ensemble import Ensemble
from romda.data_assimilation import EnKF, EnSRKF, rBA_EnKF
from romda.bias_estimators import ConstantBias, NoBias, DriftLinearBias


# --------------------------------------------------------------------------- fixtures

@pytest.fixture(scope='module')
def truth_unbiased():
    return Observations(model=VdP, t_start=0.6, t_stop=0.75, t_max=0.9, Nt_obs=30,
                        add_noise=True, noise_type='gauss, add', noise_level=0.01)


@pytest.fixture(scope='module')
def truth_biased():
    return Observations(model=VdP, t_start=0.6, t_stop=0.75, t_max=0.9, Nt_obs=30,
                        add_noise=True, noise_type='gauss, add', noise_level=0.02,
                        manual_bias='linear')


def obs_covariance(truth, std_obs=0.05):
    return np.diag(std_obs * np.ones(truth.y_obs.shape[1])) * np.max(abs(truth.y_obs), axis=0) ** 2


def run_da(ens, truth, n_analysis=6, **analysis_kwargs):
    Cdd = obs_covariance(truth)
    for k, (d, t_d) in enumerate(zip(truth.y_obs, truth.t_obs)):
        ens.forecast_step(t_end=t_d)
        ens.analysis_step(d=d, Cdd=Cdd.copy(), **analysis_kwargs)
        if k + 1 >= n_analysis:
            break
    ens.model.close()
    return ens


def make_ensemble(truth, da_method=EnKF, bias=None, m=8, **kwargs):
    return Ensemble(parent_model=VdP(dt=truth.dt),
                    parent_bias=bias,
                    da_method=da_method,
                    m=m,
                    std_phi=0.1,
                    std_alpha=dict(zeta=(40., 60.)),
                    **kwargs)


# --------------------------------------------------------------------------- ensemble basics

class TestEnsembleSetup:

    def test_init_shapes(self, truth_unbiased):
        ens = make_ensemble(truth_unbiased)
        assert ens.m == 8
        assert ens.Na == 1
        assert ens.current_state.shape == (ens.Nphi + ens.Na, 8)
        assert ens.filter is not None and not ens.filter.is_bias_aware

    def test_filter_gamma_from_regularization_factor(self, truth_unbiased):
        ens = make_ensemble(truth_unbiased, da_method=rBA_EnKF, regularization_factor=3.,
                            bias=ConstantBias)
        assert ens.filter.is_bias_aware
        assert ens.filter.gamma == 3.

    def test_bias_setter_validates(self, truth_unbiased):
        ens = make_ensemble(truth_unbiased)
        with pytest.raises(AssertionError):
            ens.bias = 'not a bias'
        cb = ConstantBias(innovation=np.zeros(ens.model.Nq), t=0., dt=ens.model.dt)
        ens.bias = cb
        assert ens.bias is cb

    def test_alpha_limits_matrix_only_estimated_params(self, truth_unbiased):
        """Regression test: the limits must correspond to est_alpha, not to all model params."""
        ens = make_ensemble(truth_unbiased)
        lims = ens.alpha_limits_matrix
        assert lims.shape == (2, ens.Na, 1)
        lo, hi = lims[0, 0, 0], lims[1, 0, 0]
        assert (lo, hi) == ens.model.alpha_lims['zeta']

    def test_valid_params_check(self, truth_unbiased):
        ens = make_ensemble(truth_unbiased)
        inside = 50. * np.ones((1, ens.m))
        outside = 500. * np.ones((1, ens.m))
        assert ens.has_valid_params(inside, ens.alpha_limits_matrix)[0]
        is_ok, idx, _ = ens.has_valid_params(outside, ens.alpha_limits_matrix)
        assert not is_ok and idx == [0]


# --------------------------------------------------------------------------- DA loops

class TestBiasUnawareDA:

    @pytest.mark.parametrize('da_method', [EnKF, EnSRKF])
    def test_da_reduces_observation_error(self, truth_unbiased, da_method):
        ens = make_ensemble(truth_unbiased, da_method=da_method)

        Cdd = obs_covariance(truth_unbiased)
        errors = []
        for k, (d, t_d) in enumerate(zip(truth_unbiased.y_obs, truth_unbiased.t_obs)):
            ens.forecast_step(t_end=t_d)
            y_f = np.mean(ens.model.get_observables(), axis=-1)
            errors.append(np.linalg.norm(y_f - d))
            ens.analysis_step(d=d, Cdd=Cdd.copy())
            y_a = np.mean(ens.model.get_observables(), axis=-1)
            # each analysis must move the observables towards the data
            assert np.linalg.norm(y_a - d) <= errors[-1] + 1e-12
            if k >= 5:
                break
        ens.model.close()
        assert np.isfinite(ens.model.hist).all()
        assert len(ens.assimilated_data.times) == 6


class TestBiasAwareDA:

    @pytest.mark.parametrize('bias_cls,bias_kwargs', [
        (NoBias, {}),
        (ConstantBias, {}),
        (DriftLinearBias, dict(decay_rate=0.1)),
    ])
    def test_bias_aware_loop_runs(self, truth_biased, bias_cls, bias_kwargs):
        ens = make_ensemble(truth_biased, da_method=rBA_EnKF, bias=bias_cls,
                            regularization_factor=1., **bias_kwargs)
        ens = run_da(ens, truth_biased)
        assert np.isfinite(ens.model.hist).all()
        assert np.isfinite(ens.bias.hist).all()
        # model and bias histories stay synchronized in time
        assert abs(ens.bias.current_time - ens.model.current_time) < ens.model.dt

    def test_bias_state_tracks_innovation(self, truth_biased):
        """After each analysis, ConstantBias must store the analysis innovation d - <y^a>."""
        ens = make_ensemble(truth_biased, da_method=rBA_EnKF, bias=ConstantBias,
                            regularization_factor=1.)
        Cdd = obs_covariance(truth_biased)
        d, t_d = truth_biased.y_obs[0], truth_biased.t_obs[0]
        ens.forecast_step(t_end=t_d)
        ens.analysis_step(d=d, Cdd=Cdd.copy())
        ya = np.mean(ens.model.get_observables(), axis=-1)
        expected_innovation = d - ya
        np.testing.assert_allclose(np.mean(ens.bias.current_bias, axis=-1),
                                   expected_innovation, rtol=1e-8)
        ens.model.close()

    def test_nobias_matches_unbiased_limit(self, truth_biased):
        """NoBias keeps b = 0 throughout the assimilation."""
        ens = make_ensemble(truth_biased, da_method=rBA_EnKF, bias=NoBias,
                            regularization_factor=1.)
        ens = run_da(ens, truth_biased, n_analysis=4)
        np.testing.assert_allclose(ens.bias.hist, 0.)

    def test_unbiased_observable_estimate(self, truth_biased):
        """The bias-corrected estimate y + b should be closer to the (biased) data than
        the raw model estimate after several bias-aware analyses."""
        ens = make_ensemble(truth_biased, da_method=rBA_EnKF, bias=ConstantBias,
                            regularization_factor=1., m=10)
        ens = run_da(ens, truth_biased, n_analysis=6)

        d = ens.assimilated_data.data[-1]
        y = np.mean(ens.model.get_observables(), axis=-1)
        b = np.mean(ens.bias.current_bias, axis=-1)
        assert np.linalg.norm(y + b - d) <= np.linalg.norm(y - d) + 1e-10

    def test_num_DA_blind_defers_bias_awareness(self, truth_biased):
        ens = make_ensemble(truth_biased, da_method=rBA_EnKF, bias=ConstantBias,
                            regularization_factor=1., num_DA_blind=2)
        ens = run_da(ens, truth_biased, n_analysis=4)
        # the fallback EnKF must have been instantiated for the blind window
        assert hasattr(ens, '_bias_blind_filter')
        assert np.isfinite(ens.model.hist).all()


class TestParameterEstimation:

    def test_parameters_stay_within_limits_or_rejected(self, truth_unbiased):
        ens = make_ensemble(truth_unbiased, da_method=EnKF)
        ens = run_da(ens, truth_unbiased, n_analysis=6)
        alpha = ens.model.hist[-1, -ens.Na:, :]
        lo, hi = ens.model.alpha_lims['zeta']
        n_rejected = len(ens.rejected_analysis.times) if hasattr(ens, '_rejected_analysis') else 0
        assert np.all((alpha >= lo) & (alpha <= hi)) or n_rejected > 0

    def test_num_SE_only_freezes_parameters(self, truth_unbiased):
        ens = make_ensemble(truth_unbiased, da_method=EnKF, num_SE_only=3)
        Cdd = obs_covariance(truth_unbiased)
        alpha0 = ens.current_state[-ens.Na:, :].copy()
        for k, (d, t_d) in enumerate(zip(truth_unbiased.y_obs, truth_unbiased.t_obs)):
            ens.forecast_step(t_end=t_d)
            ens.analysis_step(d=d, Cdd=Cdd.copy())
            alpha = ens.current_state[-ens.Na:, :]
            if k + 1 < 3:
                # state-estimation only: parameters unchanged by the analysis
                np.testing.assert_allclose(alpha, alpha0)
            if k >= 4:
                break
        ens.model.close()
