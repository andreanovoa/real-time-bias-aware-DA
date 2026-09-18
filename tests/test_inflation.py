"""Checks for `romda.estimators.inflation` (Evensen 2009, Chap. 15).

The load-bearing invariant is that each filter kernel's stashed member-space
transform reproduces its own analysis exactly (Aa = Af X̂).
"""

from types import SimpleNamespace

import numpy as np
import pytest
from romda.estimators import EnKF, EnSRKF
from romda.estimators.inflation import multiplicative_inflation
from romda.models.physical import VdP


def test_multiplicative_inflation_scales_deviations():
    A = np.random.default_rng(0).standard_normal((5, 20))
    Aa = multiplicative_inflation(A, 1.5)
    np.testing.assert_allclose(np.mean(Aa, axis=1), np.mean(A, axis=1))
    np.testing.assert_allclose(np.std(Aa, axis=1), 1.5 * np.std(A, axis=1))


def _kernel_inputs(N=7, Nq=3, m=12, seed=0):
    rng = np.random.default_rng(seed)
    M = np.zeros((Nq, N))
    M[:, -Nq:] = np.eye(Nq)
    stub = SimpleNamespace(M=M, rng=rng, _MA=lambda A: M @ A)
    return stub, rng.standard_normal((N, m)), rng.standard_normal(Nq), 0.1 * np.eye(Nq)


@pytest.mark.parametrize('kernel', [EnKF, EnSRKF])
def test_analysis_transform_reproduces_kernel(kernel):
    stub, Af, d, Cdd = _kernel_inputs()
    Aa = kernel._analysis_kernel(stub, Af, d, Cdd)
    np.testing.assert_allclose(Aa, Af @ stub._analysis_transform, atol=1e-10)


def _vdp_analysis(zeta_lims=None, **kwargs):
    """One forecast and one analysis of a VdP ensemble with one estimated parameter, zeta."""
    ens = EnSRKF(parent_model=VdP(dt=1e-3), m=8, std_phi=0.1, std_alpha=dict(zeta=(40., 60.)),
                 inflation_factor=1.5, **kwargs)
    if zeta_lims:
        ens.model.alpha_lims = dict(zeta=zeta_lims)
    alpha0 = ens.current_state[ens.Nphi:].copy()  # constant over the forecast
    ens.forecast_step(t_end=0.05)
    d = ens.model.get_observables().mean(-1) * 1.1
    Aa = ens.analysis_step(d=d, Cdd=0.05 * np.diag(d ** 2), return_analysis=True)
    ens.model.close()
    return ens, Aa, alpha0


def _anomalies(A):
    return A - A.mean(axis=-1, keepdims=True)


def test_inflate_parameters_false_inflates_the_state_only():
    ens, Aa_all, _ = _vdp_analysis()
    _, Aa_state, _ = _vdp_analysis(inflate_parameters=False)
    Nphi = ens.Nphi
    assert ens.Na == 1 and ens.inflate_parameters
    np.testing.assert_array_equal(Aa_state[:Nphi], Aa_all[:Nphi])
    np.testing.assert_allclose(Aa_all[Nphi:].mean(-1), Aa_state[Nphi:].mean(-1))
    np.testing.assert_allclose(_anomalies(Aa_all[Nphi:]), 1.5 * _anomalies(Aa_state[Nphi:]))


def test_inflate_parameters_false_in_state_only_warm_up():
    """With start_param > 0 the rows after Nphi are the observables, not the parameters."""
    ens, Aa, alpha0 = _vdp_analysis(inflate_parameters=False, start_param=2)
    assert ens.inflation_history.factors == [1.5]
    np.testing.assert_array_equal(Aa[ens.Nphi:], alpha0)
    # the observable rows must not be validated against the parameter limits
    assert not ens.rejected_analysis.times


def test_inflate_parameters_false_in_rejected_analysis():
    """A rejected analysis returns the parameters to their forecast, which is not inflated."""
    ens, Aa, alpha0 = _vdp_analysis(inflate_parameters=False, zeta_lims=(49., 51.))
    assert len(ens.rejected_analysis.times) == 1
    np.testing.assert_array_equal(Aa[ens.Nphi:], alpha0)
