"""Checks for `romda.estimators.inflation` (Evensen 2009, Chap. 15).

The load-bearing invariant is that each filter kernel's stashed member-space
transform reproduces its own analysis exactly (Aa = Af X̂).
"""

from types import SimpleNamespace

import numpy as np
import pytest
from romda.estimators import EnKF, EnSRKF
from romda.estimators.inflation import multiplicative_inflation


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
