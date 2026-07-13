# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 19:02:23 2022

@author: an553
"""

import time
import numpy as np
from scipy import linalg

rng = np.random.default_rng(6)


# =================================================================================================================== #
# =========================================== ENSEMBLE FILTERS ====================================================== #
# =================================================================================================================== #


class Filter(object):
    r"""Base class for the ensemble filters.

    A filter is constructed with the observation operator and applied as a callable
    on the augmented forecast ensemble
    $\mathbf{A}^\mathrm{f} = [\boldsymbol{\phi}; \boldsymbol{\alpha}; \mathbf{y}]$
    (state, parameters, observables), returning the analysis ensemble.

    Parameters
    ----------
    M : np.ndarray
        Observation operator matrix of shape $(N_q, N_\phi + N_\alpha + N_q)$, i.e.,
        $\mathbf{M} = [\mathbf{0} \,|\, \mathbb{I}_{N_q}]$: the observed variables
        are the trailing $N_q$ rows of the augmented state.
    gamma : float, optional
        Bias-regularization factor. ``None`` (default) marks the filter as
        bias-unaware; bias-aware filters set a non-negative value.
    """

    def __init__(self, M, gamma=None):
        self._M = M  # observation operator matrix
        self.gamma = gamma # regularization factor for bias-aware filters (if None, not bias-aware)
    
    
    def __call__(self, *args, **kwargs):
        raise NotImplementedError('Filter call method not implemented.')

    def print_parameters(self):

        print('\n ------------------ Filter Config ------------------ ', 
              f'Filter class name = {self.filter_name}',
              f'observation operator shape (Nq x Nphi+Na+Nq) = {self._M.shape}',
              f'bias aware filter? {self.is_bias_aware}', 
              sep='\n\t')
        if self.is_bias_aware:
            print(f'\tregularization factor gamma = {self.gamma}')


    def observation_operator(self, Af):
        r"""Adjust the observation operator to the size of the provided ensemble.

        The stored operator maps the full augmented state to the observables as
        $\mathbf{M} = [\mathbf{0}_{N_q \times (N_\phi + N_\alpha)} \,|\, \mathbb{I}_{N_q}]$,
        so the observed variables are always the *trailing* $N_q$ rows of the state.
        When the parameter rows have been trimmed from ``Af`` (parameter estimation
        inactive), the trailing identity columns must be preserved and only the leading
        zero block shrinks.

        Parameters
        ----------
        Af : np.ndarray
            (Augmented) forecast ensemble whose leading dimension sets the state size.

        Returns
        -------
        np.ndarray
            Observation operator matrix adjusted to the state size of ``Af``.
        """
        Nq = self._M.shape[0]
        n_state = Af.shape[0]
        if n_state == self._M.shape[1]:
            return self._M
        return np.hstack((self._M[:, :n_state - Nq], self._M[:, -Nq:]))

    @property
    def filter_name(self):
        return self.__class__.__name__ 

    @property
    def is_bias_aware(self):
        return self.gamma is not None

#  ================================================================================================================== #
class EnSRKF(Filter):
    r"""Ensemble square-root Kalman filter (deterministic EnKF).

    Updates the ensemble mean with the Kalman gain and transforms the ensemble
    deviations with the symmetric square-root of the analysis covariance, so no
    stochastic observation perturbations are needed.

    References
    ----------
    Evensen (2009). *Data Assimilation: The Ensemble Kalman Filter.* Springer.
    """

    def __init__(self, M, gamma=None):
        super().__init__(M, gamma=None)

    def __call__(self, Af, d, Cdd):
        r"""Apply the square-root analysis update.

        Parameters
        ----------
        Af : np.ndarray
            Forecast ensemble at the analysis time, shape $(N, m)$.
        d : np.ndarray
            Observation vector at the analysis time, shape $(N_q,)$.
        Cdd : np.ndarray
            Observation error covariance matrix, shape $(N_q, N_q)$.

        Returns
        -------
        np.ndarray
            Analysis ensemble (the forecast ensemble is returned unchanged if the
            analysis is not real-valued).
        """
        m = Af.shape[1]
        M = self.observation_operator(Af)

        d = np.expand_dims(d, axis=1)
        psi_f_m = np.mean(Af, 1, keepdims=True)
        Psi_f = Af - psi_f_m

        # Mapped mean and deviations
        y = np.dot(M, psi_f_m)
        S = np.dot(M, Psi_f)

        # Matrix to invert
        C = (m - 1) * Cdd + np.dot(S, S.T)
        L, Z = linalg.eig(C)[:2]
        Linv = linalg.inv(np.diag(np.real(L)))

        X2 = np.dot(linalg.sqrtm(Linv), np.dot(Z.T, S))
        E, V = linalg.svd(X2)[1:]
        V = V.T
        if len(E) != m:  # case for only one eigenvalue (q=1). The rest zeros.
            E = np.hstack((E, np.zeros(m - len(E))))
        E = np.diag(E.real)

        sqrtIE = linalg.sqrtm(np.eye(m) - np.dot(E.T, E))

        # Analysis mean
        Cm = np.dot(Z, np.dot(Linv, Z.T))
        psi_a_m = psi_f_m + np.dot(Psi_f, np.dot(S.T, np.dot(Cm, (d - y))))

        # Analysis deviations
        Psi_a = np.dot(Psi_f, np.dot(V, np.dot(sqrtIE, V.T)))
        Aa = psi_a_m + Psi_a

        if not np.isreal(Aa).all():
            print('Aa not real, returning Af')
            return Af
        
        return Aa


#  ================================================================================================================== #

class EnKF(Filter):
    r"""Stochastic (perturbed-observations) ensemble Kalman filter.

    Each ensemble member assimilates a randomly perturbed copy of the observation,
    $\mathbf{d}_j \sim \mathcal{N}(\mathbf{d}, \mathbf{C}_{dd})$, following
    Evensen (2009), Eq. (9.27).

    References
    ----------
    Evensen (2009). *Data Assimilation: The Ensemble Kalman Filter.* Springer.
    """

    def __init__(self, M, gamma=None):
        super().__init__(M, gamma=None)


    def __call__(self, Af, d, Cdd):
        r"""Apply the stochastic (perturbed-observations) analysis update.

        Parameters
        ----------
        Af : np.ndarray
            Forecast ensemble at the analysis time, shape $(N, m)$.
        d : np.ndarray
            Observation vector at the analysis time, shape $(N_q,)$. A pre-perturbed
            observation ensemble of shape $(N_q, m)$ is also accepted.
        Cdd : np.ndarray
            Observation error covariance matrix, shape $(N_q, N_q)$.

        Returns
        -------
        np.ndarray
            Analysis ensemble (the forecast ensemble is returned unchanged if the
            analysis is not real-valued).
        """
        m = Af.shape[1]
        M = self.observation_operator(Af)
        
        psi_f_m = np.mean(Af, 1, keepdims=True)
        Psi_f = Af - psi_f_m

        # Create an ensemble of observations
        if d.ndim == 2 and d.shape[-1] == m:
            D = d
        else:
            D = rng.multivariate_normal(d, Cdd, m).transpose()

        # Mapped forecast matrix M(Af) and mapped deviations M(Af')
        Y = np.dot(M, Af)
        S = np.dot(M, Psi_f)

        # Matrix to invert
        C = (m - 1) * Cdd + np.dot(S, S.T)
        Cinv = linalg.inv(C)

        X = np.dot(S.T, np.dot(Cinv, (D - Y)))

        Aa = Af + np.dot(Af, X)

        if not np.isreal(Aa).all():
            Aa = Af
            print('Aa not real')
        return Aa



#  ================================================================================================================== #
class rBA_EnKF(Filter):

    r"""Regularized bias-aware ensemble Kalman filter (r-EnKF).

    The filter minimizes a cost function with three norms — the ensemble spread, the
    distance between the *bias-corrected* estimate and the data, and the bias norm
    weighted by the regularization factor $\gamma \ge 0$. The implementation follows
    the **corrected** equations (1a)–(1b) of the 2024 erratum:

    $$
    \boldsymbol{\psi}^\mathrm{a}_j = \boldsymbol{\psi}^\mathrm{f}_j +
    \mathbf{K} \left[ (\mathbb{I} + \mathbf{J})^\mathrm{T}
    (\mathbf{d}_j - \mathbf{y}^\mathrm{f}_j)
    - \gamma\, \mathbf{C}_{dd} \mathbf{C}_{bb}^{-1} \mathbf{J}^\mathrm{T}
    \mathbf{b}^\mathrm{f} \right],
    $$

    $$
    \mathbf{K} = \mathbf{C}^\mathrm{f}_{\psi\psi} \mathbf{M}^\mathrm{T}
    \left[ \mathbf{C}_{dd}
    + (\mathbb{I} + \mathbf{J})^\mathrm{T} (\mathbb{I} + \mathbf{J})\,
    \mathbf{M} \mathbf{C}^\mathrm{f}_{\psi\psi} \mathbf{M}^\mathrm{T}
    + \gamma\, \mathbf{C}_{dd} \mathbf{C}_{bb}^{-1} \mathbf{J}^\mathrm{T} \mathbf{J}\,
    \mathbf{M} \mathbf{C}^\mathrm{f}_{\psi\psi} \mathbf{M}^\mathrm{T} \right]^{-1},
    $$

    where $\mathbf{J} = \mathrm{d}\mathbf{b} / \mathrm{d}(\mathbf{M}\boldsymbol{\psi})$
    is the Jacobian of the bias estimator.

    !!! warning "Erratum"
        Equations (15)–(16) of the published paper contain small typos in the
        transposes of the Jacobian terms. The published and corrected forms coincide
        for a single observation ($N_q = 1$). This simplified form assumes
        uncorrelated observations ($\mathbf{C}_{dd}$ diagonal), as derived in the
        erratum (`docs/2023_CMAME_Erratum.pdf`).

    Parameters
    ----------
    M : np.ndarray
        Observation operator matrix.
    gamma : float, optional
        Bias-regularization factor (default 1.0). Larger values give more weight to
        the bias norm in the cost function.

    References
    ----------
    Nóvoa, Racca & Magri (2023). Inferring unknown unknowns: Regularized bias-aware
    ensemble Kalman filter. *Comput. Methods Appl. Mech. Eng.*, 418, 116502.
    [DOI: 10.1016/j.cma.2023.116502](https://doi.org/10.1016/j.cma.2023.116502).

    Nóvoa, Racca & Magri (2024). *Erratum* — corrected Eqs. (15)–(16)
    ([PDF](https://andreanovoa.github.io/real-time-bias-aware-DA/2023_CMAME_Erratum.pdf)).
    """

    def __init__(self, M, gamma=1.0):
        super().__init__(M, gamma=gamma)

    def __call__(self, Af, d, Cdd, Cbb, b, J):
        r"""Apply the regularized bias-aware analysis update.

        Parameters
        ----------
        Af : np.ndarray
            Forecast ensemble at the analysis time, augmented with the observables,
            shape $(N, m)$.
        d : np.ndarray
            Observation vector at the analysis time, shape $(N_q,)$. If the
            observations are biased, the caller must de-bias them before the call
            (see `Ensemble.analysis_step`).
        Cdd : np.ndarray
            Observation error covariance matrix, shape $(N_q, N_q)$.
        Cbb : np.ndarray
            Bias covariance matrix, shape $(N_q, N_q)$.
        b : np.ndarray
            Bias of the forecast observables,
            $\mathbf{y} = \mathbf{M}\mathbf{A}^\mathrm{f} + \mathbf{b}$.
            Shape $(N_q,)$, $(N_q, 1)$ or $(N_q, m)$.
        J : np.ndarray
            Jacobian of the bias with respect to the observables, shape $(N_q, N_q)$.

        Returns
        -------
        np.ndarray
            Analysis ensemble (the forecast ensemble is returned unchanged if the
            analysis is not real-valued).
        """
        m = Af.shape[1]
        Nq = len(d)
        M = self.observation_operator(Af)

        Iq = np.eye(Nq)
        # Mean and deviations of the ensemble
        Psi_f = Af - np.mean(Af, 1, keepdims=True)
        S = np.dot(M, Psi_f)
        Q = np.dot(M, Af)

        # Create an ensemble of observations
        D = rng.multivariate_normal(d, Cdd, m).transpose()

        assert b.shape[0] == Nq, f"Bias vector b must have the same length as the observation vector d. Got b.shape[0] = {b.shape[0]} and d.shape[0] = {Nq}"
        assert b.ndim in [1, 2], f"Bias vector b must be either 1D or 2D. Got b.ndim = {b.ndim}"

        if b.ndim == 1:
            B = np.repeat(b[:, np.newaxis], m, axis=1)
        else:
            if b.shape[-1] == m:
                B = b.copy()
            elif b.shape[-1] == 1:
                # B = rng.multivariate_normal(b.squeeze(), Cbb, m).transpose()
                B = np.repeat(b, m, axis=1)
            else:
                raise ValueError('b must have shape (Nq,), (Nq, 1) or (Nq, m), got {}'.format(b.shape))

        # Bias-corrected model observables
        Y = Q + B

        Cqq = np.dot(S, S.T)  # covariance of observations M Psi_f Psi_f.T M.T
        if np.array_equiv(Cdd, Cbb):
            CdWb = Iq
        else:
            CdWb = np.dot(Cdd, linalg.inv(Cbb))

        Cinv = (m - 1) * Cdd + np.dot(np.dot(Iq + J.T, Iq + J), Cqq) + \
            self.gamma * np.dot(CdWb, np.dot(np.dot(J.T, J), Cqq))
        

        K = np.dot(Psi_f, np.dot(S.T, linalg.inv(Cinv)))
        Aa = Af + np.dot(K, np.dot(Iq + J.T, D - Y) - self.gamma * np.dot(CdWb, np.dot(J.T, B)))

        # Compute cost function terms (this could be commented out to increase speed)
        if np.isreal(Aa).all():
            return Aa
        else:
            print('Aa not real')
            return Af


# =================================================================================================================== #