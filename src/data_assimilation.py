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

    def __init__(self, M, gamma):
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
        """
        Adjust observation operator matrix in case of parameter estimation not active
        Inputs:
            Af: forecast ensemble at time t
        Returns:
            Observation operator matrix adjust in case of parameter estimation not active
        """
        return self._M[:, :Af.shape[0]]  

    @property
    def filter_name(self):
        return self.__class__.__name__ 

    @property
    def is_bias_aware(self):
        return self.gamma is not None

#  ================================================================================================================== #
class EnSRKF(Filter):
    """Ensemble Square-Root Kalman Filter based on Evensen (2009)
            Inputs:
                Af: forecast ensemble at time t
                d: observation at time t
                Cdd: observation error covariance matrix
                M: matrix mapping from state to observation space
            Returns:
                Aa: analysis ensemble (or Af is Aa is not real)
        """
    
    def __init__(self, M, gamma):
        super().__init__(M, gamma=None)

    def __call__(self, Af, d, Cdd):

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
        if len(E) is not m:  # case for only one eigenvalue (q=1). The rest zeros.
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
    """Ensemble Kalman Filter as derived in Evensen (2009) eq. 9.27.
            Parameters:
                Af: forecast ensemble at time t
                d: observation at time t
                Cdd: observation error covariance matrix
                M: matrix mapping from state to observation space
            Returns:
                Aa: analysis ensemble (or Af is Aa is not real)
        """
    

    def __init__(self, M, gamma):
        super().__init__(M, gamma=None)


    def __call__(self, Af, d, Cdd):
        
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

    """Regularized Bias-Aware Ensemble Kalman Filter (r-EnKF) based on the derivation in Nóvoa et al. (CMAME, 2024).  
        Inputs:
            Af: forecast ensemble at time t (augmented with Y)
            d: observation at time t
            Cdd: observation error covariance matrix
            Cbb: bias covariance matrix
            M: matrix mapping from state to observation space
            b: bias of the forecast observables (Y = MAf + B)   
            bd: bias of the observations (Dtrue = D + Bd)   
            J: derivative of the bias with respect to the input
            gamma: regularization factor for the bias term [default = 1.0]. 
                Higher values of gamma correspond to stronger regularization (i.e., more weight on the bias term in the cost function).
        Returns:
            Aa: analysis ensemble (or Af is Aa is not real)
    """

    def __init__(self, M, gamma):
        super().__init__(M, gamma=gamma)

    def __call__(self, Af, d, Cdd, Cbb, b, bd, J):

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
            BD = np.repeat(bd[:, np.newaxis], m, axis=1)
        else: 
            if b.shape[-1] == m:
                B = b.copy()
                BD = bd.copy()
            elif b.shape[-1] == 1:
                # B = rng.multivariate_normal(b.squeeze(), Cbb, m).transpose()
                B = np.repeat(b, m, axis=1)
                BD = np.repeat(bd, m, axis=1)
            else:
                raise ValueError('b must have shape (Nq,), (Nq, 1) or (Nq, m), got {}'.format(b.shape))

        # Unbias the states
        Y = Q + B
        D = D + BD

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