# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 19:02:23 2022

@author: an553
"""

import os as os
import time

import numpy as np
from scipy import linalg
# os.environ["OMP_NUM_THREADS"]= '1'
rng = np.random.default_rng(6)
data_folder = os.getcwd() + '\\data\\'


def dataAssimilation(ensemble,
                     obs, t_obs,
                     std_obs=0.05,
                     method='EnSRKF'):
    ensemble.filt = method
    dt = ensemble.dt
    ti = 0  # iterator

    print(obs[0])

    ensemble.printModelParameters()
    if ensemble.bias.name == 'ESN':
        ensemble.bias.printESNparameters()
    print('\n -------------------- ASSIMILATION PARAMETERS -------------------- \n',
          '\t Filter = {0}  \n\t bias = {1} \n'.format(method, ensemble.bias.name),
          '\t m = {} \n'.format(ensemble.m),
          '\t Time between analysis = {0:.2} s \n'.format(t_obs[-1] - t_obs[-2]),
          '\t Inferred params = {0} \n'.format(ensemble.est_p),
          '\t Ensemble standard deviation = {0}'.format(ensemble.std_psi)
          )
    if method == 'EnKFbias':
        print('\t Bias penalisation factor k = {}\n'.format(ensemble.bias.k))
    print(' --------------------------------------------')

    # ----------------------------- FORECAST UNTIL FIRST OBS ----------------------------- ##
    time1 = time.time()
    # if t_obs[ti] - ensemble.t > 1.:
    #     t1 = t_obs[ti] - (t_obs[-1] - t_obs[-2])*2
    #     Nt = int(np.round((t1 - ensemble.t) / dt))
    #     ensemble = forecastStep(ensemble, Nt, averaged=True, alpha=ensemble.alpha0)
    #
    # Nt = int(np.round((t_obs[ti] - ensemble.t) / dt))
    # ensemble = forecastStep(ensemble, Nt, averaged=False)

    if ensemble.start_ensemble_forecast > 0:
        t1 = t_obs[ti] - ensemble.start_ensemble_forecast
        Nt = int(np.round((t1 - ensemble.t) / dt))
        ensemble = forecastStep(ensemble, Nt, averaged=True, alpha=ensemble.alpha0)

    Nt = int(np.round((t_obs[ti] - ensemble.t) / dt))
    ensemble = forecastStep(ensemble, Nt, averaged=False)


    # print('dt', dt, 'dtESN', ensemble.bias.upsample * dt, 'dtA', t_obs[1]-t_obs[0])

    # if ensemble.bias.L == 100 and ensemble.std_psi == 0.25:
    # plt.figure()
    # plt.plot(ensemble.hist_t, np.mean(ensemble.getObservableHist()[0], -1))
    # plt.xlim([ensemble.t-0.05, ensemble.t])
    # plt.show()

    print('Elapsed time to first observation: ' + str(time.time() - time1) + ' s')

    # --------------------------------- ASSIMILATION LOOP -------------------------------- ##
    num_obs = len(t_obs)
    time1 = time.time()
    print_i = int(len(t_obs) / 4) * np.array([1, 2, 3])
    print('Assimilation progress: 0 % ', end="")

    ensemble.activate_bias_aware = False
    ensemble.activate_parameter_estimation = False

    # Define observation covariance matrix
    Cdd_norm = np.diag((std_obs * np.ones(np.size(obs[ti]))))
    # Cdd = np.diag((std_obs * np.ones(np.size(obs[ti])) * max(abs(obs[ti]))))
    while True:
        if ti >= ensemble.num_DA_blind:
            ensemble.activate_bias_aware = True
        if ti >= ensemble.num_SE_only:
            ensemble.activate_parameter_estimation = True
        # ------------------------------  PERFORM ASSIMILATION ------------------------------ #
        # Analysis step
        Cdd = Cdd_norm * abs(obs[ti])
        # print(Cdd)
        Aa, J = analysisStep(ensemble, obs[ti], Cdd, method)

        # Store cost function
        ensemble.hist_J.append(J)

        # Update state with analysis
        ensemble.psi = Aa
        ensemble.hist[-1] = Aa

        # Update bias as d - y^a
        y = ensemble.getObservables()  # TODO: investigate changes using y=y^a and not y=M psi^a
        b = obs[ti] - np.mean(y, -1)
        ensemble.bias.b = b
        # ensemble.bias.hist[-1] = b

        if len(ensemble.hist_t) != len(ensemble.hist):
            raise Exception('something went wrong')
        # ------------------------------ FORECAST TO NEXT OBSERVATION ---------------------- #
        # next observation index
        ti += 1
        if ti >= num_obs:
            print('100% ----------------\n')
            break
        elif ti in print_i:
            print(np.int(np.round(ti / len(t_obs) * 100, decimals=0)), end="% ")

        # Parallel forecast
        Nt = int(np.round((t_obs[ti] - ensemble.t) / dt))
        ensemble = forecastStep(ensemble, Nt)

    print('Elapsed time during assimilation: ' + str(time.time() - time1) + ' s')

    return ensemble

# =================================================================================================================== #


def forecastStep(case, Nt, averaged=False, alpha=None):
    """ Forecast step in the data assimilation algorithm. The state vector of
        one of the ensemble members is integrated in time
        Inputs:
            case: ensemble forecast as a class object
            Nt: number of timesteps to forecastself.N_units
            esn: class ESN
            truth: class containing the true state (only used for washout)
        Returns:
            psi: forecast state
            t: forecast time
            bias: output from ESN (U and r) if estimating bias, otherwise None
    """
    # TODO: modify the forecast so that Process 1 for case and process 2 for bias
    #  [might not be doable on washout]

    # Forecast ensemble and update the history
    psi, t = case.timeIntegrate(Nt=Nt, averaged=averaged, alpha=alpha)
    case.updateHistory(psi, t)
    # Forecast ensemble bias and update its history
    if case.bias is not None:
        y = case.getObservableHist(Nt + 1)
        b, t_b = case.bias.timeIntegrate(t=t, y=y)
        case.bias.updateHistory(b, t_b)
    return case


def analysisStep(case, d, Cdd, filt='EnSRKF'):
    """ Analysis step in the data assimilation algorithm. First, the ensemble
        is augmented with parameters and/or bias and/or state
        Inputs:
            case: ensemble forecast as a class object
            d: observation at time t
            Cdd: observation error covariance matrix
            filt: desired filter to use. Default EnSRKF bias-blind
        Returns:
            Aa: analysis ensemble (or Af is Aa is not real)
    """

    Af = case.psi.copy()  # state matrix [modes + params] x m
    M = case.M

    if case.est_p and not case.activate_parameter_estimation:
        Af_params = Af[-len(case.est_p):, :]
        Af = Af[:-len(case.est_p), :]
        M = M[:, :-len(case.est_p)]

    # ======================== APPLY SELECTED FILTER ======================== #

    if filt == 'EnKFbias':
        # --------------- Augment state matrix with biased Y --------------- #
        y = case.getObservables()
        Af = np.vstack((Af, y))
        # ----------------- Retrieve bias and its Jacobian ----------------- #
        b = case.bias.getBias(y)
        db = case.bias.stateDerivative(y)

        # -------------- Define bias Covariance and the weight -------------- #
        k = case.bias.k

        # Bias covariance matrix
        # Cbb = np.dot(B, B.T) / (m - 1.)
        # PsiB = b - np.mean(b, -1, keepdims=True)
        # Cbb = np.dot(PsiB, PsiB.T) / (case.m - 1.)
        Cbb = Cdd.copy()
        # Cbb = np.eye(len(d))

        if case.activate_bias_aware:
            Aa, J = EnKFbias(Af, d, Cdd, Cbb, k, M, b, db)
            # print('Activated bias-aware')
        else:
            Aa, J = EnKF(Af, d, Cdd, M)

    else:
        # --------------- Augment state matrix with biased Y --------------- #
        y = case.getObservables()
        # # TODO create a second bias aware filter that does what IN22. This will debug ESN implementation
        # #  ADD BIAS TO Y [obsolete] #
        # # If model bias provided, add the bias to the observables and assimilate
        # # the observations on the unbiased \tilde{y}. This is the JFM(2022) method.
        # if case.bias is not None:
        #     b = case.bias.getBias()
        #     y += np.expand_dims(b, 1)

        Af = np.vstack((Af, y))
        if filt == 'EnSRKF':
            Aa, J = EnSRKF(Af, d, Cdd, M)
        elif filt == 'EnKF':
            Aa, J = EnKF(Af, d, Cdd, M)
        else:
            raise ValueError('Filter ' + filt + ' not defined.')

    # ============================ CHECK PARAMETERS AND INFLATE =========================== #
    Aa = inflateEnsemble(Aa, case.inflation)
    if case.est_p:
        if not case.activate_parameter_estimation:
            return np.concatenate((Aa[:-np.size(y, 0), :], Af_params)), J
        else:
            isphysical = checkParams(Aa, case)
            if not isphysical:
                Aa = inflateEnsemble(Af, 1.05)
                if not checkParams(Aa, case):
                    print('booooo')
                    Aa = Af.copy()
                J = np.array([None] * 4)
                return Aa[:-np.size(y, 0), :], J

    return Aa[:-np.size(y, 0), :], J


# =================================================================================================================== #
def inflateEnsemble(A, rho):
    A_m = np.mean(A, -1, keepdims=True)
    return A_m + rho * (A - A_m)


def checkParams(Aa, case):
    isphysical = True
    ii = len(case.psi0)
    for param in case.est_p:
        lims = case.param_lims[param]
        vals = Aa[ii, :]
        if lims[0] is not None:  # lower bound
            isphysical = all([isphysical, all(vals >= lims[0])])
        if lims[1] is not None:  # upper bound
            isphysical = all([isphysical, all(vals <= lims[1])])
        ii += 1
    return isphysical


# =================================================================================================================== #
def EnSRKF(Af, d, Cdd, M):
    """Ensemble Square-Root Kalman Filter as derived in Evensen (2009)
        Inputs:
            Af: forecast ensemble at time t
            d: observation at time t
            Cdd: observation error covariance matrix
            M: matrix mapping from state to observation space
        Returns:
            Aa: analysis ensemble (or Af is Aa is not real)
    """
    m = np.size(Af, 1)
    d = np.expand_dims(d, axis=1)
    psi_f_m = np.mean(Af, 1, keepdims=True)
    Psi_f = Af - psi_f_m

    # Mapped mean and deviations
    y = np.dot(M, psi_f_m)
    S = np.dot(M, Psi_f)

    # Matrix to invert
    C = (m - 1) * Cdd + np.dot(S, S.T)
    [L, Z] = linalg.eig(C)
    L = np.diag(L.real)
    Linv = linalg.inv(L)

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

    J = np.array([None] * 4)
    if np.isreal(Aa).all():
        Ya = Aa[-len(d):]
        Wdd = linalg.inv(Cdd)
        Cpp = np.dot(Psi_f, Psi_f.T)
        Wpp = linalg.pinv(Cpp)

        J[0] = np.dot(np.mean(Af - Aa, -1).T, np.dot(Wpp, np.mean(Af - Aa, -1)))
        J[1] = np.dot(np.mean(d - Ya, -1).T, np.dot(Wdd, np.mean(d - Ya, -1)))

        dJdpsi = np.dot(Wpp, Af - Aa) + np.dot(M.T, np.dot(Wdd, Ya - d))
        J[3] = abs(np.mean(dJdpsi) / 2.)
        # print(J[3])
        return Aa, J
    else:
        print('Aa not real')
        return Af, J


# =================================================================================================================== #


def EnKF(Af, d, Cdd, M):
    """Ensemble Kalman Filter as derived in Evensen (2009) eq. 9.27.
        Inputs:
            Af: forecast ensemble at time t
            d: observation at time t
            Cdd: observation error covariance matrix
            M: matrix mapping from state to observation space
        Returns:
            Aa: analysis ensemble (or Af is Aa is not real)
    """
    m = np.size(Af, 1)

    psi_f_m = np.mean(Af, 1, keepdims=True)
    Psi_f = Af - psi_f_m

    # Create an ensemble of observations
    D = rng.multivariate_normal(d, Cdd, m).transpose()
    # D = np.repeat(np.expand_dims(d, 1), m, axis=1)


    # Mapped forecast matrix M(Af) and mapped deviations M(Af')
    Y = np.dot(M, Af)
    S = np.dot(M, Psi_f)

    # Matrix to invert
    C = (m - 1) * Cdd + np.dot(S, S.T)
    Cinv = linalg.inv(C)

    X = np.dot(S.T, np.dot(Cinv, (D - Y)))

    Aa = Af + np.dot(Af, X)

    J = np.array([None] * 4)
    if np.isreal(Aa).all():
        Ya = Aa[-len(d):]
        Cpp = np.dot(Psi_f, Psi_f.T)
        Wdd = linalg.inv(Cdd)
        Wpp = linalg.pinv(Cpp)

        J[0] = np.dot(np.mean(Af - Aa, -1).T, np.dot(Wpp, np.mean(Af - Aa, -1)))
        J[1] = np.dot(np.mean(np.expand_dims(d, -1) - Ya, -1).T, np.dot(Wdd, np.mean(np.expand_dims(d, -1) - Ya, -1)))

        dJdpsi = np.dot(Wpp, Af - Aa) + np.dot(M.T, np.dot(Wdd, Ya - D))
        J[3] = abs(np.mean(dJdpsi) / 2.)
        return Aa, J
    else:
        print('Aa not real')
        return Af, J


# =================================================================================================================== #


def EnKFbias(Af, d, Cdd, Cbb, k, M, b, dbdy):
    """ Bias-aware Ensemble Kalman Filter.
        Inputs:
            Af: forecast ensemble at time t (augmented with Y) [N x m]
            d: observation at time t_interpt [q x 1]
            Cdd: observation error covariance matrix [q x q]
            M: matrix mapping from state to observation space [q x N]
            b: bias of the forecast observables (\tilde{Y} = Y + B) [q x 1]
            dbdy: derivative of the bias with respect to the input
        Returns:
            Aa: analysis ensemble (or Af is Aa is not real)
    """
    m = np.size(Af, 1)
    q = len(d)

    Iq = np.eye(q)
    # Mean and deviations of the ensemble
    Psi_f = Af - np.mean(Af, 1, keepdims=True)
    S = np.dot(M, Psi_f)
    Y = np.dot(M, Af)



    # Create an ensemble of observations
    D = rng.multivariate_normal(d, Cdd, m).transpose()
    # B = rng.multivariate_normal(b, Cbb, m).transpose()
    # D = np.repeat(np.expand_dims(d, 1), m, axis=1)
    B = np.repeat(np.expand_dims(b, 1), m, axis=1)


    if np.array_equiv(Cdd, Cbb):
        Cqq = np.dot(S, S.T)
        Cinv = linalg.inv((m - 1) * Cdd + np.dot(np.dot(Iq + dbdy, Iq + dbdy)+k*np.dot(dbdy, dbdy), Cqq))
        K = np.dot(np.dot(Psi_f, S.T), Cinv)
        Aa = Af + np.dot(K, np.dot(Iq+dbdy, D-Y) - k*np.dot(dbdy, B))
    else:


        # Inverse of covariance
        Wbb = k * linalg.pinv(Cbb)
        Wdd = linalg.pinv(Cdd)
        # Compute W and Q
        W = np.dot((Iq + dbdy).T, np.dot(Wdd, (Iq + dbdy))) + np.dot(dbdy.T, np.dot(Wbb, dbdy))
        Q = np.dot((Iq + dbdy).T, np.dot(Wdd, (D + np.dot(dbdy, Y) - B))) + np.dot(dbdy.T,
                                                                                   np.dot(Wbb, (np.dot(dbdy, Y) - B)))
        C = linalg.pinv(W)
        K = np.dot(np.dot(Psi_f, S.T), linalg.inv((m - 1) * C + np.dot(S, S.T)))
        Aa = Af + np.dot(K, np.dot(C, Q) - np.dot(M, Af))



    # Inverse of covariance




    J = np.array([None] * 4)
    if np.isreal(Aa).all():
        Ya = Aa[-len(d):]
        Wdd = linalg.inv(Cdd)
        Cpp = np.dot(Psi_f, Psi_f.T)
        Wpp = linalg.pinv(Cpp)
        Wbb = linalg.pinv(Cbb)

        J[0] = np.dot(np.mean(Af - Aa, -1).T, np.dot(Wpp, np.mean(Af - Aa, -1)))
        J[1] = np.dot(np.mean(np.expand_dims(d, -1) - Ya, -1).T, np.dot(Wdd, np.mean(np.expand_dims(d, -1) - Ya, -1)))

        # J[2] = k * np.dot(np.mean(b, -1).T, np.dot(Wbb, np.mean(b, -1)))
        J[2] = k * np.dot(b.T, np.dot(Wbb, b))

        dbdpsi = np.dot(M.T, dbdy.T)
        dydpsi = dbdpsi + M.T
        ba = np.expand_dims(b, 1) + np.dot(dbdy, Ya - Y)

        dJdpsi = np.dot(Wpp, Af - Aa) + np.dot(dydpsi, np.dot(Wdd, Ya + ba - D)) + np.dot(dbdpsi, np.dot(Wbb, ba))
        J[3] = abs(np.mean(dJdpsi) / 2.)

        return Aa, J
    else:
        print('Aa not real')
        return Af, J

    # #  Ensemble bias implementation
    # Aa = np.zeros(np.shape(Af))
    # for mi in range(m):
    #     W = np.dot((Iq + dB_dY[mi]).T, np.dot(Wdd, (Iq + dB_dY[mi]))) + \
    #         np.dot(dB_dY[mi].T, np.dot(Wbb, dB_dY[mi]))
    #
    #     Q = np.dot((Iq + dB_dY[mi]).T, np.dot(Wdd, (D[:, mi] + np.dot(dB_dY[mi], Y[:, mi]) - B[:, mi]))) + \
    #         np.dot(dB_dY[mi].T, np.dot(Wbb, (np.dot(dB_dY[mi], Y[:, mi]) - B[:, mi])))
    #
    #     C = linalg.inv(W)
    #     X = np.dot(C, Q) - np.dot(M, Af[:, mi])
    #
    #     Aa[:, mi] = Af[:, mi] + np.dot(Psi_f, np.dot(S.T, np.dot(linalg.inv((m - 1) * C + np.dot(S, S.T)), X)))

# =================================================================================================================== #
