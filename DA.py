# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 19:02:23 2022

@author: an553
"""

import time
import numpy as np
from scipy import linalg

rng = np.random.default_rng(6)


def dataAssimilation(ensemble, y_obs, t_obs, std_obs=0.2, wash_obs=None, wash_t=None):
    dt, ti = ensemble.dt, 0
    dt_obs = t_obs[-1] - t_obs[-2]

    # -----------------------------  Print simulation parameters ----------------------------- ##
    ensemble.print_model_parameters()
    ensemble.bias.print_bias_parameters()
    print_DA_parameters(ensemble, t_obs)

    # ----------------------------- FORECAST UNTIL FIRST OBS ----------------------------- ## TODO: clean up this first stage
    time1 = time.time()
    if wash_t is not None:
        t0 = wash_t[0] - dt_obs * ensemble.start_ensemble_forecast
        for key, val in zip(['wash_obs', 'wash_time'], [wash_obs, wash_t]):
            setattr(ensemble.bias, key, val)
    else:
        t0 = t_obs[ti] - dt_obs * ensemble.start_ensemble_forecast

    if t0 < t_obs[ti]:
        Nt = int(np.round((t0 - ensemble.t) / dt))
        ensemble = forecastStep(ensemble, Nt, averaged=True, alpha=ensemble.alpha0)

    # actual_std = abs(np.std(ensemble.psi, axis=-1) / np.mean(ensemble.psi, axis=-1))
    # prefix = ''
    # mean = np.mean(ensemble.psi, axis=-1)
    # if any(abs(actual_std[:ensemble.Nphi] - ensemble.std_psi) > 0.05):
    #     prefix = '(Inflated phi)'
    #     psi_inflated = ensemble.addUncertainty(mean[:ensemble.Nphi], ensemble.std_psi, ensemble.m, method='normal')
    #     ensemble.psi[:ensemble.Nphi] = psi_inflated
    #     ensemble.hist[-1] = ensemble.psi
    #     actual_std = abs(np.std(ensemble.psi, axis=-1) / np.mean(ensemble.psi, axis=-1))

    # print('Ensemble initial spread: {:.3f}, {} {}'.format(np.mean(actual_std[:ensemble.Nphi]),
    #                                                       actual_std[ensemble.Nphi:], prefix))

    Nt = int(np.round((t_obs[ti] - ensemble.t) / dt))
    ensemble = forecastStep(ensemble, Nt, averaged=False)
    print('Elapsed time to first observation: ' + str(time.time() - time1) + ' s')

    assert ensemble.t == t_obs[ti]

    # # ---------------------------------- REMOVE TRANSIENT -------------------------------- ##
    # i_transient = np.argmin(abs(ensemble.hist_t - ensemble.t_transient))
    # for structure in [ensemble, ensemble.bias]:
    #     if hasattr(structure, 'upsample'):
    #         i_transient = int(i_transient / structure.upsample)
    #     for key in ['hist', 'hist_t']:
    #         setattr(structure, key, getattr(structure, key)[i_transient:])
    # --------------------------------- ASSIMILATION LOOP -------------------------------- ##
    num_obs = len(t_obs)
    ensemble.activate_bias_aware, ensemble.activate_parameter_estimation = False, False
    time1, print_i = time.time(), int(len(t_obs) / 10) * np.array([range(10)])

    # Define observation covariance matrix
    Cdd_norm = np.diag((std_obs * np.ones(ensemble.Nq)))
    Cdd = Cdd_norm * np.max(abs(y_obs), axis=0) ** 2

    print('Assimilation progress: \n\t0 % ', end="")
    while True:
        if ti >= ensemble.num_DA_blind:
            ensemble.activate_bias_aware = True
        if ti >= ensemble.num_SE_only:
            ensemble.activate_parameter_estimation = True
        # ------------------------------  PERFORM ASSIMILATION ------------------------------ #
        # Analysis step
        # Cdd = Cdd_norm * abs(obs[ti]) # ????????? think this geometrically

        Aa, J = analysisStep(ensemble, y_obs[ti], Cdd)

        # Update state with analysis
        ensemble.psi = Aa
        ensemble.hist[-1] = Aa

        # Update bias as d - y^a  # TODO: Bayesian framework
        y = ensemble.getObservables()
        ensemble.bias.resetBias(y_obs[ti] - np.mean(y, -1))

        # Store cost function
        ensemble.hist_J.append(J)

        assert ensemble.hist_t[-1] == ensemble.bias.hist_t[-1]

        # ------------------------------ FORECAST TO NEXT OBSERVATION ---------------------- #
        # next observation index
        ti += 1
        if ti >= num_obs:
            print('100% ----------------\n')
            break
        elif ti in print_i:
            print(int(np.round(ti / len(t_obs) * 100, decimals=0)), end="% ")

        Nt = int(np.round((t_obs[ti] - ensemble.t) / dt))
        ensemble = forecastStep(ensemble, Nt)  # Parallel forecast

    print('Elapsed time during assimilation: ' + str(time.time() - time1) + ' s')
    return ensemble


# =================================================================================================================== #


def forecastStep(case, Nt, averaged=False, alpha=None):
    """ Forecast step in the data assimilation algorithm. The state vector of
        one of the ensemble members is integrated in time
        Inputs:
            case: ensemble forecast as a class object
            Nt: number of time steps to forecast
            averaged: is the ensemble being forcast averaged?
            alpha: changeable parameters of the problem 
        Returns:
            case: updated case forecast Nt time steps
    """

    # Forecast ensemble and update the history

    psi, t = case.timeIntegrate(Nt=Nt, averaged=averaged, alpha=alpha)
    case.updateHistory(psi, t)

    # Forecast ensemble bias and update its history
    if case.bias is not None:
        y = case.getObservableHist(Nt)
        b, t_b = case.bias.timeIntegrate(t=t, y=y)
        case.bias.updateHistory(b, t_b)

    if case.hist_t[-1] != case.bias.hist_t[-1]:
        raise AssertionError('t assertion', case.hist_t[-1], case.bias.hist_t[-1])
    return case


def analysisStep(case, d, Cdd):
    """ Analysis step in the data assimilation algorithm. First, the ensemble
        is augmented with parameters and/or bias and/or state
        Inputs:
            case: ensemble forecast as a class object
            d: observation at time t
            Cdd: observation error covariance matrix
        Returns:
            Aa: analysis ensemble (or Af is Aa is not real)
    """

    Af = case.psi.copy()  # state matrix [modes + params] x m
    M = case.M.copy()
    Cdd = Cdd.copy()

    if case.est_a and not case.activate_parameter_estimation:
        Af = Af[:-case.Na, :]
        M = M[:, :-case.Na]

    # --------------- Augment state matrix with biased Y --------------- #
    y = case.getObservables()
    Af = np.vstack((Af, y))
    # ======================== APPLY SELECTED FILTER ======================== #
    if case.filter == 'EnSRKF':
        Aa, cost = EnSRKF(Af, d, Cdd, M, get_cost=case.get_cost)
    elif case.filter == 'EnKF':
        Aa, cost = EnKF(Af, d, Cdd, M, get_cost=case.get_cost)
    elif case.filter == 'rBA_EnKF':
        # ----------------- Retrieve bias and its Jacobian ----------------- #
        b = case.bias.getBias
        J = case.bias.stateDerivative()
        # -------------- Define bias Covariance and the weight -------------- #
        k = case.regularization_factor

        Cbb = Cdd.copy()  # Bias covariance matrix same as obs cov matrix for now
        if case.activate_bias_aware:
            Aa, cost = rBA_EnKF(Af, d, Cdd, Cbb, k, M, b, J, get_cost=case.get_cost)
        else:
            Aa, cost = EnKF(Af, d, Cdd, M, get_cost=case.get_cost)
    else:
        raise ValueError('Filter ' + case.filter + ' not defined.')

    # ============================ CHECK PARAMETERS AND INFLATE =========================== #
    if not case.est_a:
        return Aa[:-case.Nq, :], cost
    else:
        if not case.activate_parameter_estimation:
            Af_params = Af[-case.Na:, :]
            Aa = inflateEnsemble(Aa, case.inflation)
            return np.concatenate((Aa[:-case.Nq, :], Af_params)), cost
        else:
            is_physical, idx_alpha, d_alpha = checkParams(Aa, case)
            if is_physical:
                Aa = inflateEnsemble(Aa, case.inflation)
                return Aa[:-case.Nq, :], cost
            case.is_not_physical()  # Count non-physical parameters

            print('not physical')
            if not hasattr(case, 'rejected_analysis'):
                case.rejected_analysis = []

            if not case.constrained_filter:
                print('reject-inflate')
                Aa = inflateEnsemble(Af, case.reject_inflation, d=d, additive=True)
                case.rejected_analysis.append([(case.t, np.dot(case.Ma, Aa), np.dot(case.Ma, Af), None)])
                return Aa[:-case.Nq, :], cost
            else:
                # Try assimilating the parameters themselves
                Alphas = np.dot(case.Ma, Af)[idx_alpha]
                M_alpha = np.vstack([M, case.Ma[idx_alpha]])

                Caa = np.eye(len(idx_alpha)) * np.var(Alphas, axis=-1)  # ** 2
                # Store
                case.rejected_analysis.append([(case.t, np.dot(case.Ma, Aa),
                                                np.dot(case.Ma, Af), (d_alpha, Caa))])

                C_zeros = np.zeros([case.Nq, len(idx_alpha)])
                Cdd_alpha = np.block([[Cdd, C_zeros],
                                      [C_zeros.T, Caa]])
                d_alpha = np.concatenate([d, d_alpha])

                if case.filter == 'EnSRKF':
                    Aa, cost = EnSRKF(Af, d_alpha, Cdd_alpha, M_alpha, get_cost=case.get_cost)
                elif case.filter == 'EnKF':
                    Aa, cost = EnKF(Af, d_alpha, Cdd_alpha, M_alpha, get_cost=case.get_cost)
                elif case.filter == 'rBA_EnKF':
                    if case.activate_bias_aware:
                        Aa, cost = rBA_EnKF(Af, d_alpha, Cdd_alpha, Cbb, k, M_alpha, b, J, get_cost=case.get_cost)
                    else:
                        Aa, cost = EnKF(Af, d_alpha, Cdd_alpha, M_alpha, get_cost=case.get_cost)

                # # double check point in case the inflation takes the ensemble out of parameter range
                if checkParams(Aa, case)[0]:
                    print('\t ok c-filter case')
                    Aa = inflateEnsemble(Aa, case.inflation)
                    return Aa[:-case.Nq, :], cost

                print('!', end="")
                print('not ok c-filter case')
                Aa = inflateEnsemble(Af, case.inflation)
                return Aa[:-case.Nq, :], cost


# =================================================================================================================== #
def inflateEnsemble(A, rho, d=None, additive=False):
    if additive:
        # y_pert = d * 1e-4
        A[:len(d)] += d * (rho - 1)
    #     print(A.shape, d.shape, np.mean(A, axis=-1))
    #     raise ValueError
    #
    #     return A + pert
    # else:
    A_m = np.mean(A, -1, keepdims=True)
    return A_m + rho * (A - A_m)


def checkParams(Aa, case):
    alphas, lower_bounds, upper_bounds, ii = [], [], [], 0
    for param in case.est_a:
        lims = case.param_lims[param]
        vals = Aa[case.Nphi + ii, :]
        lower_bounds.append(lims[0])
        upper_bounds.append(lims[1])
        alphas.append(vals)
        ii += 1

    break_low = [lims is not None and any(val < lims) for val, lims in zip(alphas, lower_bounds)]
    break_up = [lims is not None and any(val > lims) for val, lims in zip(alphas, upper_bounds)]

    is_physical = True
    if not any(np.append(break_low, break_up)):
        return is_physical, None, None

    #  -----------------------------------------------------------------
    is_physical = False

    if not case.constrained_filter:
        if any(break_low):
            idx = np.argwhere(break_low).squeeze(axis=-1)
            for idx_ in idx:
                alpha_ = alphas[idx_]
                mean_, min_ = np.mean(alpha_), np.min(alpha_)
                bound_ = lower_bounds[idx_]
                if mean_ >= bound_:
                    print('t = {:.3f} reject-inflate: min {} = {:.2f} < {:.2f}'.format(case.t,
                                                                                       case.est_a[idx_], min_, bound_))
                else:
                    print('t = {:.3f} reject-inflate: mean {} = {:.2f} < {:.2f}'.format(case.t,
                                                                                        case.est_a[idx_], mean_,
                                                                                        bound_))
        if any(break_up):
            idx = np.argwhere(break_up)
            if len(idx.shape) > 1:
                idx = idx.squeeze(axis=-1)
            for idx_ in idx:
                alpha_ = alphas[idx_]
                mean_, max_ = np.mean(alpha_), np.max(alpha_)
                bound_ = upper_bounds[idx_]
                if mean_ <= bound_:
                    print('t = {:.3f} reject-inflate: max {} = {:.2f} > {:.2f}'.format(case.t,
                                                                                       case.est_a[idx_], max_, bound_))
                else:
                    print('t = {:.3f} reject-inflate: mean {} = {:.2f} > {:.2f}'.format(case.t,
                                                                                        case.est_a[idx_], mean_,
                                                                                        bound_))
        return is_physical, None, None

    #  -----------------------------------------------------------------
    idx_alpha, d_alpha = [], []
    if any(break_low):
        idx = np.argwhere(break_low).squeeze(axis=0)
        for idx_ in idx:
            idx_alpha.append(idx_)
            alpha_ = alphas[idx_]
            mean_, min_ = np.mean(alpha_), np.min(alpha_)
            bound_ = lower_bounds[idx_]
            if mean_ >= bound_:
                d_alpha.append(np.max(alpha_) + np.std(alpha_))

                print('t = {:.3f}: min {} = {:.2f} < {:.2f}. d_alpha = {:.2f}'.format(case.t, case.est_a[idx_],
                                                                                      min_, bound_, d_alpha[-1]))
            else:
                d_alpha.append(bound_ + 2 * np.std(alphas[idx_]))
                print('t = {:.3f}: mean {} = {:.2f} < {:.2f}. d_alpha = {:.2f}'.format(case.t, case.est_a[idx_],
                                                                                       mean_, bound_, d_alpha[-1]))

    if any(break_up):
        idx = np.argwhere(break_up)
        if len(idx.shape) > 1:
            idx = idx.squeeze(axis=-1)
        for idx_ in idx:
            idx_alpha.append(idx_)
            alpha_ = alphas[idx_]

            mean_, max_ = np.mean(alpha_), np.max(alpha_)
            bound_ = upper_bounds[idx_]

            # min_ = np.min(alphas[idx_])
            if mean_ < bound_:
                # vv = alpha_[np.argwhere(alpha_ < bound_)]
                d_alpha.append(np.min(alpha_) - np.std(alpha_))
                # elif min_ < bound_:
                #     d_alpha.append(min_)
                print('t = {:.3f}: max {} = {:.2f} > {:.2f}. d_alpha = {:.2f}'.format(case.t, case.est_a[idx_],
                                                                                      max_, bound_, d_alpha[-1]))
            else:
                d_alpha.append(bound_ - 2 * np.std(alphas[idx_]))

                print('t = {:.3f}: mean {} = {:.2f} > {:.2f}. d_alpha = {:.2f}'.format(case.t, case.est_a[idx_],
                                                                                       mean_, bound_, d_alpha[-1]))

    return is_physical, np.array(idx_alpha, dtype=int), d_alpha


# =================================================================================================================== #
# =========================================== ENSEMBLE FILTERS ====================================================== #
# =================================================================================================================== #

def EnSRKF(Af, d, Cdd, M, get_cost=False):
    """Ensemble Square-Root Kalman Filter based on Evensen (2009)
        Inputs:
            Af: forecast ensemble at time t
            d: observation at time t
            Cdd: observation error covariance matrix
            M: matrix mapping from state to observation space
            get_cost: do you want to compute the cost function?
        Returns:
            Aa: analysis ensemble (or Af is Aa is not real)
            cost: (optional) calculation of the DA cost function and its derivative
    """
    m = np.size(Af, 1)  # ensemble size
    d = np.expand_dims(d, axis=1)
    psi_f_m = np.mean(Af, 1, keepdims=True)
    Psi_f = Af - psi_f_m

    # Mapped mean and deviations
    y = np.dot(M, psi_f_m)
    S = np.dot(M, Psi_f)

    # Matrix to invert
    C = (m - 1) * Cdd + np.dot(S, S.T)
    L, Z = linalg.eig(C)
    Linv = linalg.inv(np.diag(L.real))

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

    cost = np.array([None] * 4)
    if np.isreal(Aa).all():
        if get_cost:  # Compute cost function terms
            Ya = Aa[-len(d):]
            Wdd = linalg.inv(Cdd)
            Cpp = np.dot(Psi_f, Psi_f.T)
            Wpp = linalg.pinv(Cpp)

            cost[0] = np.dot(np.mean(Af - Aa, -1).T, np.dot(Wpp, np.mean(Af - Aa, -1)))
            cost[1] = np.dot(np.mean(d - Ya, -1).T, np.dot(Wdd, np.mean(d - Ya, -1)))

            dJdpsi = np.dot(Wpp, Af - Aa) + np.dot(M.T, np.dot(Wdd, Ya - d))
            cost[3] = abs(np.mean(dJdpsi) / 2.)
        return Aa, cost
    else:
        print('Aa not real')
        return Af, cost


def EnKF(Af, d, Cdd, M, get_cost=False):
    """Ensemble Kalman Filter as derived in Evensen (2009) eq. 9.27.
        Inputs:
            Af: forecast ensemble at time t
            d: observation at time t
            Cdd: observation error covariance matrix
            M: matrix mapping from state to observation space
            get_cost: do you want to compute the cost function?
        Returns:
            Aa: analysis ensemble (or Af is Aa is not real)
            cost: (optional) calculation of the DA cost function and its derivative
    """
    m = np.size(Af, 1)

    psi_f_m = np.mean(Af, 1, keepdims=True)
    Psi_f = Af - psi_f_m

    # Create an ensemble of observations
    D = rng.multivariate_normal(d, Cdd, m).transpose()

    # Mapped forecast matrix M(Af) and mapped deviations M(Af')
    Y = np.dot(M, Af)
    S = np.dot(M, Psi_f)

    # Matrix to invert
    C = (m - 1) * Cdd + np.dot(S, S.T)
    Cinv = linalg.inv(C)

    X = np.dot(S.T, np.dot(Cinv, (D - Y)))

    Aa = Af + np.dot(Af, X)

    cost = np.array([None] * 4)
    if np.isreal(Aa).all():
        if get_cost:  # Compute cost function terms
            Ya = Aa[-len(d):]
            Cpp = np.dot(Psi_f, Psi_f.T)
            Wdd = linalg.inv(Cdd)
            Wpp = linalg.pinv(Cpp)

            cost[0] = np.dot(np.mean(Af - Aa, -1).T, np.dot(Wpp, np.mean(Af - Aa, -1)))
            cost[1] = np.dot(np.mean(np.expand_dims(d, -1) - Ya, -1).T,
                             np.dot(Wdd, np.mean(np.expand_dims(d, -1) - Ya, -1)))
            dJdpsi = np.dot(Wpp, Af - Aa) + np.dot(M.T, np.dot(Wdd, Ya - D))
            cost[3] = abs(np.mean(dJdpsi) / 2.)
        return Aa, cost
    else:
        print('Aa not real')
        return Af, cost


def rBA_EnKF(Af, d, Cdd, Cbb, k, M, b, J, get_cost=False):
    """ Bias-aware Ensemble Kalman Filter.
        Inputs:
            Af: forecast ensemble at time t (augmented with Y) [N x m]
            d: observation at time t [Nq x 1]
            Cdd: observation error covariance matrix [Nq x Nq]
            Cbb: bias covariance matrix [Nq x Nq]
            k: bias penalisation factor
            M: matrix mapping from state to observation space [Nq x N]
            b: bias of the forecast observables (Y = MAf + B) [Nq x 1]
            J: derivative of the bias with respect to the input [Nq x Nq]
            get_cost: do you want to compute the cost function?
        Returns:
            Aa: analysis ensemble (or Af is Aa is not real)
            cost: (optional) calculation of the DA cost function and its derivative
    """
    Nm = np.size(Af, 1)
    Nq = len(d)

    Iq = np.eye(Nq)
    # Mean and deviations of the ensemble
    Psi_f = Af - np.mean(Af, 1, keepdims=True)
    S = np.dot(M, Psi_f)
    Q = np.dot(M, Af)

    # Create an ensemble of observations
    D = rng.multivariate_normal(d, Cdd, Nm).transpose()
    B = rng.multivariate_normal(b, Cbb, Nm).transpose()
    # B = np.repeat(np.expand_dims(b, 1), Nm, axis=1)

    Y = Q + B

    Cqq = np.dot(S, S.T)  # covariance of observations M Psi_f Psi_f.T M.T
    if np.array_equiv(Cdd, Cbb):
        CdWb = Iq
    else:
        CdWb = np.dot(Cdd, linalg.inv(Cbb))

    Cinv = (Nm - 1) * Cdd + np.dot(Iq + J,
                                   np.dot(Cqq, (Iq + J).T)) + k * np.dot(CdWb,
                                                                         np.dot(J, np.dot(Cqq, J.T)))
    K = np.dot(Psi_f, np.dot(S.T, linalg.inv(Cinv)))
    Aa = Af + np.dot(K, np.dot(Iq + J, D - Y) - k * np.dot(CdWb, np.dot(J, B)))

    # Compute cost function terms (this could be commented out to increase speed)
    cost = np.array([None] * 4)
    if np.isreal(Aa).all():
        if get_cost:  # Compute cost function terms
            ba = b + np.dot(J, np.mean(np.dot(M, Aa) - Q, -1))
            Ya = np.dot(M, Aa) + np.expand_dims(ba, -1)
            Wdd = linalg.inv(Cdd)
            Wpp = linalg.pinv(np.dot(Psi_f, Psi_f.T))
            Wbb = k * linalg.inv(Cbb)

            cost[0] = np.dot(np.mean(Af - Aa, -1).T, np.dot(Wpp, np.mean(Af - Aa, -1)))
            cost[1] = np.dot(np.mean(np.expand_dims(d, -1) - Ya, -1).T,
                             np.dot(Wdd, np.mean(np.expand_dims(d, -1) - Ya, -1)))
            cost[2] = np.dot(ba.T, np.dot(Wbb, ba))

            dbdpsi = np.dot(M.T, J.T)
            dydpsi = dbdpsi + M.T

            Ba = np.repeat(np.expand_dims(ba, 1), Nm, axis=1)
            dJdpsi = np.dot(Wpp, Af - Aa) + np.dot(dydpsi, np.dot(Wdd, Ya - D)) + np.dot(dbdpsi, np.dot(Wbb, Ba))
            cost[3] = abs(np.mean(dJdpsi) / 2.)
        return Aa, cost
    else:
        print('Aa not real')
        return Af, cost


# =================================================================================================================== #


def print_DA_parameters(ensemble, t_obs):
    print('\n -------------------- ASSIMILATION PARAMETERS -------------------- \n',
          '\t Filter = {0}  \n\t bias = {1} \n'.format(ensemble.filter, ensemble.bias.name),
          '\t m = {} \n'.format(ensemble.m),
          '\t Time steps between analysis = {} \n'.format(ensemble.dt_obs),
          '\t Inferred params = {0} \n'.format(ensemble.est_a),
          '\t Inflation = {0} \n'.format(ensemble.inflation),
          '\t Ensemble std(psi0) = {}\n'.format(ensemble.std_psi),
          '\t Ensemble std(alpha0) = {}\n'.format(ensemble.std_a),
          '\t Number of analysis steps = {}, t0={}, t1={}'.format(len(t_obs), t_obs[0], t_obs[-1])
          )
    if ensemble.filter == 'rBA_EnKF':
        print('\t Bias penalisation factor k = {}\n'.format(ensemble.regularization_factor))
    print(' --------------------------------------------')
