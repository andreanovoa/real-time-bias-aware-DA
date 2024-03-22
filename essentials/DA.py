# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 19:02:23 2022

@author: an553
"""

import time
import numpy as np
from scipy import linalg

rng = np.random.default_rng(6)


def dataAssimilation(ensemble, y_obs, t_obs, std_obs=0.2, **kwargs):

    # Print simulation parameters ##
    ensemble.print_model_parameters()
    ensemble.bias.print_bias_parameters()
    print_DA_parameters(ensemble, t_obs)

    # FORECAST UNTIL FIRST OBS ##
    time1 = time.time()
    Nt = int(np.round((t_obs[0] - ensemble.get_current_time) / ensemble.dt))

    ensemble = forecastStep(ensemble, Nt, **kwargs)

    if ensemble.bias_bayesian_update and ensemble.bias.N_ens != ensemble.m:
        raise AssertionError('Wrong ESN initialisation')

    print('Elapsed time to first observation: ' + str(time.time() - time1) + ' s')

    #  ASSIMILATION LOOP ##
    ti, ensemble.activate_bias_aware, ensemble.activate_parameter_estimation = 0, False, False
    time1 = time.time()
    print_i = int(len(t_obs) / 10) * np.array([range(10)])

    # Define observation covariance matrix
    Cdd = np.diag((std_obs * np.ones(ensemble.Nq))) * np.max(abs(y_obs), axis=0) ** 2

    print('Assimilation progress: \n\t0 % ', end="")
    while True:
        ensemble.activate_bias_aware = ti >= ensemble.num_DA_blind
        ensemble.activate_parameter_estimation = ti >= ensemble.num_SE_only

        # ------------------------------  PERFORM ASSIMILATION ------------------------------ #
        Aa = analysisStep(ensemble, y_obs[ti], Cdd)  # Analysis step

        # -------------------------  UPDATE STATE AND BIAS ESTIMATES------------------------- #
        ensemble.update_history(Aa[:-ensemble.Nq, :], update_last_state=True)

        # Update bias using the mean analysis innovation i^a = d - <y^a>
        Ya = ensemble.get_observables()
        ia = np.expand_dims(y_obs[ti], -1) - np.mean(Ya, -1, keepdims=True)

        # Update the bias state
        if not ensemble.bias.bayesian_update or ensemble.bias.name == 'NoBias':
            updated_state = dict(b=ia)
        else:
            i_data = np.expand_dims(y_obs[ti], axis=-1) - np.mean(Ya, -1, keepdims=True)
            u_hat, r_hat = ensemble.bias.reconstruct_state(i_data, Cdd=Cdd,
                                                           inflation=ensemble.bias.inflation, update_reservoir=False)

            if not ensemble.bias.update_reservoir:
                updated_state = dict(b=u_hat)
            else:
                updated_state = dict(b=u_hat, r=r_hat)

        ensemble.bias.update_history(**updated_state, update_last_state=True)

        # ------------------------------ FORECAST TO NEXT OBSERVATION ---------------------- #
        ti += 1
        if ti >= len(t_obs):
            print('100% ----------------\n')
            break
        elif ti in print_i:
            print(int(np.round(ti / len(t_obs) * 100, decimals=0)), end="% ")

        Nt = int(np.round((t_obs[ti] - ensemble.get_current_time) / ensemble.dt))
        # Parallel forecast
        ensemble = forecastStep(ensemble, Nt)

        assert ensemble.hist_t[-1] == ensemble.bias.hist_t[-1]
    print('Elapsed time during assimilation: ' + str(time.time() - time1) + ' s')
    return ensemble



# =================================================================================================================== #


def forecastStep(case, Nt, **kwargs):
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
    psi, t = case.time_integrate(Nt)
    case.update_history(psi, t)

    # Forecast ensemble bias and update its history
    if case.bias is not None:
        y = case.get_observable_hist(Nt)
        b, t_b = case.bias.time_integrate(t=t, y=y, **kwargs)
        case.bias.update_history(b, t_b)

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

    Af = case.get_current_state  # state matrix [modes + params] x m
    M = case.M.copy()
    Cdd = Cdd.copy()

    if case.est_a and not case.activate_parameter_estimation:
        Af = Af[:-case.Na, :]
        M = M[:, :-case.Na]

    # --------------- Augment state matrix with biased Y --------------- #
    y = case.get_observables()
    Af = np.vstack((Af, y))
    # ======================== APPLY SELECTED FILTER ======================== #
    if case.filter == 'EnSRKF':
        Aa = EnSRKF(Af, d, Cdd, M)
    elif case.filter == 'EnKF':
        Aa = EnKF(Af, d, Cdd, M)
    elif case.filter == 'rBA_EnKF':
        # ----------------- Retrieve bias and its Jacobian ----------------- #
        b = case.bias.get_current_bias
        J = case.bias.state_derivative()

        if case.bias.biased_observations:
            bd = np.mean(b - case.bias.get_current_innovations, axis=-1)
            d += bd


        # -------------- Define bias Covariance and the weight -------------- #
        k = case.regularization_factor
        Cbb = Cdd.copy()  # Bias covariance matrix same as obs cov matrix for now

        if case.activate_bias_aware:
            Aa = rBA_EnKF(Af, d, Cdd, Cbb, k, M, b, J)
        else:
            Aa = EnKF(Af, d, Cdd, M)
    else:
        raise ValueError('Filter ' + case.filter + ' not defined.')

    # ============================ CHECK PARAMETERS AND INFLATE =========================== #
    if not case.est_a:
        Aa = inflateEnsemble(Af, case.inflation, d=d, additive=True)
    else:
        if not case.activate_parameter_estimation:
            Af_params = Af[-case.Na:, :]
            Aa = inflateEnsemble(Af, case.inflation, d=d, additive=True)
            Aa = np.concatenate((Aa[:-case.Nq, :], Af_params, Aa[-case.Nq:, :]))
        else:
            is_physical, idx_alpha, d_alpha = checkParams(Aa, case)
            if is_physical:
                Aa = inflateEnsemble(Aa, case.inflation)
            else:
                case.is_not_physical()  # Count non-physical parameters
                if not hasattr(case, 'rejected_analysis'):
                    case.rejected_analysis = []
                if not case.constrained_filter:
                    print('reject-inflate')
                    Aa = inflateEnsemble(Af, case.reject_inflation, d=d, additive=True)
                    case.rejected_analysis.append([(case.get_current_time, np.dot(case.Ma, Aa), np.dot(case.Ma, Af), None)])
                else:
                    raise NotImplementedError('Constrained filter yet to test')
                    # # Try assimilating the parameters themselves
                    # Alphas = np.dot(case.Ma, Af)[idx_alpha]
                    # M_alpha = np.vstack([M, case.Ma[idx_alpha]])
                    # Caa = np.eye(len(idx_alpha)) * np.var(Alphas, axis=-1)  # ** 2
                    # # Store
                    # case.rejected_analysis.append([(case.t, np.dot(case.Ma, Aa),
                    #                                 np.dot(case.Ma, Af), (d_alpha, Caa))])
                    # C_zeros = np.zeros([case.Nq, len(idx_alpha)])
                    # Cdd_alpha = np.block([[Cdd, C_zeros], [C_zeros.T, Caa]])
                    # d_alpha = np.concatenate([d, d_alpha])
                    #
                    # if case.filter == 'EnSRKF':
                    #     Aa = EnSRKF(Af, d_alpha, Cdd_alpha, M_alpha)
                    # elif case.filter == 'EnKF':
                    #     Aa = EnKF(Af, d_alpha, Cdd_alpha, M_alpha)
                    # elif case.filter == 'rBA_EnKF':
                    #     if case.activate_bias_aware:
                    #         Aa = rBA_EnKF(Af, d_alpha, Cdd_alpha, Cbb, k, M_alpha, b, J)
                    #     else:
                    #         Aa = EnKF(Af, d_alpha, Cdd_alpha, M_alpha)
                    #
                    # # double check point in case the inflation takes the ensemble out of parameter range
                    # if checkParams(Aa, case)[0]:
                    #     print('\t ok c-filter case')
                    #     Aa = inflateEnsemble(Aa, case.inflation)
                    # else:
                    #     print('! not ok c-filter case')
                    #     Aa = inflateEnsemble(Af, case.inflation)

            # Aa = Aa[:-case.Nq, :]
    return Aa


# =================================================================================================================== #
def inflateEnsemble(A, rho, d=None, additive=False):
    if type(additive) is str and 'add' in additive:
        A[:len(d)] += np.array([d * (rho - 1)]).T

    A_m = np.mean(A, -1, keepdims=True)
    return A_m + rho * (A - A_m)


def checkParams(Aa, case):
    alphas, lower_bounds, upper_bounds, ii = [], [], [], 0
    for param in case.est_a:
        lims = case.params_lims[param]
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
                    print('t = {:.3f} r-i: min {} = {:.2f} < {:.2f}'.format(case.get_current_time,
                                                                            case.est_a[idx_], min_, bound_))
                else:
                    print('t = {:.3f} r-i: mean {} = {:.2f} < {:.2f}'.format(case.get_current_time,
                                                                             case.est_a[idx_], mean_, bound_))
        if any(break_up):
            idx = np.argwhere(break_up)
            if len(idx.shape) > 1:
                idx = idx.squeeze(axis=-1)
            for idx_ in idx:
                alpha_ = alphas[idx_]
                mean_, max_ = np.mean(alpha_), np.max(alpha_)
                bound_ = upper_bounds[idx_]
                if mean_ <= bound_:
                    print('t = {:.3f} r-i: max {} = {:.2f} > {:.2f}'.format(case.get_current_time,
                                                                            case.est_a[idx_], mean_, bound_))
                else:
                    print('t = {:.3f} r-i: mean {} = {:.2f} > {:.2f}'.format(case.get_current_time,
                                                                             case.est_a[idx_], mean_, bound_))
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

                print('t = {:.3f}: min{}={:.2f}<{:.2f}. d_alph={:.2f}'.format(case.get_current_time, case.est_a[idx_],
                                                                              min_, bound_, d_alpha[-1]))
            else:
                d_alpha.append(bound_ + 2 * np.std(alphas[idx_]))
                print('t = {:.3f}: mean{}={:.2f}<{:.2f}. d_alph={:.2f}'.format(case.get_current_time, case.est_a[idx_],
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
                print('t = {:.3f}: max{}={:.2f}>{:.2f}. d_alph={:.2f}'.format(case.get_current_time, case.est_a[idx_],
                                                                              max_, bound_, d_alpha[-1]))
            else:
                d_alpha.append(bound_ - 2 * np.std(alphas[idx_]))

                print('t = {:.3f}: mean{}={:.2f}>{:.2f}. d_alph={:.2f}'.format(case.get_current_time, case.est_a[idx_],
                                                                               mean_, bound_, d_alpha[-1]))

    return is_physical, np.array(idx_alpha, dtype=int), d_alpha

# =================================================================================================================== #
# =========================================== ENSEMBLE FILTERS ====================================================== #
# =================================================================================================================== #


def EnSRKF(Af, d, Cdd, M):
    """Ensemble Square-Root Kalman Filter based on Evensen (2009)
        Inputs:
            Af: forecast ensemble at time t
            d: observation at time t
            Cdd: observation error covariance matrix
            M: matrix mapping from state to observation space
        Returns:
            Aa: analysis ensemble (or Af is Aa is not real)
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

    if not np.isreal(Aa).all():
        Aa = Af
        print('Aa not real')
    return Aa


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


def rBA_EnKF(Af, d, Cdd, Cbb, k, M, b, J):
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
        Returns:
            Aa: analysis ensemble (or Af is Aa is not real)
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

    if b.ndim > 1 and b.shape[-1] == Nm:
        B = b
    else:
        if b.ndim == 1:
            b = np.expand_dims(b, axis=1)
        # B = rng.multivariate_normal(b.squeeze(), Cbb, Nm).transpose()
        B = np.repeat(b, Nm, axis=1)

    Y = Q + B

    Cqq = np.dot(S, S.T)  # covariance of observations M Psi_f Psi_f.T M.T
    if np.array_equiv(Cdd, Cbb):
        CdWb = Iq
    else:
        CdWb = np.dot(Cdd, linalg.inv(Cbb))

    Cinv = ((Nm - 1) * Cdd + np.dot(Iq + J, np.dot(Cqq, (Iq + J).T)) +
            k * np.dot(CdWb, np.dot(J, np.dot(Cqq, J.T))))

    K = np.dot(Psi_f, np.dot(S.T, linalg.inv(Cinv)))
    Aa = Af + np.dot(K, np.dot(Iq + J, D - Y) - k * np.dot(CdWb, np.dot(J, B)))


    if not np.isreal(Aa).all():
        print('Aa not real')
        Aa = Af

    return Aa


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
