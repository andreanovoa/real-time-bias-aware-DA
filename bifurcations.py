# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 10:36:33 2022

@author: an553
"""

import pylab as plt
import numpy as np
from scipy.signal import find_peaks
import os as os
import pickle
import TAModels

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)
plt.rc('legend', facecolor='white', framealpha=1, edgecolor='white')


def one_dim_sweep(model, model_params: dict, sweep_p: str, range_p: list, plot=False):
    data_folder = 'data/' + model.name + '/'
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    y_all = []
    cases = []
    for p in range_p:
        model_params[sweep_p] = p
        case = model(model_params)
        filename = data_folder + TAfilename(case)
        # load or save
        if os.path.isfile(filename):
            with open(filename, 'rb') as f:
                case = pickle.load(f)
        else:
            psi_i, t = case.timeIntegrate(Nt=int(6. / case.dt))
            case.updateHistory(psi_i, t)
            with open(filename, 'wb') as f:
                pickle.dump(case, f)

        obs = case.getObservableHist()[0]
        if len(obs.shape) > 2:
            obs = np.squeeze(obs, axis=1)
        obs = obs[-int(1. / case.dt):, 0]
        # store
        y_all.append(obs)
        cases.append(case)
    # Plot 1D bigurcation diagram
    if plot:
        plt.figure()
        for y, val in zip(y_all, range_p):
            for k in [1, -1]:
                peaks = find_peaks(y * k)
                peaks = y[peaks[0]]
                plt.plot(np.ones(len(peaks)) * val, peaks, '.', color='b')
                plt.xlabel('$\\' + sweep_p + '$')
                plt.ylabel(case.getObservableHist()[1])
        plt.tight_layout()
        plt.show()

    return cases, y_all


# ---------------------------------------------------------------------------------------------------- #
def QR(M, N_exp):
    """ Compute an orthogonal basis, Q, and the exponential change
        in the norm along each element of the basis, S.
    """
    M = np.squeeze(M)
    Q = [None] * N_exp
    S = np.empty(N_exp)

    S[0] = np.linalg.norm(M[0])
    Q[0] = M[0] / S[0]

    for i in range(1, N_exp):

        # orthogonalize
        temp = 0
        for j in range(i):
            temp += np.dot(Q[j], M[i]) * Q[j]
        Q[i] = M[i] - temp

        # normalize
        S[i] = np.linalg.norm(Q[i])  # increase of the perturbation along i-th direction
        Q[i] /= S[i]

    return Q, np.log(S)


# ---------------------------------------------------------------------------------------------------- #
def TAfilename(case):
    name = case.name + '_' + case.law
    for key, value in case.getParameters().items():
        name += '_' + key + '{:.3e}'.format(value)
    return name


# ---------------------------------------------------------------------------------------------------- #
def Lyap_Classification(exponents):
    tol = 1e-8
    if exponents[0] < -tol:
        return 0  # Fixed point
    if exponents[0] > tol:
        return 3  # Chaotic
    if exponents[1] < -tol:
        return 1  # Limit cycle
    if abs(exponents[1]) < tol:
        return 2  # Quasiperiodic
    return print('something went wrong')


def VdP(psi):
    eta, mu = psi[:2]
    omega, nu, kappa, gamma, beta = 2 * np.pi * 120., 7., 3.4, 1.7, 70.
    law = 'tan'

    deta_dt = mu
    dmu_dt = - omega ** 2 * eta
    if law == 'cubic':  # Cubic law
        dmu_dt += mu * (2. * nu - kappa * eta ** 2)
    elif law == 'tan':  # arc tan model
        dmu_dt += mu * (beta ** 2 / (beta + kappa * eta ** 2) - beta + 2 * nu)
    else:
        raise TypeError("Undefined heat release law. Choose 'cubic' or 'tan'.")
        # dmu_dt  +=  mu * (2.*P['nu'] + P['kappa'] * eta**2 - P['gamma'] * eta**4) # higher order polinomial
    return np.hstack([deta_dt, dmu_dt])


def RK4(y0, dt, N, func, t=None, params=None):
    ''' 4th order explicit Tunge-Kutta integration method '''

    for i in range(N):
        k1 = dt * func(t, y0, params)
        k2 = dt * func(t, y0 + k1 / 2, params)
        k3 = dt * func(t, y0 + k2 / 2, params)
        k4 = dt * func(t, y0 + k3, params)

        y0 += (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return y0


# ---------------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    TAmodel = TAModels.VdP
    TAdict = {'law': 'cubic'}
    param = 'nu'
    range_param = np.arange(6, 7, 1.)

    # plot 1D diagram
    og_cases = one_dim_sweep(TAmodel, TAdict, param, range_param, plot=False)[0]

    # og_cases = [TAmodel(TAdict)]

    # compute 2 first Lyapunov exponents
    N_exp = 2
    eps = 1.e-10  # multiplication factor to make the orthonormalized perturbation infinitesimal
    N_orth = 10
    dt = og_cases[0].dt
    N = int(5. / dt)
    N_loops = N // N_orth
    N_dim = og_cases[0].N

    Lambdas = []
    for case in og_cases:
        q0 = case.psi.squeeze()
        SS = np.empty((N_loops, N_exp))  # initialize lyapunov exponents
        # N_exp randomly perturbed initial conditions
        q0_pert = []
        QQ = []
        QQp = []
        aa = []
        for _ in range(N_exp):
            q_per = np.random.rand(N_dim)
            aa.append(q_per / np.linalg.norm(q_per))

        # ------------------------- Compute lyapunov exponents ------------------------- #
        print('Lyapunov exponents computation : 0 % ', end="")
        S = 0
        for jj in range(N_loops):

            # q0 = RK4(q0, dt, N_orth, VdP)  # unperturbed initial condition on the attractor

            # perturb initial condition with orthonormal basis
            q0_pert = [q0 + eps * a for a in aa]

            # case.psi = q0
            # psi = case.timeIntegrate(N_orth)[0]
            # q0 = psi[-1]
            q0 = RK4(q0, dt, N_orth, case.timeDerivative, params=case.govEqnDict())

            for ii in range(N_exp):
                q0_pert[ii] = RK4(q0_pert[ii], dt, N_orth, case.timeDerivative, params=case.govEqnDict())

                # q0_pert[ii] = psi_pert[-1]

                # q0_pert[ii] = RK4(q0_pert[ii], dt, N_orth, VdP)  # unperturbed initial condition on the attractor

            # compute the final value of the N_exp perturbations
            a = [(q - q0).squeeze() / eps for q in q0_pert]
            # print(a[0].shape)
            # orthornormalize basis and compute exponents
            aa, S1 = QR(a, N_exp)

            # skip the first step, which does not start from the orthonormalized basis
            if jj > 0:
                S += S1
                SS[jj] = S / (jj * dt * N_orth)
                if jj % (N_loops // 5) == 0:
                    print(round(jj / N_loops * 100), end="% ")
        print('100 %')

        ## Compute Kaplan-Yorke dimension
        Lyap_exp = SS[-1]
        print('Lyapunov exponents      ', Lyap_exp)

        if Lyap_exp.sum() > 0:
            print('Error: not enough exponents have been computed. Increase N_exp to compute KY-dimension')

        else:
            sums = np.cumsum(Lyap_exp)  # cumulative sum of the Lyapunov exponents
            arg = np.argmax(sums < 0)  # index for which the cumulative sum becomes negative

            KY_dim = arg + sums[arg - 1] / np.abs(Lyap_exp[arg])

            print('Kaplan-Yorke dimension  ', KY_dim)

        ### Plot convergence of the exponents
        plt.rcParams["figure.figsize"] = (10, 5)
        plt.rcParams["font.size"] = 25

        plt.plot(np.arange(N_loops) * dt * N_orth, SS)
        plt.xlabel('Time')
        plt.ylabel('Lyapunov Exponents')
        plt.tight_layout(pad=0.2)

        # plt.figure()
        # plt.plot(np.array(QQ)[:, 0])
        # plt.plot(np.array(QQp)[:, 0])

        plt.show()
