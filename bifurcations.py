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
from scipy.integrate import odeint, solve_ivp
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)
plt.rc('legend', facecolor='white', framealpha=1, edgecolor='white')

import time
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


# ==================================================================================================== #
if __name__ == '__main__':

    # ----------------------------- Select working model ----------------------------- #
    TAmodel = TAModels.VdP
    TAdict = {'law': 'cubic'}

    # ------------------------------- Plot 1D diagram ------------------------------- #
    param = 'nu'  # desired parameter to sweep
    range_param = np.arange(6, 7, 1.)
    og_cases = one_dim_sweep(TAmodel, TAdict, param, range_param, plot=False)[0]

    # ------------------------- Compute lyapunov exponents -------------------------- #
    N_exp = 2   # compute the N_exp-first Lyapunov exponents
    eps = 1e-6  # multiplication factor to make the orthonormalized perturbation infinitesimalv
    N_dim = og_cases[0].N

    Lambdas = []
    for case in og_cases:
        N = int(100. / case.dt)
        N_orth = int(5 / case.dt)
        # NOTE:  N_orth*dt should easily be 5 Lyapunov times with eps small enough. Because in 5 lyapunov
        # times the magnitude of the perturbation increases only be 2^5=32 times on average.
        # NOTE 2: if N_orth too large, do not trust exponents, only sign as the slope can be misleading due to the
        # chaotic saturation and/or machine precision

        N_loops = N // N_orth
        # initial state
        q0 = case.psi
        t1 = case.t
        SS = np.empty((N_loops, N_exp))  # initialize lyapunov exponents
        # N_exp randomly perturbed initial conditions
        q0_pert = []
        aa = []
        for _ in range(N_exp):
            q_per = np.random.rand(N_dim)
            aa.append(q_per / np.linalg.norm(q_per))

        plt.figure()
        fun = case.timeDerivative
        params = case.govEqnDict()
        print('Lyapunov exponents computation : 0 % ', end="")
        S = 0
        time_1 = time.time()
        time_int = 0.
        for jj in range(N_loops):
            t0 = t1
            t1 = t0 + N_orth * case.dt

            # t_vect = np.linspace(t0, t1, N_orth + 1)
            # perturb initial condition with orthonormal basis and propagate them
            q0_pert = [q0.squeeze() + eps * a for a in aa]
            # integrate perturbed cases
            t1111 = time.time()
            for ii in range(N_exp):
                sol = solve_ivp(fun, t_span=([t0, t1]), y0=q0_pert[ii], args=(params,))#, t_eval=t_vect)
                q0_pert[ii] = sol.y[:, -1]

            sol = solve_ivp(fun, t_span=([t0, t1]), y0=q0.squeeze(), args=(params,))#, t_eval=t_vect)
            q0 = sol.y[:, -1]
            # print('integration time = ', time.time() - t1111)
            time_int += time.time() - t1111
            plt.plot(sol.t, sol.y.T)

            t1111 = time.time()
            # compute the final value of the N_exp perturbations
            a = [(q - q0) / eps for q in q0_pert]
            # orthornormalize basis and compute exponents
            aa, S1 = QR(a, N_exp)

            # skip the first step, which does not start from the orthonormalized basis
            if jj > 0:
                S += S1
                SS[jj] = S / (jj * case.dt * N_orth)
                if jj % (N_loops // 5) == 0:
                    print(round(jj / N_loops * 100), end="% ")
        print('100 %')
        print('total time  = ', time.time() - time_1)
        print('integration time  = ', time_int)

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

        plt.figure()
        ### Plot convergence of the exponents
        plt.rcParams["figure.figsize"] = (10, 5)
        plt.rcParams["font.size"] = 25

        plt.plot(np.arange(N_loops) * case.dt * N_orth, SS)
        plt.xlabel('Time')
        plt.ylabel('Lyapunov Exponents')
        plt.tight_layout(pad=0.2)

        # plt.figure()
        # plt.plot(np.array(QQ)[:, 0])
        # plt.plot(np.array(QQp)[:, 0])

        plt.show()
