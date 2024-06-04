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
from scipy.integrate import solve_ivp

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)
plt.rc('legend', facecolor='white', framealpha=1, edgecolor='white')


def one_dim_sweep(model, model_params: dict, sweep_p: str, range_p: list,  t_max=8., save=True):

    y_all, cases = [], []
    for p in range_p:
        model_params[sweep_p] = p
        case_p = model(model_params)
        if save:
            data_folder = 'data/' + model.name + '_avgproperties/'
            if not os.path.isdir(data_folder):
                os.makedirs(data_folder)
            filename = data_folder + TAfilename(case_p)
            # load or save
            if os.path.isfile(filename):
                with open(filename, 'rb') as f:
                    case_p = pickle.load(f)
            else:
                psi_i, tt = case_p.time_integrate()
                case_p.update_history(psi_i, tt)
                with open(filename, 'wb') as f:
                    pickle.dump(case_p, f)
        else:
            psi_i, tt = case_p.time_integrate()
            case_p.update_history(psi_i, tt)

        obs, lbl = case_p.get_observable_hist(int(case_p.t_transient / case_p.dt)), case_p.obs_labels
        if len(obs.shape) > 2:
            obs, lbl = obs[:, 0], lbl[0]
        # store
        y_all.append(obs - np.mean(obs))
        cases.append(case_p)
    return cases, y_all

def plot_1D_diagram(y_all, range_p, sweep_p, categories=None):

    fig, ax_bif = plt.subplots(1, 1, layout="tight")
    # fig, ax = plt.subplots(int(np.ceil(len(range_p)/3)), 3, figsize=[20, 12], layout="tight")
    # ax = ax.flatten()
    kk = 0
    if categories is None:
        categories = [0] * len(range_p)
    colors = ['b', 'r', 'c', 'y']
    for yy, val, cat in zip(y_all, range_p, categories):
        for k in [1, -1]:
            peaks = find_peaks(yy.squeeze() * k)
            peaks = yy[peaks[0]]
            ax_bif.plot(np.ones(len(peaks)) * val, peaks, '.', color=colors[cat], markersize=3)
            ax_bif.set(xlabel='$' + sweep_p + '$', ylabel='y')
            # xscale='log', title='C1={}, C2={}'.format(case_p.C1, case_p.C2))
        # ax[kk].plot(yy[:5000])
        # ax[kk].set(ylabel=lbl, xlabel='timestep', title=val)
        kk += 1
    plt.show()
# ---------------------------------------------------------------------------------------------------- #
def QR(M, N_exp):
    """ Compute an orthogonal basis, Q, and the exponential change
        in the norm along each element of the basis, S.
    """
    M, Q, S = np.squeeze(M), [None] * N_exp, np.empty(N_exp)
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
def Lyap_Classification(cases):
    tol = 0.01

    N_exp = 2  # compute the N_exp-first Lyapunov exponents
    eps = 1e-8  # multiplication factor to make the orthonormalized perturbation infinitesimalv
    lambdas = []
    for case in cases:
        q0, t1 = case.psi, case.t  # initial state
        q0_hist = case.get_observable_hist(int(case.t_transient / 2 / case.dt))

        # Compute peaks in the timeseries
        max_peaks, std_peaks = 0., 0.
        for dim in range(len(case.obs_labels)):
            for k in [1, -1]:
                peaks = find_peaks(q0_hist[:, dim].squeeze() * k)
                peaks = q0_hist[peaks[0], dim] - np.mean(q0_hist[:, dim])
                if len(peaks) > 0:
                    max_peaks = max(max(abs(peaks)), max_peaks)
                    std_peaks = max(np.std(peaks), std_peaks)
        # Check if FP and/or LC
        if max_peaks < .2:
            print('FP')
            lambdas.append(0)
        elif std_peaks < 1.:
            print('LC, max amplitude = ', max_peaks, ', std = ', std_peaks)
            lambdas.append(1)
        else:  # Check if CH or QP
            N_orth, N_loops = 1000, 1000
            # NOTE:  N_orth*dt should easily be 5 Lyapunov times with eps small enough. Because in 5 lyapunov
            # times the magnitude of the perturbation increases only be 2^5=32 times on average.
            # NOTE 2: if N_orth too large, do not trust exponents, only sign as the slope can be misleading due to the
            # chaotic saturation and/or machine precision

            orth_basis = [q_per / np.linalg.norm(q_per) for q_per in np.random.rand(N_exp, case.N)]
            q0_pert, S, SS = [], 0, np.empty((N_loops, N_exp))
            for jj in range(N_loops):
                t0 = t1
                t1 = t0 + N_orth * case.dt
                # perturb initial condition with orthonormal basis and propagate them
                q0_pert = [q0.squeeze() + eps * a for a in orth_basis]
                # integrate perturbed cases
                for ii in range(N_exp):
                    sol = solve_ivp(case.time_derivative, t_span=([t0, t1]), y0=q0_pert[ii],
                                    args=(case.govEqnDict(), case.alpha0))
                    q0_pert[ii] = sol.y[:, -1]

                sol = solve_ivp(case.time_derivative, t_span=([t0, t1]), y0=q0.squeeze(),
                                args=(case.govEqnDict(), case.alpha0))
                q0 = sol.y[:, -1]

                # compute the final value of the N_exp perturbations, orthornormalize basis and compute exponents
                basis = [(q - q0) / eps for q in q0_pert]
                orth_basis, S1 = QR(basis, N_exp)

                # skip the first step, which does not start from the orthonormalized basis
                if jj > 0:
                    S += S1
                    SS[jj] = S / (jj * case.dt * N_orth)

            if SS[-1][0] > tol:
                print('CH, exponents = ', SS[-1])
                lambdas.append(3)
            elif SS[-1][1] < -tol:
                print('LC, exponents = ', SS[-1])
                lambdas.append(1)
            else:
                print('QP, exponents = ', SS[-1])
                lambdas.append(2)


            # plt.figure()
            # plt.plot(np.arange(N_loops) * case.dt * N_orth, SS)
            # plt.xlabel('Time')
            # plt.ylabel('Lyapunov Exponents')
            # plt.tight_layout(pad=0.2)
            # plt.show()

    return lambdas

# ==================================================================================================== #
if __name__ == '__main__':

    # ==================== RIJKE MODEL ==================== #
    # TAmodel = TAModels.Rijke
    # TAdict = dict(law='sqrt', beta=3.6, tau=2.E-3, C1=0.06, C2=0.01 )
    #
    # # param = 'tau'  # desired parameter to sweep
    # # range_param = np.linspace(0.1, 1., 11) * 1E-3
    #
    # param = 'beta'  # desired parameter to sweep
    # range_param = np.linspace(0, 3.5, 15)
    #
    # # param = 'C1'  # desired parameter to sweep
    # # range_param = np.linspace(0.005, 0.1, 39)
    #
    # og_cases = one_dim_sweep(TAmodel, TAdict, param, range_param, plot=True)[0]

    # ==================== LORENTZ 63 ==================== #
    TAmodel = TAModels.Lorenz63
    TAdict = dict(rho=20., sigma=10., beta=1.8, dt=2E-2)
    param = 'beta'  # desired parameter to sweep
    range_param = np.linspace(0, 2, 50)
    og_cases, og_y = one_dim_sweep(TAmodel, TAdict, param, range_param, save=False, t_max=1000)
    plt.close()

    # ------------------------- Compute lyapunov exponents -------------------------- #
    exponents = Lyap_Classification(og_cases)
    plot_1D_diagram(og_y, range_param, param, categories=exponents)
