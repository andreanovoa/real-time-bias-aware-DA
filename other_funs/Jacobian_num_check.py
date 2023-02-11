import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.linalg as la

name = 'results/2022-09-06_Rijke_Wave_Rijke_sqrt'
with open(name, 'rb') as f:
    parameters = pickle.load(f)
    createEnsemble(parameters['forecast_model'])
    truth = pickle.load(f)
    filter_ens = pickle.load(f)

rin = filter_ens.bias.r.transpose()
bin = filter_ens.bias.b.transpose()
Win = filter_ens.bias.Win
Wout = filter_ens.bias.Wout
W = filter_ens.bias.W
norm = filter_ens.bias.norm
sigma_in = filter_ens.bias.sigma_in
bias_in = filter_ens.bias.bias_in
bias_out = filter_ens.bias.bias_out
rho = filter_ens.bias.rho
Nq = filter_ens.bias.N_dim
Nr = filter_ens.bias.N_unit

WCout = la.lstsq(Wout[0][:-1], W[0].transpose())[0]


def Jacobian(b_in, r_in):

    Win_1 = Win[0][:-1, :].transpose()
    Wout_1 = Wout[0][:-1, :].transpose()

    b_aug = np.concatenate((b_in / norm, np.array([bias_in])))


    # Option(i)
    # Forecast the reservoir state
    # r_out = np.tanh(np.dot(b_aug * sigma_in, Win[0]) + rho * np.dot(b_in, WCout))

    # Option(ii)
    r_out = np.tanh(np.dot(b_aug * sigma_in, Win[0]) + rho * np.dot(r_in, W[0]))


    T = 1 - r_out**2

    # Option(i)
    # drout_dbin = sigma_in * Win_1 / norm + rho * WCout.transpose()

    # Option(ii)
    drout_dbin = sigma_in * Win_1 / norm

    J = np.dot(Wout_1, drout_dbin * np.expand_dims(T, 1))
    return J



def step(b, r):  # ________________________________________________________
    """ Advances one ESN time step.
        Returns:
            new reservoir state (no bias_out)
    """
    # Normalise input data and augment with input bias (ESN symmetry parameter)
    b_aug = np.concatenate((b / norm, np.array([bias_in])))

    # Forecast the reservoir state
    # Option(i)
    # Forecast the reservoir state
    # r_out = np.tanh(np.dot(b_aug * sigma_in, Win[0]) + rho * np.dot(b, WCout))

    # Option(ii)
    r_out = np.tanh(np.dot(b_aug * sigma_in, Win[0]) + rho * np.dot(r, W[0]))

    # output bias added
    r_aug = np.concatenate((r_out, np.array([bias_out])))
    # compute output from ESN
    b_out = np.dot(r_aug, Wout[0])
    return b_out, r_out


if __name__ == '__main__':

    bout, rout = step(bin, rin)
    J = Jacobian(bin, rin)

    plt.figure()
    for eps_i in np.linspace(5, -3, 100):
        eps = 10**eps_i
        dbdy = np.zeros([Nq, Nq])
        for i in range(Nq):
            bin_tilde = bin.copy()
            bin_tilde[i] += eps


            bout_tilde, _ = step(bin_tilde, rin.copy())

            dbdy[:, i] = (bout_tilde - bout.copy()) / eps

        error1 = np.linalg.norm(J - dbdy) / np.linalg.norm(J)
        plt.plot(eps, error1, 'bo')

    plt.xlabel('epsilon')
    plt.ylabel('error')
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.show()