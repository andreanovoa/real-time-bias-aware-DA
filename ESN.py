# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 17:14:52 2022

@author: an553
"""

'''   
    TODO: 
        - Add a function to test ESN
'''

import os as os
import numpy as np
import matplotlib.pyplot as plt


class ESN:
    attrs = ['norm', 'Win', 'Wout', 'W', 'dt', 'N_wash', 'N_unit', 'N_dim',
             'bias_in', 'bias_out', 'rho', 'sigma_in', 'upsample',
             't_train', 't_val', 't_wash']

    def __init__(self, filename, train=False, training_params=None):
        # filename    =   os.getcwd() + '\\data\\' + filename
        if train:
            self.trainESN(filename, training_params)
            filename = filename + '_ESN'

        file = np.load(filename + '.npz')  # load .npz output from training_main
        for attr in file.files:
            setattr(self, attr, file[attr])

    @classmethod
    def trainESN(self, filename, training_params):
        dic_params = training_params.copy()
        dic_params['filename'] = filename
        exec(open("main_training.py").read(), dic_params)

    @property  # Bias prediction
    def U(self):
        if not hasattr(self, '_U'):
            self._U = np.zeros(self.N_dim)  # initialise to zeros
        return self._U

    @U.setter
    def U(self, new_U):
        self._U = new_U

    @property  # Reservoir state
    def r(self):
        if not hasattr(self, '_r'):
            self._r = np.zeros(self.N_unit)  # initialise reservoir to zeros
        return self._r

    @r.setter
    def r(self, new_r):
        self._r = new_r

    def updateReservoir(self, new_U, new_r):  # ________________________________
        """ Updates the reservoir state and bias """
        self.U = new_U
        self.r = new_r

    def step(self):  # ________________________________________________________
        """ Advances one ESN time step.
            Returns:
                new augmented reservoir state (new state with bias_out)
        """
        # input is normalized and input bias added
        U_aug = np.hstack((self.U / self.norm, np.array([self.bias_in])))
        # hyperparameters are explicit here
        r_post = np.tanh(np.dot(U_aug * self.sigma_in, self.Win) + \
                         self.rho * np.dot(self.r, self.W))
        # output bias added
        r_aug = np.hstack((np.squeeze(r_post), np.array([self.bias_out])))

        return r_aug

    def openLoop(self, U_wash):  # ____________________________________________
        """ Initialises ESN in open-loop.
            Input:
                - U_wash: washout input time series
            Returns:
                - U:  prediction from ESN during open loop
                - ra: time series of augmented reservoir states
        """
        N = U_wash.shape[0] - 1  # AN: I added the -1 because we don't want forecast, just initialisation
        ra = np.empty((N + 1, self.N_unit + 1))
        U = np.empty((N + 1, self.N_dim))
        ra[0] = np.concatenate((self.r, np.array([self.bias_out])))
        for i in np.arange(1, N + 1):
            self.updateReservoir(U_wash[i - 1], ra[i - 1, :self.N_unit])
            ra[i] = self.step()
            U[i] = np.dot(ra[i], self.Wout)
            # update reservoir with predicted step
        self.updateReservoir(U[i], ra[i, :self.N_unit])
        return U, ra

    def closedLoop(self, Nt):  # ______________________________________________
        """ Advances ESN in closed-loop.
            Input:
                - Nt: number of forecast time steps
            Returns:
                - U:  forecast time series
                - ra: time series of augmented reservoir states
        """
        Nt = int(Nt)
        ra = np.empty((Nt + 1, self.N_unit + 1))
        U = np.empty((Nt + 1, self.N_dim))
        ra[0] = np.concatenate((self.r, np.array([self.bias_out])))
        U[0] = np.dot(ra[0], self.Wout)
        if U[0] == self.U:
            for i in np.arange(1, Nt + 1):
                self.updateReservoir(U[i - 1], ra[i - 1, :self.N_unit])
                ra[i] = self.step()
                U[i] = np.dot(ra[i], self.Wout)
        else:
            print('closed loop - sowething is wrong')
        return U[1:], ra[1:]


def testESN(file):
    esn = ESN(file + '_ESN')
    data = loadBias(file)  # data    =   np.load(file + '.npz')
    t_data = np.arange(0., len(data), 1) * esn.dt / esn.upsample

    wash = data[::esn.upsample]
    t_wash = t_data[::esn.upsample]

    t1 = len(wash) // 2
    t2 = t1 + esn.N_wash + 1
    # Forecase ESN in open and closed loops
    Uo, ro = esn.openLoop(wash[t1:t2])

    Uc, rc = esn.closedLoop(100)
    # Plot test
    plt.figure(figsize=[10, 5])
    plt.plot(t_wash[t1:t2], wash[t1:t2], 'o', color='grey',
             linewidth=5, alpha=0.4, label='washout')

    plt.plot(t_data, data, '-', color='grey',
             linewidth=5, alpha=0.4, label='truth')

    plt.plot(t_wash[t1:t2], Uo, '-', marker='.', label='open loop')
    plt.plot(t_wash[t2:t2 + 100], Uc, '-', marker='+', label='closed loop')

    plt.legend(loc='best')
    plt.ylim([min(wash) * 1.1, max(wash) * 1.1])
    plt.xlim([t_wash[t1 - 10], t_wash[t2 + 110]])

    return


def loadBias(file):
    b = np.load(file + '.npz')
    return b['bias']


# %% ===========================================================================
if __name__ == '__main__':

    from old import VdP as Model
    from Util import createObservations
    from datetime import date

    datafolder = os.getcwd() + '\\data\\'
    filename = 'bias_VdP_' + str(date.today())
    tmax = 5.

    HOM = createObservations(Model, law='atan', name='HOM', t_max=tmax)
    LOM = createObservations(Model, law='cubic', name='LOM', t_max=tmax)

    if not os.path.isfile(datafolder + filename + '.npz'):
        bias = HOM[0].hist[:, 0, :] - LOM[0].hist[:, 0, :]
        np.savez(datafolder + filename, bias=bias)
    else:
        bias = loadBias(datafolder + filename)

    fig, ax = plt.subplots(2, 1, figsize=[15, 10], tight_layout=True)
    ax[0].plot(HOM[0].hist_t, HOM[0].hist[:, 0], label='HOM')
    ax[0].plot(LOM[0].hist_t, LOM[0].hist[:, 0], 'y', label='LOM')
    ax[0].set(ylabel='$\\eta$', xlabel='$t$', xlim=[tmax-1., tmax-.9])
    ax[0].legend(loc='best')
    ax[1].plot(LOM[0].hist_t, bias, 'mediumpurple')
    ax[1].set(ylabel='bias', xlabel='$t$', xlim=[14., 14.1])

    if not os.path.isfile(datafolder + filename + '_ESN.npz'):
        esn = ESN(filename, train=True,
                  training_params={'t_train': 1.0, 't_val': 0.2,
                                   't_wash': 0.05, 'upsample': 5,
                                   'test_run': False})
        # %%
    testESN(datafolder + filename)
