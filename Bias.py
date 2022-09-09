import os
# os.environ["OMP_NUM_THREADS"]= '1'
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la


class Bias:

    def __init__(self, b, t):
        self.b = b
        self.t = t
        self.hist = None
        self.hist_t = None

    def updateHistory(self, b, t):
        if self.hist is not None:
            self.hist = np.concatenate((self.hist, b))
            self.hist_t = np.concatenate((self.hist_t, t))
        else:
            self.hist = np.array([self.b])
            self.hist_t = np.array([self.t])
        self.b = self.hist[-1]
        self.t = self.hist_t[-1]

    def updateCurrentState(self, b, t):
        self.b = b
        self.t = t


# =================================================================================================================== #


class LinearBias(Bias):
    name = 'LinearBias'
    attrs = dict(b1=[0.2],
                 b2=[0.5],
                 k=1.,
                 dt=1E-4)

    def __init__(self, y, t, Bdict=None):
        if Bdict is None:
            Bdict = {}

        for key, val in self.attrs.items():
            setattr(self, key, val)

        for key, val in Bdict.items():
            setattr(self, key, val)

        self.q = np.size(y, 0)
        self.m = np.size(y, 1)

        if len(np.shape(self.b1)) < 2:
            self.b1 = self.b1 * np.ones(self.m)
            self.b2 = self.b2 * np.ones(self.m)
            # self.b1 = self.b1 * np.ones((self.q, self.m))
            # self.b2 = self.b2 * np.ones((self.q, self.m))

        self.Nw = 2 * self.q
        # Initialise bias parent
        b = self.getBias(y)

        super().__init__(b, t)

    def getBias(self, *args):
        y = args[0]
        B = self.b1 * y + self.b2
        return B

    def updateWeights(self, weights):
        self.b1 = weights[0]
        self.b2 = weights[1]

    # @property
    def getWeights(self):
        return [self.b1, self.b2], ['b1', 'b2']

    def stateDerivative(self, y):
        dbdy = [self.b1[i] * np.eye(self.q) for i in range(np.size(y, -1))]
        return np.array(dbdy)

    def timeIntegrate(self, Nt=100, y=None):

        t_b = np.linspace(self.t, self.t + Nt * self.dt, Nt + 1)

        # b = np.zeros((len(t_b), np.size(self.b, 0), np.size(self.b, 1)))
        b = self.getBias(y)

        return b, t_b[1:]


# =================================================================================================================== #


class ESN(Bias):
    #    attrs = ['norm', 'Win', 'Wout', 'W', 'dt_ESN', 'N_wash', 'N_unit', 'N_dim',
    #             'bias_in', 'bias_out', 'rho', 'sigma_in', 'upsample',
    #             't_train', 't_val', 't_wash']
    name = 'ESN'
    training_params = {'t_train': 1.2,
                       't_val': 0.2,
                       't_wash': 0.025,
                       'dt_ESN': 5E-4,
                       'test_run': True,
                       'k': 1.
                       }

    def __init__(self, y, t, Bdict=None):
        if Bdict is None:
            Bdict = {}
        else:
            for key, val in self.training_params.items():
                if key not in Bdict.keys():
                    setattr(self, key, val)
                    Bdict[key] = val
                else:
                    setattr(self, key, Bdict[key])

        # ------------------------ Define bias data filename ------------------------ #
        if Bdict['filename'][:4] != 'data/':
            Bdict['filename'] = 'data/' + Bdict['filename']
        if Bdict['filename'][:-len('_bias')] != '_bias':
            Bdict['filename'] = Bdict['filename'] + '_bias'

        # --------------------- Train a new ESN if not in folder --------------------- #
        ESN_filename = Bdict['filename'][:-len('bias')] + 'ESN.npz'
        if not os.path.isfile(ESN_filename):
            # print('Training ESN')
            # Load or create bias data
            if 'trainData' in Bdict.keys():
                # print('\t saving bias data')
                bias = Bdict['trainData']
                # Delete unnecessary data. Keep only wash + training + val (+ test)
                N_wtv = int((self.t_wash + self.t_train + self.t_val) / Bdict['dt'])
                if self.test_run:
                    N_wtv += int(self.t_wash * 10 / Bdict['dt'])
                if N_wtv > len(bias):
                    N_wtv = len(bias)

                np.savez(Bdict['filename'], bias[-N_wtv:])
            else:

                # TODO add smg that checks that the loaded params and the defined match.

                print('\t loading bias data')
                Bdict['trainData'] = np.load(Bdict['filename'] + '.npz')['bias']
            # Run training main script
            exec(open("main_training.py").read(), Bdict)
        else:
            print('Loaded trained ESN')

        # --------------------------- Load trained ESN --------------------------- #
        fileESN = np.load(ESN_filename)  # load .npz output from training_main
        for attr in fileESN.files:
            setattr(self, attr, fileESN[attr])

        # --------------------- Create washout observed data ---------------------- #
        #

        self.washout_obs = Bdict['washout_obs'][-self.N_wash * self.upsample - 1::self.upsample].squeeze()
        self.washout_t = Bdict['washout_t'][-self.N_wash * self.upsample - 1::self.upsample]
        self.washout_obs = self.washout_obs[1:]
        self.washout_t = self.washout_t[1:]

        assert self.washout_t[-1] == Bdict['washout_t'][-1]

        # plt.figure()
        # plt.plot(self.washout_t, self.washout_obs[:,0], '-o')
        # plt.plot(Bdict['washout_t'], np.squeeze(Bdict['washout_obs'][:,0]))
        # plt.plot(self.washout_t[0], self.washout_obs[0,0], '*')
        # plt.show()

        # -----------  Initialise reservoir state and its history to zeros ------------ #
        self.r = np.zeros(self.N_unit)
        self.hist_r = np.array([self.r])

        # --------------------------  Initialise parent Bias  ------------------------- #
        b = np.zeros(self.N_dim)

        super().__init__(b, t)

    def getBias(self, *args):
        return self.b

    def getWeights(self):  # TODO maybe
        pass

    def updateWeights(self, weights):  # TODO maybe
        pass

    def getReservoirState(self):
        return self.b, self.r

    def updateReservoir(self, r):
        self.hist_r = np.concatenate((self.hist_r, r))
        self.r = r[-1]

    @property
    def WCout(self):
        if not hasattr(self, '_WCout'):
            self.WCout = la.lstsq(self.Wout[0][:-1], self.W[0])
        return self._WCout

    # TODO Jacobian
    def stateDerivative(self, y):

        # Get current state
        bin, rin = self.getReservoirState()

        Win_1 = self.Win[0][:-1, :].transpose()
        Wout_1 = self.Wout[0][:-1, :].transpose()

        # Option(i) rin function of bin:
        # b_aug = np.concatenate((bin / self.norm, np.array([self.bias_in])))
        # rout = np.tanh(np.dot(b_aug * self.sigma_in, self.Win[0]) + self.rho * np.dot(bin, self.WCout()))
        # drout_dbin = sigma_in * Win_1 / self.norm + self.rho * self.WCout.transpose()

        # Option(ii) rin constant:
        _, rout = self.step(bin, rin)
        drout_dbin = self.sigma_in * Win_1 / self.norm


        # Compute Jacobian
        T = 1 - rout ** 2
        J = np.dot(Wout_1, drout_dbin * np.expand_dims(T, 1))

        return J

    def timeIntegrate(self, Nt=100, y=None):

        Nt = int(Nt // self.upsample)
        t_b = np.linspace(self.t, self.t + Nt * self.dt_ESN, Nt + 1)

        if len(self.hist) == 1:
            # observable washout data
            wash_obs = self.washout_obs  # truth, observables at high frequency [I MUST UPSAMPLE BEFORE - INIT]

            # forecast model washout data
            wash_model = np.mean(y[-self.N_wash * self.upsample::self.upsample], -1)

            # bias washout, the input data to open loop
            washout = wash_obs - wash_model

            # open loop initialisation of the ESN
            u_open, r_open = self.openLoop(washout)

            b = np.zeros((Nt + 1, self.N_dim))
            r = np.zeros((Nt + 1, self.N_unit))

            b[-self.N_wash:] = u_open
            r[-self.N_wash:] = r_open

            assert len(b) == len(t_b)

        else:
            b, r = self.closedLoop(Nt)

        # update reservoir history
        self.updateReservoir(r[1:])

        return b[1:], t_b[1:]

    def step(self, b, r):  # ________________________________________________________
        """ Advances one ESN time step.
            Returns:
                new reservoir state (no bias_out)
        """
        # Normalise input data and augment with input bias (ESN symmetry parameter)
        b_aug = np.concatenate((b / self.norm, np.array([self.bias_in])))
        # Forecast the reservoir state
        r_out = np.tanh(np.dot(b_aug * self.sigma_in, self.Win[0]) + self.rho * np.dot(r, self.W[0]))
        # output bias added
        r_aug = np.concatenate((r_out, np.array([self.bias_out])))
        # compute output from ESN
        b_out = np.dot(r_aug, self.Wout[0])
        return b_out, r_out

    def openLoop(self, b_wash):  # ____________________________________________
        """ Initialises ESN in open-loop.
            Input:
                - U_wash: washout input time series
            Returns:
                - U:  prediction from ESN during open loop
                - r: time series of reservoir states
        """
        Nt = b_wash.shape[0] - 1

        r = np.empty((Nt + 1, self.N_unit))
        b = np.empty((Nt + 1, self.N_dim))
        b[0], r[0] = self.getReservoirState()
        for i in range(Nt):
            b[i + 1], r[i + 1] = self.step(b_wash[i], r[i])

        plt.figure()
        plt.plot(self.washout_t, b_wash[:, 0], '-o', label='washout data')
        plt.plot(self.washout_t, b[:, 0], '-x', label='open-loop')

        return b, r

    def closedLoop(self, Nt):  # ______________________________________________
        """ Advances ESN in closed-loop.
            Input:
                - Nt: number of forecast time steps
            Returns:
                - U:  forecast time series
                - ra: time series of augmented reservoir states
        """
        Nt = int(Nt)
        r = np.empty((Nt + 1, self.N_unit))
        b = np.empty((Nt + 1, self.N_dim))
        b[0], r[0] = self.getReservoirState()
        for i in range(Nt):
            b[i + 1], r[i + 1] = self.step(b[i], r[i])

        t_b = np.linspace(self.t, self.t + Nt * self.dt_ESN, Nt + 1)

        plt.plot(t_b, b[:, 0], '-+', color='green', label='closed-loop')
        # plt.legend()
        # plt.show()

        return b, r

    # TODO at some point
    # @classmethod
    # def trainESN(cls, filename, training_params):
    #     dic_params = training_params.copy()
    #     dic_params['filename'] = filename
    #     exec(open("main_training.py").read(), dic_params)
