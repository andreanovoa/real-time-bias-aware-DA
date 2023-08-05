import numpy as np
from EchoStateNetwork import EchoStateNetwork
from Util import interpolate


class Bias:
    attrs = dict(augment_data=False)

    def __init__(self, b, t, dt, **kwargs):
        self.b = b
        self.t = t
        self.dt = dt
        self.hist = None
        self.hist_t = None

        for key in Bias.attrs:
            if key not in kwargs.keys():
                raise ValueError('Must provide {} to initialise ESN bias model'.format(key))
            else:
                setattr(self, key, kwargs[key])

    def updateHistory(self, b, t, reset=False):
        if self.hist is not None and not reset:
            self.hist = np.concatenate((self.hist, b))
            self.hist_t = np.concatenate((self.hist_t, t))
        else:
            self.hist = np.array([self.b])
            self.hist_t = np.array([self.t])
        self.b = self.hist[-1]
        self.t = self.hist_t[-1]

    def getOutputs(self):
        out = dict(name=self.name,
                   hist=self.hist,
                   hist_t=self.hist_t)
        for key in self.attrs.keys():
            out[key] = getattr(self, key)
        return out

    def updateCurrentState(self, b, t):
        self.b = b
        self.t = t

    @property
    def getBias(self):
        return self.b

# =================================================================================================================== #


class NoBias(Bias):
    name = 'None'

    def __init__(self, y, t, dt, **kwargs):
        super().__init__(b=np.zeros(len(y)), t=t, dt=dt, **kwargs)

    def resetBias(self, value):
        self.b = value

    def stateDerivative(self):
        return np.zeros([len(self.b), len(self.b)])

    def timeIntegrate(self, t, y=None, t_end=0):
        return np.zeros([len(t), len(self.b)]), t

    def print_bias_parameters(self):
        print('\n ----------------  Bias model parameters ---------------- ',
              '\n Bias model: {}'.format(self.name))


# =================================================================================================================== #

class ESN(Bias, EchoStateNetwork):
    name = 'ESN'

    def __init__(self, y, t, dt, **kwargs):
        # --------------------------  Initialise parent Bias  ------------------------- #
        Bias.__init__(self, b=np.zeros(len(y)), t=t, dt=dt, **kwargs)

        # ---------------------  Initialise parent EchoStateNetwork  --------------------- #
        EchoStateNetwork.__init__(self, y=np.zeros(len(y)), dt=dt, **kwargs)

        self.initialised = False
        self.trained = False

    def resetBias(self, value):
        self.b = value
        self.reset_state(u=value)

    def stateDerivative(self, open_loop_J=True):
        # Compute ESN Jacobian
        J = self.JacobianESN(open_loop_J=open_loop_J)
        return -J

    def timeIntegrate(self, t, y=None):
        Nt = int(round(len(t) / self.upsample))
        t_b = np.linspace(self.t, self.t + Nt * self.dt_ESN, Nt + 1)
        b = np.zeros((Nt + 1, self.N_dim))
        # b, r = [np.zeros((Nt + 1, n)) for n in [self.N_dim, self.N_units]]

        # If the time is before the washout initialization, return zeros
        if t[-1] < self.wash_time[-1]:
            return b[1:], t_b[1:]

        if not self.initialised:
            # Run washout phase in open-loop
            washout = self.format_washout(y, t_b)
            b_open, r_open = self.openLoop(washout)
            # Store open-loop forecast
            b[-Nt-1:-Nt+self.N_wash-1] = b_open[:-1]
            # r[-Nt-1:-Nt+self.N_wash-1] = r_open[:-1]
            self.reset_state(u=b_open[-1], r=r_open[-1])
            # Flag initialised
            self.initialised = True
            Nt -= (self.N_wash + 1)

        # Run closed-loop
        b[-Nt-1:], r_closed = self.closedLoop(Nt)

        # update ESN physical and reservoir states
        self.reset_state(u=b[-1], r=r_closed[-1])

        return b[1:], t_b[1:]

    def print_bias_parameters(self):
        print('\n ----------------  Bias model parameters ---------------- ',
              '\n Bias model: {}'.format(self.name),
              '\n Data filename: {}'.format(self.filename),
              '\n Training time: {} s, \t Validation time: {} s'.format(self.t_train, self.t_val),
              '\n Washout steps: {}, \t Upsample: {}'.format(self.N_wash, self.upsample),
              '\n Num of neurones: {}, \t Run test?: {}'.format(self.N_units, self.test_run),
              '\n Augment data?: {}, \t Num of training datasets: {}'.format(self.augment_data, self.L),
              '\n Connectivity: {}, \t Tikhonov parameter: {}'.format(self.connect, self.tikh),
              '\n Spectral radius: {}, \t Input scaling: {}'.format(self.rho, self.sigma_in)
              )

    def format_washout(self, y, t_b):
        # observable washout data
        wash_obs = self.wash_data[::self.upsample]  # truth, observables at high frequency

        # forecast model washout data
        wash_model = np.mean(y[::self.upsample], -1)
        wash_model = interpolate(t_b, wash_model, self.wash_time[::self.upsample], bound=False)

        # bias washout, the input data to open loop
        return wash_obs - wash_model

    def train_bias_model(self, train_data, val_strategy=EchoStateNetwork.RVC_Noise):
        if self.augment_data:
            train_data = np.vstack([train_data,
                                    train_data * 1e-1,
                                    train_data * -1e-2])
        self.trainESN(train_data, validation_strategy=val_strategy)
# =================================================================================================================== #
