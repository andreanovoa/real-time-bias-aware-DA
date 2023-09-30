import numpy as np
from EchoStateNetwork import EchoStateNetwork
from Util import interpolate


class Bias:
    attrs = dict(augment_data=False,
                 est_b=False,
                 bias_form='None')

    def __init__(self, b, t, dt, **kwargs):
        self.b = b
        self.t = t
        self.dt = dt
        self.precision_t = int(-np.log10(dt)) + 2

        # ========================== CREATE HISTORY ========================== ##
        self.hist = np.array([self.b])
        self.hist_t = np.array([self.t])

        # ========================= DEFINE ESSENTIALS ========================== ##
        for key, val in Bias.attrs.items():
            if key in kwargs.keys():
                setattr(self, key, kwargs[key])
            else:
                setattr(self, key, val)

    def updateHistory(self, b, t, reset=False):
        if not reset:
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
        if hasattr(self, 'L'):
            self.name += '_L{}'.format(self.L)

    def resetBias(self, value):
        self.b = value
        self.reset_state(u=value)

    def stateDerivative(self):
        return -self.Jacobian(open_loop_J=True)  # Compute ESN Jacobian

    def timeIntegrate(self, t, y=None, wash_t=None, wash_obs=None):
        interp_flag = False
        Nt = len(t) // self.upsample
        if len(t) % self.upsample:
            Nt += 1
            interp_flag = True
        t_b = np.round(self.t + np.arange(0, Nt+1) * self.dt_ESN, self.precision_t)

        # If the time is before the washout initialization, return zeros
        if self.initialised:
            b, r = self.closedLoop(Nt)
        elif wash_t is None:
            b = np.zeros((Nt + 1, self.N_dim))
            r = np.zeros((Nt + 1, self.N_units))
        else:
            b = np.zeros((Nt + 1, self.N_dim))
            t1 = np.argmin(abs(t_b - wash_t[0]))
            Nt -= t1
            # Flag initialised
            self.initialised = True

            # Run washout phase in open-loop
            wash_model = interpolate(t, np.mean(y, -1), wash_t)
            b_open, r = self.openLoop(wash_obs - wash_model)
            b[t1:t1+self.N_wash+1] = b_open
            Nt -= self.N_wash
            # Run the rest of the time window in closed-loop
            if Nt > 0:
                # Store open-loop forecast
                self.reset_state(u=b[-1], r=r[-1])
                b_close, r = self.closedLoop(Nt)
                b[t1 + self.N_wash + 1:] = b_close[1:]

        if interp_flag:
            b[-1] = interpolate(t_b[-Nt:], b[-Nt:], t[-1])
            r[-1] = interpolate(t_b[-Nt:], r[-Nt:], t[-1])
            t_b[-1] = t[-1]

        # update ESN physical and reservoir states
        self.reset_state(u=b[-1], r=r[-1])

        return b[1:], t_b[1:]

    def print_bias_parameters(self):
        print('\n -----------------------  Bias model parameters -------------------- ',
              '\n Bias model: {}'.format(self.name),
              '\n Data filename: {}'.format(self.filename),
              '\n Training time: {:.6} s, \t Validation time: {:.6} s'.format(self.t_train, self.t_val),
              '\n Washout steps: {}, \t Upsample: {}'.format(self.N_wash, self.upsample),
              '\n Num of neurones: {}, \t Run test?: {}'.format(self.N_units, self.perform_test),
              '\n Augment data?: {}, \t Num of training datasets: {}'.format(self.augment_data, self.L),
              '\n Connectivity: {}, \t\t Tikhonov parameter: {}'.format(self.connect, self.tikh),
              '\n Spectral radius: {:.6}, \t Input scaling: {:.6}'.format(self.rho, self.sigma_in)
              )

    def train_bias_model(self, train_data, val_strategy=EchoStateNetwork.RVC_Noise,
                         plot_training=True, folder='./'):
        self.train(train_data, validation_strategy=val_strategy,
                   plot_training=plot_training, folder=folder)
        self.trained = True
# =================================================================================================================== #
