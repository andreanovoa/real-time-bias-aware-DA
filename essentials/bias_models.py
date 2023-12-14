from ML_models.EchoStateNetwork import EchoStateNetwork
from essentials.Util import interpolate
import numpy as np


class Bias:
    attrs = dict(augment_data=False,
                 est_b=False,
                 bias_form='None',
                 upsample=1)

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
            self.hist = np.array([self.getBias()])
            self.hist_t = np.array([self.t])

        self.b = self.getBias(state=self.hist[-1])
        self.t = self.hist_t[-1]

    def updateCurrentState(self, b, t):
        self.b = b
        self.t = t

    def getBias(self, state=None):
        if state is None:
            return self.b
        else:
            return state

    def resetBias(self, value):
        self.b = value

# =================================================================================================================== #


class NoBias(Bias):
    name = 'None'

    def __init__(self, y, t, dt, **kwargs):
        super().__init__(b=np.zeros(y.shape), t=t, dt=dt, **kwargs)

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
        # --------------------  Initialise parent EchoStateNetwork  ------------------- #
        EchoStateNetwork.__init__(self, y=np.zeros(len(y)), dt=dt, **kwargs)
        # --------------------------  Initialise parent Bias  ------------------------- #
        Bias.__init__(self, b=np.zeros(y.shape), t=t, dt=dt, **kwargs)
        # Flags
        self.initialised = False
        self.trained = False
        if 'store_ESN_history' in kwargs.keys():
            self.store_ESN_history = kwargs['store_ESN_history']
        else:
            self.store_ESN_history = False

    def resetBias(self, value):
        self.b = value
        self.reset_state(u=value)

    def stateDerivative(self):
        J = self.Jacobian(open_loop_J=True)  # Compute ESN Jacobian
        db_dinput = J[self.observed_idx, :]
        return -db_dinput

    def timeIntegrate(self, t, y=None, wash_t=None, wash_obs=None):
        if not self.trained:
            raise NotImplementedError('ESN model not trained')

        interp_flag = False
        Nt = len(t) // self.upsample
        if len(t) % self.upsample:
            Nt += 1
            interp_flag = True
        t_b = np.round(self.t + np.arange(0, Nt+1) * self.dt_ESN, self.precision_t)

        # If the time is before the washout initialization, return zeros
        if self.initialised:
            u, r = self.closedLoop(Nt)
        else:
            u = np.zeros((Nt + 1, self.N_dim, self.N_ens))
            r = np.zeros((Nt + 1, self.N_units, self.N_ens))
            if wash_t is not None:
                t1 = np.argmin(abs(t_b - wash_t[0]))
                Nt -= t1
                # Flag initialised
                self.initialised = True
                # Run washout phase in open-loop
                wash_model = interpolate(t, y, wash_t)
                if wash_obs.ndim < wash_model.ndim:
                    wash_obs = np.expand_dims(wash_obs, -1)
                washout = wash_obs - wash_model

                u_open, r_open = self.openLoop(washout)
                u[t1:t1+self.N_wash+1] = u_open
                r[t1:t1+self.N_wash+1] = r_open
                Nt -= self.N_wash
                # Run the rest of the time window in closed-loop
                if Nt > 0:
                    # Store open-loop forecast
                    self.reset_state(u=u_open[-1], r=r_open[-1])
                    u_close, r_close = self.closedLoop(Nt)
                    u[t1 + self.N_wash + 1:] = u_close[1:]
                    r[t1 + self.N_wash + 1:] = r_close[1:]
        # Interpolate the final point if the upsample is not multiple of dt
        if interp_flag:
            u[-1] = interpolate(t_b[-Nt:], u[-Nt:], t[-1])
            r[-1] = interpolate(t_b[-Nt:], r[-Nt:], t[-1])
            t_b[-1] = t[-1]

        # update ESN physical and reservoir states, and store the history if requested
        self.reset_state(u=u[-1], r=r[-1])

        return u[1:], t_b[1:]

    def print_bias_parameters(self):
        print('\n ---------------- {} bias model parameters --------------- '.format(self.name))
        keys_to_print = sorted(['t_train', 't_val', 'N_wash', 'rho', 'sigma_in', 'upsample',
                                'N_units', 'perform_test', 'augment_data', 'L', 'connect', 'tikh'])
        for key in keys_to_print:
            try:
                print('\t {} = {:.6}'.format(key, getattr(self, key)))
            except ValueError:
                print('\t {} = {}'.format(key, getattr(self, key)))

    def train_bias_model(self, train_data, val_strategy=None, plot_training=True, folder='./'):
        if val_strategy is None:
            val_strategy = EchoStateNetwork.RVC_Noise
        self.train(train_data, validation_strategy=val_strategy, plot_training=plot_training, folder=folder)
        self.trained = True

    def getBias(self, state=None, get_full_state=False):
        if get_full_state:
            return self.getReservoirState()[0]
        else:
            if len(self.observed_idx) != self.N_dim:
                bias_idx = [a for a in np.arange(self.N_dim) if a not in self.observed_idx]
            else:
                bias_idx = np.arange(self.N_dim)

            if state is None:
                state = self.getReservoirState()[0]
                return state[bias_idx]
            else:  # the input is a timeseries
                if state.ndim > 1:
                    return state[:, bias_idx]
                else:
                    return state[bias_idx]
# =================================================================================================================== #



