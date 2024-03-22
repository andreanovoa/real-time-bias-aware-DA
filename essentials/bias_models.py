import matplotlib.pyplot as plt

from ML_models.EchoStateNetwork import EchoStateNetwork
from essentials.Util import interpolate
import numpy as np


class Bias:

    def __init__(self, b, t, dt, **kwargs):
        self.dt = dt
        self.precision_t = int(-np.log10(dt)) + 2
        self.upsample = 1
        self.m = 1
        self.augment_data = False

        # Default to not perform bayesian update to state
        self.bayesian_update = False
        self.biased_observations = False
        self.filter = None
        self.inflation = None

        # ========================= Re-DEFINE ESSENTIALS ========================== ##
        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, kwargs[key])

        # ========================== CREATE HISTORY ========================== ##
        if b.ndim == 1:
            b = np.expand_dims(b, axis=-1)
        if self.biased_observations:
            b = np.concatenate([b, b], axis=0)

        self.hist = np.array([b])
        self.hist_t = np.array([t])

        # Add keys to print out
        self.keys_to_print = ['bayesian_update', 'biased_observations', 'upsample', 'N_ens']
        if self.bayesian_update:
            self.keys_to_print += ['filter', 'inflation']

    @property
    def N_ens(self):
        return self.hist.shape[-1]

    @property
    def get_current_time(self):
        return self.hist_t[-1]

    @property
    def get_current_bias(self):
        current_state = self.hist[-1]
        return self.get_bias(state=current_state)

    @property
    def get_current_innovations(self):
        current_state = self.hist[-1]
        return self.get_innovations(state=current_state)

    def get_bias(self, state, **kwargs):
        return state

    def get_innovations(self, state, **kwargs):
        return state

    def get_ML_state(self, **kwargs):
        return None

    def print_bias_parameters(self):
        print('\n ---------------- {} bias model parameters --------------- '.format(self.name))
        for key in sorted(self.keys_to_print):
            val = getattr(self, key)
            if type(val) is float:
                print('\t {} = {:.6}'.format(key, val))
            else:
                print('\t {} = {}'.format(key, val))

    def update_history(self, b, t=None, reset=False, update_last_state=False, **kwargs):

        assert self.hist.ndim == 3

        if not reset and not update_last_state:
            if b is None or t is None:
                raise AssertionError('both t and b must be defined')
            self.hist = np.concatenate((self.hist, b))
            self.hist_t = np.concatenate((self.hist_t, t))
        elif update_last_state:
            if b is not None:
                self.hist[-1, self.observed_idx] = b
                if hasattr(self, 'reset_state'):
                    if 'u' not in kwargs.keys():
                        kwargs['u'] = b
                    self.reset_state(**kwargs)
            else:
                raise ValueError('psi must be provided')
            if t is not None:
                self.hist_t[-1] = t
        else:
            if t is None:
                t = self.get_current_time
            self.hist_t = np.array([t])
            self.hist = np.array([b])
            if self.hist.ndim < 3:
                self.hist = np.expand_dims(self.hist, axis=-1)
            if hasattr(self, 'reset_state'):
                r = np.zeros((self.N_units, self.N_ens))
                self.reset_state(u=b, r=r)


# =================================================================================================================== #


class NoBias(Bias):
    name = 'NoBias'

    def __init__(self, y, t, dt, **kwargs):
        super().__init__(b=np.zeros(y.shape), t=t, dt=dt, **kwargs)
        self.N_dim = self.hist.shape[1]
        self.observed_idx = np.arange(self.N_dim)

    def state_derivative(self):
        return np.zeros([self.N_dim, self.N_dim])

    def time_integrate(self, t, **kwargs):
        return np.zeros([len(t), self.N_dim, self.N_ens]), t

# =================================================================================================================== #


class ESN(Bias, EchoStateNetwork):
    name = 'ESN'

    def __init__(self, y, t, dt, **kwargs):
        self.update_reservoir = False

        # --------------------------  Initialise parent Bias  ------------------------- #
        Bias.__init__(self, b=y, t=t, dt=dt, **kwargs)

        # --------------------  Initialise parent EchoStateNetwork  ------------------- #
        EchoStateNetwork.__init__(self, y=self.hist[0], dt=dt, **kwargs)

        # Flags
        self.initialised = False
        self.trained = False

        # Add keys to print data
        extra_keys = ['t_train', 't_val', 'N_wash', 'rho', 'sigma_in',
                      'N_units', 'perform_test', 'L', 'connect', 'tikh',
                      'update_reservoir', 'observed_idx']
        self.keys_to_print += extra_keys

    def reset_bias(self, u, r=None):
        self.reset_state(u=u, r=r)

    def state_derivative(self):
        u, r = [np.mean(xx, axis=-1, keepdims=True) for xx in self.get_reservoir_state()]
        J = self.Jacobian(open_loop_J=True, state=(u, r))  # Compute ESN Jacobian
        db_din = J[np.array(self.bias_idx), np.array([self.bias_idx]).T]
        return -db_din

    def time_integrate(self, t, y=None, wash_t=None, wash_obs=None):
        if not self.trained:
            raise NotImplementedError('ESN model not trained')

        interp_flag = False
        Nt = len(t) // self.upsample
        if len(t) % self.upsample:
            Nt += 1
            interp_flag = True
        t_b = np.round(self.get_current_time + np.arange(0, Nt+1) * self.dt_ESN, self.precision_t)

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
                washout = wash_obs - np.mean(wash_model, axis=-1)

                u_open, r_open = self.openLoop(washout, inflation=self.inflation)
                u[t1:t1+self.N_wash+1] = u_open
                r[t1:t1+self.N_wash+1] = r_open
                Nt -= self.N_wash

                # Run the rest of the time window in closed-loop
                if Nt > 0:
                    # Store open-loop forecast
                    self.reset_state(u=self.outputs_to_inputs(full_state=u_open[-1]), r=r_open[-1])
                    u_close, r_close = self.closedLoop(Nt)
                    u[t1 + self.N_wash + 1:] = u_close[1:]
                    r[t1 + self.N_wash + 1:] = r_close[1:]
        # Interpolate the final point if the upsample is not multiple of dt
        if interp_flag:
            u[-1] = interpolate(t_b[-Nt:], u[-Nt:], t[-1])
            r[-1] = interpolate(t_b[-Nt:], r[-Nt:], t[-1])
            t_b[-1] = t[-1]

        # update ESN physical and reservoir states, and store the history if requested
        self.reset_state(u=self.outputs_to_inputs(full_state=u[-1]), r=r[-1])

        return u[1:], t_b[1:]

    def train_bias_model(self, **train_data):
        data = train_data['data']
        del train_data['data']
        dict_items = train_data.copy().items()
        for key, val in dict_items:
            if hasattr(self, key):
                setattr(self, key, val)
                del train_data[key]

        self.train(data, **train_data)
        self.trained = True
        if self.bayesian_update:
            self.update_history(b=np.zeros((self.N_dim, self.m)), reset=True)
            self.initialise_state(data=data,  N_ens=self.m)

    def get_ML_state(self, concat_reservoir_state=False):
        u, r = self.get_reservoir_state()
        if concat_reservoir_state:
            return np.concatenate([u, r], axis=0)
        else:
            return u

    @property
    def bias_idx(self):
        if self.biased_observations:
            return [a for a in np.arange(self.N_dim) if a not in self.observed_idx]
        else:
            return self.observed_idx

    def get_bias(self, state, mean=True):
        if mean:
            state = np.mean(state, axis=-1, keepdims=True)

        if state.shape[0] == self.N_dim:
            return state[self.bias_idx]
        elif state.shape[1] == self.N_dim:
            return state[:, self.bias_idx]
        else:
            raise AssertionError('state shape = {}'.format(state.shape))

    def get_innovations(self, state, mean=True):
        if mean:
            state = np.mean(state, axis=-1, keepdims=True)

        if state.shape[0] == self.N_dim:
            return state[self.observed_idx]
        elif state.shape[1] == self.N_dim:
            return state[:, self.observed_idx]
        else:
            raise AssertionError('state shape = {}'.format(state.shape))
# =================================================================================================================== #



