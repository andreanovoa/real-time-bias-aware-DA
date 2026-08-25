from copy import deepcopy
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from romda.models import HistoryTracker, Integrator, Model
from romda.plotting import categorical_cmap


def _ensrkf_update(Af: np.ndarray, d: np.ndarray, Cdd: np.ndarray, M: np.ndarray) -> np.ndarray:
    r"""Stateless ensemble square-root Kalman filter update, for the Bayesian bias update.

    Implements the same analysis kernel as `romda.estimators.ensembles.EnSRKF`, but
    takes an explicit measurement matrix `M` instead of reading one from a `Model`
    (the bias state has no associated `Model`, so it cannot use `EnSRKF` directly).
    Kept private and self-contained (no `romda`-internal imports) to avoid a
    circular import between `romda.bias_estimators` and `romda.estimators`.

    Parameters
    ----------
    Af : ndarray, shape (N, m)
        Forecast ensemble.
    d : ndarray, shape (Nq,)
        Observation vector.
    Cdd : ndarray, shape (Nq, Nq)
        Observation-noise covariance.
    M : ndarray, shape (Nq, N)
        Measurement operator.

    Returns
    -------
    ndarray, shape (N, m)
        Analysis ensemble.
    """
    m = Af.shape[1]
    d = np.atleast_1d(d)
    Cdd = np.atleast_2d(Cdd)

    psi_f_m = np.mean(Af, 1, keepdims=True)
    Psi_f = Af - psi_f_m

    y = M @ psi_f_m
    S = M @ Psi_f

    C = (m - 1) * Cdd + S @ S.T
    Cinv = inv(C)

    ma = psi_f_m + Psi_f @ S.T @ Cinv @ (d[:, None] - y)

    T = np.eye(m) - S.T @ Cinv @ S
    ev, evec = np.linalg.eigh(T)
    sqrtT = evec @ np.diag(np.sqrt(np.maximum(ev, 0.0))) @ evec.T

    Aa = ma + Psi_f @ sqrtT
    return Aa if np.isreal(Aa).all() else Af


class Bias:
    r"""Base class for the model-bias estimators used in bias-aware data assimilation.

    A bias estimator provides three things to the assimilation loop:

    1. a **forecast** of the bias between analyses (`time_integrate`), driven by its
       internal forecaster (an ESN, a constant map, a linear model, ...);
    2. the **Jacobian** of the bias with respect to the observables
       (`state_derivative`), $\mathbf{J} = \mathrm{d}\mathbf{b}/\mathrm{d}\mathbf{q}$,
       required by the regularized bias-aware EnKF;
    3. an **update rule** from the analysis innovation
       (`update_state_from_innovation`), optionally Bayesian (an internal EnSRKF on
       the bias state).

    The estimator state has $N_\mathrm{dim}$ components: $[\mathbf{b}]$ if the
    observations are unbiased, or $[\mathbf{b}; \mathbf{i}]$ (bias and innovations,
    $N_\mathrm{dim} = 2 N_q$) if `biased_observations` is set. Child classes may add
    hidden components (e.g., the ESN reservoir).

    Parameters
    ----------
    innovation : np.ndarray
        Initial innovation/bias estimate, used to set the observable dimension $N_q$.
    t : float
        Initial time.
    dt : float
        Time step of the output history.
    **kwargs
        Class-attribute overrides (see Attributes) and forecaster options.

    Attributes
    ----------
    upsample : int
        Upsampling factor of the internal forecaster time step relative to ``dt``.
    L : int
        Number of trajectories in the training dataset (data-driven estimators).
    augment_data : bool or int
        Whether (and how much) to augment the training data.
    bayesian_update : bool
        If True, the innovation update is a Bayesian (EnSRKF) update of the full
        estimator state; otherwise the innovation is assigned directly.
    biased_observations : bool
        If True, the observations themselves are assumed biased and the estimator
        tracks bias and innovations separately.
    force_retrain : bool
        If True, retrain the forecaster even if a cached configuration exists.
    """

    upsample = 1
    L = 1
    augment_data = False

    forecaster_type = None  # This should be set in child classes to specify the expected type of the forecaster model, e.g., ESN_model for ESN_bias.
    bayesian_update = False         # Default to not perform bayesian update to state
    biased_observations = False  # Whether observations are biased or not
    force_retrain = False

    keys_to_print = ['bayesian_update', 'upsample', 'biased_observations', 'L', 'augment_data_length']
    extra_keys_to_print = []

    def __init__(self, innovation, t, dt, **kwargs):


        # ===================== ASSIGN PROVIDED KWARGS ======================= ##
        keys = list(kwargs.keys())
        [setattr(self, key, kwargs.pop(key)) for key in keys if hasattr(self, key)]

        self.keys_to_print += self.extra_keys_to_print

        # ================== Setup dimensions ================= ##

        self.precision_t = int(-np.log10(dt)) + 2
        self.dt = dt
        self.Nq = self._format_state(innovation).shape[1]


        # ================== Initialize Forecaster & HISTORY ================= ##

        self.init_forecaster(**kwargs)

        bias_state = self.initialize_bias_state

        assert bias_state.shape[-2] == self.N, f"Bias state shape {bias_state.shape} does not match expected (Nt, N = {self.N}, Nens)."
        self.update_history(bias_state, t=t, reset=True)


    @property
    def name(self):
        return self.__class__.__name__

    @property
    def N_ens(self):
        if not hasattr(self, '_N_ens'):
            return 1
        return self._N_ens

    @N_ens.setter
    def N_ens(self, value):
        if value <= 0:
            raise ValueError("Number of ensemble members must be positive.")
        if hasattr(self, 'history'):
            Warning("Changing N_ens after initialization may lead to inconsistencies in the history.")
        self._N_ens = value

    @property
    def augment_data_length(self):
        augment_data = self.augment_data
        if augment_data:
            if isinstance(augment_data, int) and augment_data > 1:
                return augment_data
            return 2
        return 1

    @property
    def initialize_bias_state(self):
        """
        Only used at initialization. If the forecaster is a model, this should be handled by the child class.
        The state has N_dim components: [bias] if the observations are unbiased,
        or [bias; innovations] if the observations are biased (N_dim = 2 * Nq).
        """
        return np.zeros((self.N_dim, self.N_ens))

    @property
    def forecaster(self):
        assert hasattr(self, '_forecaster'), 'Forecaster not initialized yet.'
        return self._forecaster # type: ignore #should be an instance of a forecaster model, e.g., ESN_model

    @forecaster.setter
    def forecaster(self, obj):
        if self.forecaster_type is None:
            raise ValueError('Child class must specify forecaster_type.')
        assert isinstance(obj, self.forecaster_type), f'Forecaster must be an instance of {self.forecaster_type}, but got {type(obj)}.'
        self._forecaster = obj

    @property
    def history(self) -> HistoryTracker:
        return self._forecaster.history  # type: ignore

    @property
    def integrator(self) -> Integrator:
        """
        This is the integrator used by the model of the bias. E.g., DiscreteIntegrator if using ESN_model as forecaster.
        """
        return self._forecaster.integrator  # type: ignore

    @property
    def N_dim(self):
        if not self.biased_observations:
            return self.Nq
        else:
            return 2 * self.Nq

    @property
    def N(self):
        return self.N_dim + self.N_hidden

    @property
    def N_hidden(self):
        # Number of hidden units in the bias model, e.g., for ESN bias model. This is added to the state dimension N_dim to get the total state dimension N.
        return 0


    @property
    def bias_idx(self):
        return np.arange(self.Nq)

    @property
    def observed_idx(self):
        if self.biased_observations:
            return self.bias_idx + self.Nq
        else:
            return self.bias_idx

    def init_forecaster(self, **kwargs):
        """
        .....
        """
        raise NotImplementedError('Bias child classes must implement _init_forecaster() method.')

    def state_derivative(self):
        """
        Returns the derivative of the bias state, which is used for time integration.
        This is computed by the forecaster model.
        """
        raise NotImplementedError('Bias child classes must implement state_derivative property, typically computed by the forecaster model.')

    def washout_phase(self, d_wash, t_wash, **kwargs) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """
        Optional method to initialize the bias model if needed, e.g., by running a washout phase with given data.
        By default, does nothing, but can be implemented in child classes if needed.
        """
        return None

    @property
    def washout_data(self):
        """
        Returns the washout data used for initializing the bias model, which is typically obtained from the washout phase using the validation data. This property can be used to access the washout data for further processing or analysis.

        Returns
        -------
            Tuple of (washout_data, washout_time) where:
                - washout_data: np.ndarray - Washout data used for initializing the bias model.
                - washout_time: np.ndarray - Time points corresponding to the washout data.
        """
        return getattr(self, '_washout_data', (None, None))



    @washout_data.setter
    def washout_data(self, value):
        assert isinstance(value, tuple) and len(value) == 2, "Washout data must be a tuple of (washout_data, washout_time)."
        self._washout_data = value

    def _format_state(self, b):
        """
        Ensure b has shape (nt, nb, nens)
        """
        if b.ndim == 3:
            return b # already (nt, nb, nens)
        elif b.ndim == 1:
            b_repeated = np.repeat(b[:, np.newaxis], self.N_ens, axis=1)  # (nb,) -> (nb, nens)
            return b_repeated[np.newaxis, :, :]  # (nb, nens) -> (1, nb, nens) Add extra dimension for time

        elif b.ndim == 2 and (b.shape[-1] == self.N_ens or b.shape[0] == self.N_dim):
            return b[np.newaxis, :, :]  # (nb, nens) -> (1, nb, nens) # Add extra dimension for time
        elif b.ndim == 2 and b.shape[-1] != self.N_ens:
            return np.repeat(b[:, :, np.newaxis], self.N_ens, axis=2)  # (nt, nb) -> (nt, nb, nens)
        else:
            raise AssertionError(f'b must have 1, 2 or 3 dimensions, got {b.ndim}=({b.shape})')


    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        """Setter for the time step."""
        if value <= 0:
            raise ValueError("Time step must be positive.")
        self._dt = np.round(value, self.precision_t)

    @property
    def hist(self):
        """Returns only the valid (non-empty) portion of the history buffer."""
        return self.history.hist

    @property
    def hist_t(self):
        """Returns only the valid portion of the time history."""
        return self.history.hist_t

    @property
    def current_state(self):
        """Returns the current state (last entry in history)."""
        return self.history.current_state.copy()

    @property
    def current_time(self):
        """Returns the current time (last entry in time history)."""
        return self.history.current_time


    @property
    def config(self):
        _config = dict()
        for key in self.keys_to_print:
             if hasattr(self, key):
                _config[key] = getattr(self, key)

        return _config

    @property
    def current_bias(self):
        """Returns the current (ensemble-mean) bias computed from the current state.
        Shape: (Nq, 1) -- the bias is defined on the ensemble mean."""
        return self.get_bias(state=self.current_state, mean=True)[0, :, :]

    @property
    def current_innovations(self):
        """Returns the current (ensemble-mean) innovations computed from the current state.
        Shape: (Nq, 1)."""
        return self.get_innovations(state=self.current_state, mean=True)[0, :, :]


    def get_bias(self, state, mean=False):
        state = self._format_state(state)
        if mean:
            state = np.mean(state, axis=-1, keepdims=True)

        return state[:, self.bias_idx, :]


    def get_innovations(self, state, mean=False):
        state = self._format_state(state)
        if mean:
            state = np.mean(state, axis=-1, keepdims=True)

        return state[:, self.observed_idx, :]


    def get_bias_hist(self, mean=False):
        return self.get_bias(state=self.hist, mean=mean)

    def time_integrate(self, Nt):
        return self.integrator.advance(Nt=Nt)


    def update_history(self, state, t=None, reset=False, modify_saved_states=False, **kwargs):
        state = self._format_state(state) # Ensure shape (nt, nstate, nens)

        # Ensure time array matches nt
        if t is None:
            t = (np.arange(state.shape[0]) * self.dt).round(self.precision_t) + self.current_time
        if isinstance(t, float):
            t = np.array([t])
        assert t.size == state.shape[0], f"Length of t ({t.size}) must match number of time steps in state ({state.shape[0]})."

        if modify_saved_states:
            assert state.shape[0] == 1, "When modify_saved_states is True, state must have only one time step (shape[0] == 1)."

        self.history.update_history(state, t=t, reset=reset, modify_saved_states=modify_saved_states)
        self.update_history_aux(state=state, reset=reset, modify_saved_states=modify_saved_states, **kwargs)

    def update_history_aux(self, **kwargs):
        """Auxiliary method to update any additional history attributes in child classes if needed."""
        pass


    def update_state_from_innovation(self, input_innovation):
        """
        Optional method to perform a Bayesian update to the state using the bias model. This can be implemented in child classes if needed, e.g., for ESN bias model.
        By default, does nothing, but can be implemented in child classes if needed.

        Parameters
        ----------
        input_innovation : np.ndarray
            Analysis innovation ensemble, shape ``(Nq, m)``, ``(Nq, 1)`` or ``(Nq,)``.

        Returns
        -------
        np.ndarray
            Updated estimator state, shape ``(N, N_ens)``.
        """
        input_innovation = self._format_state(input_innovation) # Ensure shape (nt, nstate, nens)
        assert input_innovation.shape[0] == 1, "Input innovation must have only one time step (shape[0] == 1) for state_from_innovation method."

        forecast_state = self.current_state


        if self.bayesian_update:
            mean_innovation = np.mean(input_innovation[0], axis=-1)  # Average innovation across ensemble (obs_dim, Nens) -> (obs_dim,)
            # cov_innovation = (inn_uncertainty * np.max(np.abs(mean_innovation)))**2 * np.eye(mean_innovation.shape[0])  # Diagonal covariance of innovation (obs_dim, obs_dim)
            # cov_innovation = np.cov(input_innovation[0], rowvar=True)  # Full covariance of innovation (obs_dim, obs_dim)
            # cov_innovation = np.atleast_2d(cov_innovation)
            conv_inn = input_innovation[0] - mean_innovation[:, np.newaxis]  # Centered innovations (obs_dim, Nens)
            cov_innovation = np.atleast_2d(np.cov(conv_inn, rowvar=True))  # Full covariance of centered innovations (obs_dim, obs_dim)

            updated_state = _ensrkf_update(Af=forecast_state, d=mean_innovation, Cdd=cov_innovation, M=self.DA_method)
        else:
            updated_state = forecast_state.copy()
            mean_innovation = np.mean(input_innovation[0], axis=-1, keepdims=True)  # Average innovation across ensemble (obs_dim, Nens) -> (obs_dim, 1)
            if input_innovation.shape[-1] == self.N_ens:
                # One innovation per bias-ensemble member: assign directly
                updated_state[self.observed_idx, :] = input_innovation[0]

            elif self.N_ens == 1 or input_innovation.shape[-1] == 1:
                # Assign the same mean innovation to all ensemble members
                updated_state[self.observed_idx, :] = np.repeat(mean_innovation, self.N_ens, axis=-1)

            else:  # ensemble sizes differ => resample from the innovation statistics
                cov_innovation = np.cov(input_innovation[0], rowvar=True)
                mean_innovation = mean_innovation.flatten()
                cov_innovation = np.atleast_2d(cov_innovation)
                resampled_innovation = np.random.multivariate_normal(mean_innovation, cov_innovation, size=self.N_ens).T  # Resample innovations for each ensemble member (obs_dim, Nens)
                updated_state[self.observed_idx, :] = resampled_innovation

        return updated_state


    @property
    def DA_method(self):
        """Cached measurement operator `M` for the Bayesian innovation update (`_ensrkf_update`)."""
        if not self.bayesian_update:
            raise ValueError("Why is this being accessed when bayesian_update is False?")
        elif not hasattr(self, '_DA_method'):
            observation_operator = np.zeros((len(self.observed_idx), self.N))
            observation_operator[:, self.observed_idx] = np.eye(len(self.observed_idx))
            self._DA_method = observation_operator
            print(f"Initialized Bayesian-update observation operator for bias model. M shape: {observation_operator.shape}")

        return self._DA_method

    def print_bias_parameters(self, indent=2):
        print(f'\n{self.__class__.__name__}')
        print(f'{"=" * len(self.__class__.__name__)}')
        for key in sorted(set(self.keys_to_print)):
            if hasattr(self, key):
                val = getattr(self, key)
                if type(val) is float:
                    print(f'{" " * indent}{key} = {val:.6}')
                else:
                    print(f'{" " * indent}{key} = {val}')
        if isinstance(self.forecaster, Model):
            print(f'{" " * indent}====Forecaster model ====')
            self.forecaster.print_parameters(show_header=False,
                                             indent=indent*2)


    def copy(self):
        return deepcopy(self)

    def close(self):
        """Close any resources used by the bias model, e.g., forecaster model."""
        if hasattr(self, '_forecaster') and hasattr(self._forecaster, 'close'):
            self._forecaster.close()


    # ================= Visualization methods ================== #
    def visualize_bias_and_innovations(self, state=None, t=None, plot_members=True, ylims=None):
        if state is None:
            state = self.hist

        inn = self.get_innovations(state=state, mean=not plot_members)
        b = self.get_bias(state=state, mean=not plot_members)

        if t is None:
            t = self.hist_t[-len(inn):]

        lbls =( [f'inn_{ii}' for ii in range(inn.shape[1])],
              [f'b_{ii}' for ii in range(b.shape[1])])
        ttls = ['Innovations', 'Bias']


        t_mid = len(t) - len(t)//8
        lims = [[0, t_mid ],  [t_mid, len(t)-1]  ]

        Nq = inn.shape[1]
        nens = inn.shape[-1]
        cols = categorical_cmap(nc=2, nsc=nens, continuous=False)
        cols = [cols[ii * nens:(ii + 1) * nens] for ii in range(2)]
        ci = 0
        # Plot the time evolution of the observables
        for y, lbl, ttl, cs in zip([inn, b], lbls, ttls, cols):

            fig = plt.figure(figsize=(8, Nq+1), layout="constrained")
            plt.suptitle(f'{ttl} time evolution')

            axs = fig.subplots(Nq, 2, sharey='row', sharex='col')
            if self.Nq == 1:
                axs = [axs]


            for ii, ax in enumerate(axs):
                for jj, lim in enumerate(lims):
                    lines = ax[jj].plot(t[lim[0]:lim[1]], y[lim[0]:lim[1], ii])
                    for line, color in zip(lines, cs):
                        line.set_color(color)
                    if ii == self.Nq-1:
                        ax[jj].set(xlabel='$t$', xlim=[t[lim[0]], t[lim[1]]])

                ax[0].set(ylabel=lbl[ii])
                if ylims is not None:
                    ax[0].set(ylim=ylims)

            ci += 1


