import numpy as np

from .bias import Bias
from models import HistoryTracker, ConstantIntegrator
from types import SimpleNamespace


class ConstantBias(Bias):
    """
    Constant (persistent) bias estimator.

    The bias is held constant between analysis steps, i.e., the forecast model of the bias
    is db/dt = 0. At each analysis step, the bias state is reset to the latest innovation
    (see Bias.update_state_from_innovation). This is the classic persistent-bias assumption,
    and is a special case of DriftLinearBias with zero drift and zero linear coupling.

    Parameters
    ----------
    innovation : np.ndarray
        Initial innovation/bias estimate, shape (Nq,), (Nq, N_ens) or (1, Nq, N_ens).
    t : float
        Initial time.
    dt : float
        Time step of the output history.
    k : float or np.ndarray, optional
        If provided, the initial bias state is set to the constant value(s) k instead of
        the provided innovation.
    """

    upsample = 1  # The forecast is constant, so no upsampling is needed

    def __init__(self, innovation, t, dt, k=None, **kwargs):

        if k is not None:
            innovation = np.ones(np.shape(innovation)) * k  # Constant initial bias

        self._b0 = np.asarray(innovation, dtype=float)

        super().__init__(innovation=innovation, t=t, dt=dt, **kwargs)

    @property
    def initialize_bias_state(self):
        b0 = self._b0
        if b0.ndim == 3:
            b0 = b0[0]
        elif b0.ndim == 1:
            b0 = b0[:, np.newaxis]

        if b0.shape[-1] != self.N_ens:
            b0 = np.repeat(np.mean(b0, axis=-1, keepdims=True), self.N_ens, axis=1)

        if self.biased_observations and b0.shape[0] == self.Nq:
            # State is [bias; innovations]: initialize both blocks with the same values
            b0 = np.concatenate([b0, b0], axis=0)

        assert b0.shape[0] == self.N_dim, \
            f'Initial bias state has {b0.shape[0]} components, expected N_dim={self.N_dim}.'
        return b0

    def state_derivative(self):
        n_bias = self.Nq
        return np.zeros([n_bias, n_bias])

    def init_forecaster(self, **kwargs):
        """
        Constant forecaster: no underlying model, the state is simply held in time.
        """
        self._forecaster = SimpleNamespace()

        initial_capacity = kwargs.pop('initial_capacity', 1000)
        self._forecaster.history = HistoryTracker(initial_capacity=initial_capacity)

        #  INITIALISE INTEGRATOR STRATEGY ================== ##
        self._forecaster.integrator = ConstantIntegrator(self)


class NoBias(ConstantBias):
    """
    Placeholder bias estimator that always returns zero bias.
    Useful to run the bias-aware machinery in its unbiased limit.
    """

    def __init__(self, innovation, t, dt, **kwargs):
        kwargs.pop('k', None)
        super().__init__(innovation=innovation, t=t, dt=dt, k=0., **kwargs)

    def update_state_from_innovation(self, input_innovation):
        # NoBias never updates: the bias remains identically zero
        return self.current_state


if __name__ == '__main__':
    nb = ConstantBias(innovation=np.ones((2, 7)), t=0.0, dt=0.1)

    state, t = nb.time_integrate(Nt=10)
    nb.update_history(state, t)

    reset_constant = np.array([np.random.rand(nb.N_dim)]).T * np.ones_like(state[-1])
    nb.update_history(reset_constant, update_last_state=True)

    state, t = nb.time_integrate(Nt=10)
    nb.update_history(state, t)

    print(nb.history.hist.shape, nb.get_bias_hist())
