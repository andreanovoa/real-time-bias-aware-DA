import numpy as np

from .bias import Bias
from romda.models import DiscreteIntegrator, HistoryTracker
from types import SimpleNamespace


class DriftLinearBias(Bias):
    """
    Data-driven bias model with temporal drift and linear state dependence.

    Combines
    1. Constant bias (as in ConstantBias)
    2. Temporal drift: linear growth with time (drift_velocity)
    3. Linear state dependence: linear_matrix @ bias_state

    The bias evolution is modeled as:
        db/dt = drift_velocity + linear_matrix @ b

    This discrete integration becomes:
        b_{t+dt} = b_t + dt * (drift_velocity + linear_matrix @ b_t)
                 = (I + dt * linear_matrix) @ b_t + dt * drift_velocity

    Attributes:
        drift_velocity (np.ndarray): Constant drift term, shape (n_bias,) or (n_bias, 1)
        linear_matrix (np.ndarray): Linear coupling matrix, shape (n_bias, n_bias)
        decay_rate (float): Alternative to linear_matrix for simple exponential decay:
                           linear_matrix = -decay_rate * I
    """

    biased_observations = False
    correlation_based_training = False
    augment_data = False

    extra_keys_to_print = ['decay_rate']

    def __init__(self, innovation, t, dt,
                 drift_velocity=None,
                 linear_matrix=None,
                 decay_rate=None, **kwargs):
        """
        Initialize DriftLinearBias.

        Parameters:
        -----------
        innovation : np.ndarray
            Initial innovation/bias state, shape (n_bias,) or (n_bias, n_ens)
        t : float
            Initial time
        dt : float
            Time step
        drift_velocity : np.ndarray, optional
            Linear drift term. If None, defaults to zero.
        linear_matrix : np.ndarray, optional
            Linear coupling matrix (n_bias, n_bias). If None and decay_rate is not None,
            linear_matrix is set to -decay_rate * I.
        decay_rate : float, optional
            Exponential decay rate. Only used if linear_matrix is None.
            decay_rate > 0 means bias decays to zero.
        **kwargs : dict
            Additional arguments passed to parent Bias class
        """

        # Store linear dynamics parameters before calling parent init
        innovation = np.atleast_1d(np.asarray(innovation, dtype=float))
        if innovation.ndim == 1:
            innovation = innovation.reshape(-1, 1)
        elif innovation.ndim == 3:
            innovation = innovation[0]
        n_bias = innovation.shape[0]
        self._b0 = innovation
        self.decay_rate = decay_rate

        # Initialize drift_velocity
        if drift_velocity is None:
            self.drift_velocity = np.zeros((n_bias, 1))
        else:
            drift_velocity = np.atleast_1d(drift_velocity).reshape(-1, 1)
            if drift_velocity.shape[0] != n_bias:
                raise ValueError(f"drift_velocity dimension {drift_velocity.shape[0]} != innovation dimension {n_bias}")
            self.drift_velocity = drift_velocity

        # Initialize linear_matrix
        if linear_matrix is not None:
            linear_matrix = np.atleast_2d(linear_matrix)
            if linear_matrix.shape != (n_bias, n_bias):
                raise ValueError(f"linear_matrix shape {linear_matrix.shape} != ({n_bias}, {n_bias})")
            self.linear_matrix = linear_matrix
        elif decay_rate is not None:
            # Simple exponential decay: -decay_rate * I
            self.linear_matrix = -decay_rate * np.eye(n_bias)
        else:
            # Default: no linear coupling
            self.linear_matrix = np.zeros((n_bias, n_bias))

        super().__init__(innovation=innovation, t=t, dt=dt, **kwargs)

    @property
    def initialize_bias_state(self):
        b0 = self._b0
        if b0.shape[-1] != self.N_ens:
            b0 = np.repeat(np.mean(b0, axis=-1, keepdims=True), self.N_ens, axis=1)
        if self.biased_observations and b0.shape[0] == self.Nq:
            b0 = np.concatenate([b0, b0], axis=0)
        assert b0.shape[0] == self.N_dim, \
            f'Initial bias state has {b0.shape[0]} components, expected N_dim={self.N_dim}.'
        return b0

    @property
    def dt_step(self):
        """Integrator time step (same as output dt for this model)."""
        return self.dt

    def state_derivative(self):
        """
        Jacobian of the bias with respect to its state, i.e., the linear coupling matrix of
            db/dt = drift_velocity + linear_matrix @ b

        Returns
        -------
        np.ndarray
            The linear coupling matrix (n_bias, n_bias)
        """
        return self.linear_matrix

    def init_forecaster(self, **kwargs):
        """
        Initialize the forecaster with simple discrete time stepping.
        """
        self._forecaster = SimpleNamespace()

        initial_capacity = kwargs.pop('initial_capacity', 1000)
        self._forecaster.history = HistoryTracker(initial_capacity=initial_capacity)

        # Use DiscreteIntegrator for time stepping
        self._forecaster.integrator = DiscreteIntegrator(self)

    def time_step(self, Nt: int = 100, **kwargs):
        """
        Advance bias state for Nt discrete time steps using Euler method.

        For each step:
            b_{t+dt} = b_t + dt * (drift_velocity + linear_matrix @ b_t)

        Parameters
        ----------
        Nt : int
            Number of time steps to advance

        Returns
        -------
        tuple
            (states, times) where:
            - states : np.ndarray, shape (Nt+1, N_dim, n_ens) - all states including initial
            - times : np.ndarray, shape (Nt+1,) - all time stamps
        """
        # Generate output times
        t_out = np.round(
            self.current_time + np.arange(Nt + 1) * self.dt,
            self.precision_t
        )

        # Initialize storage
        current_state = self.current_state.copy()
        states_list = [current_state]

        # Integrate forward Nt steps
        for _ in range(Nt):
            # Extract bias portion if biased_observations
            if self.biased_observations:
                n_bias = current_state.shape[0] // 2
                b_bias = current_state[:n_bias, :]
                b_innov = current_state[n_bias:, :]
            else:
                b_bias = current_state
                b_innov = None

            # Euler step
            b_bias_new = b_bias + self.dt * (
                self.drift_velocity + self.linear_matrix @ b_bias
            )

            # Reconstruct full state if biased_observations
            if self.biased_observations:
                current_state = np.vstack([b_bias_new, b_innov])
            else:
                current_state = b_bias_new

            states_list.append(current_state.copy())

        # Stack into array: (Nt+1, N_dim, n_ens)
        states_array = np.stack(states_list, axis=0)

        return states_array, t_out

    def print_bias_parameters(self):
        """Print bias model parameters including drift and linear terms."""
        super().print_bias_parameters()
        print('\n ------------ DriftLinearBias specific parameters ----------- ')

        if self.drift_velocity is not None:
            print('\t drift_velocity shape: {}'.format(self.drift_velocity.shape))
            print('\t drift_velocity = {}'.format(self.drift_velocity.ravel()[:5]))  # Show first 5

        if self.linear_matrix is not None:
            print('\t linear_matrix shape: {}'.format(self.linear_matrix.shape))
            evals = np.linalg.eigvalsh(self.linear_matrix)
            print('\t eigenvalues range: [{:.4f}, {:.4f}]'.format(evals.min(), evals.max()))


if __name__ == '__main__':

    n_bias = 3
    innovation = np.random.randn(n_bias, 1) * 0.1
    drift_velocity = np.array([[0.01], [0.02], [0.03]])  # Time-dependent growth
    decay_rate = 0.05  # 5% decay per time unit

    db = DriftLinearBias(
        innovation=innovation,
        t=0.0,
        dt=0.1,
        drift_velocity=drift_velocity,
        decay_rate=decay_rate
    )

    print("Initial state shape:", db.current_state.shape)
    print("Initial bias:", db.current_bias)

    # Integrate forward in time
    state, t = db.time_integrate(Nt=100)
    db.update_history(state, t)
    print("\nAfter 100 steps:")
    print("State shape:", state.shape)
    print("Final time:", t[-1])
    print("Final bias:", db.current_bias)

    # Print parameters
    db.print_bias_parameters()
