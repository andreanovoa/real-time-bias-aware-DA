

import numpy as np


class HistoryTracker:
    """ Mixin class to add history tracking functionality to models.
    """
    
    # ________________________ History accessors ________________________ #

    @property
    def hist(self):
        """Returns only the valid (non-empty) portion of the history buffer."""
        return self._hist[:self.current_ti]

    @property
    def hist_t(self):
        """Returns only the valid portion of the time history."""
        return self._hist_t[:self.current_ti]
    
    @property
    def capacity(self):
        return self._hist_t.shape[0]

    @property
    def current_state(self):
        return self.hist[self.current_ti - 1]

    @property
    def current_time(self):
        return self.hist_t[self.current_ti - 1]
    
    @property
    def current_ti(self):
        return self._ti

    @current_ti.setter
    def current_ti(self, value: int):
        self._ti = value


    def __init__(self, initial_capacity=1000):
        """ Initialises the history arrays.
        Args:
            initial_capacity: Initial capacity of the history arrays.
        """

        self._initial_capacity = initial_capacity
        self.current_ti = 0  # Current time index in history


    def _reset_history(self, new_history, t_reset):
        """Resets the history arrays to the provided new_history and t_reset.
        Args:
            new_history: New state history to set (Nt, N, m)
            t_reset: New time history to set (Nt,)
        """

        Nt = max(new_history.shape[0], self._initial_capacity)

        # Initialize the history arrays
        self._hist = np.empty((Nt, new_history.shape[1], new_history.shape[2]))
        self._hist_t = np.empty((Nt,))
        # Store the reset history

        self._hist[:new_history.shape[0]] = new_history
        self._hist_t[:t_reset.shape[0]] = t_reset
        self.current_ti = new_history.shape[0]

    def _reset_last_state(self, new_state, t=None):
        """Resets only the last state in the history arrays to the provided new_state and t."""
        if new_state.shape[0]> 1:
            raise ValueError("new_state must contain only one time step to reset the last state.")
        else:
            # print(f"changing last state in history: {self._hist[self.current_ti - 1]} to new value: ", new_state.flatten())

            self._hist[self.current_ti - 1] = new_state[0]
        if t is not None:
            # print(f"changing last time in history: {self._hist_t[self.current_ti - 1]} to new value: ", t[0])
            self._hist_t[self.current_ti - 1] = t[0]


    def update_history(self, state: np.ndarray, t: np.ndarray, reset=False, update_last_state=False):
        assert state.shape[0] == t.shape[0], f"Length of t ({t.shape}) must match number of time steps in state ({state.shape})."
        if reset: # Reset the full history 
            self._reset_history(state, t)
        
        elif update_last_state: # Update only the last state in history
            self._reset_last_state(new_state=state, t=t)
        else:
            t0 = self.current_ti
            t1 = t0 + state.shape[0]

            if t1 > self.capacity:
                self._increase_hist_size(Nt=state.shape[0]*10)

            self._hist[t0:t1] = state
            self._hist_t[t0:t1] = t
            self.current_ti = t1


    def _increase_hist_size(self, Nt=None):
        """
        With this I avoid np.concatenate every time I want to add new data to history. 
        """
        
        if Nt is None: 
            Nt = self._initial_capacity

        new_capacity = self.capacity + Nt

        # Create new, larger arrays
        new_hist = np.empty((new_capacity, self._hist.shape[1], self._hist.shape[2]))
        new_hist_t = np.empty((new_capacity,))

        # Copy existing data (expensive operation, but done rarely)
        new_hist[:self.capacity] = self._hist
        new_hist_t[:self.capacity] = self._hist_t

        # Update attributes
        self._hist = new_hist
        self._hist_t = new_hist_t


