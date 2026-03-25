import numpy as np

from bias import Bias
from integrator import ConstantIntegrator
from history import HistoryTracker
from types import SimpleNamespace


# TODO: This is currently just a placeholder for the constant bias model, 
# which is a special case of the DriftLinearBias with zero drift and zero linear coupling. 

# TODO: k shiould be a parameter which can be estimated within the data assimilation loop. So, 
# we need the option to have an ensemble of k values, which can be updated during the data assimilation loop. 
# This would allow us to capture uncertainty in the constant bias estimate and potentially improve the performance 
# of the bias correction.

class ConstantBias(Bias):

    upsample = 20 # High upsample in time as constant forecast steps do not affect computation

    def __init__(self, innovation, t, dt, k=None, **kwargs):
        
        raise NotImplementedError('This classneeds debugging.')
        if k is not None:
            innovation = np.ones(innovation.shape) * k  # Constant innovation

        super().__init__(innovation=innovation, t=t, dt=dt, **kwargs)
    

    def state_derivative(self):
        n_bias = self.N_dim // 2 if self.biased_observations else self.N_dim
        return np.zeros([n_bias, n_bias])


    def init_forecaster(self, **kwargs):
        """
        Default constant forecaster, with no model.
        
        """
        self._forecaster = SimpleNamespace()  # Create empty forecaster object

        if 'initial_capacity' in kwargs.keys():
            initial_capacity = kwargs.pop('initial_capacity')
        else:
            initial_capacity = max(1000, state.shape[0]*10)
        self._forecaster.history = HistoryTracker(initial_capacity=initial_capacity)


        #  INITIALISE INTEGRATOR STRATEGY ================== ##
        self._forecaster.integrator = ConstantIntegrator(self)



if __name__ == '__main__':
    nb = ConstantBias(innovation=np.ones((2,7)), t=0.0, dt=0.1)

    state, t = nb.time_integrate(Nt=10)
    nb.update_history(state, t)
    

    reset_constant = np.array([np.random.rand(nb.N_dim)]).T * np.ones_like(state[-1])
    nb.update_history(reset_constant, update_last_state=True)

    state, t = nb.time_integrate(Nt=10)
    nb.update_history(state, t)
    

    print(nb.history.hist.shape, nb.get_bias_hist())