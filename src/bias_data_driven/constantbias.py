from .base_dd import *
from integrator import ConstantIntegrator
from history import HistoryTracker
from types import SimpleNamespace



class ConstantBias(Bias):
    biased_observations = True  # Whether observations are biased or not
    def __init__(self, innovation, t, dt, k=None, **kwargs):
        
        if k is not None:
            innovation = np.ones(innovation.shape) * k  # Constant innovation

        super().__init__(innovation=innovation, t=t, dt=dt, **kwargs)
        
    def state_derivative(self):
        return np.zeros([self.N_bias, self.N_bias])


    def _init_forecaster(self, state, **kwargs):
        """
        default _init_forecaster, with no model
        
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