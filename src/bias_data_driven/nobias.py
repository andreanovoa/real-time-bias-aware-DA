from base import *
from integrator import ConstantIntegrator
from history import HistoryTracker
from types import SimpleNamespace



class NoBias(Bias):

    def __init__(self, y, t, dt, **kwargs):
        super().__init__(innovation=np.zeros(y.shape), t=t, dt=dt, **kwargs)
        self.N_dim = self.hist.shape[1]
        self.observed_idx = np.arange(self.N_dim)

    def state_derivative(self):
        return np.zeros([self.N_dim, self.N_dim])


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
    nb = NoBias(y=np.zeros((10,)), t=0.0, dt=0.1)
    print(nb)

    state, t = nb.time_integrate(Nt=1000)
    nb.update_history(state, t)
    
    print(nb.history.hist.shape)