
from .base import *
from models_data_driven import ESN_model



class ESN_bias(DataDrivenBias):

    def __init__(self,
                 rom: Type[Model],  
                 reference_data: Union[Type[Observations], List[Type[Observations]]] = None,
                 filename: str = None,
                 **kwargs):

        # ---------------  Initialize ESN_model and Bias via DataDrivenBias ------------------- #

        super().__init__(rom=rom, reference_data=reference_data, 
                         forecaster_class= ESN_model, filename=filename, 
                         
                         **kwargs)


         # ----------------- Initialize reservoir state and reset Bias history ---------------------- #
        state0 = self.forecaster.initialize_from_val_data(N_ens=self.N_ens)

        self.forecaster.reservoir_state = state0[self.N_dim:self.N_dim+self.N_units, :]
        self.forecaster.update_history(state0, reset=True)
    
    @property
    def N_dim(self):
        return self._forecaster.N_dim
    
    @property
    def N_units(self):
        return self._forecaster.N_units

    def state_derivative(self):
        esn = self.forecaster
        
        r_mean = np.mean(esn.reservoir_state, axis=-1, keepdims=True) 
        u_mean = esn.reservoir_to_physical(r_mean)
        esn_J = esn.Jacobian(open_loop_J=True, state=(u_mean, r_mean))  # Compute ESN Jacobian

        

        db_din = esn_J[np.array(self.bias_idx), np.array([self.bias_idx]).T]
        return -db_din
    


if __name__ == '__main__':

    from observations import Observations
    from models_physical import VdP
    import numpy as np
    rng = np.random.default_rng(0)



    # The manual bias is a function of state and/or time
    def manual_bias(y, t):
        # Linear function of the state
        # return .2 * y + .3 * np.max(y, axis=0), 'linear'
        # Periodic function of the state
        return 0.5 * np.max(y, axis=0) * np.cos(2 * y / np.max(y, axis=0)), 'periodic'
        # Time-varying bias
        # return .4 * y * np.sin((np.expand_dims(t, -1) * np.pi * 2) ** 2), 'time'



    truth = Observations(model=VdP(), 
                        t_start=10*VdP.t_CR,
                        t_stop=15*VdP.t_CR,
                        t_max=2*VdP.t_transient,
                        Nt_obs=30,
                        add_noise=True,
                        noise_type='gauss, add',
                        noise_level=0.02,
                        manual_bias=manual_bias
                        )

    from ensemble import Ensemble
    from data_assimilation import EnSRKF, EnKF, rBA_EnKF



    alpha0 = dict(zeta=(40, 50.),
                beta=(50, 60),
                kappa=(3, 4),)

    ensemble = Ensemble(# Data assimilation parameters
                        da_method=rBA_EnKF, 
                        regularization_factor=0.,
                        inflation_factor=1.0,
                        # Model parameters
                        parent_model=VdP,      
                        m=10,               # Number of ensemble members
                        std_phi=0.1,        # Initial uncertainty in the state
                        std_alpha=alpha0,       # Initial uncertainty in the parameters
                        )

    esnb = ESN_bias(t=0.0, 
                    rom=ensemble.model, reference_data=truth,
                    L=12,
                    std_phi=0.1,        # Initial uncertainty in the state
                    std_alpha=alpha0,       # Initial uncertainty in the parameters
                    )
    print(esnb)

    state, t = esnb.time_integrate(Nt=1000)
    esnb.update_history(state, t)
    
    print(esnb.history.hist.shape)