
from .base_dd import *
from models_data_driven import ESN_model
from config.esn_config import ESNConfig, find_matching_config, load_esn_model_from_config, save_esn_model_to_config, save_esn_model_to_config


class ESN_bias(DataDrivenBias):

    def __init__(self,
                 rom: Model,  
                 reference_data = None,
                 **kwargs):

        # ---------------  Initialize ESN_model and Bias via DataDrivenBias ------------------- #

        super().__init__(rom=rom, 
                         reference_data=reference_data, 
                         forecaster_class=ESN_model, 
                         **kwargs)


         # ----------------- Initialize reservoir state and reset Bias history ---------------------- #
        state0 = self.forecaster.initialize_from_val_data(N_ens=self.N_ens) # this method belongd to ESN_model

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
    

    def load_forecaster(self, hash=None, data=None, **kwargs):
        """Load ESN_model forecaster from disk using hash or kwargs if hash is None.  
        Arguments:
            hash: Optional[str] - Hash corresponding to the ESN_model forecaster configuration to load
            data: Optional[np.ndarray] - Data to use for creating the ESN_model forecaster if hash is not provided. 
            kwargs: Additional keyword arguments that can be used to create the ESN_model forecaster configuration if hash is not provided. 
                This can include specific hyperparameters for the ESN_model or parameters used to create a hash based on the data characteristics.
        Returns:
            ESN_model instance if loading is successful, None otherwise.

        """

        if hash is None:
            initial_params = kwargs.copy()
            initial_params['data'] = data
            # Create ESNConfig from kwargs and compute hash to find matching saved model configuration
            # ensute only the relevant kwargs are used to create the ESNConfig and hash
            # for k in list(initial_params.keys()):
            #     if k not in ESNConfig.__dict__:
            #         initial_params.pop(k)
            # print(f'Attempting to load ESN_model with configuration: {initial_params}')
            query_config = ESNConfig.from_init_params(**initial_params)
            query_hash = query_config.to_hash()

            # print(query_config)
        else:            
            query_hash = hash

        return load_esn_model_from_config(q=query_hash)


    def create_forecaster(self, data, **kwargs):

        # Train new model
        esn_model = ESN_model(data=data, **kwargs)


        # query_config = ESNConfig.from_esn_model(esn_model)
        
        # print(f'Created ESNConfig: {query_config}')

        # Save model configuration to disk and return hash
        save_esn_model_to_config(esn_model)

        return esn_model


    def save_forecaster(self, forecaster):
        """Save ESN_model forecaster to disk and return the hash corresponding to the saved model configuration. 
        Arguments:
            forecaster: ESN_model instance to save
            config_dir: Optional[str] - Directory to save the ESN_model configuration file. If None, uses default directory.
            kwargs: Additional keyword arguments that can be used to create a hash based on the model configuration or training data characteristics.
        Returns:
            str: Hash corresponding to the saved ESN_model configuration.
        """

        # Save ESNConfig to disk and return hash
        return save_esn_model_to_config(forecaster)





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
                    rom=ensemble.model, 
                    reference_data=truth,
                    L=12,
                    std_phi=0.1,        # Initial uncertainty in the state
                    std_alpha=alpha0,       # Initial uncertainty in the parameters
                    )
    print(esnb)

    state, t = esnb.time_integrate(Nt=1000)
    esnb.update_history(state, t)
    
    print(esnb.history.hist.shape)