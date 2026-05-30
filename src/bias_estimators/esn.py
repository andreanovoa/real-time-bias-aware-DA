
import numpy as np
from typing import Optional

from .bias import Bias
from models import Model
from observations import Observations
from models.data_driven import ESN_model
from config.esn_config import ESNConfig, load_esn_model_from_config, save_esn_model_to_config

from utils import save_to_pickle_file
from .aux import create_bias_training_dataset, load_bias_training_dataset


class ESN_bias(Bias):

    forecaster_type = ESN_model
    biased_observations = True
    correlation_based_training = True
    augment_data = True

    def __init__(self,
                 rom: Model,  
                 reference_data = None,
                 **kwargs):

        
        super().__init__(
                        innovation=kwargs.pop('innovation', np.zeros((rom.Nq,))),
                        dt=kwargs.pop('dt', rom.dt),
                        t=kwargs.pop('t', rom.t),
                        rom=rom,
                        reference_data=reference_data,
                        **kwargs)

        assert isinstance(self.forecaster, ESN_model), "forecaster must be an instance of ESN_model"


    @property
    def initialize_bias_state(self):
        """
        Initializes the reservoir state of the ESN_model forecaster using the validation data. This method is called during the initialization of the bias model to set the initial state of the ESN_model forecaster based on the validation data, which can help improve the training and performance of the bias model.
        Arguments:
            N_ens: int - Number of ensemble members to initialize in the reservoir state.

        Returns:
            Initialized reservoir state for the ESN_model forecaster.
        """
        assert hasattr(self, 'forecaster'), "Forecaster must be initialized before calling initialize_from_val_data."
        
        return self.forecaster.initialize_from_val_data(N_ens=self.N_ens) # this method belongd to ESN_model
    
        
    def washout_phase(self, d_wash, t_wash, **kwargs):
        """
        Arguments:
            d_wash: np.ndarray - Washout data to use for initializing the bias model.
            t_wash: np.ndarray - Time points corresponding to the washout data.

        Returns:
            None
        """
        assert hasattr(self, 'forecaster'), "Forecaster must be initialized before calling washout_phase."
        

        esn = self.forecaster #type: ESN_model

        # Make sure first dimension is time
        assert d_wash.ndim == 2, f"Washout data must be a 2D array with shape (Nt, Nq) or (Nq, Nt), got {d_wash.shape}."
        if d_wash.shape[0] != len(t_wash):
            u_wash = d_wash.copy().T
        else:
            u_wash = d_wash.copy()
        
        # apply upsample and cut if needed to match the washout time points
        u_wash, t_wash = [xx[::esn.upsample][:esn.N_wash+1] for xx in [u_wash, t_wash]]


        # store washout data for potential future plotting
        self.washout_data = (u_wash, t_wash)


        # Get current reservoir state and corresponding physical state from the ESN_model forecaster
        Nt = len(t_wash) + 1
        r_open = esn.reservoir_state
        u_out, r_out = np.empty((Nt, self.N_dim, r_open.shape[1])), np.empty((Nt, *r_open.shape))

        r_out[0]= esn.reservoir_state
        u_out[0] = esn.reservoir_to_physical(r_out[0])
        
        # Open-loop reservoir
        for kk in range(len(t_wash)):
            u_open, r_open = esn.step(u_wash[kk], r_out[kk])
            u_out[kk+1], r_out[kk+1] = u_open, r_open

        #store final state into the initialization arrays

        psi = esn.build_psi(u=u_out, r=r_out)
        return psi[1:], t_wash


    
    @property
    def N_hidden(self):
        if not hasattr(self, '_forecaster'):
            return 0
        return self._forecaster.N_units


    def state_derivative(self):
        esn = self.forecaster #type: ESN_model
        state = self.current_state
        u, r = state[:self.N_dim], state[-self.N_hidden:]
        r_mean = np.mean(r, axis=-1, keepdims=True) 
        u_mean = np.mean(u, axis=-1, keepdims=True)

        esn_J = esn.Jacobian(open_loop_J=True, u_in=u_mean, r_in=r_mean)  # Compute ESN Jacobian

        return -esn_J[np.array(self.bias_idx), np.array([self.bias_idx]).T]


    def init_forecaster(self,
                        training_data_filename: Optional[str] = None,
                        **kwargs):
        
        cfg = self.config.copy()
        cfg.update(kwargs)

        cfg['N_dim'] = self.N_dim
        cfg['training_data_filename'] = training_data_filename
        cfg['dt'] = self.dt
        rom = kwargs.get('rom') 

        assert rom is not None, "ROM object must be provided."

        # add traingin times if not provided in kwargs, with default values based on the ROM time scales
        t_test_default = 5 * rom.t_CR if kwargs.get('perform_test', True) else 0
        for key, default_value in zip(['t_train', 't_val', 't_test'], [rom.t_transient / 2, rom.t_CR, t_test_default]):
            if key not in cfg.keys(): 
                cfg[key] = kwargs.get(key, default_value)
        
        cfg['rom'] = rom

        min_training_time = sum([cfg[key] for key in ['t_train', 't_val', 't_test']])
        self.minimum_training_steps = int(np.ceil(min_training_time / self.dt))

        if not hasattr(self, 'L'):
            self.L = rom.m

        # Load or create training dataset for bias model
        self.forecaster = self.load_or_create_forecaster(**cfg) #type: ESN_model


    def load_or_create_forecaster(self, hash=None, reference_data=None, rom=None, **kwargs) -> ESN_model:
        """Load ESN_model forecaster from disk using hash or kwargs if hash is None.  
        Arguments:
            hash: Optional[str] - Hash corresponding to the ESN_model forecaster configuration to load
            reference_data: Optional[np.ndarray] - Reference data to use for creating the ESN_model forecaster if hash is not provided. 
            rom: Optional[Model] - ROM object to use for creating the ESN_model forecaster if hash is not provided.
            kwargs: Additional keyword arguments that can be used to create the ESN_model forecaster configuration if hash is not provided. 
                This can include specific hyperparameters for the ESN_model or parameters used to create a hash based on the data characteristics.
        Returns:
            ESN_model instance if loading is successful, None otherwise.

        """

        cfg = kwargs.copy()

        if hash is None:
            query_config = ESNConfig.from_init_params(**kwargs)
            query_hash = query_config.to_hash()
        else:            
            query_hash = hash

        # Try to load forecaster configuration from disk
        loaded_case = load_esn_model_from_config(q=query_hash)
    

        if loaded_case is not None and not kwargs.get('force_retrain', False):
            assert isinstance(loaded_case, ESN_model), f'Loaded case must be an instance of ESN_model, but got {type(loaded_case)}.'
            assert loaded_case.trained is True, f'{loaded_case.name} model must be trained after initialization.'
            return loaded_case

        elif rom is None or reference_data is None:
            raise ValueError('Both rom and reference_data must be provided to create a new ESN_bias model')
        
        else:
            # Create new forecaster. First, create training dataset for bias model, then use it to create the forecaster.
            train_data_dict = self.load_or_create_bias_training_dataset(training_data_filename=kwargs.get('training_data_filename'),
                                                                        rom=rom,
                                                                        reference_data=reference_data,
                                                                        std_alpha=cfg.get('std_alpha'),
                                                                        std_phi=cfg.get('std_phi'))
            if train_data_dict is None:
                raise RuntimeError('Failed to load or create training data for bias model.')

            # Create a new instance of the ESN_model to use as forecaster
            cfg.update(train_data_dict)

            # print('CONFIGURATION')
            # print('data ', cfg['data'].shape)
            # print('state ', cfg['state'].shape)
            # print('------------------')

            new_esn_model = ESN_model(**cfg) # Note: ESN_model trains itself during initialization using the provided training data, so we don't need a separate training step here. If the ESN_model implementation changes in the future to require a separate training step, this code will need to be updated accordingly.

            # Save model configuration to disk and return it
            _ = save_esn_model_to_config(new_esn_model)

            return new_esn_model



    def load_or_create_bias_training_dataset(self, 
                                             training_data_filename=None, 
                                             rom=None, 
                                             reference_data=None, 
                                             std_phi=None, 
                                             std_alpha=None):
        """
        Load training dataset for bias model from disk if available, otherwise create a new training dataset using the provided 
        rom and reference_data, and save it to disk if a filename is provided.
        
        These functions are defined in aux.py and handle the creation of the training dataset based on the ROM and reference data, 
        including any necessary preprocessing, augmentation, and formatting for training the bias model.
        
        Arguments:
            Mandatory if loading dataset:
                - training_data_filename: Optional[str] - Filename to load/save the training dataset for the bias model.
            Mandatory if creating training dataset:
                - rom: Optional[ROM] - ROM to use for creating the training dataset if it needs to be created.
                - reference_data: Optional[np.ndarray] - Reference data to use for creating the training dataset if it needs to be created.
            Options for creating training dataset:
                - std_phi: Optional[float] - Standard deviation of phi for creating the training dataset if it needs to be created.
                - std_alpha: Optional[float] - Standard deviation of alpha for creating the training dataset if it needs to be created.
        """

        train_data_dict = load_bias_training_dataset(filename=training_data_filename,
                                                    necessary_properties=self.config.copy(),
                                                    minimum_training_steps=self.minimum_training_steps,
                                                    augment_data_length=self.augment_data_length,
                                                    L=self.L)


        if train_data_dict is not None:
            assert isinstance(train_data_dict, dict), 'ERROR: Loaded training data for bias model must be a dictionary.'
            return train_data_dict 

        print('Creating training data for bias model...')
        
        assert rom is not None and reference_data is not None, 'ROM and reference data must be provided to create bias training dataset.'


        
        train_data_dict = create_bias_training_dataset(
                                config=self.config,
                                rom=rom,
                                reference_data=reference_data,
                                std_phi=std_phi,
                                std_alpha=std_alpha,
                                L=self.L,
                                augment_data_length=self.augment_data_length,
                                minimum_training_steps=self.minimum_training_steps,
                                correlation_based_training=self.correlation_based_training,
                                biased_observations=self.biased_observations,
                            )

        if training_data_filename is not None:
            save_to_pickle_file(training_data_filename, train_data_dict)

        return train_data_dict







if __name__ == '__main__':

    from observations import Observations
    from models.physical import VdP
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
                    dt=0.1,
                    innovation=np.zeros((1, ensemble.model.Nq)),
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