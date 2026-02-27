from bias import Bias
from observations import Observations, plt
from model import Model

from typing import Dict, Optional, Type, List, Tuple, Union
from utils import mean_vector_to_ensemble, save_to_pickle_file, load_from_pickle_file, check_valid_file, correlation
from typeguard import typechecked

import numpy as np



class DataDrivenBias(Bias):
    """
    Abstract Base Class for data-driven bias estimation models.
    Child classes must implement the following methods:
        - train_forecaster() --> to train the data-driven bias model
        - initialize_forecaster() --> to initialize the data-driven bias model
        - state_derivative() --> to compute the state derivative of the data-driven bias model (for bias-aware DA)
    """

    biased_observations = True
    correlation_based_training = True   
    augment_data = True
    N_ens = 1

    @typechecked
    def __init__(self, 
                 rom: Model,  
                #  forecaster_class: Union[Model, Type[Model]],
                 reference_data: Union[Observations, List[Observations], None] = None,
                 **kwargs):
        

        kwargs_keys = list(kwargs.keys())
        for kwy in kwargs_keys:
            if hasattr(self, kwy) or kwy in self.extra_keys_to_print:
                setattr(self, kwy, kwargs.pop(kwy))


            

        # ------------------  Initialise parent Bias  ----------------------- #
        
        super().__init__( 
                        innovation=kwargs.pop('innovation', np.zeros((rom.Nq, 1))), 
                        t=kwargs.pop('t', rom.current_time),
                        dt=kwargs.pop('dt', rom.dt),
                        reference_data=reference_data,
                        rom=rom,
                        **kwargs)


    def init_forecaster(self, 
                         state: np.ndarray,
                         rom: Model, 
                         reference_data: Union[Observations, List[Observations], None],
                         training_data_filename: Optional[str] = None,
                         **kwargs): 
        """
        Initialize or load the data-driven bias model (forecaster).
        Inputs:
            rom: The forecast reduced order model instance for which estimate the bias
            reference_data: List of Observations instances or a single instance
            training_data_filename: Path to load/save the training data for the data-driven bias model
            kwargs: Additional parameters for training data generation and model training
        """
        forecaster_config = {**self.config, **kwargs}
        forecaster_config['N_dim'] = state.shape[1]
        forecaster_config['training_data_filename'] = training_data_filename

        # Set the default training time based on the ROM properties if not provided in kwargs
        forecaster_config['t_train'] = kwargs.get('t_train', rom.t_transient/2)
        forecaster_config['t_val'] = kwargs.get('t_val', rom.t_CR)
        forecaster_config['t_test'] = kwargs.get('t_test', 5*rom.t_CR if kwargs.get('perform_test', True) else 0)
        forecaster_config['dt'] = self.dt
        
        # Compute minimum training steps
        min_training_time = forecaster_config['t_train'] + forecaster_config['t_val'] + forecaster_config['t_test']
        self.minimum_training_steps = int(np.ceil(min_training_time / self.dt)) 


        # check if an ESN_bias model can be loaded from file using the provided filename or kwargs (e.g., hyperparameters used for training)
        forecaster_case = self.load_forecaster(**forecaster_config)

        if forecaster_case is None:
            if rom is None or reference_data is None:
                raise ValueError('Both rom and reference_data must be provided  to create a new DataDrivenBias model')
            
            # Create training data if not loaded from file and train new model
            train_data_dict = self.load_or_create_bias_training_dataset(rom=rom,
                                                                        reference_data=reference_data, 
                                                                        **forecaster_config)
            if train_data_dict is None:
                raise RuntimeError('Failed to load or create training data for bias model.')

            # Initialize the data-driven bias model with the training data
            forecaster_config.update(train_data_dict)
            forecaster_case = self.create_forecaster(**forecaster_config)
            
        assert forecaster_case.trained is True, f'{forecaster_case.name} model must be trained after initialization.'

        self._forecaster = forecaster_case


    # ============================ METHODS TO BE IMPLEMENTED BY CHILDREN CLASSES ======================== #

    def load_forecaster(self, **kwargs):
        raise NotImplementedError('Child of DataDrivenBias class must implement load_forecaster() method.')

    def create_forecaster(self, **kwargs):
        raise NotImplementedError('Child of DataDrivenBias class must implement create_forecaster() method.')

    def state_derivative(self):
        raise NotImplementedError('Child of DataDrivenBias class must implement state_derivative() method.')

    
    # =========================== DataDrivenBias PROPERTIES ======================== #    
    @property
    def augment_data_length(self):
        if self.augment_data:
            # how many sets of augmented data are created based on correlation between model outputs and observations)
            if self.augment_data > 1:
                return self.augment_data
            return 2 # default is to create 2 sets of data (best and worst correlation)
        else:
            return 1
    
    # ===================== METHODS TO LOAD OR CREATING THE TRAINING DATA BIAS MODEL ======================== #
    def load_or_create_bias_training_dataset(self, training_data_filename, **kwargs):
        # Try to load the training data from file
        train_data_dict = self._load_bias_training_dataset(training_data_filename)

        if train_data_dict is not None:
            print(f'Loaded training data for bias model from file: {training_data_filename}')
            return train_data_dict
        else:
            print(f'Creating training data for bias model...')
            assert 'rom' in kwargs and 'reference_data' in kwargs, 'rom and reference_data must be provided to create training data for bias model.'
            
            train_data_dict = self._create_bias_training_dataset(rom=kwargs.pop('rom'), 
                                                                reference_data=kwargs.pop('reference_data'), 
                                                                std_phi=kwargs.get('std_phi', None),
                                                                std_alpha=kwargs.get('std_alpha', None)
                                                                )
            if training_data_filename is not None:
                save_to_pickle_file(training_data_filename, train_data_dict)
            return train_data_dict


    def _load_bias_training_dataset(self, filename: str = None):
        if filename is not None:
            try:
                loaded_train_data = load_from_pickle_file(filename)
                try:
                    necessary_properties = self.config.copy()

                    if check_valid_file(loaded_train_data, 
                                        necessary_properties):
                        
                        _U = loaded_train_data['data']

                        if _U.shape[1] < self.minimum_training_steps:
                            print('Re-run multi-parameter training data: Increase the length of the training data')
                        elif self.augment_data and _U.shape[0] == self.L:
                            print('Re-run multi-parameter training data: need data augment ')
                        else:
                            print('OK: Loaded training dataset for bias model.')
                            return loaded_train_data # Only return if all checks passed
                except TypeError:
                    print(f'File {filename} type = {type(loaded_train_data)} is not dict')
            except FileNotFoundError:
                print(f'Run multi-parameter training data: file {filename} not found')
        # If loading fails or no filename provided, we reach this point and return None
        return None


    def _create_bias_training_dataset(
                                    self,
                                    rom: Type[Model],
                                    reference_data: Union[Type[Observations], List[Type[Observations]]], 
                                    # Additional parameters for training data generation
                                    std_phi: float = None,
                                    std_alpha: Union[float, Dict[str, Union[float,  List[float]]]] = None,
                                    ):

        """
        Multi-parameter data generation for ESN training.
        - If the observations are biased, the bias estimator must predict  
            (1) the innovations, i.e., the difference between the raw data and the model (observable)
            (2) the difference between the truth and the model, which is the actual model bias (non observable)
        - If there is data augmentation, the training data are augmented by 
            (a) scaling the innovations by different factors (only if not correlation_based_training)
            (b) correlating the model outputs with the observations to create different training sets
        - If multiple experimental datasets are provided, the training data from each dataset are concatenated.

        Inputs:
            rom: The forecast reduced order model instance for which estimate the bias
            std_phi: Standard deviation for sampling state variables
            std_alpha: Standard deviation or min-max range for sampling parameters
            reference_data: List of Observations instances or a single instance
        Returns:
            train_data: Dictionary containing training data and relevant parameters
            training_keys = ['upsample',
                            'L',
                            'augment_data',
                            'correlation_based_training',
                            'biased_observations']
            data shape: (N_datasets * L * augment_data_length, Nt_min, N_dim)
        """

        # ========================= Generate model states and prepare reference data ========================= #
        


        # Ensure L is set for ensemble generation
        if not hasattr(self, 'L'):
            self.L = rom.m
       
        y_model_L = _sample_model_states(rom=rom, L=self.L, 
                                         minimum_training_steps=self.minimum_training_steps,
                                         std_phi=std_phi, std_alpha=std_alpha) # Nt x Nq x L

        print('\n\n Preparing reference data for training...')
        print('y_model_L shape:', y_model_L.shape)

        y_raw, y_true = _prepare_reference_data(reference_data, minimum_training_steps=self.minimum_training_steps) # Lists of Nt x Nq x 1
        Nq = y_model_L.shape[1] # number of observed variables
        Nt_min = self.minimum_training_steps

        # ========================= Create training data ========================= #
        if not self.correlation_based_training:   # (Nóvoa & Magri 2023 CMAME)

            innovations_all, model_bias_all = [], []
            for yr, yt in zip(y_raw, y_true):
                innovations = (yr - y_model_L[-Nt_min:]).transpose((2, 0, 1))  # shape (L x Nt x Nq)
                innovations_all.append(innovations)
                
                if self.augment_data:
                    innovations_all.append(innovations * 1e-1)
                    innovations_all.append(innovations * -1e-2)

                if self.biased_observations:
                    model_bias = (yt - y_model_L[-Nt_min:]).transpose((2, 0, 1))  # shape (L x Nt x Nq)
                    model_bias_all.append(model_bias)
                    if self.augment_data:
                        model_bias_all.append(model_bias * 1e-1)
                        model_bias_all.append(model_bias * -1e-2)                

        else:  #  (Nóvoa et al. 2024 JFM) 
            # Here, we create augmented data sets based on correlation between model outputs and observations.
            # The augment_data_length determines how many sets are created (best, mid, worst correlations).
            
            innovations_all, model_bias_all = [], []
            for yr, yt in zip(y_raw, y_true):
                ym_L = _correlate_data(y_model_L, yr, self.augment_data_length, self.minimum_training_steps) # shape (Nt_min x Nq x L * augment_data_length)

                innovations = (yr - ym_L).transpose((2, 0, 1))  # shape (L x Nt x Nq)
                innovations_all.append(innovations)
                if self.biased_observations:
                    model_bias = (yt - ym_L).transpose((2, 0, 1))  # shape (L x Nt x Nq)
                    model_bias_all.append(model_bias)

        # Combine the innovations (and model biases) #

        if not self.biased_observations:
            train_data = np.concatenate(innovations_all, axis=0)
            print('train_data shape:', train_data.shape)
            print(innovations_all[0].shape, len(innovations_all ))

            observed_idx = np.arange(Nq)
        else:
            innovations_all = np.concatenate(innovations_all, axis=0)
            model_bias_all = np.concatenate(model_bias_all, axis=0) 
            train_data = np.concatenate([model_bias_all,
                                         innovations_all], axis=2)
            observed_idx = Nq + np.arange(Nq) # indices of innovations in the state vector            

        # =============================== Save train_data dict ================================ #
        # Save key keywords
        train_data_dict = {key:val for key, val in self.config.items()} 
        # add the extra kwargs used to create the training data
        # train_data_dict.update({key: val for key, val in kwargs.items() if key not in train_data_dict.keys()})
        # add training data and observed indices
        train_data_dict.update(data=train_data,
                               observed_idx=observed_idx,
                               )
        return train_data_dict        



def _prepare_reference_data(reference_data, minimum_training_steps) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare reference data for training data generation.
        Inputs:
            reference_data: List of Observations instances or a single instance
        Returns:
            y_raw: List of raw observation arrays (each of shape Nt x Nq x 1)
            y_true: List of clean observation arrays (each of shape Nt x Nq x 1)
        """
        if isinstance(reference_data, Observations):
            reference_data = [reference_data]

        y_raw, y_true = [], []
        for rfd in reference_data:
            # ensure ndim = 3
            yr, yt = rfd.y_raw.copy(), rfd.y_true.copy()
            if yr.ndim == 2:
                yr = yr[:, :, np.newaxis]
            if yt.ndim == 2:
                yt = yt[:, :, np.newaxis]

            y_raw.append(yr[-minimum_training_steps:])
            y_true.append(yt[-minimum_training_steps:])

        return y_raw, y_true



    # def plot_training_dataset(self, plot_data):
    #     _L, _Nt, ndim = plot_data.shape

    #     if self.biased_observations:
    #         ncol = 2
    #     else:
    #         ncol = 1

    #     ndim = int(round(ndim // ncol))
    #     nrow = int(min(ndim, 10))
    #     t_data = np.arange(0, _Nt) * self.dt
    #     times = [0, self.t_train, self.t_train + self.t_val, t_data[-1]]

    #     Lis = np.sort(np.random.choice(_L, size=min(5, _L), replace=False))

    #     _, axs_all = plt.subplots(nrows=nrow * len(Lis), ncols=ncol, 
    #                                 figsize=(8*ncol+2, 1.*nrow*len(Lis)), sharex=True, 
    #                                 sharey='row', layout='constrained')
        
    #     if not isinstance(axs_all, np.ndarray):
    #         axs_all = [axs_all]
        

    #     for row, Li in enumerate(Lis):
    #         axs = axs_all[nrow * row:nrow * (row + 1)]

    #         if self.biased_observations:
    #             axs = axs.T.flatten()
    #             if row == 0:
    #                 [axs[_ii].set(title=_ttl) for _ii, _ttl in zip([0, nrow], ['Model bias', 'Innovations'])]
            
            
    #         for kk, ax in enumerate(axs):
    #             if kk < nrow:
    #                 ax.plot(t_data, plot_data[Li, :, kk], lw=.8, color='k')
    #             else:
    #                 ax.plot(t_data, plot_data[Li, :, kk - nrow + ndim], lw=.8, color='k')


    #             [ax.axvspan(times[_ii], times[_ii+1], facecolor=_c, alpha=0.3, zorder=-100,
    #                         label=_lbl) for _ii, _c, _lbl in zip(range(3), ['orange', 'red', 'navy'],
    #                                                             ['Train', 'Validate', 'Test'])]
    #         axs[-1].legend(loc='upper left', bbox_to_anchor=(1.01, 1.05), frameon=False, title=f'Li={Li}/{_L}', fontsize='small')


    # #     plt.show()


    # def visualize_hist(self, b=None, t=None):
    #     if b is None:
    #         b = self.get_bias()
    #     if t is None:
    #         t = self.hist_t[-len(b):]

    #     lbl = [f'$b_{{{i}}}$' for i in range(self.Nq)]

    #     # Plot the time evolution of the observables
    #     t_zoom = int(self.t_CR / self.dt)

    #     fig = plt.figure(figsize=(8, self.Nq+1), layout="constrained")
    #     plt.suptitle('Observables time evolution')
    #     axs = fig.subplots(self.Nq, 2, sharey='row', sharex='col')
    #     if self.Nq == 1:
    #         axs = [axs]

    #     for ii, ax in enumerate(axs):
    #         ax[0].plot(t, b[:, ii])
    #         ax[1].plot(t[-t_zoom:], b[-t_zoom:, ii])
    #         ax[0].set(ylabel=lbl[ii])
    #         if ii == self.Nq-1:
    #             ax[0].set(xlabel='$t$', xlim=[t[0], t[-t_zoom]])
    #             ax[1].set(xlabel='$t$', xlim=[t[-t_zoom], t[-1]])



# 
# ==================== Module-level helper functions for training data generation ==================== #

def _correlate_data(y_L_model: np.ndarray, 
                    _y_raw: np.ndarray, 
                    augment_data_length: int, 
                    minimum_training_steps: int) -> np.ndarray:
    """
    Create training data based on correlation between model outputs and observations.
    
    Inputs:
        y_L_model: Model output history (Nt x Nq x L)
        _y_raw: Raw observation history (Nt x Nq x 1)
        augment_data_length: Number of augmented datasets (1, 2, or 3)
        minimum_training_steps: Minimum number of training steps required
    
    Returns:
        train_data_model: Correlated model data (Nt_min x Nq x L * augment_data_length)
    """
    len_augment_set = augment_data_length
    Nt_min = minimum_training_steps

    Nt, Nq, L = y_L_model.shape
    N_corr = Nt - Nt_min

    train_data_model = np.zeros([Nt_min, Nq, L * len_augment_set])

    lags = np.linspace(start=0, stop=N_corr, num=N_corr, dtype=int)

    y_raw_c = _y_raw[:N_corr, ..., 0] - np.mean(_y_raw[:N_corr, ..., 0], axis=0, keepdims=True)
    y_model_c = y_L_model[:2*N_corr, :, :] - np.mean(y_L_model[:2*N_corr, :, :], axis=0, keepdims=True)
    
    # normalize
    epsilon = 1e-8
    y_raw_c /= (np.max(np.abs(y_raw_c), axis=0, keepdims=True) + epsilon)
    y_model_c /= (np.max(np.abs(y_model_c), axis=0, keepdims=True) + epsilon)

    shifted_y_model_list = [y_L_model[lag : N_corr + lag] for lag in range(N_corr)]
    
    correlations = np.array([correlation(y_raw_c, yy) for yy in shifted_y_model_list])

    # for each lag, find the best, worst, and mid correlation indices and store the corresponding data
    for ii in range(L):
        _corrs = correlations[:, ii]
        best_lag = lags[np.argmax(_corrs)]

        base_col = len_augment_set * ii
        train_data_model[:, :, base_col] = y_L_model[best_lag:best_lag + Nt_min, :, ii]

        if len_augment_set >= 2:
            worst_lag = lags[np.argmin(_corrs)]
            mid_lag = int(np.mean([best_lag, worst_lag]))
            train_data_model[:, :, base_col + 1] = y_L_model[mid_lag:mid_lag + Nt_min, :, ii]
            if len_augment_set >= 3:
                train_data_model[:, :, base_col + 2] = y_L_model[worst_lag:worst_lag + Nt_min, :, ii]

    return train_data_model


def _sample_model_states(rom, 
                         L: int,
                         minimum_training_steps: int,
                         std_phi: float = None,
                         std_alpha: Union[float, Dict[str, Union[float, List[float]]]] = None) -> np.ndarray:
    """
    Sample model states from the forecast model for training data generation.
    
    Inputs:
        rom: Forecast model instance
        L: Ensemble size
        minimum_training_steps: Minimum number of training steps
        std_phi: Standard deviation for sampling state variables
        std_alpha: Standard deviation or min-max range for sampling parameters
    
    Returns:
        Observable history array (Nt_min+2*N_CR x Nq x L)
    """
    model = rom.copy()
    
    if std_phi is None:
        std_phi = np.std(model.current_state[:model.Nphi, :], axis=-1)
    if std_alpha is None:
        std_alpha = {}
        for i, key in enumerate(model.est_alpha):
            param = model.current_state[model.Nphi + i, :]
            std_alpha[key] = [min(param), max(param)]

    def sample_ensemble(_psi0, _L):
        new_phi = mean_vector_to_ensemble(rng=model.rng, 
                                          mean_vec=_psi0[:model.Nphi], 
                                          std=std_phi, 
                                          m=_L,
                                          method='uniform')
        
        new_alpha = mean_vector_to_ensemble(rng=model.rng, 
                                            mean_vec=_psi0[model.Nphi:model.Nphi+model.Na], 
                                            std=std_alpha, 
                                            m=_L,
                                            method='uniform')
        
        return np.concatenate([new_phi, new_alpha], axis=0)

    if model.m != L:
        psi0 = np.mean(model.current_state.copy(), axis=-1)
        psi0_ens = sample_ensemble(psi0, L)
        model.update_history(psi=psi0_ens[np.newaxis, :, :], t=0., reset=True)

    # Forecast to post-transient
    Nt = int(np.round(model.t_transient / model.dt, model.precision_t)) - 1
    psi, t = model.time_integrate(Nt=Nt)
    model.update_history(psi=psi, t=t, reset=True)

    y_L_model = model.get_observable_hist()
    psi = psi[-1, :, :]

    # Remove and replace fixed points
    tol = 1e-1
    N_CR = int(round(model.t_CR / model.dt))
    range_y = np.max(np.max(y_L_model[-N_CR:], axis=0) - np.min(y_L_model[-N_CR:], axis=0), axis=0)
    idx_FP = (range_y < tol)

    if len(np.flatnonzero(idx_FP)) / len(idx_FP) >= 0.2:
        idx_FP[np.flatnonzero(idx_FP)[0]] = 0
        psi0 = psi[:, ~idx_FP]
        new_psi0 = sample_ensemble(np.mean(psi0, axis=-1), len(np.flatnonzero(idx_FP)))
        psi0 = np.concatenate([psi0, new_psi0], axis=-1)
    else:
        psi0 = psi

    # Forecast fixed-point-free post-transient ensemble
    model.update_history(psi=psi0[np.newaxis, :, :], reset=True)
    psi, t = model.time_integrate(Nt=minimum_training_steps + N_CR)
    model.update_history(psi=psi, t=t, reset=True)
    model.close()

    return model.get_observable_hist()
