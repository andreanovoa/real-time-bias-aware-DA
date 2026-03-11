import numpy as np

from typing import Dict, List, Optional, Tuple, Union

from observations import Observations
from model import Model
from utils import mean_vector_to_ensemble, correlation, check_valid_file, load_from_pickle_file


from typeguard import typechecked

@typechecked
def prepare_reference_data(reference_data: Union[Observations, List[Observations]],
                           minimum_training_steps: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Prepare reference data for training the bias model. This function extracts the raw observations and true values from 
    the provided reference data, ensuring that they have the correct shape for training.
    Parameters:
    -----------
    reference_data : list of Observations or single Observations
        The reference data to prepare. Can be a single Observations object or a list of Observations objects.
    minimum_training_steps : int
        The minimum number of training steps required. This is used to ensure that the prepared data has enough 
        time steps for training the bias model.

    Returns: Tuple[List[np.ndarray], List[np.ndarray]]
        - y_raw: A list of numpy arrays containing the raw observations from the reference data, each array has 
                shape (minimum_training_steps, Nq, 1) where Nq is the number of observed variables.
        - y_true: A list of numpy arrays containing the true values from the reference data, each array has 
                shape (minimum_training_steps, Nq, 1) where
    """


    if isinstance(reference_data, Observations):
        reference_data = [reference_data]

    y_raw, y_true = [], []
    for rfd in reference_data:
        yr, yt = rfd.y_raw.copy(), rfd.y_true.copy()
        if yr.ndim == 2:
            yr = yr[:, :, np.newaxis]
        if yt.ndim == 2:
            yt = yt[:, :, np.newaxis]

        y_raw.append(yr[-minimum_training_steps:])
        y_true.append(yt[-minimum_training_steps:])

    return y_raw, y_true


@typechecked
def correlate_data(y_L_model: np.ndarray,
                    y_raw: np.ndarray,
                    augment_len: int,
                    minimum_training_steps: int,
                ) -> np.ndarray:
    
    """
    Function that correlates the model-generated data with the raw observations to create a training dataset for the bias model.
    The function 
    1. computes the correlation between the model-generated data and the raw observations, 
    2. identifies the best lag for each observed variable, and 
    3. constructs a training dataset based on the model data at these optimal lags. 
    If data augmentation is specified, it also includes additional samples based on the worst lag and a mid-point lag. 

    Parameters:
        y_L_model : np.ndarray
            The model-generated data, with shape (Nt, Nq, L) 
        y_raw : np.ndarray
            The raw observations, with shape (Nt, Nq, 1)
        augment_len : int
            The number of augmented samples to create for each observed variable. 
            If augment_len=1, only the best lag is used. If augment_len=2, both the best lag and a mid-point lag are used.
            If augment_len=3, the worst lag is also included.
        minimum_training_steps : int
            The minimum number of training steps required. 
    """
    
    Nt_min = minimum_training_steps
    Nt, Nq, L = y_L_model.shape
    N_corr = Nt - Nt_min

    train_data_model = np.zeros([Nt_min, Nq, L * augment_len])
    lags = np.linspace(start=0, stop=N_corr, num=N_corr, dtype=int)

    # Use only the first Ncorr time steps for computation time and to ensure we have enough data for training after lagging
    y_raw_c = y_raw[:N_corr, ..., 0] - np.mean(y_raw[:N_corr, ..., 0], axis=0, keepdims=True)
    epsilon = 1e-8
    y_raw_c /= (np.max(np.abs(y_raw_c), axis=0, keepdims=True) + epsilon)

    shifted_y_model_list = [y_L_model[lag: N_corr + lag] for lag in range(N_corr)]
    correlations = np.array([correlation(y_raw_c, yy) for yy in shifted_y_model_list])

    for ii in range(L):
        corr_values = correlations[:, ii]
        best_lag = lags[np.argmax(corr_values)]

        base_col = augment_len * ii
        train_data_model[:, :, base_col] = y_L_model[best_lag:best_lag + Nt_min, :, ii]

        if augment_len >= 2:
            worst_lag = lags[np.argmin(corr_values)]
            mid_lag = int(np.mean([best_lag, worst_lag]))
            train_data_model[:, :, base_col + 1] = y_L_model[mid_lag:mid_lag + Nt_min, :, ii]
            if augment_len >= 3:
                train_data_model[:, :, base_col + 2] = y_L_model[worst_lag:worst_lag + Nt_min, :, ii]

    return train_data_model


@typechecked
def sample_model_states(rom: Model,
                        L: int,
                        minimum_training_steps: int,
                        std_phi: Optional[Union[float, np.ndarray]] = None,
                        std_alpha: Optional[Union[float, Dict[str, Union[float, List[float]]]]] = None,
                    ) -> np.ndarray:
    """
    
    Sample model states from the ROM to create a training dataset for the bias model. This function
    1. initializes the ROM with an ensemble of states (with smapled ICs and parameters if specified), 
    2. integrates the ROM forward in time to generate model data, and 
    3. processes this data to create a training dataset for the bias model. 
    
    Parameters:
        rom : Model
            The reduced-order model (ROM) to sample states from. 
        L : int
            The number of samples to generate. 
        minimum_training_steps : int
            The minimum number of time steps to integrate the ROM for to generate enough data for training the bias model.
        std_phi : float, optional
            The standard deviation to use when sampling the initial conditions (ICs) for the ensemble.
            - None -> the standard deviation of the current state of the ROM will be used.
        std_alpha : float or dict, optional
            The standard deviation to use when sampling the parameters for the ensemble. 
            - float -> it will be used for all parameters
            - dict -> it should have keys corresponding to the parameter names and values specifying the standard deviation for each parameter.

    """

    model = rom.copy()

    if std_phi is None:
        std_phi = np.std(model.current_state[:model.Nphi, :], axis=-1)
        
    if std_alpha is None:
        std_alpha = {}
        for i, key in enumerate(model.est_alpha):
            param = model.current_state[model.Nphi + i, :]
            std_alpha[key] = [min(param), max(param)]

    assert std_phi is not None, "std_phi must be specified or computed from the model state."
    assert std_alpha is not None, "std_alpha must be specified or computed from the model state."

    def sample_ensemble(psi0_mean, ensemble_size):
        new_phi = mean_vector_to_ensemble(
            rng=model.rng,
            mean_vec=psi0_mean[:model.Nphi],
            std=std_phi, 
            m=ensemble_size,
            method='uniform',
        )
        if psi0_mean.shape[0] > model.Nphi:

            new_alpha = mean_vector_to_ensemble(
                rng=model.rng,
                mean_vec=psi0_mean[model.Nphi:model.Nphi + model.Na],
                std=std_alpha,
                m=ensemble_size,
                method='uniform',
            )

            return np.concatenate([new_phi, new_alpha], axis=0)
        else:
            return new_phi
        
    if model.m != L:
        psi0 = np.mean(model.current_state.copy(), axis=-1)
        psi0_ens = sample_ensemble(psi0, L)
        model.update_history(psi=psi0_ens[np.newaxis, :, :], t=0.0, reset=True)

    Nt = int(np.round(model.t_transient / model.dt, model.precision_t)) - 1
    psi, t = model.time_integrate(Nt=Nt)
    model.update_history(psi=psi, t=t, reset=True)

    y_L_model = model.get_observable_hist()
    psi_last = psi[-1, :, :]

    tol = 1e-1
    N_CR = int(round(model.t_CR / model.dt))
    range_y = np.max(np.max(y_L_model[-N_CR:], axis=0) - np.min(y_L_model[-N_CR:], axis=0), axis=0)
    idx_fixed = range_y < tol

    if len(np.flatnonzero(idx_fixed)) / len(idx_fixed) >= 0.2:
        idx_fixed[np.flatnonzero(idx_fixed)[0]] = 0
        psi0 = psi_last[:, ~idx_fixed]
        new_psi0 = sample_ensemble(np.mean(psi0, axis=-1), len(np.flatnonzero(idx_fixed)))
        psi0 = np.concatenate([psi0, new_psi0], axis=-1)
    else:
        psi0 = psi_last

    model.update_history(psi=psi0[np.newaxis, :, :], reset=True)
    psi, t = model.time_integrate(Nt=minimum_training_steps + N_CR)
    model.update_history(psi=psi, t=t, reset=True)
    model.close()

    return model.get_observable_hist()


def load_bias_training_dataset(
    filename,
    necessary_properties: dict,
    minimum_training_steps: int,
    augment_data_length: int,
    L: int,
):
    

    if filename is None:
        return None

    try: #Check if file exists... 
        loaded_train_data = load_from_pickle_file(filename)
    except FileNotFoundError or AssertionError:
        print(f'Run multi-parameter training data: file {filename} not  found or does not contain a dictionary')
        return None
    
    #Check if loaded file is valid... 
    if not isinstance(loaded_train_data, dict):
        print(f'Loaded file is invalid: {type(loaded_train_data)}: {loaded_train_data}')
        return None
    elif not check_valid_file(loaded_train_data, necessary_properties):
        return None
    else:
        # Check if the data has enough time steps and matches the expected shape based on augment_data_length and L
        data = loaded_train_data['data']
        if data.shape[1] < minimum_training_steps:
            print('Re-run multi-parameter training data: Increase the length of the training data')
            return None
        if augment_data_length > 1 and data.shape[0] != L * augment_data_length:
            print('Re-run multi-parameter training data: augment_data_length does not match the number of samples in the loaded training data')
            return None

        print('OK: Loaded training dataset for bias model.')
        return loaded_train_data



def create_bias_training_dataset(config: dict,
                                rom: Model,
                                reference_data,
                                minimum_training_steps: int,
                                L: int,
                                augment_data_length: int,
                                correlation_based_training: bool,
                                biased_observations: bool,
                                std_phi: Optional[float] = None,
                                std_alpha: Optional[Union[float, Dict[str, Union[float, List[float]]]]] = None,
                            ) -> dict:
    
    y_model_L = sample_model_states(rom=rom,
                                    L=L,
                                    minimum_training_steps=minimum_training_steps,
                                    std_phi=std_phi,
                                    std_alpha=std_alpha)

    print('\n\n Preparing reference data for training...')
    print('y_model_L shape:', y_model_L.shape)

    y_raw, y_true = prepare_reference_data(reference_data, minimum_training_steps=minimum_training_steps)
    augment = augment_data_length > 1

    if not correlation_based_training:
        innovations_all, model_bias_all = [], []
        for yr, yt in zip(y_raw, y_true):
            innovations = (yr - y_model_L[-minimum_training_steps:]).transpose((2, 0, 1))
            innovations_all.append(innovations)

            if augment:
                innovations_all.append(innovations * 1e-1)
                innovations_all.append(innovations * -1e-2)

            if biased_observations:
                model_bias = (yt - y_model_L[-minimum_training_steps:]).transpose((2, 0, 1))
                model_bias_all.append(model_bias)
                if augment:
                    model_bias_all.append(model_bias * 1e-1)
                    model_bias_all.append(model_bias * -1e-2)
    else:
        innovations_all, model_bias_all = [], []
        for yr, yt in zip(y_raw, y_true):
            ym_L = correlate_data(y_model_L, yr, augment_data_length, minimum_training_steps)
            innovations = (yr - ym_L).transpose((2, 0, 1))
            innovations_all.append(innovations)

            if biased_observations:
                model_bias = (yt - ym_L).transpose((2, 0, 1))
                model_bias_all.append(model_bias)

    if not biased_observations:
        train_data = np.concatenate(innovations_all, axis=0)
    else:
        innovations_all = np.concatenate(innovations_all, axis=0)
        model_bias_all = np.concatenate(model_bias_all, axis=0)
        train_data = np.concatenate([model_bias_all, innovations_all], axis=2)

    train_data_dict = {key: val for key, val in config.items()}
    train_data_dict.update(data=train_data)
    return train_data_dict
