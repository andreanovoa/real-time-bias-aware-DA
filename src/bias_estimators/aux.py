import numpy as np

from typing import Dict, List, Optional, Tuple, Union

from observations import Observations
from models import Model
from utils import mean_vector_to_ensemble, correlation, check_valid_file, load_from_pickle_file, CR
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt

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
                        std_alpha: Optional[Union[float, Dict[str, Union[float, List[float], Tuple[float, float]]]]] = None,
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
    elif not isinstance(std_alpha, dict):
        # A scalar std_alpha means a relative uncertainty around the nominal values.
        # Convert it to a dict of [min, max] ranges for the estimated parameters
        # (or all model parameters if none are being estimated).
        rel = float(std_alpha)
        params = model.est_alpha if len(model.est_alpha) > 0 else model.params
        std_alpha = {}
        for key in params:
            val = getattr(model, key)
            bounds = sorted([val * (1. - rel), val * (1. + rel)])
            std_alpha[key] = bounds

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
        if std_alpha is not None:

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
        

    psi0 = np.mean(model.current_state.copy(), axis=-1)
    Nt = int(np.round(model.t_transient / model.dt, model.precision_t)) - 1

    # Add parameters to the state vector if parameter uncertanty is givemn
    if std_alpha is not None and psi0.shape[0] == model.Nphi:
        assert isinstance(std_alpha, dict), "std_alpha must be a dict if parameter uncertainty is specified."
        model.ensemble = dict(Na=len(std_alpha), est_alpha=list(std_alpha.keys()), m=L)
        psi0 = np.hstack([psi0, np.zeros((len(std_alpha),))])
        for ii, key in enumerate(std_alpha.keys()):
            psi0[model.Nphi + ii] = getattr(model, key)
        psi0_ens = sample_ensemble(psi0, L)
        model.update_history(psi=psi0_ens[np.newaxis, :, :], t=0.0, reset=True)
        
    elif model.m != L:
        psi0_ens = sample_ensemble(psi0, L)
        model.update_history(psi=psi0_ens[np.newaxis, :, :], t=0.0, reset=True)

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
                                correlation_based_training: bool,
                                augment_data_length: int,
                                biased_observations: bool,
                                std_phi: Optional[float] = None,
                                std_alpha: Optional[Union[float, Dict[str, Union[float, List[float]]]]] = None,
                            ) -> dict:
    
    """
    Parameters:
        config: dict containing the configuration for creating the training dataset. 
        rom: Model object representing the reduced-order model to sample states from.
        reference_data: Observations object or list of Observations objects containing the reference data to prepare for training.
        minimum_training_steps: int specifying the minimum number of time steps required for training the bias model.
        L: int specifying the number of samples to generate from the ROM for training.
        correlation_based_training: bool indicating whether to use correlation-based training (i.e., correlating the model-generated data with the raw observations to create the training dataset).
        augment_data_length: int specifying the number of augmented samples to create for each observed variable.
            - If augment_data_length=1, only the best lag is used. 
            - If augment_data_length=2 both the best lag and a mid-point lag are used. If augment_data_length=3, the worst lag is also included.
        biased_observations: bool indicating whether to include the model bias (i.e., the difference between the true values and the model-generated data) in the training dataset.
        std_phi: float specifying the standard deviation to use when sampling the initial conditions for the ensemble. 
            - If None, the standard deviation of the current state of the ROM will be used.
        std_alpha: float or dict specifying the standard deviation to use when sampling the parameters for the ensemble. 
            - If float, it will be used for all parameters. 
            - If dict, it should have keys corresponding to the parameter names and values specifying the standard deviation for each parameter.
    Returns:
        train_data_dict: dict containing the training data for the bias model, along with the configuration used to create it. 
            - 'data': np.ndarray containing the training data for the bias model. shape (L * augment_data_length, Ndim) where Ndim = Nq if not biased_observations else 2 * Nq.
            - 'y_model': np.ndarray containing the model-generated data used for training. shape (L, minimum_training_steps, Nq).
            - Additional keys corresponding to the entries in the input config dictionary
    """

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
        y_model = []
        for yr, yt in zip(y_raw, y_true):
            ym_L = correlate_data(y_model_L, yr, augment_data_length, minimum_training_steps)
            y_model.append(ym_L.copy())
            
            innovations = (yr - ym_L).transpose((2, 0, 1)) #shape (L, minimum_training_steps, Nq)
            innovations_all.append(innovations)


            if biased_observations:
                model_bias = (yt - ym_L).transpose((2, 0, 1))
                model_bias_all.append(model_bias)
        y_model_L = np.concatenate(y_model, axis=0)

    if not biased_observations:
        train_data = np.concatenate(innovations_all, axis=0)
    else:
        innovations_all = np.concatenate(innovations_all, axis=0)
        model_bias_all = np.concatenate(model_bias_all, axis=0)
        train_data = np.concatenate([model_bias_all, innovations_all], axis=2)
    # Save to dictionary
    train_data_dict = {key: val for key, val in config.items()}
    train_data_dict.update(data=train_data,
                           y_model=y_model_L)
    return train_data_dict




def plot_train_data(truth, bias_data, t_CR):

    L, _, _ = bias_data['data'].shape

    Nt = int(t_CR / truth.dt)
    i0_t = np.argmin(np.abs(truth.t_true - truth.t_obs[0]))

    # Build a common valid time window and select the segment before first observation.
    n_common = min(
        len(truth.t_true),
        truth.y_true.shape[0],
        truth.b_true.shape[0],
        bias_data['y_model'].shape[0],
        bias_data['data'].shape[1],
    )
    i_end = min(max(i0_t, 1), n_common)
    i_start = max(i_end - Nt, 0)
    if i_end - i_start < 2:
        i_end = n_common
        i_start = max(i_end - Nt, 0)

    yt = truth.y_true[i_start:i_end]
    bt = truth.b_true[i_start:i_end]
    yr = bias_data['y_model'][i_start:i_end].transpose(2, 0, 1)
    Nq = yt.shape[1]
    br = bias_data['data'][:, i_start:i_end, :Nq]
    tt = truth.t_true[i_start:i_end]

    if len(tt) == 0:
        raise ValueError('Selected plotting window is empty. Check t_CR and training_data dimensions.')


    RS = []
    for ii in range(L):
        RS.append(np.linalg.norm(br[ii][:, 0]) / np.sqrt(len(yt)))

    RS = np.asarray(RS, dtype=float)
    true_RMS = np.linalg.norm(bt[:, 0]) / np.sqrt(len(yt))

    # Plot training data (single row) --------------------------
    fig = plt.figure(figsize    =[12, 2.7], layout='constrained')
    axs = fig.subplots(1, 2)
    
    # Robust color mapping: clip outliers; if RMS are nearly equal, force distinct member colors.
    if np.ptp(RS) < 1e-12:
        color_values = np.linspace(0.0, 1.0, L)
        norm = Normalize(vmin=0.0, vmax=1.0)
        cmap = plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap('viridis'))
        cbar_extend = 'neither'
        cbar_title = 'Member'
    else:
        lo, hi = np.percentile(RS, [5, 95])
        if np.isclose(lo, hi):
            lo = float(np.min(RS))
            hi = float(np.max(RS))
        color_values = np.clip(RS, lo, hi)
        norm = Normalize(vmin=float(lo), vmax=float(hi))
        cmap = plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap('viridis'))
        cbar_extend = 'both'
        cbar_title = '$\\mathrm{RMS}$'

    xlim = [tt[0], tt[-1]]

    axs[0].plot(tt, yt[:, 0], color='silver', linewidth=6, alpha=.8)
    axs[1].plot(tt, bt[:, 0], color='silver', linewidth=4, alpha=.8)

    for ii in range(L):
        clr = cmap.to_rgba(color_values[ii])
        axs[0].plot(tt, yr[ii][:, 0], color=clr)
        axs[1].plot(tt, br[ii][:, 0], color=clr)

    axs[0].legend(['Truth'], bbox_to_anchor=(0., 0.25), loc='upper left')
    axs[1].legend(['True RMS $={0:.3f}$'.format(true_RMS)], bbox_to_anchor=(0., 0.25), loc='upper left')
    axs[0].set(xlabel='$t$', ylabel='$\\eta$', xlim=xlim)
    axs[1].set(xlabel='$t$', ylabel='$b$', xlim=xlim)

    clb = fig.colorbar(cmap, ax=axs, orientation='vertical', extend=cbar_extend)
    clb.ax.set_title(cbar_title)