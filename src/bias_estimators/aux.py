"""Helpers for building, caching, and plotting the training dataset used by data-driven bias estimators (e.g. `ESN_bias`)."""

from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from romda.models import Model
from romda.observations import Observations
from romda.utils import check_valid_file, correlation, load_from_pickle_file, mean_vector_to_ensemble
from typeguard import typechecked


@typechecked
def prepare_reference_data(reference_data: Union[Observations, list[Observations]],
                           minimum_training_steps: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Prepare reference data for training the bias model.

    Extracts the raw observations and true values from the reference data, trimmed
    to the trailing `minimum_training_steps` time steps.

    Parameters
    ----------
    reference_data : Observations or list of Observations
        The reference data to prepare.
    minimum_training_steps : int
        Minimum number of training steps required; each returned array is trimmed to
        this length.

    Returns
    -------
    y_raw : list of ndarray, each shape (minimum_training_steps, Nq, 1)
        Raw observations from the reference data.
    y_true : list of ndarray, each shape (minimum_training_steps, Nq, 1)
        True (noise-free) values from the reference data.
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

    """Correlate model-generated data with raw observations to build a training set.

    1. Computes the correlation between the model-generated data and the raw
       observations.
    2. Identifies the best lag for each observed variable.
    3. Constructs a training dataset from the model data at these optimal lags,
       optionally augmented with a mid-point and/or worst-lag sample.

    Parameters
    ----------
    y_L_model : ndarray, shape (Nt, Nq, L)
        Model-generated data (`L` independent trajectories).
    y_raw : ndarray, shape (Nt, Nq, 1)
        Raw observations.
    augment_len : int
        Number of augmented samples per observed variable: 1 uses only the best
        lag; 2 adds a mid-point lag; 3 also adds the worst lag.
    minimum_training_steps : int
        Minimum number of training steps required.

    Returns
    -------
    ndarray, shape (minimum_training_steps, Nq, L * augment_len)
        Training data assembled from the best- (and optionally mid-/worst-) lag
        model trajectories.
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
                        std_alpha: Optional[Union[float, dict[str, Union[float, list[float], tuple[float, float]]]]] = None,
                    ) -> np.ndarray:
    """Sample model states from the ROM to build a training dataset for the bias model.

    1. Initializes the ROM with an ensemble of states (sampled initial conditions and,
       if specified, parameters).
    2. Integrates the ROM forward in time to generate model data.
    3. Processes this data into a training dataset for the bias model.

    Parameters
    ----------
    rom : Model
        The reduced-order model to sample states from.
    L : int
        Number of samples (ensemble members) to generate.
    minimum_training_steps : int
        Minimum number of time steps to integrate the ROM for, to generate enough
        data for training the bias model.
    std_phi : float, optional
        Standard deviation for sampling the initial conditions. If None (default),
        the standard deviation of the ROM's current state is used.
    std_alpha : float or dict, optional
        Standard deviation for sampling the parameters. A float is used as a
        multiplier on the current-state standard deviation, giving a
        ``[mean - std_alpha*std, mean + std_alpha*std]`` range for each estimated
        parameter; a dict should have parameter names as keys and per-parameter
        standard deviations as values. If None (default), the range is taken from
        the current state of the ROM.

    Returns
    -------
    ndarray
        Observable history of the (re-)integrated ROM ensemble, used as the model
        data for training.
    """

    model = rom.copy()

    if std_phi is None:
        std_phi = np.std(model.current_state[:model.Nphi, :], axis=-1)

    if std_alpha is None:
        std_alpha = {}
        for i, key in enumerate(model.est_alpha):
            param = model.current_state[model.Nphi + i, :]
            std_alpha[key] = [min(param), max(param)]
    elif isinstance(std_alpha, float):
        # A scalar std_alpha is a multiplier on the ensemble standard deviation:
        # build a [mean - const*std, mean + const*std] range for each estimated
        # parameter from the current state of the ROM.
        const = std_alpha
        std_alpha = {}
        for i, key in enumerate(model.est_alpha):
            param = model.current_state[model.Nphi + i, :]
            mean_param = np.mean(param)
            std_alpha[key] = [mean_param - const * np.std(param), mean_param + const * np.std(param)]

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
        if std_alpha:

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

    # Add parameters to the state vector if parameter uncertainty is given
    if std_alpha and psi0.shape[0] == model.Nphi:
        assert isinstance(std_alpha, dict), "std_alpha must be a dict if parameter uncertainty is specified."
        model.ensemble_cfg = dict(Na=len(std_alpha), est_alpha=list(std_alpha.keys()), m=L)
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
    expected_Ndim: Optional[int] = None,
):
    """Load a cached bias-training dataset from disk, if valid.

    Returns None (triggering a fresh dataset build) if `filename` is None, the file
    is missing/invalid, its configuration doesn't match `necessary_properties`, or
    its shape doesn't match `minimum_training_steps` / `augment_data_length` / `L` /
    `expected_Ndim`.

    Parameters
    ----------
    filename : str or None
        Path to the cached training-dataset pickle file.
    necessary_properties : dict
        Configuration the cached dataset must match (see `check_valid_file`).
    minimum_training_steps : int
        Minimum number of time steps the cached data must contain.
    augment_data_length : int
        Expected augmentation factor of the cached data.
    L : int
        Expected number of samples of the cached data.
    expected_Ndim : int, optional
        Expected trailing (observable) dimension of the cached data.

    Returns
    -------
    dict or None
        The cached training-data dictionary, or None if unavailable/invalid.
    """
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
        if expected_Ndim is not None and data.shape[-1] != expected_Ndim:
            print(f'Re-run multi-parameter training data: cached output dimension {data.shape[-1]} does not match expected {expected_Ndim}')
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
                                std_alpha: Optional[Union[float, dict[str, Union[float, list[float]]]]] = None,
                            ) -> dict:

    """Build a training dataset for the bias model from ROM samples and reference data.

    Parameters
    ----------
    config : dict
        Configuration for creating the training dataset; echoed into the returned
        dictionary.
    rom : Model
        The reduced-order model to sample states from.
    reference_data : Observations or list of Observations
        Reference data to prepare for training.
    minimum_training_steps : int
        Minimum number of time steps required for training the bias model.
    L : int
        Number of samples to generate from the ROM.
    correlation_based_training : bool
        If True, correlate the model-generated data with the raw observations (via
        `correlate_data`) rather than pairing them directly at matching time steps.
    augment_data_length : int
        Number of augmented samples per observed variable: 1 uses only the best
        lag/direct pairing; 2 adds a mid-point (or scaled) sample; 3 also adds the
        worst lag (or oppositely-scaled) sample.
    biased_observations : bool
        Whether to also include the model bias (true minus model-generated data) in
        the training dataset, alongside the innovations.
    std_phi : float, optional
        Standard deviation for sampling initial conditions; see `sample_model_states`.
    std_alpha : float or dict, optional
        Standard deviation for sampling parameters; see `sample_model_states`.

    Returns
    -------
    dict
        Training data for the bias model, plus the entries of `config`:

        - ``'data'`` : ndarray, shape (L * augment_data_length, minimum_training_steps, Ndim)
          where ``Ndim = Nq`` if not `biased_observations`, else ``2 * Nq``.
        - ``'y_model'`` : ndarray, shape (L, minimum_training_steps, Nq) — the
          model-generated data used for training.
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
    """Plot the observable and bias training samples against the truth.

    Shows one window before the first observation, colouring each training sample
    (of `bias_data`) by its bias RMS.

    Parameters
    ----------
    truth : Observations
        Reference truth used to select the plotting window and overlay the true
        observable/bias signals.
    bias_data : dict
        Training-data dictionary as returned by `create_bias_training_dataset`.
    t_CR : float
        Characteristic (e.g. oscillation) time scale, used to set the plotting
        window length.
    """
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
    axs[1].legend([f'True RMS $={true_RMS:.3f}$'], bbox_to_anchor=(0., 0.25), loc='upper left')
    axs[0].set(xlabel='$t$', ylabel='$\\eta$', xlim=xlim)
    axs[1].set(xlabel='$t$', ylabel='$b$', xlim=xlim)

    clb = fig.colorbar(cmap, ax=axs, orientation='vertical', extend=cbar_extend)
    clb.ax.set_title(cbar_title)
