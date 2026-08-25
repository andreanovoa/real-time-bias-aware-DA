"""
Created on Wed May 11 09:45:48 2022

@author: Andrea Nóvoa @andrea_novoa
"""
import os
import pickle
import re
import zipfile
from functools import lru_cache
from typing import Optional

import matplotlib as mpl
import matplotlib.backends.backend_pdf as plt_pdf
import matplotlib.pyplot as plt
import numpy as np
import requests
import scipy.ndimage as ndimage

# shared model-layer helpers now live in (and are re-exported from) dynamodels
from dynamodels.utils import (  # noqa: F401
    Cheb,
    allowed_kwargs_for_func,
    interpolate,
    load_from_mat_file,
    mean_vector_to_ensemble,
    normalized_alpha,
    normalized_time,
    normalized_y,
    save_to_mat_file,
)
from dynamodels.utils import create_dataset as _dm_create_dataset
from matplotlib import colors
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from tqdm import tqdm

rng = np.random.default_rng(6)


def convert_to_python_type(obj, *, float_ndigits=12):
    """Convert numpy types to native Python types, with canonical float rounding."""
    if obj is None:
        return "none"

    if isinstance(obj, np.generic):
        if np.issubdtype(type(obj), np.integer):
            return int(obj)
        elif np.issubdtype(type(obj), np.floating):
            return round(float(obj), float_ndigits)
        elif np.issubdtype(type(obj), np.bool_):
            return bool(obj)
        elif np.issubdtype(type(obj), np.complexfloating):
            c = complex(obj)
            return (round(c.real, float_ndigits), round(c.imag, float_ndigits))
        else:
            return obj.item()

    elif isinstance(obj, float):
        return round(obj, float_ndigits)

    elif isinstance(obj, np.ndarray):
        return [convert_to_python_type(x, float_ndigits=float_ndigits) for x in obj.tolist()]
    elif isinstance(obj, tuple):
        return [convert_to_python_type(item, float_ndigits=float_ndigits) for item in obj]
    elif isinstance(obj, list):
        return [convert_to_python_type(item, float_ndigits=float_ndigits) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_python_type(v, float_ndigits=float_ndigits) for k, v in obj.items()}

    # if Path, change to sttring
    elif isinstance(obj, os.PathLike):
        return str(obj)

    return obj


def cut_signals(t, *signals, min_time=None, max_time=None):
    """Trim time array `t` and any number of aligned `signals` to ``[min_time, max_time]``."""
    i0 = 0 if min_time is None else np.argmin(abs(t - min_time))
    i1 = len(t) - 1 if max_time is None else np.argmin(abs(t - max_time))
    t_cut = t[i0:i1]
    signals_cut = [sig[i0:i1].copy() if sig is not None else None for sig in signals]
    return t_cut, signals_cut


def add_noise_to_flow(U, noise_level=0.05, noise_type="gauss", spatial_smooth=0.):
    """Add noise to a velocity field.

    Parameters
    ----------
    U : np.ndarray
        3D array ``(Nt, Nx, Ny)`` for a single velocity component, or 4D array
        ``(2, Nt, Nx, Ny)`` for both components (U, V). NaNs (e.g. inside a
        cylinder) are excluded from the noise-amplitude scaling.
    noise_level : float, optional
        Standard deviation of the noise, as a fraction of the maximum absolute
        velocity. Default 0.05.
    noise_type : str, optional
        ``'gauss'`` for Gaussian (white) noise, or ``'pink'``/``'brown'``/
        ``'blue'``/``'violet'`` for coloured noise (see `colour_noise`).
        Default ``'gauss'``.
    spatial_smooth : float, optional
        Standard deviation for Gaussian spatial smoothing of the noise (0
        disables smoothing). Default 0.

    Returns
    -------
    np.ndarray
        Noisy velocity field, same shape as `U`.
    """
    # mask nan values (e.g., inside cylinder) to avoid affecting noise scaling
    fluid_mask = ~np.isnan(U)
    U = np.where(fluid_mask, U, 0.)

    if U.ndim == 4:
        assert U.shape[0] == 2, "Expected first dimension of size 2 for (U, V) components."
    elif U.ndim == 3:
        U = U[np.newaxis, ...]  # Add component dimension for uniform processing
    else:
        raise ValueError("Input U must be a 3D array (Nt x Nx x Ny) or a 4D array (2 x Nt x Nx x Ny).")

    # Compute noise amplitude
    noise_amp = noise_level * np.max(np.abs(U[fluid_mask]))
    rng_noise = np.random.default_rng()

    if noise_type == "gauss":
        noise_U = rng_noise.normal(scale=noise_amp, size=U.shape)
    else:
        noise_U = np.fft.irfftn(
            np.fft.rfftn(rng_noise.standard_normal(U.shape)) * noise_amp
            * colour_noise(U.shape, noise_colour=noise_type), s=U.shape).real


    # Apply optional spatial smoothing
    if spatial_smooth > 0:
        sigma = (0, 0, spatial_smooth, spatial_smooth)
        noise_U = ndimage.gaussian_filter(noise_U, sigma=sigma)


    U_noisy = U + noise_U
    U_noisy[~fluid_mask] = np.nan

    return U_noisy




def find_first_ascending_folder(start_dir, target_names):
    """Ascend from `start_dir` looking for any of the `target_names` folders.

    Returns
    -------
    tuple
        ``(parent_path, found_folder)``, or ``(None, False)`` if none is found
        before reaching the filesystem root.

    Examples
    --------
    >>> parent, found = find_first_ascending_folder('.', ['src', 'dev'])
    >>> if parent:
    ...     print(f"Found {found} in {parent}")
    """
    dir_path = os.path.abspath(start_dir)
    while True:
        existing = [name for name in target_names if name in os.listdir(dir_path)]
        if existing:
            return dir_path, existing[0]
        parent = os.path.dirname(dir_path)
        if parent == dir_path:
            return None, False
        dir_path = parent


def get_project_root( root='.'):
    """Return the project root directory."""

    project_root, found = find_first_ascending_folder(root, ['src', 'dev'])
    if found == 'dev':

        project_root = f'{project_root}/real_public'
        print('On dev folder, root=' , project_root)

    elif not found:
        raise FileNotFoundError("Project root directory not found. Ensure you are in the correct directory structure.")

    return project_root



def set_working_directories(subfolder='', root='.'):
    """Resolve the data/results/figures folders for a given case `subfolder`.

    Parameters
    ----------
    subfolder : str, optional
        Case-specific subfolder appended to each base directory.
    root : str, optional
        Directory to start searching for the project root from (see
        `get_project_root`). Default ``'.'``.

    Returns
    -------
    tuple of str
        ``(data_folder, results_folder, figs_folder)``.
    """


    if subfolder[-1] != '/':
        subfolder += '/'

    # Get project root, i.e., where real_time_DA is
    project_root = get_project_root(root)

    #  Set results and fgures folders
    if 'tutorials' in os.getcwd():
        results_folder = f'{project_root}/results/tutorials/{subfolder}'
        figs_folder = f'{project_root}/docs/figs/{subfolder}'
    else:
        results_folder = f'{project_root}/results/{subfolder}'
        figs_folder = f'{project_root}/results/figs/{subfolder}'

    #  Set data folder. Note: some data is not provided in the repository.

    data_folder = f'{project_root}/data/{subfolder}'

    return data_folder, results_folder, figs_folder



def colour_noise(dims, noise_colour='pink', beta=2, ff=None):
    r"""Generate a spectral filter for coloured noise, for use with `np.fft.irfft`.

    Parameters
    ----------
    dims : int or sequence of int
        Number of time steps or spatial points along each dimension.
    noise_colour : str, optional
        Type of noise: ``'white'`` (flat spectrum), ``'pink'`` ($1/f^\beta$,
        long-range correlation; a trailing number overrides `beta`, e.g.
        ``'pink1.5'``), ``'brown'`` ($1/f^2$), ``'blue'`` ($\sqrt{f}$) or
        ``'violet'`` ($f$). Default ``'pink'``.
    beta : float, optional
        Decay exponent used for ``'pink'`` noise (overridden if `noise_colour`
        has a trailing number). Default 2.
    ff : optional
        Unused; reserved.

    Returns
    -------
    np.ndarray
        Spectral filter, real-FFT shaped: length ``dims // 2 + 1`` if `dims` is an
        int, or the same shape as `dims` with the last axis halved if `dims` is a
        sequence.
    """


    # Frequency arrays for each dimension
    if isinstance(dims, int):
        ff_dims = [np.fft.rfftfreq(dims)]
    else:
        ff_dims = [np.fft.fftfreq(Nt) for Nt in dims[:-1]] + [np.fft.rfftfreq(dims[-1])]


    def create_spectrum(freqs, noise_type):
        """Create 1D spectrum for a single dimension."""
        spectrum = np.ones_like(freqs, dtype=np.float32)
        noise_type = noise_type.lower()

        # Handle DC component first
        # find the zero frequency components into a mask

        mask = (freqs == 0) # tthuis should return a boolean array where True indicates the zero frequency component

        if 'white' in noise_type:
            spectrum[:] = 1.0
        elif 'blue' in noise_type:
            spectrum = np.sqrt(np.abs(freqs))
        elif 'violet' in noise_type:
            spectrum = np.abs(freqs)
        elif 'pink' in noise_type:
            numbers = re.findall(r'\d+\.*\d*', noise_type)
            beta_used = float(numbers[0]) if numbers else beta
            spectrum = 1 / np.where(mask, np.inf, np.abs(freqs) ** (1/beta_used))
        elif 'brown' in noise_type:
            spectrum = 1 / np.where(mask, np.inf, np.abs(freqs))
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")


        # Zero DC component and normalize
        spectrum[mask] = 0.
        if np.any(spectrum[~mask]):
            spectrum /= np.sqrt(np.mean(spectrum[~mask]**2))
        return spectrum

    # Reshape each frequency array to match the dimensions
    # and combine them into a single array
    S_dims = []

    for ii, ff in enumerate(ff_dims):
        S = create_spectrum(ff, noise_colour)
        shape = [1] * len(ff_dims)
        shape[ii] = -1  # -1 preserves original size
        S_dims.append(S.reshape(shape))

    S = S_dims[0]
    if len(S_dims) > 1:
        for spec in S_dims[1:]:
            S = S * spec


    return S





def check_valid_file(load_case, params_dict):
    """Check that a loaded case (dict or object) matches the expected `params_dict` values.

    Parameters
    ----------
    load_case : dict or object
        Loaded case to validate; attributes/keys are read via `getattr`/``[...]``.
    params_dict : dict
        Expected parameter name/value pairs.

    Returns
    -------
    bool
        True if every parameter present in `load_case` matches `params_dict`.
    """
    # check that true and forecast model input_parameters
    # print('Test if loaded file is valid', end='')
    is_mapping = isinstance(load_case, dict)

    def _has(key):
        return (key in load_case) if is_mapping else hasattr(load_case, key)

    def _get(key):
        return load_case[key] if is_mapping else getattr(load_case, key)

    for key, val in params_dict.items():
        if _has(key):
            print('\n\t', key, val, _get(key), end='')
            if len(np.shape([val])) == 1:
                if _get(key) != val:
                    print('\t <--- Re-init model!')
                    return False
            else:
                if any([x1 != x2 for x1, x2 in zip(_get(key), val)]):
                    print('\t <--- Re-init model!')
                    return False
    # print('... OK\n')
    return True




@lru_cache(maxsize=10)
def getEnvelope(timeseries_x, timeseries_y, fill_value=0):
    """Return a linear interpolant through the peaks of `timeseries_y` (its envelope)."""
    peaks, _ = find_peaks(timeseries_y, distance=200)
    return interp1d(timeseries_x[peaks], timeseries_y[peaks], bounds_error=False, fill_value=fill_value)


def save_to_pickle_file(filename, *args):
    """Pickle each of `args` sequentially into `filename`, creating parent directories."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        for arg in args:
            pickle.dump(arg, f)


def load_from_pickle_file(filename):
    """Load all objects sequentially pickled into `filename` (see `save_to_pickle_file`).

    Returns
    -------
    object, list or False
        The single unpickled object, a list of objects if more than one was
        stored, or False if `filename` does not exist.
    """
    if not os.path.exists(filename):
        return False

    args = []
    with open(filename, 'rb') as f:
        while True:
            try:
                arg = pickle.load(f)
                args.append(arg)
            except EOFError:
                break
    if len(args) == 1:
        return args[0]
    else:
        return args


def save_figs_to_pdf(pdf_name, figs=None):
    """Save `figs` (default: all open figures) to a multi-page PDF, closing each after saving."""
    pdf_file = plt_pdf.PdfPages(pdf_name)
    if figs is None:
        figs = [plt.figure(ii) for ii in plt.get_fignums()]
    elif not isinstance(figs, list):
        figs = [figs]

    for fig in figs:
        pdf_file.savefig(fig, dpi=300)  # Save figure to PDF
        plt.close(fig)

    pdf_file.close()  # Close results pdf


def add_pdf_page(pdf, fig_to_add, close_figs=True):
    """Append `fig_to_add` as a page to an open `matplotlib.backends.backend_pdf.PdfPages`."""
    pdf.savefig(fig_to_add)
    if close_figs:
        plt.close(fig_to_add)




def fun_PSD(dt, X):
    """Compute the Power Spectral Density of one or more signals.

    Parameters
    ----------
    dt : float
        Sampling time.
    X : np.ndarray
        Signal(s), shape ``(Nq, Nt)`` (1D signals are promoted to ``(1, Nt)``; a 2D
        array is transposed if its first dimension is larger than its second, i.e.
        the longer axis is assumed to be time).

    Returns
    -------
    f : np.ndarray
        Frequencies, shape ``(Nt // 2,)``.
    PSD : list of np.ndarray
        Power Spectral Density of each row of `X`.
    """
    if X.ndim == 2:
        if X.shape[0] > X.shape[1]:
            X = X.T
    elif X.ndim == 1:
        X = np.expand_dims(X, axis=0)
    else:
        raise AssertionError('X must be 2 dimensional')

    len_x = X.shape[-1]
    f = np.linspace(0.0, 1.0 / (2.0 * dt), len_x // 2)
    PSD = []
    for x in X:
        yt = np.fft.fft(x)
        PSD.append(2.0 / len_x * np.abs(yt[0:len_x // 2]))

    return f, PSD


def plot_train_data(truth, y_ref, t_ref, t_CR, folder):
    """Plot and save (as SVG) a summary of bias-estimator training data vs. the truth.

    Parameters
    ----------
    truth : dict
        Reference case with keys ``'t'``, ``'y'``, ``'b'``, ``'dt'``, ``'t_obs'``.
    y_ref : np.ndarray
        Training data/estimates, shape ``(Nt, Nq, L)``.
    t_ref : np.ndarray
        Time array matching `y_ref`.
    t_CR : float
        Length of the plotted window around the first observation time.
    folder : str
        Output directory for the saved figure (created if missing).
    """
    L = y_ref.shape[-1]
    y = y_ref[:len(truth['t'])]  # train_ens.getObservableHist(Nt=len(truth['t']))
    t = t_ref[:len(truth['t'])]

    Nt = int(t_CR / truth['dt'])
    i0_t = np.argmin(abs(truth['t'] - truth['t_obs'][0]))
    i0_r = np.argmin(abs(t_ref - truth['t_obs'][0]))

    yt = truth['y'][i0_t - Nt:i0_t]
    bt = truth['b'][i0_t - Nt:i0_t]
    yr = y[i0_r - Nt:i0_r]
    tt = t_ref[i0_r - Nt:i0_r]

    RS = []
    for ii in range(y.shape[-1]):
        R = CR(yt, yr[:, :, ii])[1]
        RS.append(R)

    true_RMS = CR(yt, yt - bt)[1]

    # Plot training data -------------------------------------
    fig = plt.figure(figsize=[12, 4.5], layout="constrained")
    sub_figs = fig.subfigures(2, 1, height_ratios=[1, 1])
    axs_top = sub_figs[0].subplots(1, 2)
    axs_bot = sub_figs[1].subplots(1, 2)
    norm = colors.Normalize(vmin=true_RMS, vmax=1.5)
    cmap = plt.cm.ScalarMappable(norm=norm, cmap=mpl.colormaps['viridis'])
    cmap.set_clim(true_RMS, 1.5)
    axs_top[0].plot(tt, yt[:, 0], color='silver', linewidth=6, alpha=.8)
    axs_top[-1].plot(tt, bt[:, 0], color='silver', linewidth=4, alpha=.8)
    xlims = [[truth['t_obs'][0] - t_CR, truth['t_obs'][0]],
             [truth['t_obs'][0], truth['t_obs'][0] + t_CR * 2]]

    for ii in range(y.shape[-1]):
        clr = cmap.to_rgba(RS[ii])
        axs_top[0].plot(tt, yr[:, 0, ii], color=clr)
        norm_bias = (truth['y'][:, 0] - y[:, 0, ii])
        for ax in [axs_bot, axs_top]:
            ax[-1].plot(t, norm_bias, color=clr)

    max_y = np.max(abs(yt[:, 0] - bt[:, 0]))

    axs_top[0].plot(tt, yt[:, 0], color='silver', linewidth=4, alpha=.5)
    axs_top[-1].plot(tt, bt[:, 0], color='silver', linewidth=4, alpha=.5)
    axs_bot[0].plot(t, truth['b'][:, 0] / max_y * 100, color='silver', linewidth=4, alpha=.5)
    axs_top[0].legend(['Truth'], bbox_to_anchor=(0., 0.25), loc="upper left")
    axs_top[1].legend([f'True RMS $={true_RMS:.3f}$'], bbox_to_anchor=(0., 0.25), loc="upper left")
    axs_top[0].set(xlabel='$t$', ylabel='$\\eta$', xlim=xlims[0])
    axs_bot[0].set(xlabel='$t$', ylabel='$b$ normalized [\\%]', xlim=xlims[-1])

    axs_bot[-1].set(xlabel='$t$', ylabel='$b$', xlim=xlims[-1])
    axs_top[-1].set(xlabel='$t$', ylabel='$b$', xlim=xlims[0])

    for ax in [axs_bot, axs_top]:
        clb = fig.colorbar(cmap, ax=ax, orientation='vertical', extend='max')
        clb.ax.set_title('$\\mathrm{RMS}$')

    os.makedirs(folder, exist_ok=True)
    plt.savefig(folder + f'L{L}_training_data.svg', dpi=350)
    plt.close()


def CR(y_true, y_est):
    """Return the (Pearson) correlation and the relative RMS error of `y_est` vs. `y_true`.

    Returns
    -------
    C : float
        Pearson correlation coefficient.
    R : float
        Root-mean-square error, normalized by ``norm(y_true)``.
    """
    # time average of both quantities
    y_tm = np.mean(y_true, 0, keepdims=True)
    y_em = np.mean(y_est, 0, keepdims=True)

    # correlation
    C = (np.sum((y_est - y_em) * (y_true - y_tm)) /
         np.sqrt(np.sum((y_est - y_em) ** 2) * np.sum((y_true - y_tm) ** 2)))
    # root-mean square error
    R = np.sqrt(np.sum((y_true - y_est) ** 2) / np.sum(y_true ** 2))
    return C, R



def correlation(y_true, y_est):
    """Compute the Pearson correlation coefficient of `y_est` vs. `y_true`, per ensemble member.

    Parameters
    ----------
    y_true : np.ndarray
        Reference data, shape ``(N_time, Nq)`` or ``(N_time, Nq, 1)``.
    y_est : np.ndarray
        Estimate, shape ``(N_time, Nq, N_ens)``.

    Returns
    -------
    np.ndarray
        Correlation coefficient per ensemble member, shape ``(N_ens,)`` (0 where
        the denominator is numerically zero).
    """
    y_true = np.asarray(y_true)
    y_est = np.asarray(y_est)

    # Accept 2D y_true (Nt, Nq) and promote to (Nt, Nq, 1)
    if y_true.ndim == 2:
        y_true = y_true[..., np.newaxis]
    if y_true.ndim != 3 or y_est.ndim != 3:
        raise ValueError(f'Expected y_true with ndim 2 or 3 and y_est with ndim 3, got {y_true.ndim}, {y_est.ndim}')

    # Check compatible time and spatial dimensions
    if y_true.shape[0] != y_est.shape[0] or y_true.shape[1] != y_est.shape[1]:
        raise ValueError(f'Incompatible shapes: y_true and y_est must share first two dimensions (time, spatial). {y_true.shape} vs {y_est.shape}')

    # Compute means
    y_tm = np.mean(y_true, axis=0, keepdims=True)  # shape (1, Nq, 1)
    y_em = np.mean(y_est, axis=0, keepdims=True)   # shape (1, Nq, N_ens)

    # Centered signals
    y_true_centered = y_true - y_tm # shape (Nt, Nq, 1)
    y_est_centered = y_est - y_em # shape (Nt, Nq, N_ens)

    # Numerator and denominator for Pearson r
    numerator = np.sum(y_est_centered * y_true_centered, axis=(0,1))  # shape (N_ens)
    denom = np.sqrt(np.sum(y_est_centered ** 2, axis=(0,1)) * np.sum(y_true_centered ** 2, axis=(0,1)))  # shape (N_ens)

    # Safe division: set correlation to 0 where denom is (near) zero
    with np.errstate(divide='ignore', invalid='ignore'):
        r_values = numerator / denom
        r_values = np.where(denom < 1e-10, 0.0, r_values)

    return r_values
def create_dataset_from_model(model_class, num_lyap_times=300, noise_level=0.02, seed=0, **kwargs):
    """`dynamodels.utils.create_dataset_from_model` cached under this repo's
    ``data/<model class name>/`` folder."""
    data_folder = set_working_directories(f'{model_class.__name__}/')[0]
    return _dm_create_dataset(model_class, data_folder, num_lyap_times=num_lyap_times,
                              noise_level=noise_level, seed=seed, **kwargs)


def create_Lorenz63_dataset(noise_level=0.02, num_lyap_times=300, seed=0, **kwargs):
    """Deprecated: `create_dataset_from_model` with `Lorenz63`. Kept for old notebooks."""
    from romda.models.physical import Lorenz63
    return create_dataset_from_model(Lorenz63, num_lyap_times=num_lyap_times,
                                     noise_level=noise_level, seed=seed, **kwargs)



def download_zenodo_file(download_url, data_folder='./', filename=None):
    """Download a file from Zenodo into `data_folder`, if not already present.

    If the downloaded file is a zip archive, it is also unzipped in place (see
    `unzip_file`).

    Parameters
    ----------
    download_url : str
        Direct URL to the Zenodo file, e.g.
        ``'https://zenodo.org/records/000000/files/example.py'``.
    data_folder : str, optional
        Folder where the file should be saved. Default ``'./'``.
    filename : str, optional
        Filename to save as. Defaults to the filename inferred from `download_url`.
    """
    # https://zenodo.org/records/000000/files/example.py

    os.makedirs(data_folder, exist_ok=True)
    if filename is None:
        filename = download_url.split('/')[-1].split('?')[0]
    file_path = os.path.join(data_folder, filename)

    # Download if missing, with progress bar
    download_flag = True
    if os.path.exists(file_path):
        print(f"File '{filename}' already exists. Skipping download.")
        if not filename.endswith('.zip'):
            download_flag = False
        elif zipfile.is_zipfile(file_path):
            download_flag = False

    if download_flag:
        print(f"Downloading {filename} from Zenodo...")
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        with open(file_path, 'wb') as f, tqdm(desc=filename,
                                              total=total_size,
                                              unit='B',
                                              unit_scale=True,
                                              unit_divisor=1024) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        if zipfile.is_zipfile(file_path):
            unzip_file(file_path, output_folder=data_folder, remove_first_folder=True)

def unzip_file(file_path, output_folder=None, remove_first_folder=True):
    """Extract a zip archive to `output_folder`.

    Parameters
    ----------
    file_path : str
        Path to the zip file.
    output_folder : str
        Folder where the contents are extracted.
    remove_first_folder : bool, optional
        If True, strip the archive's top-level folder from each extracted path.
        Default True.
    """
    # If it's a zip, check if already unzipped before extracting
    with zipfile.ZipFile(file_path, 'r') as zip_ref:

        members = [m for m in zip_ref.namelist() if not m.startswith('__MACOSX/') and not m.startswith('.DS_Store')]

        if not members:
            print("Zip file is empty, skipping extraction.")
            return

        # Find the top-level folder for each member
        zip_name = members[0].split('/')[0]
        for member in members:
            if member.endswith('/'):
                continue   # Skip directories
            if remove_first_folder:
                target = member[len(zip_name)+1:]
            else:
                target = member

            dest_path = f'{output_folder}/{target}'
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with zip_ref.open(member) as source, open(dest_path, 'wb') as target_file:
                target_file.write(source.read())
        print(f"Unzipped '{zip_name}.zip' in '{output_folder}'")


def get_annular_data(data_folder: Optional[str] = None):
    """
    Download and unzip the annular data from Zenodo if not already present.
    """

    if data_folder is None:
        laod_dir = set_working_directories('annular/')[0] #type: str
    else:
        laod_dir = data_folder

    zenodo_dir = "https://zenodo.org/records/15609832/files"

    download_zenodo_file(f'{zenodo_dir}/annular.zip?download=1',
                         laod_dir,
                        filename='annular_data.zip')

    download_zenodo_file(f'{zenodo_dir}/README.md?download=1', laod_dir)



def get_wake_data(data_folder: Optional[str] = None, case='circle_re_100'):
    """
    Download and unzip the bluff bodies wake flow data from Zenodo if not already present.
    """

    if data_folder is None:
        laod_dir = set_working_directories('wakes/')[0] #type: str
    else:
        laod_dir = data_folder

    zenodo_dir = "https://zenodo.org/records/15623774/files"

    download_zenodo_file(f'{zenodo_dir}/{case}.mat?download=1',
                         laod_dir, filename=f'{case}.mat')

    download_zenodo_file(f'{zenodo_dir}/README.md?download=1', laod_dir)





def load_cylinder_dataset(noise_type = 'gauss', noise_level = 0.1, smoothing = 0.1,
                          root_folder='.', visualize=False):
    """Load (downloading if needed) the cylinder-wake flow dataset, adding noise once and caching the result.

    Downloads the ``circle_re_100`` wake dataset (see `get_wake_data`) on first
    use, adds noise via `add_noise_to_flow`, and caches the noisy/clean pair to a
    ``.mat`` file so subsequent calls with the same parameters just reload it.

    Parameters
    ----------
    noise_type : str, optional
        Forwarded to `add_noise_to_flow`. Default ``'gauss'``.
    noise_level : float, optional
        Forwarded to `add_noise_to_flow`. Default 0.1.
    smoothing : float, optional
        Spatial smoothing, forwarded to `add_noise_to_flow` as `spatial_smooth`.
        Default 0.1.
    root_folder : str, optional
        Passed to `set_working_directories` to locate the data/results folders.
        Default ``'.'``.
    visualize : bool, optional
        If True, also generate a comparison animation via `visualize_flow_data`.
        Default False.

    Returns
    -------
    all_data : np.ndarray
        Clean velocity data, shape ``(Nt, Nx, Ny, 2)``.
    all_data_noisy : np.ndarray
        Noisy velocity data, same shape.
    new_results_dir : str
        Directory where the cached dataset (and optional animation) is stored.
    """
    data_folder, results_folder = set_working_directories('wakes/', root=root_folder)[:2]



    new_results_dir = f'{results_folder}/data_noise{noise_level}{noise_type}_smoothing{smoothing}/'

    # new_data_dir = f'{data_folder}/data_noise{noise_level}{noise_type}_smoothing{smoothing}/'
    os.makedirs(new_results_dir, exist_ok=True)

    data_name = f'{new_results_dir}00_data.mat'

    if not os.path.exists(data_name):

        if not os.path.exists(data_folder + 'circle_re_100.mat'):
            # Download the dataset if it does not exist

            # pri(f'folder/file not found {data_folder}')
            get_wake_data(data_folder, case='circle_re_100')

        # Load dataset
        mat = load_from_mat_file(data_folder + 'circle_re_100.mat')

        U, V = [mat[key] for key in ['ux', 'uy']]  # Nu, Nt, Ny, Nx

        U_noisy, V_noisy = add_noise_to_flow(np.array([U, V]),
                                             noise_level=noise_level,
                                             noise_type=noise_type,
                                             spatial_smooth=smoothing)

        all_data = np.array([U, V])                     # Nu, Nt, Ny, Nx
        all_data_noisy = np.array([U_noisy, V_noisy])   # Nu, Nt, Ny, Nx

        #  Change order of dimensions
        all_data = all_data.transpose(1, 2, 3, 0)               # Nt, Nx, Ny, Nu
        all_data_noisy = all_data_noisy.transpose(1, 2, 3, 0)   # Nt, Nx, Ny, Nu

        save_to_mat_file(data_name, dict(all_data=all_data,
                                         all_data_noisy=all_data_noisy))
    else:
        print(f'Loading...{data_name}')
        dataset = load_from_mat_file(data_name)
        all_data, all_data_noisy = [dataset[key] for key in ['all_data', 'all_data_noisy']]

    if visualize:
        visualize_flow_data(all_data, all_data_noisy, simulation_dir=new_results_dir)


    return all_data, all_data_noisy, new_results_dir





def visualize_flow_data(X_true, X_noisy, simulation_dir=''):
    """Build (if missing) and display a GIF comparing true and noisy flow fields.

    Parameters
    ----------
    X_true : np.ndarray
        True flow field, shape ``(..., 2)`` with the last axis indexing the
        (u, v) velocity components.
    X_noisy : np.ndarray
        Noisy flow field, same shape as `X_true`.
    simulation_dir : str, optional
        Directory where the GIF (``00_data.gif``) is saved/read from.
    """

    # Define the GIF name

    gif_name = f'{simulation_dir}00_data.gif'

    # Visualize the flow fields
    if not os.path.exists(gif_name):
        datasets = {
            '$u_x$': X_true[...,0],
            '$u_y$': X_true[...,1],
            '$\\tilde{u}_x$': X_noisy[...,0],
            '$\\tilde{u}_y$': X_noisy[...,1]
        }
        anim = animate_flowfields(datasets, n_frames=200, step=2, figsize=(6, 4))
        anim.save(gif_name)

    # Display in notebook (IPython imported lazily so the package does not
    # require IPython outside notebook contexts, e.g. in the docs CI build)
    from IPython.display import Image, display
    display(Image(filename=gif_name))



def animate_flowfields(datsets,
                       time=None,
                       n_frames=40, cmaps=None, rms_cmap='Reds', std_cmap='Blues', step=1,
                       rows=False, figsize=None, assimilated_data=None):
    """Create a `matplotlib.animation.FuncAnimation` of side-by-side flow fields.

    Parameters
    ----------
    datsets : dict
        Flow-field datasets, each of shape ``(Ny, Nx, Nt)`` (transposed in place
        if needed so the vertical dimension is the larger one, and required to
        share spatial dimensions). Keys are used as subplot titles; a title
        containing ``'RMS'`` or ``'std'`` selects `rms_cmap`/`std_cmap`.
    time : np.ndarray, optional
        1D array of time values matching the last axis of each dataset. Required
        (together with `assimilated_data`) to overlay observation markers.
    n_frames : int, optional
        Number of frames (used only when `assimilated_data` is None). Default 40.
    cmaps : list of str, optional
        Colormap per dataset. Defaults to ``'viridis'`` for all.
    rms_cmap : str, optional
        Colormap for datasets whose title contains ``'RMS'``. Default ``'Reds'``.
    std_cmap : str, optional
        Colormap for datasets whose title contains ``'std'``. Default ``'Blues'``.
    step : int, optional
        Frame stride (ignored if `assimilated_data` provides ``'t_obs'``). Default 1.
    rows : bool, optional
        If True, stack subplots in a column; otherwise in a row. Default False.
    figsize : tuple of float, optional
        Figure size. Defaults to a size scaled by the number of datasets.
    assimilated_data : dict, optional
        If given, drives the animated frames (overriding `n_frames`/`step`):

        - ``'t_obs'`` : array-like of observation times at which data are
          assimilated; a red marker is shown at these frames.
        - ``'xy'`` : array of shape ``(N_sensors, 2)`` with sensor coordinates
          ``[x_col, y_row]``.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The animation object (figure is closed; call `.save` or display it).
    """

    if cmaps is None:
        cmaps = ['viridis'] * len(datsets)

    if rows:
        if figsize is None:
            figsize = (1.5 * len(datsets), 4)
        fig, axs = plt.subplots(len(datsets), 1, sharex=True, sharey=True,
                                figsize=figsize, layout='constrained')
        cbar_orientation = 'vertical'
    else:
        if figsize is None:
            figsize = (4, 1.5 * len(datsets))
        fig, axs = plt.subplots(1, len(datsets), sharex=True, sharey=True,
                                figsize=figsize, layout='constrained')
        cbar_orientation = 'horizontal'

    if len(datsets) == 1:
        axs = [axs]

    # Transpose datasets if needed so vertical dim > horizontal dim
    for key, D in datsets.items():
        if D.shape[0] < D.shape[1]:
            datsets[key] = D.transpose(1, 0, 2)

    for key, D in datsets.items():
        ref = list(datsets.values())[0]
        if D.shape[0] != ref.shape[0] or D.shape[1] != ref.shape[1]:
            raise ValueError("All datasets must have the same spatial dimensions.")

    # ------------------------------------------------------------------ #
    #  Build frame_indices: driven by observations when available         #
    # ------------------------------------------------------------------ #
    dots, t_obs_set = [], set()

    if assimilated_data is not None and time is not None:
        show_obs  = True
        t_obs     = np.asarray(assimilated_data.get('t_obs', []))
        sensor_xy = np.asarray(assimilated_data.get('xy', []))

        # Map observation times to nearest indices in `time`
        obs_indices = np.searchsorted(time, t_obs)

        # Uniform stride across entire time range; always include observation indices
        regular_indices = np.arange(0, len(time), step)
        frame_indices = np.unique(np.concatenate((regular_indices, obs_indices)))
        frame_indices.sort()
        frame_indices = frame_indices.clip(0, len(time) - 1).tolist()
        frame_indices = frame_indices[:n_frames]  # Limit to n_frames if too many

        # Keep observation time values for dot-visibility test
        t_obs_set = set(t_obs.tolist())

        for ax in axs:
            sc = ax.scatter(
                *((sensor_xy[:, 0], sensor_xy[:, 1]) if len(sensor_xy) else ([], [])),
                c='red', s=40, marker='o', zorder=5, visible=False)
            dots.append(sc)
    else:
        show_obs      = False
        time          = np.arange(list(datsets.values())[0].shape[-1])
        frame_indices = list(range(0, min(n_frames, len(time)), step))

    # ------------------------------------------------------------------ #
    #  Build initial pcolormesh artists                                   #
    # ------------------------------------------------------------------ #
    ims = []
    for ax, (ttl, D), cmap in zip(axs, datsets.items(), cmaps):
        if 'RMS' in ttl:
            ims.append(ax.pcolormesh(D[..., 0], rasterized=True,
                                     cmap=plt.get_cmap(rms_cmap), vmin=0, vmax=1))
        elif 'std' in ttl.lower():
            norm = colors.Normalize(vmin=np.nanmin(D), vmax=np.nanmax(D))
            ims.append(ax.pcolormesh(D[..., 0], rasterized=True,
                                     cmap=plt.get_cmap(std_cmap), norm=norm))
        else:
            norm = colors.Normalize(vmin=np.nanmin(D), vmax=np.nanmax(D))
            ims.append(ax.pcolormesh(D[..., 0], rasterized=True,
                                     cmap=plt.get_cmap(cmap), norm=norm))

        ax.set(xticks=[], yticks=[])
        fig.colorbar(ims[-1], ax=ax, orientation=cbar_orientation, label=ttl)

    # ------------------------------------------------------------------ #
    #  Animation update function                                          #
    # ------------------------------------------------------------------ #
    def animate(ti):
        frame = frame_indices[ti]
        for im, D in zip(ims, datsets.values()):
            im.set_array(D[..., frame])
        if show_obs:
            is_obs = time[frame] in t_obs_set
            for sc in dots:
                sc.set_visible(is_obs)
        print(f'Frame {ti + 1}/{len(frame_indices)}', flush=True, end='\r')
        return ims + dots

    plt.close(fig)
    return FuncAnimation(fig, animate, frames=len(frame_indices), cache_frame_data=False)



def get_figsize_based_on_domain(domain, total_subplots, max_cols=5, total_width=6):
    """Compute a subplot-grid layout and figure size matching a spatial domain's aspect ratio.

    Parameters
    ----------
    domain : sequence of float
        ``[xmin, xmax, ymin, ymax]``.
    total_subplots : int
        Number of subplots to arrange.
    max_cols : int, optional
        Maximum number of columns (prevents excessively wide layouts). Default 5.
    total_width : float, optional
        Desired total figure width, in inches. Default 6.

    Returns
    -------
    figsize : tuple of float
        ``(fig_width, fig_height)``.
    ncols : int
        Number of subplot columns.
    nrows : int
        Number of subplot rows.
    """
    x_span = abs(domain[1] - domain[0])
    y_span = abs(domain[3] - domain[2])
    aspect_ratio = y_span / x_span if x_span != 0 else 1

    ncols = min(max_cols, total_subplots)
    nrows = int(np.ceil(total_subplots / ncols))

    per_subplot_width = total_width / ncols
    per_subplot_height = per_subplot_width * aspect_ratio

    fig_width = total_width
    fig_height = per_subplot_height * nrows

    return (fig_width, fig_height), ncols, nrows





# --------------------------- notebook figure collector ---------------------------

_savefig_figures = {}


def savefig(fig, name, figs_dir=None, **kwargs):
    """Collect a notebook's figures into one PDF, ``figs/<nn>.pdf`` (from the ``NN_``
    name prefix), rewriting the file each call so re-runs replace pages, not append.

    ``figs_dir`` defaults to ``figs/`` next to the caller (the current directory)."""
    figs = figs_dir or os.path.join(os.getcwd(), 'figs')
    os.makedirs(figs, exist_ok=True)
    nb = name.split('_')[0]
    _savefig_figures.setdefault(nb, {})[name] = fig

    path = os.path.join(figs, f'{nb}.pdf')
    with plt_pdf.PdfPages(path) as pdf:
        for f in _savefig_figures[nb].values():
            pdf.savefig(f, bbox_inches='tight', **kwargs)
    return path
