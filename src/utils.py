# -*- coding: utf-8 -*-
"""
Created on Wed May 11 09:45:48 2022

@author: Andrea Nóvoa @andrea_novoa
"""
import os
import numpy as np
import pickle
from functools import lru_cache
import matplotlib as mpl
from matplotlib import colors
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.ndimage as ndimage

from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import matplotlib.backends.backend_pdf as plt_pdf
import re
import requests
from tqdm import tqdm
import zipfile

from typing import List, Tuple, Union, Dict, Type, Optional
from numpy.typing import NDArray

from PIL import Image

import inspect

from IPython.display import Image, display

from matplotlib.animation import FuncAnimation



rng = np.random.default_rng(6)


def allowed_kwargs_for_func(func, kwargs):
    sig = inspect.signature(func)
    accepted = set(sig.parameters)
    return {k: v for k, v in kwargs.items() if k in accepted}




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


def mean_vector_to_ensemble(rng: np.random.Generator,
                            mean_vec: NDArray[np.floating],
                            std: Union[float,
                                       NDArray[np.floating], 
                                       List[float], 
                                       Dict[str, Union[float, List[float], Tuple[float, float]]]],
                            m: int,
                            method: str = 'uniform',
                            ensure_mean_at_init: bool = False) -> np.ndarray:
    """
    Adds uncertainty to a mean state vector/value for ensemble generation.
    Returns an array of shape (state_dim, m).
    
    """
    if method not in ['uniform', 'normal']:
        raise ValueError(f'Distribution "{method}" not supported. Choose "uniform" or "normal".')
     
    mean_vec = np.asarray(mean_vec.copy())
    
    # Case 1: std is a dictionary (for estimated parameters 'alpha')
    if isinstance(std, dict):
        ensemble_ = []
        for sa in std.values():
            if method == 'uniform':
                # For uniform, std values are [min_val, max_val]
                ensemble_.append(rng.uniform(low=sa[0], high=sa[1], size=m)) #type: ignore
            else: # normal
                # Use mean of bounds as location, and half the range as a heuristic scale (std)
                loc = np.mean(sa)
                if isinstance(sa, list) and len(sa) == 2:
                    scale = (sa[1] - sa[0]) / 4.0
                else:
                    scale = loc * 0.5
                ensemble_.append(rng.normal(loc=loc, scale=scale, size=m))
        ensemble_ = np.array(ensemble_) # Shape: (num_params, m)

    # Case 2: std is a single float or a different std for each component (relative standard deviation for state or parameters)
    elif isinstance(std, float) or isinstance(std, np.ndarray):
        if method == 'uniform':
            # ensure std is an array with. compatible shape
            if isinstance(std, float):
                std = std * np.ones_like(mean_vec) 
            if std.ndim == 1:
                std = std[:, np.newaxis]

            
            perturbation = 1.0 + rng.uniform(-std, std, size=(mean_vec.size, m))
            ensemble_ = mean_vec[:, np.newaxis] * perturbation
        
        else: # normal (using multivariate normal for state vector)

            # Multiplicative uniform perturbation: mean * (1 +/- std)
            # print(f'Creating normal ensemble with std={std} for mean_vec of shape {mean_vec.shape} and m={m}')

            if np.iscomplexobj(mean_vec):
                # Handle complex state by perturbing real and imaginary parts independently
                # ensure we have a numpy array so static type checkers accept real/imag access
                real_mu = np.real(mean_vec)
                imag_mu = np.imag(mean_vec)
                real_part = rng.multivariate_normal(real_mu, np.diag((real_mu * std) ** 2), size=m).T
                imag_part = rng.multivariate_normal(imag_mu,  np.diag((imag_mu * std) ** 2), size=m).T
                ensemble_ = real_part + 1j * imag_part
            else:
                # Covariance matrix is diagonal, perturbation scaled by mean and relative std
                cov = np.diag((mean_vec * std) ** 2)
                ensemble_ = rng.multivariate_normal(mean_vec, cov, size=m).T
        
    else:
        raise TypeError(f'Initial std must be a float or a dict, not {type(std)}')


    # Replace the first member with the unperturbed mean
    if ensure_mean_at_init and ensemble_ is not None:
        ensemble_[:, 0] = mean_vec

    return ensemble_


def set_cylinder_truth(case, X_filter, X_filter_true, Nt_obs = 25, visualize=False):
    N_test = X_filter.shape[-1]

    if case.sensor_locations is not None:
        data_obs = X_filter.copy().reshape(-1, N_test)[case.sensor_locations].T
        data_obs_true = X_filter_true.copy().reshape(-1, N_test)[case.sensor_locations].T
    else:
        data_obs = case.project_data_onto_Psi(data=X_filter)
        data_obs_true = case.project_data_onto_Psi(data=X_filter_true)

    dt = case.dt
    t_true = np.arange(0, N_test)  * dt

    t_start = .5
    t_stop = min(5., t_true[-10])
    

    obs_idx = np.arange(t_start // dt, t_stop // dt + 1, Nt_obs, dtype=int) + 1


    # Nt_extra = len(t_true[obs_idx[-1]:])

    _truth = dict(y_raw=data_obs,
                  y_true=data_obs_true, 
                  t=t_true, 
                  dt=dt,
                  t_obs=t_true[obs_idx], 
                  y_obs=data_obs[obs_idx], 
                  dt_obs=Nt_obs * dt,
                  Nt_extra=int(case.t_CR // case.dt),
                  )
    
    return _truth



def add_noise_to_flow(U, noise_level=0.05, noise_type="gauss", spatial_smooth=0.):
    """
    Adds noise to a 3D velocity field (Nt x Nx x Ny).

    Args
        U : numpy.ndarray
            3D array representing the velocity component (Nt x Nx x Ny).
            or 4D array (2 x Nt x Nx x Ny) for both velocity components (U, V).
        
        noise_level : float, optional, default=0.05
            The standard deviation of the noise as a fraction of the maximum absolute velocity.
        
        noise_type : str, optional, default="gauss"
            The type of noise to apply:
            - 'gauss': Gaussian (white) noise.
            - 'pink', 'brown', 'blue', 'violet': Colored noise.

        spatial_smooth : float, optional, default=0
            Standard deviation for Gaussian smoothing (0 means no smoothing).

    Returns
        U_noisy : numpy.ndarray
            Noisy velocity field with the same shape as U.
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
    """
    Ascends from start_dir looking for any of the target_names folders.
    Returns (parent_path, found_folder), or (None, None) if not found.

    # Usage Example:
    parent, found = find_first_ascending_folder('.', ['src', 'dev'])
    if parent:
        print(f"Found {found} in {parent}")
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
    """
    Returns:
    - (data_folder, results_folder, figs_folder)
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
    """
    Generates a 1D spectral filter for colored noise.

    Args:
        dims (int or list): Number of time steps or spatial points.
        noise_colour (str): Type of noise ('white', 'pink', 'brown', 'blue', 'violet').
            - 'white'   -> Flat power spectrum (uncorrelated).
            - 'pink'    -> 1/f noise (long-range correlation).
            - 'brown'   -> 1/f^2 noise (strong low-frequency correlation).
            - 'blue'    -> Increases with frequency (anti-correlated noise).
            - 'violet'  -> Stronger high-frequency noise.
        beta (float, optional): Controls the decay of the power spectrum (used in pink noise).

    Returns:
        np.ndarray: 1D array of length Nt//2+1 (for rfft) with the noise filter in Fourier space.
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
def Cheb(Nc, lims=(0, 1), getg=False):
    """ Compute the Chebyshev collocation derivative matrix (D)
        and the Chevyshev grid of (N + 1) points in [ [0,1] ] / [-1,1]
    """
    g = - np.cos(np.pi * np.arange(Nc + 1, dtype=float) / Nc)
    c = np.hstack([2., np.ones(Nc - 1), 2.]) * (-1) ** np.arange(Nc + 1)
    X = np.outer(g, np.ones(Nc + 1))
    dX = X - X.T
    D = np.outer(c, 1 / c) / (dX + np.eye(Nc + 1))
    D -= np.diag(D.sum(1))

    # Modify
    if lims[0] == 0:
        g = (g + 1.) / 2.
    if getg:
        return D, g
    else:
        return D


def RK4(t, q0, func, *kwargs):
    """ 4th order RK for autonomous systems described by func """
    dt = t[1] - t[0]
    N = len(t) - 1
    qhist = [q0]
    for i in range(N):
        k1 = dt * func(dt, q0, kwargs)
        k2 = dt * func(dt, q0 + k1 / 2, kwargs)
        k3 = dt * func(dt, q0 + k2 / 2, kwargs)
        k4 = dt * func(dt, q0 + k3, kwargs)
        q0 = q0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        qhist.append(q0)

    return np.array(qhist)


def interpolate(t_y, y, t_eval, fill_values: Optional[tuple[float, float]] = None):
    # interpolator = PchipInterpolator(t_y, y)

    if fill_values is None:
        fill_values = (y[0], y[-1])

    interpolator = interp1d(t_y, y,
                            axis=0,  # interpolate along columns
                            bounds_error=False,
                            kind='linear',
                            fill_value=fill_values # type: ignore #tuple[float, float]
                            )
    return interpolator(t_eval)


def getEnvelope(timeseries_x, timeseries_y, fill_value=0):
    peaks, _ = find_peaks(timeseries_y, distance=200)
    return interp1d(timeseries_x[peaks], timeseries_y[peaks], bounds_error=False, fill_value=fill_value)
    # return u_p


def save_to_pickle_file(filename, *args):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        for arg in args:
            pickle.dump(arg, f)


def load_from_pickle_file(filename):

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


def load_from_mat_file(filename, squeeze_me=True):
    return sio.loadmat(filename, appendmat=True, squeeze_me=squeeze_me)


def save_to_mat_file(filename, data: dict, oned_as='column', do_compression=True):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    sio.savemat(filename, data, oned_as=oned_as, do_compression=do_compression)

def save_figs_to_pdf(pdf_name, figs=None):

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
    pdf.savefig(fig_to_add)
    if close_figs:
        plt.close(fig_to_add)




def fun_PSD(dt, X):
    # Function that computes the Power Spectral Density.
    # - Inputs:
    #       - dt: sampling time
    #       - X: signal(s) to compute the PSD (Nq x Nt)
    # - Outputs:
    #       - f: corresponding frequencies
    #       - PSD: Power Spectral Density (Nq list)
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
    axs_top[1].legend(['True RMS $={0:.3f}$'.format(true_RMS)], bbox_to_anchor=(0., 0.25), loc="upper left")
    axs_top[0].set(xlabel='$t$', ylabel='$\\eta$', xlim=xlims[0])
    axs_bot[0].set(xlabel='$t$', ylabel='$b$ normalized [\\%]', xlim=xlims[-1])

    axs_bot[-1].set(xlabel='$t$', ylabel='$b$', xlim=xlims[-1])
    axs_top[-1].set(xlabel='$t$', ylabel='$b$', xlim=xlims[0])

    for ax in [axs_bot, axs_top]:
        clb = fig.colorbar(cmap, ax=ax, orientation='vertical', extend='max')
        clb.ax.set_title('$\\mathrm{RMS}$')

    os.makedirs(folder, exist_ok=True)
    plt.savefig(folder + 'L{}_training_data.svg'.format(L), dpi=350)
    plt.close()


def CR(y_true, y_est):
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
    """Calculates the Pearson correlation coefficient (r-value) for each ensemble member.
    Inputes:
        y_true: np.ndarray of shape (N_time, Nq) or (N_time, Nq, 1)
        y_est: np.ndarray of shape (N_time, Nq, N_ens)
    Returns:
        r_values: np.ndarray of length N_ens with the correlation coefficients. 
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
        raise ValueError('Incompatible shapes: y_true and y_est must share first two dimensions (time, spatial). {} vs {}'.format(y_true.shape, y_est.shape))

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



# I used this for the CMAME paper, but it is not used in the current version of the code.
#  I keep it here for reference and possible future use.
def get_error_metrics(results_folder):
    raise NotImplementedError('To be redefined withh the new project architecture')

# def get_error_metrics(results_folder):
#     print('computing error metrics...')
#     out = dict(Ls=[], ks=[])

#     L_dirs, k_files = [], []
#     LLL = os.listdir(results_folder)
#     for Ldir in LLL:
#         if os.path.isdir(results_folder + Ldir + '/') and Ldir[0] == 'L':
#             L_dirs.append(results_folder + Ldir + '/')
#             out['Ls'].append(float(Ldir.split('L')[-1]))

#     for ff in os.listdir(L_dirs[0]):
#         k = float(ff.split('_k')[-1])
#         out['ks'].append(k)
#         k_files.append(ff)

#     # sort ks and Ls
#     idx_ks = np.argsort(np.array(out['ks']))
#     out['ks'] = [out['ks'][i] for i in idx_ks]
#     out['k_files'] = [k_files[i] for i in idx_ks]

#     idx = np.argsort(np.array(out['Ls']))
#     out['L_dirs'] = [L_dirs[i] for i in idx]
#     out['Ls'] = [out['Ls'][i] for i in idx]

#     # Output quantities
#     keys = ['R_biased_DA', 'R_biased_post',
#             'C_biased_DA', 'C_biased_post',
#             'R_unbiased_DA', 'R_unbiased_post',
#             'C_unbiased_DA', 'C_unbiased_post']
#     for key in keys:
#         out[key] = np.empty([len(out['Ls']), len(out['ks'])])

#     print(out['Ls'])
#     print(out['ks'])

#     ii = -1
#     for Ldir in out['L_dirs']:
#         ii += 1
#         print('L = ', out['Ls'][ii])
#         jj = -1
#         for ff in out['k_files']:
#             jj += 1
#             # Read file
#             truth, filter_ens = load_from_pickle_file(Ldir + ff)[1:]
#             truth = truth.copy()

#             print('\t k = ', out['ks'][jj], '({}, {})'.format(filter_ens.bias.L, filter_ens.regularization_factor))
#             # Compute biased and unbiased signals
#             y, t = filter_ens.get_observable_hist(), filter_ens.hist_t
#             b, t_b = filter_ens.bias.hist, filter_ens.bias.hist_t
#             y_mean = np.mean(y, -1)

#             # Unbiased signal error
#             if hasattr(filter_ens.bias, 'upsample'):
#                 y_unbiased = interpolate(t, y_mean, t_b) + b
#                 y_unbiased = interpolate(t_b, y_unbiased, t)
#             else:
#                 y_unbiased = y_mean + b

#             # if jj == 0:
#             N_CR = int(filter_ens.t_CR // filter_ens.dt)  # Length of interval to compute correlation and RMS
#             i0 = np.argmin(abs(t - truth['t_obs'][0]))  # start of assimilation
#             i1 = np.argmin(abs(t - truth['t_obs'][-1]))  # end of assimilation

#             # cut signals to interval of interest
#             y_mean, t, y_unbiased = y_mean[i0 - N_CR:i1 + N_CR], t[i0 - N_CR:i1 + N_CR], y_unbiased[i0 - N_CR:i1 + N_CR]

#             if ii == 0 and jj == 0:
#                 i0_t = np.argmin(abs(truth['t'] - truth['t_obs'][0]))  # start of assimilation
#                 i1_t = np.argmin(abs(truth['t'] - truth['t_obs'][-1]))  # end of assimilation
#                 y_truth, t_truth = truth['y'][i0_t - N_CR:i1_t + N_CR], truth['t'][i0_t - N_CR:i1_t + N_CR]
#                 y_truth_b = y_truth - truth['b'][i0_t - N_CR:i1_t + N_CR]

#                 out['C_true'], out['R_true'] = CR(y_truth[-N_CR:], y_truth_b[-N_CR:])
#                 out['C_pre'], out['R_pre'] = CR(y_truth[:N_CR], y_mean[:N_CR])
#                 out['t_interp'] = t[::N_CR]
#                 scale = np.max(y_truth, axis=0)
#                 for key in ['error_biased', 'error_unbiased']:
#                     out[key] = np.empty([len(out['Ls']), len(out['ks']), len(out['t_interp']), y_mean.shape[-1]])

#             # End of assimilation
#             for yy, key in zip([y_mean, y_unbiased], ['_biased_DA', '_unbiased_DA']):
#                 C, R = CR(y_truth[-N_CR * 2:-N_CR], yy[-N_CR * 2:-N_CR])
#                 out['C' + key][ii, jj] = C
#                 out['R' + key][ii, jj] = R

#             # After Assimilaiton
#             for yy, key in zip([y_mean, y_unbiased], ['_biased_post', '_unbiased_post']):
#                 C, R = CR(y_truth[-N_CR:], yy[-N_CR:])
#                 out['C' + key][ii, jj] = C
#                 out['R' + key][ii, jj] = R

#             # Compute mean errors
#             b_obs = y_truth - y_mean
#             b_obs_u = y_truth - y_unbiased
#             ei, a = -N_CR, -1
#             while ei < len(b_obs) - N_CR - 1:
#                 a += 1
#                 ei += N_CR
#                 out['error_biased'][ii, jj, a, :] = np.mean(abs(b_obs[ei:ei + N_CR]), axis=0) / scale
#                 out['error_unbiased'][ii, jj, a, :] = np.mean(abs(b_obs_u[ei:ei + N_CR]), axis=0) / scale

#             save_to_pickle_file(results_folder + 'CR_data', out)





def create_Lorenz63_dataset(noise_level=0.02, num_lyap_times=300, seed=0, **kwargs):
    """
    Create or load a Lorenz-63 time series dataset, optionally add Gaussian observation noise,
    and persist the result to disk for reuse.

    Parameters
    ----------
    noise_level : float, optional.  Default is 0.02.
        Relative standard deviation of additive Gaussian noise applied to each observable.
        The actual noise standard deviation for variable j is noise_level * std(clean_data[:, j]).
    num_lyap_times : int, optional. Default 300.
        The total number of time steps produced is num_lyap_times * N_lyap where N_lyap = int(model.t_lyap / model.dt).
    seed : int, optional
        Seed for the numpy.random.default_rng used to generate observation noise. Default is 0.
    **kwargs
        Additional keyword arguments forwarded to the Lorenz63 model initializer (models.physical.lorenz63).

    Returns
    -------
    tuple
        (dataset, filepath)
        - dataset : dict with keys
            - clean_data : ndarray, shape (Nt, n_vars)
                Clean model observables (no observation noise).
            - noisy_data : ndarray, shape (Nt, n_vars)
                Clean data with added Gaussian noise as described above.
            - t : ndarray, shape (Nt,)
                Time vector corresponding to the rows of the data arrays.
            - N_lyap : int
                Number of timesteps per Lyapunov time (computed as int(model.t_lyap / model.dt)).
        - filepath : str
            Full path of the .mat file used for loading/saving the dataset.

    Raises
    ------
    FileNotFoundError
        Propagated if any required file operations fail in an unexpected manner (the function
        itself catches the expected "dataset not present" case and proceeds to generate data).
    ImportError
        If models.physical.lorenz63 or helper functions (set_working_directories, load_from_mat_file,
        save_to_mat_file) are not available, an ImportError or NameError may be raised.

    Example
    -------
    >>> dataset, path = create_Lorenz63_dataset(noise_level=0.05, num_lyap_times=200, seed=42, sigma=10.0)
    >>> print(dataset['clean_data'].shape, dataset['t'].shape, path)

    """

    from models.physical import Lorenz63

    # Load or create training data from the Lorenz 63 model
    data_folder = set_working_directories('Lorenz/')[0]
    model = Lorenz63(**kwargs)


    # Default filename
    filename = model.filename
    filename += f"Nlyap{num_lyap_times}_noise{noise_level}_seed{seed}"

    t_lyap = model.t_lyap
    dt = model.dt
    N_lyap = int(t_lyap / dt)

    try:
        dataset = load_from_mat_file(data_folder + filename)
        print('Loaded case')
    except FileNotFoundError:



        # Create a time series from the model
        model.create_long_timeseries(Nt=num_lyap_times * N_lyap)

        # Get the model observables and time
        t = model.hist_t
        all_data = model.get_observable_hist()[..., 0].copy()
        all_data_clean = all_data.copy()

        # Add noise to the data
        rng_noise = np.random.default_rng(seed)
        U_std = np.std(all_data, axis=0)
        for dd in range(all_data.shape[1]):
            all_data[:, dd] += rng_noise.normal(loc=0, scale=noise_level * U_std[dd], size=all_data.shape[0])

        # Save data for future use
        dataset = dict(clean_data=all_data_clean,
                       noisy_data=all_data,
                       t=t,
                       N_lyap=N_lyap)
        save_to_mat_file(data_folder + filename, dataset)

    return dataset, data_folder + filename



def download_zenodo_file(download_url, data_folder='./', filename=None):
    """
    Download a file from Zenodo to the data folder if it is missing.
    - Inputs:
        download_url: Direct URL to the Zenodo file 
        data_folder: Folder where the file should be saved
        filename: Optional, specify the filename; if None, it is inferred from the URL
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
    """
    Unzip a file to the specified data folder.
    - Inputs:
        file_path: Path to the zip file
        data_folder: Folder where the contents should be extracted; if None, uses the same folder as the zip file
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

    zenodo_dir = "https://zenodo.org/records/15623774/files/"
    
    download_zenodo_file(f'{zenodo_dir}/{case}.mat?download=1"', 
                         laod_dir, filename=f'{case}.mat')

    download_zenodo_file(f'{zenodo_dir}/README.md?download=1', laod_dir)





def load_cylinder_dataset(noise_type = 'gauss', noise_level = 0.1, smoothing = 0.1, 
                          root_folder='.', visualize=False):

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

        U_noisy, V_noisy = add_noise_to_flow(U, V, 
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
    """
    Visualize the true and noisy flow fields.
    
    Parameters:
    - X_true: True flow field data.
    - X_noisy: Noisy flow field data.
    - simulation_dir: Directory to save the GIF.
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

    # Display in notebook
    display(Image(filename=gif_name))



def animate_flowfields(datsets, 
                       time=None,
                       n_frames=40, cmaps=None, rms_cmap='Reds', std_cmap='Blues', step=1,
                       rows=False, figsize=None, assimilated_data=None):
    """
    Create an animation of flow fields from multiple datasets.
    Inputs:
    - datsets: Dict of datasets, containing flow field data (Each of shape: Ny x Nx x Nt). 
        The keys are used as titles for each subplot.
    - time: 1-D array of time values corresponding to the last axis of each dataset.
    - n_frames: Number of frames in the animation (used only when assimilated_data is None).
    - cmaps: List of colormaps for each dataset.
    - rms_cmap: Colormap for RMS datasets.
    - std_cmap: Colormap for standard deviation datasets.
    - assimilated_data: dict with keys:
        - 't_obs': array-like of observation *time values* at which data are assimilated.
          When provided, these drive the animation frames (n_frames / step are ignored).
        - 'xy': array of shape (N_sensors, 2) with sensor coordinates [x_col, y_row].
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
    """
    Returns ((fig_width, fig_height), ncols, nrows) where figsize respects the domain
    aspect ratio and total_width constraint across the subplot grid.

    Parameters:
    - domain: [xmin, xmax, ymin, ymax]
    - total_subplots: number of subplots to arrange
    - max_cols: maximum columns (prevents excessively wide layouts)
    - total_width: desired figure width in inches

    Returns:
    - Tuple: ((fig_width, fig_height), ncols, nrows)
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



def get_cropped_indices(original_grid, 
                        original_domain, 
                        domain_of_interest, 
                        down_sample=None):

    # Extract original and DOI boundaries
    Nx, Ny = original_grid
    x_min, x_max, y_min, y_max = original_domain
    doi_x_min, doi_x_max, doi_y_min, doi_y_max = domain_of_interest

    # Generate 1D spatial grids for original domain
    x = np.linspace(x_min, x_max, Nx)
    y = np.linspace(y_min, y_max, Ny)

    # Find indices within domain_of_interest along each axis
    x_idx = np.where((x >= doi_x_min) & (x <= doi_x_max))[0]
    y_idx = np.where((y >= doi_y_min) & (y <= doi_y_max))[0]


    if len(x_idx) == 0 or len(y_idx) == 0:
        raise ValueError('Domain of interest does not overlap with original domain grid.')


    if down_sample is not None:
        if isinstance(down_sample, int):
            down_sample = [down_sample]
        if len(down_sample) == 1:
            step_x = step_y = down_sample[0]
        elif len(down_sample) == 2:
            step_x, step_y = down_sample
        else:
            raise AssertionError(f'Too many downsample entries: {down_sample}')

        x_idx = x_idx[::step_x]
        y_idx = y_idx[::step_y]


    return np.ix_(x_idx, y_idx)




def crop_data_to_domain_of_interest(data,
                                    original_domain: Union[list[float], tuple],
                                    domain_of_interest:  Union[list[float], tuple],
                                    down_sample: Optional[Union[int, list, tuple]] = None):
        """
        Adjust the domain and grid shape for a given dataset.

        Args:
            - data: The dataset to process. Shape: (Nu, Nt), Nx, Ny
            - original_domain: (x_min, x_max, y_min, y_max) tuple for the entire data domain
            - domain_of_interest: (x_min, x_max, y_min, y_max) tuple specifying subdomain to crop to
            - down_sample: Optional down-sampling factor(s) for the cropped data

        Returns:
            - Processed dataset, new domain, new grid shape, and the index mapping.
        """

        if data.ndim <= 4 and data.ndim >= 2:
            original_grid = list(data.shape[-2:])
        else:
            raise ValueError(f'data input shape must be [(Nu, Nt) x Nx x Ny], got {data.shape}')

        # # Extract original and DOI boundaries

        cropped_grid_indices = get_cropped_indices(original_grid, original_domain, 
                                                   domain_of_interest, down_sample)

        data_cropped = data[..., cropped_grid_indices[0], cropped_grid_indices[1]].copy()

        return data_cropped, cropped_grid_indices


