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

import glob
import contextlib
from PIL import Image

rng = np.random.default_rng(6)



def add_noise_to_flow(U, V, noise_level=0.05, noise_type="gauss", spatial_smooth=0):
    """
    Adds noise to a 3D velocity field (Nt x Nx x Ny).

    Args
        U, V : numpy.ndarray
            3D arrays representing the velocity components (Nt x Nx x Ny).
        
        noise_level : float, optional, default=0.05
            The standard deviation of the noise as a fraction of the maximum absolute velocity.
        
        noise_type : str, optional, default="gauss"
            The type of noise to apply:
            - 'gauss': Gaussian (white) noise.
            - 'pink', 'brown', 'blue', 'violet': Colored noise.

        spatial_smooth : float, optional, default=0
            Standard deviation for Gaussian smoothing (0 means no smoothing).

    Returns
        U_noisy, V_noisy : numpy.ndarray
            Noisy velocity fields with the same shape as U and V.
    """
    rng = np.random.default_rng()  # Random generator

    U = U.copy()
    V = V.copy()
    
    U[np.isnan(U) | np.isinf(U)] = 0  # Replace NaN/Inf values
    V[np.isnan(V) | np.isinf(V)] = 0  # Replace NaN/Inf values

    # Compute noise amplitude
    max_vel = max(np.max(np.abs(U)), np.max(np.abs(V)))
    noise_amp = noise_level * max_vel

    def generate_noise(shape, noise_type):
        """Generates 3D noise (Nt x Nx x Ny)."""
        # Nt, Nx, Ny = shape
        if noise_type == "gauss":
            return rng.normal(scale=noise_amp, size=shape)
        else:
            noise_white = np.fft.rfftn(rng.standard_normal(shape)) * noise_amp
            S = colour_noise(shape, noise_colour=noise_type)
            noise_colored = noise_white * S
            return np.fft.irfftn(noise_colored, s=shape).real


    # Generate noise
    noise_U = generate_noise(U.shape, noise_type)
    noise_V = generate_noise(V.shape, noise_type)

    # Apply optional spatial smoothing
    if spatial_smooth > 0:
        sigma = (0, spatial_smooth, spatial_smooth)
        noise_U = ndimage.gaussian_filter(noise_U, sigma=sigma)
        noise_V = ndimage.gaussian_filter(noise_V, sigma=sigma)

    # Add noise to velocity field
    
    U_noisy = U + noise_U
    V_noisy = V + noise_V

    return U_noisy, V_noisy



def set_working_directories(subfolder=''):
    
    tutorial = 'tutorials' in os.getcwd()

    def find_root(start_dir):
            dir_path = os.path.abspath(start_dir)
            while True:
                if 'src' in os.listdir(dir_path) and 'scripts' in os.listdir(dir_path):
                    return dir_path
                parent = os.path.dirname(dir_path)
                if parent == dir_path:  # Reached system root
                    return None
                dir_path = parent

    # Find the project root directory
    project_root = find_root(os.getcwd()) 
    if project_root is None:
        raise FileNotFoundError("Project root directory not found. Ensure you are in the correct directory structure.")

    #  Set results and fgures folders
    if tutorial:
        results_folder = os.path.join(project_root, 'results/tutorials', subfolder)
        figs_folder = os.path.join(project_root, 'docs/figs', subfolder)
    else:
        results_folder = os.path.join(project_root, 'results', subfolder)
        figs_folder = os.path.join(project_root, 'results/figs', subfolder)

    #  Set data folder. Note: some data is not provided in the repository.  
    # if os.path.isdir('/Users/andreanovoa'):
    #     data_folder = os.path.join('/Users/andreanovoa', 'data', subfolder)
    # elif os.path.isdir('/Users/anovoama/'):
    #     data_folder = os.path.join('/Users/anovoama', 'data', subfolder)
    # else:
    data_folder = os.path.join(project_root, 'data', subfolder)



    # # Check data folder exists
    # if not os.path.isdir(data_folder):
    #     raise FileNotFoundError(f"Data folder {data_folder} does not exist.")


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
    print('Test if loaded file is valid', end='')
    for key, val in params_dict.items():
        if hasattr(load_case, key):
            print('\n\t', key, val, getattr(load_case, key), end='')
            if len(np.shape([val])) == 1:
                if getattr(load_case, key) != val:
                    print('\t <--- Re-init model!')
                    return False
            else:
                if any([x1 != x2 for x1, x2 in zip(getattr(load_case, key), val)]):
                    print('\t <--- Re-init model!')
                    return False
    return True


def categorical_cmap(nc, nsc, cmap="tab10", continuous=False):
    # number of categories(nc) and the number of subcategories(nsc)
    # and returns a colormap with nc * nsc different colors, where for
    # each category there are nsc colors of same hue.

    if nc > plt.get_cmap(cmap).N:
        raise ValueError("Too many categories for colormap.")
    if continuous:
        ccolors = plt.get_cmap(cmap)(np.linspace(0, 1, nc))
    else:
        ccolors = plt.get_cmap(cmap)(np.arange(nc, dtype=int))
    cols = np.zeros((nc * nsc, 3))
    for i, c in enumerate(ccolors):
        chsv = mpl.colors.rgb_to_hsv(c[:3])
        arhsv = np.tile(chsv, nsc).reshape(nsc, 3)
        arhsv[:, 1] = np.linspace(chsv[1], 0.25, nsc)
        arhsv[:, 2] = np.linspace(chsv[2], 1, nsc)
        rgb = mpl.colors.hsv_to_rgb(arhsv)
        cols[i * nsc:(i + 1) * nsc, :] = rgb
    return cols


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


def interpolate(t_y, y, t_eval, fill_values=None):
    # interpolator = PchipInterpolator(t_y, y)

    if fill_values is None:
        fill_values = (y[0], y[-1])

    interpolator = interp1d(t_y, y,
                            axis=0,  # interpolate along columns
                            bounds_error=False,
                            kind='linear',
                            fill_value=fill_values)
    return interpolator(t_eval)


def getEnvelope(timeseries_x, timeseries_y, fill_value=0):
    peaks, peak_properties = find_peaks(timeseries_y, distance=200)
    u_p = interp1d(timeseries_x[peaks], timeseries_y[peaks], bounds_error=False, fill_value=fill_value)
    return u_p


def save_to_pickle_file(filename, *args):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        for arg in args:
            pickle.dump(arg, f)


def load_from_pickle_file(filename):
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



def folder_to_gif(folder, img_type='.png', gif_name='movie.gif'):
    """
    Convert all the images inside a folder into a gif. 
    
    """
    if img_type[0] != '.':
        img_type = f'.{img_type}'
    
    fp_in = folder + f'*{img_type}'
    fp_out = folder + gif_name
    
    # use exit stack to automatically close opened images
    with contextlib.ExitStack() as stack:
    
        # lazily load images
        imgs = (stack.enter_context(Image.open(f)) for f in sorted(glob.glob(fp_in)))
    
        # extract  first image from iterator
        img = next(imgs)
    
        # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
        img.save(fp=fp_out, 
                 format='GIF', 
                 append_images=imgs,
                 save_all=True, 
                 duration=200, 
                 loop=0)


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
    norm = mpl.colors.Normalize(vmin=true_RMS, vmax=1.5)
    cmap = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis)
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


def get_error_metrics(results_folder):
    print('computing error metrics...')
    out = dict(Ls=[], ks=[])

    L_dirs, k_files = [], []
    LLL = os.listdir(results_folder)
    for Ldir in LLL:
        if os.path.isdir(results_folder + Ldir + '/') and Ldir[0] == 'L':
            L_dirs.append(results_folder + Ldir + '/')
            out['Ls'].append(float(Ldir.split('L')[-1]))

    for ff in os.listdir(L_dirs[0]):
        k = float(ff.split('_k')[-1])
        out['ks'].append(k)
        k_files.append(ff)

    # sort ks and Ls
    idx_ks = np.argsort(np.array(out['ks']))
    out['ks'] = np.array(out['ks'])[idx_ks]
    out['k_files'] = [k_files[i] for i in idx_ks]

    idx = np.argsort(np.array(out['Ls']))
    out['L_dirs'] = [L_dirs[i] for i in idx]
    out['Ls'] = np.array(out['Ls'])[idx]

    # Output quantities
    keys = ['R_biased_DA', 'R_biased_post',
            'C_biased_DA', 'C_biased_post',
            'R_unbiased_DA', 'R_unbiased_post',
            'C_unbiased_DA', 'C_unbiased_post']
    for key in keys:
        out[key] = np.empty([len(out['Ls']), len(out['ks'])])

    print(out['Ls'])
    print(out['ks'])

    ii = -1
    for Ldir in out['L_dirs']:
        ii += 1
        print('L = ', out['Ls'][ii])
        jj = -1
        for ff in out['k_files']:
            jj += 1
            # Read file
            truth, filter_ens = load_from_pickle_file(Ldir + ff)[1:]
            truth = truth.copy()

            print('\t k = ', out['ks'][jj], '({}, {})'.format(filter_ens.bias.L, filter_ens.regularization_factor))
            # Compute biased and unbiased signals
            y, t = filter_ens.get_observable_hist(), filter_ens.hist_t
            b, t_b = filter_ens.bias.hist, filter_ens.bias.hist_t
            y_mean = np.mean(y, -1)

            # Unbiased signal error
            if hasattr(filter_ens.bias, 'upsample'):
                y_unbiased = interpolate(t, y_mean, t_b) + b
                y_unbiased = interpolate(t_b, y_unbiased, t)
            else:
                y_unbiased = y_mean + b

            # if jj == 0:
            N_CR = int(filter_ens.t_CR // filter_ens.dt)  # Length of interval to compute correlation and RMS
            i0 = np.argmin(abs(t - truth['t_obs'][0]))  # start of assimilation
            i1 = np.argmin(abs(t - truth['t_obs'][-1]))  # end of assimilation

            # cut signals to interval of interest
            y_mean, t, y_unbiased = y_mean[i0 - N_CR:i1 + N_CR], t[i0 - N_CR:i1 + N_CR], y_unbiased[i0 - N_CR:i1 + N_CR]

            if ii == 0 and jj == 0:
                i0_t = np.argmin(abs(truth['t'] - truth['t_obs'][0]))  # start of assimilation
                i1_t = np.argmin(abs(truth['t'] - truth['t_obs'][-1]))  # end of assimilation
                y_truth, t_truth = truth['y'][i0_t - N_CR:i1_t + N_CR], truth['t'][i0_t - N_CR:i1_t + N_CR]
                y_truth_b = y_truth - truth['b'][i0_t - N_CR:i1_t + N_CR]

                out['C_true'], out['R_true'] = CR(y_truth[-N_CR:], y_truth_b[-N_CR:])
                out['C_pre'], out['R_pre'] = CR(y_truth[:N_CR], y_mean[:N_CR])
                out['t_interp'] = t[::N_CR]
                scale = np.max(y_truth, axis=0)
                for key in ['error_biased', 'error_unbiased']:
                    out[key] = np.empty([len(out['Ls']), len(out['ks']), len(out['t_interp']), y_mean.shape[-1]])

            # End of assimilation
            for yy, key in zip([y_mean, y_unbiased], ['_biased_DA', '_unbiased_DA']):
                C, R = CR(y_truth[-N_CR * 2:-N_CR], yy[-N_CR * 2:-N_CR])
                out['C' + key][ii, jj] = C
                out['R' + key][ii, jj] = R

            # After Assimilaiton
            for yy, key in zip([y_mean, y_unbiased], ['_biased_post', '_unbiased_post']):
                C, R = CR(y_truth[-N_CR:], yy[-N_CR:])
                out['C' + key][ii, jj] = C
                out['R' + key][ii, jj] = R

            # Compute mean errors
            b_obs = y_truth - y_mean
            b_obs_u = y_truth - y_unbiased
            ei, a = -N_CR, -1
            while ei < len(b_obs) - N_CR - 1:
                a += 1
                ei += N_CR
                out['error_biased'][ii, jj, a, :] = np.mean(abs(b_obs[ei:ei + N_CR]), axis=0) / scale
                out['error_unbiased'][ii, jj, a, :] = np.mean(abs(b_obs_u[ei:ei + N_CR]), axis=0) / scale

            save_to_pickle_file(results_folder + 'CR_data', out)





def create_Lorenz63_dataset(noise_level=0.02, num_lyap_times=300, seed=0, **kwargs):
    from src.models_physical import Lorenz63
    # Load or create training data from the Lorenz 63 model
    data_folder = set_working_directories('Lorenz/')[0]

    model = Lorenz63(**kwargs)




    # Default filename
    filename = f"{''.join([f'{key}{val:.2f}_' for key, val in model.get_default_params.items()])}Nlyap{num_lyap_times}_noise{noise_level}_seed{seed}"

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

    return dataset



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
        if zipfile.is_zipfile(filename):
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

            dest_path = os.path.join(output_folder, target)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with zip_ref.open(member) as source, open(dest_path, 'wb') as target_file:
                target_file.write(source.read())
        print(f"Unzipped '{zip_name}.zip' in '{output_folder}'")


def get_annular_data(data_folder=None):
    """
    Download and unzip the annular data from Zenodo if not already present.
    """
    if data_folder is None:
        data_folder = set_working_directories('annular/')

    zip_file = os.path.join(data_folder, 'annular_data.zip')
    zenodo_dir = "https://zenodo.org/records/15609832/files"
    
    download_zenodo_file(f'{zenodo_dir}/annular.zip?download=1', data_folder, 
                            filename='annular_data.zip')

    download_zenodo_file(f'{zenodo_dir}/README.md?download=1', data_folder)

    

def get_circle_data(data_folder=None):
    """
    Download and unzip the cylindre flow data from Zenodo if not already present.
    """
    if data_folder is None:
        data_folder = set_working_directories('circle/')

    zenodo_dir = "https://zenodo.org/records/15623774/files/"
    
    download_zenodo_file(f'{zenodo_dir}/circle_re_100.mat?download=1"', data_folder, 
                            filename='circle_re_100.mat')

    download_zenodo_file(f'{zenodo_dir}/README.md?download=1', data_folder)

