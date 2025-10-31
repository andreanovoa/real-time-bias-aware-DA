# %%
import numpy as np
from bias import *


import numpy as np
import os # Need this for create_observations

from model import Model

from utils import load_from_pickle_file, save_to_pickle_file, load_from_mat_file, colour_noise, fun_PSD

rng = np.random.default_rng() 




class Observations():

    # Class Attributes (Defaults)
    noise_type = 'gauss, add'
    Nt_obs = 20
    std_obs = 0.05
    
    t_min = 0.0 # Start time for the whole simulation

    # Instance Attributes (Defaults - Will be set in __init__)
    post_processed = False 
    manual_bias = None

    t_min = 0.0
    t_max = None
    t_start = None
    t_stop = None

    true_model = None
    data_folder=None
    add_noise = True

    def __init__(self, model, **kwargs):
        """
        Initializes the Observations object, loading or creating truth data,
        applying bias, adding noise, and interpolating to observation times.
        """
        # 1. Update instance attributes with any passed kwargs
        model_dict = kwargs.copy()  # Copy to avoid modifying original kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                del model_dict[key]  # Remove from kwargs to avoid passing it down'

        if self.data_folder is None:

            self.data_folder = os.path.join(os.getcwd() + '/data/')

        # 2. Generate or Load Truth Data
        self.y_true, self.t_true, self.name_truth = self._create_observations(model, **model_dict)

        # 3. Add noise and bias if requested
        self.b, self.name_bias = self._get_bias()
        self.y_true += self.b 
        self.y_raw = self._apply_noise()

        # 4. Compute Observation Times
        # Adjust all times by t_min if t_min > 0 (to start t_true[0] at 0)
        if self.t_min > 0:
            t_true -= self.t_min
            if self.t_start is not None:
                self.t_start -= self.t_min
            if self.t_stop is not None:
                self.t_stop -= self.t_min

        # Calculate indices
        start_idx = np.searchsorted(self.t_true, self.t_start)
        stop_idx = np.searchsorted(self.t_true, self.t_stop, side='right') - 1
        
        # Original logic: step from start_idx to stop_idx (inclusive)
        obs_idx = np.arange(start_idx, stop_idx + 1, self.Nt_obs, dtype=int)
        
        # 5. Save Final observation data
        self.t_obs = self.t_true[obs_idx]
        self.y_obs = self.y_raw[obs_idx]
        self.dt_obs = self.Nt_obs * (self.t_true[1] - self.t_true[0])


    def _get_bias(self):
        """Applies manual or default bias to zero.
            Options:
            - manual_bias = None (no bias)
            - manual_bias = string (predefined bias types: 'linear', 'periodic', 'time', 'cosine')
            - manual_bias = function(y_true, t_true) returning (b_true, name_bias)
            Returns: 
            - b_true: bias array, 
            - name_bias: string description of bias type
        """
        y_true = self.y_true.copy()
        t_true = self.t_true.copy()
        b_true = y_true * 0.
        manual_bias = self.manual_bias
        
        if manual_bias is None:
            name_bias = 'No_bias'

        elif isinstance(manual_bias, str):
            name_bias = manual_bias
            # Use cleaner np.max(y_true, axis=0) or np.ptp(y_true, axis=0) for scaling
            y_max_over_time = np.max(y_true, axis=0)
            t_expanded = np.expand_dims(t_true, axis=-1)
            
            if manual_bias == 'time':
                b_true = .4 * y_true * np.sin((t_expanded * np.pi * 2) ** 2)
            elif manual_bias == 'periodic':
                # Use a small constant to prevent division by zero in case y_max_over_time is 0
                max_safe = y_max_over_time + 1e-10 
                b_true = 0.2 * y_max_over_time * np.cos(2 * y_true / max_safe)
            elif manual_bias == 'linear':
                b_true = .1 * y_max_over_time + .3 * y_true
            elif manual_bias == 'cosine':
                b_true = np.cos(y_true)
            else:
                raise ValueError(f"Bias '{manual_bias}' not recognized. Choose [linear, periodic, time, cosine].")
        else:
            # The manual bias is a function of state and/or time
            b_true, name_bias = manual_bias(y_true, t_true)
        
        return b_true, name_bias


    def _apply_noise(self):
        """Adds noise to the biased truth data.
            Options:
            - if add_noise is False, returns clean data
            - std_obs: standard deviation of the noise (as fraction of max signal)
            - noise_type: 'gauss' or 'coloured' for Gaussian or coloured noise
            - 'add' or 'mult' for additive or multiplicative noise
            Returns: noisy data.
        """

        y_true = self.y_true.copy()
        if not self.add_noise:
            return y_true 
        else:
            # def create_noisy_signal(y_clean, noise_level=0.1, noise_type='gauss, add'):
            if y_true.ndim == 2:
                y_true = np.expand_dims(y_true, -1)

            Nt, q, L = y_true.shape
            y_noisy = y_true.copy()

            for ll in range(L):
                # Type/color of the noise
                if 'gauss' in self.noise_type.lower():
                    noise = rng.multivariate_normal(np.zeros(q), np.eye(q) * self.std_obs ** 2, Nt)
                else:
                    i0 = Nt % 2 != 0  # Add extra step if odd
                    noise = np.zeros([Nt, q])
                    for ii in range(q):
                        noise_white = np.fft.rfft(rng.standard_normal(Nt + i0) * self.std_obs)
                        S = colour_noise(Nt + i0, noise_colour=self.noise_type)
                        S = noise_white * S  # Normalize S
                        noise[:, ii] = np.fft.irfft(S)[i0:]  # transform back into time domain
                # Additive or multiplicative noise
                if 'add' in self.noise_type.lower():
                    y_noisy[:, :, ll] += noise * np.max(abs(y_true[:, :, ll]), axis=0)
                else:
                    y_noisy[:, :, ll] += noise * y_noisy[:, :, ll]

            y_noisy = y_noisy.squeeze()

            if y_noisy.ndim == 1:
                y_noisy = np.expand_dims(y_noisy, axis=-1)

            return y_noisy



    def _create_observations(self, model, **kwargs):

        def object_is_a_Model(obj):
            if isinstance(obj, type):
                return issubclass(obj, Model)
            else:
                return isinstance(obj, Model)


        """Creates or loads truth data from a model or file."""
        if object_is_a_Model(model):

            # Instantiate the model if a class is provided
            if isinstance(model, type):
                model = model(**kwargs)  

            # Define Time Windows based on Model properties
            self.t_start = self.t_start or model.t_transient
            self.t_stop = self.t_stop or (self.t_start + 3 * model.t_CR)
            self.t_max = self.t_max or (self.t_stop + self.t_start) 

            # ============================================================
            # Add key input_parameters to filename
            suffix = ''
            for key, val in kwargs.items(): 
                if key in model.alpha_labels.keys():
                    if type(val) is str:
                        suffix += val + '_'
                    else:
                        suffix += key + '{:.2e}'.format(val) + '_'


            name_truth = f'Truth_{model.name}_{suffix}'
            full_path = os.path.join(self.data_folder, name_truth) if self.data_folder else name_truth
            
            if os.path.isfile(full_path):
                true_model = load_from_pickle_file(full_path)
                print('Loaded true data model: ' + name_truth, true_model)
            else:
                true_model = model.copy()

            # Forecast to t_max if necessary and save file
            if true_model.hist_t[-1] < self.t_max:
                
                Nt_forecast = int((self.t_max - true_model.hist_t[-1]) / true_model.dt) + 1
                psi, t = true_model.time_integrate(Nt_forecast)
                true_model.update_history(psi, t)

                true_model.close()
                if len(full_path) > 0:
                    save_to_pickle_file(full_path, true_model)

            
            # ============================================================
            # Retrieve observables
            y_true = true_model.get_observable_hist()
            y_true = np.squeeze(y_true, axis=-1)
            t_true = true_model.hist_t
            name_truth = name_truth
            

        elif isinstance(model, str):            
            # Load Data from File
            full_path = os.join(self.data_folder, model) if self.data_folder else model
            try:
                if 'rijke' in model:
                    mat = load_from_mat_file(full_path)
                    y_true, t_true = [mat[key].transpose() for key in ['p_mic', 't_mic']]                    
                elif 'annular' in model:
                    mat = load_from_mat_file(full_path)
                    y_raw, y_true, t_true = [mat[key] for key in ['y_raw', 'y_filtered', 't']]
                else:
                    raise FileNotFoundError
            except FileNotFoundError:
                raise FileNotFoundError(f'File {model} not defined in folder {self.data_folder}.')
            
            name_truth = 'Truth_Exp_' + model.split('/')[-1]
            
        else:
            raise ValueError("Model must be either a Model instance or a string filename.")

        return y_true, t_true, name_truth



    @staticmethod
    def plot_truth(case, fig_width=10, window=None, xlim=None, plot_time=False, Nq=None, filename=None, f_max=None):
    
        
        y_raw, y_true, t_true, y_obs, t_obs, b = [getattr(case, key) for key in ['y_raw', 'y_true', 't_true', 'y_obs', 't_obs', 'b']]
        

        dt = t_true[1] - t_true[0]
        plot_bias = np.mean(b ** 2) > 0.

        if plot_bias:
            noise = y_raw - (y_true - b)
        else:
            noise = y_raw - y_true
        if Nq is None:
            Nq = y_true.shape[1]
        if t_obs is None:
            t0 = 0
            if window is None:
                t1 = int(case.true_model.t_transient // dt)
            else:
                t1 = int(window // dt)
        else:
            t0 = int((t_obs[0]) // dt)
            if window is None:
                if case.true_model is not None:
                    t1 = int((t_obs[-1] + case.true_model.t_CR) // dt)
                    t1 = min(t1, len(t_true) - 1)
                else:
                    t1 = int((t_obs[-1]) // dt)
            else:
                t1 = int((t_obs[0] + window) // dt)

        if xlim is None:
            xlim = [t_true[t0], t_true[t1]]



        max_y = np.max(abs(y_raw[:t1 - t0]))

        fig1 = plt.figure(figsize=(fig_width, 2 * Nq), layout="constrained")
        subfigs = fig1.subfigures(nrows=1, ncols=4, width_ratios=[2, 0.5, 1, 1])
        labels = ['Raw', 'True', 'Difference']
        y_labels = ['$\\tilde{y}, y$', '', '$(\\tilde{y}-y)$']
        cols = ['tab:blue', 'mediumseagreen', 'tab:purple']
        c_b = 'tab:orange'

        ax_01 = subfigs[0].subplots(Nq, 2, sharex='all', sharey='row')
        ax_4 = subfigs[-1].subplots(Nq, 1, sharex='all', sharey='row')

        # Plot zoomed timeseries of raw, post-processed and noise
        if Nq == 1:
            axss = [[ax_01[0]], [ax_01[1]], [ax_4]]
        else:
            axss = [ax_01[:, 0], ax_01[:, 1], ax_4]

        dashes = (10, 1)
        for ax, yy, ttl, lbl, c in zip(axss, [y_raw, y_true, noise], labels, y_labels, cols):
            ax[0].set(title=ttl)
            ax[-1].set(xlabel='$t$', xlim=xlim)
            for qi in range(Nq):
                ax[qi].plot(t_true, yy[:, qi], color=c)
                ax[qi].axhline(np.mean(yy[:, qi]), color=c)
                if ttl[0] == 'R' and y_obs is not None:
                    ax[qi].plot(t_obs, y_obs[:, qi], 'ro', ms=3)
                elif ttl[0] == 'T' and plot_bias:
                    ax[qi].plot(t_true, yy[:, qi] - b[:, qi], color=c_b, dashes=dashes)
                    ax[qi].axhline(np.mean(yy[:, qi] - b[:, qi]), color=c_b, dashes=dashes)
                if len(lbl) > 1:
                    ax[qi].set(ylabel=lbl + '$_{}$'.format(qi))

        # Plot probability density src and power spectral densities
        ax_pdf = subfigs[1].subplots(Nq, 1, sharey='row', sharex='all')
        ax_PSD = subfigs[2].subplots(Nq, 1, sharex='all', sharey='all')
        if Nq == 1:
            ax_pdf = [ax_pdf]
            ax_PSD = [ax_PSD]

        binwidth = 0.05 * max_y
        bins = np.arange(-max_y, max_y + binwidth, binwidth)
        for yy, ttl, lbl, c in zip([y_raw, y_true], labels[:2], y_labels[:2], cols[:2]):
            ax_pdf[0].set(title='PDF')
            ax_pdf[-1].set(xlabel='$p$')
            ax_PSD[-1].set(xlabel='$f$')
            for qi in range(Nq):
                ax_pdf[qi].hist(yy[:, qi], bins=bins, density=True, orientation='horizontal',
                                color=c, label=lbl + '$_{}$'.format(qi), histtype='step')
                ax_pdf[qi].hist(yy[:, qi], bins=bins, density=True, orientation='horizontal',
                                color=c, label=lbl + '$_{}$'.format(qi), histtype='stepfilled', alpha=.7)
                
                ax_pdf[qi].hist(y_obs[:, qi], bins=bins, density=True, orientation='horizontal', ls='--',
                                color='red', histtype='step',lw=1)
                if Nq == 1:
                    ylims = ax_01[qi].get_ylim()
                else:
                    ylims = ax_01[qi, 0].get_ylim()
                ax_pdf[qi].set(yticklabels=[], ylim=ylims)
            f, PSD = fun_PSD(dt, yy.squeeze())
            for qi in range(Nq):
                ax_PSD[qi].semilogy(f, PSD[qi], color=c, label=lbl + '$_{}$'.format(qi))
            ax_PSD[0].set(title='PSD', xlim=[0, f_max])
        if plot_bias:
            f, PSD = fun_PSD(dt, (y_true - b).squeeze())
            for qi in range(Nq):
                ax_pdf[qi].hist(y_true[:, qi] - b[:, qi], bins=bins, density=True, orientation='horizontal',
                                color=c_b, histtype='step', lw=.5)
                ax_PSD[qi].semilogy(f, PSD[qi], color=c_b, dashes=dashes)

        # Plot full timeseries if requested
        figs2 = []
        if plot_time:
            for yy, name, c in zip([y_raw, y_true], labels[:2], cols[:2]):
                y_true, t_true = [zz[t0:] for zz in [yy, t_true]]
                max_y = np.max(abs(y_true))
                fig2 = plt.figure(figsize=(12, 2 * Nq), layout="constrained")
                subfigs = fig2.subfigures(nrows=1, ncols=2, width_ratios=[1, 0.5])
                for sf, xlims in zip(subfigs, [(t_true[0], t_true[-1]), (t_true[-1000], t_true[-1])]):
                    ax = sf.subplots(Nq, 1, sharex='all')
                    if Nq == 1:
                        ax = [ax]
                    ax[0].set(title=name)
                    ax[-1].set(xlabel='$t$', xlim=xlims)
                    for qi in range(Nq):
                        ax[qi].plot(t_true, y_true[:, qi], color=c)
                        ax[qi].set(ylim=[-max_y, max_y])
                figs2.append(fig2)
        # Show or save plots
        if filename is None:
            plt.show()
        else:
            if filename[-len('.pdf'):] != '.pdf':
                filename += '.pdf'
            os.makedirs('/'.join(filename.split('/')[:-1]), exist_ok=True)
            pdf_file = plt_pdf.PdfPages(filename)
            pdf_file.savefig(fig1)
            plt.close(fig1)
            for fig in figs2:
                pdf_file.savefig(fig)
                plt.close(fig)
            pdf_file.close()  # Close results pdf
            



# %%
if __name__ == "__main__":

    from models_physical import Lorenz63
    truth = Observations(model=Lorenz63, t_start=10.0, t_stop=40.0, Nt_obs=10, std_obs=0.5, noise_type='gauss, add', manual_bias=None)


# %%
    truth.plot_truth(truth)

# %%
