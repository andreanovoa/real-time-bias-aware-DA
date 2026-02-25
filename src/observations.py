# %%
from typing import Type, Union
import numpy as np
import matplotlib.pyplot as plt
import os  

from model import Model

from utils import load_from_pickle_file, save_to_pickle_file, load_from_mat_file, colour_noise, fun_PSD
from typeguard import typechecked


rng = np.random.default_rng() 




class Observations():

    # Class Attributes (Defaults)
    noise_type = 'gauss, add'
    Nt_obs = 20
    noise_level = 0.05
    

    # Instance Attributes (Defaults - Will be set in __init__)
    post_processed = False 
    manual_bias = None

    t_min = 0.0  
    t_max = None
    t_start = None
    t_stop = None

    true_parameters = None
    results_folder = None
    add_noise = True

    _frozen = False

    @typechecked
    def __init__(self, model: Union[Model, Type[Model]], **kwargs):
        """
        Initializes the Observations object, loading or creating truth data,
        applying bias, adding noise, and interpolating to observation times.
        """
        # 1. Update instance attributes with any passed kwargs
        model_dict = kwargs.copy()  
        for key in kwargs.keys():
            if hasattr(Observations, key):
                setattr(self, key, model_dict.pop(key))

        # 2. Generate or Load Truth Data
        self.y_raw, self.y_true, self.t_true, self.name_truth = self._create_observations(model, **model_dict)
        self.dt = self.t_true[1] - self.t_true[0]

        # 3. Add noise and bias if requested (in this order and only if y_raw is None)
        self._set_bias()
        self._apply_noise()


        # 4. Compute Observation Times
        # Adjust all times by t_min if t_min > 0 (to start t_true[0] at 0)
        if self.t_min > 0:
            self.t_true -= self.t_min
            if self.t_start is not None:
                self.t_start -= self.t_min
            if self.t_stop is not None:
                self.t_stop -= self.t_min
        
        self.update_obs_idx(self.t_start, self.t_stop, self.Nt_obs)


        # Calculate indices
        self._frozen = True  # Freeze attributes to prevent further modification
        print('Observations initialized.')


    @property
    def obs_idx(self):  
        """Allows reading the calculated observation index."""
        if self._obs_idx is None:
             raise AttributeError("obs_idx has not been calculated yet. Call update_obs_idx first.")
        return self._obs_idx
        

        

    def update_obs_idx(self, t_start=None, t_stop=None, Nt_obs=None):
        """
        Calculates and updates the observation index based on the provided times and sampling rate.
        This is the dedicated method for modification.
        """

        # Use existing attributes if parameters are not provided
        for key, val in zip(['t_start', 't_stop', 'Nt_obs'], [t_start, t_stop, Nt_obs]):
            if val is None:                
                val = getattr(self, key)
            else:
                setattr(self, key, val)


        start_idx = np.searchsorted(self.t_true, self.t_start)
        stop_idx = np.searchsorted(self.t_true, self.t_stop, side='right') - 1

        self._obs_idx =  np.arange(start_idx, stop_idx + 1, self.Nt_obs, dtype=int)



    @property
    def dt_obs(self):
        return self.Nt_obs * self.dt
    
    @property
    def y_obs(self):
        return self.y_raw[self.obs_idx,...,0]
    
    @property
    def t_obs(self):
        return self.t_true[self.obs_idx]
    

    # --- Properties for Frozen Data ---

    @property
    def y_raw(self):
        return self._y_raw

    @y_raw.setter
    def y_raw(self, value):
        if self._frozen:
            raise AttributeError("Cannot modify 'y_raw': Attributes are frozen after initialization.")
        self._y_raw = value

    @property
    def y_true(self):
        return self._y_true

    @y_true.setter
    def y_true(self, value):
        if self._frozen:
            raise AttributeError("Cannot modify 'y_true': Attributes are frozen after initialization.")
        self._y_true = value

    @property
    def t_true(self):
        return self._t_true

    @t_true.setter
    def t_true(self, value):
        if self._frozen:
            raise AttributeError("Cannot modify 't_true': Attributes are frozen after initialization.")
        self._t_true = value

    @property
    def name_truth(self):
        return self._name_truth

    @name_truth.setter
    def name_truth(self, value):
        if self._frozen:
            raise AttributeError("Cannot modify 'name_truth': Attributes are frozen after initialization.")
        self._name_truth = value



    def _set_bias(self):
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
            if self.y_raw is None:
                self.y_raw = y_true.copy()


        elif isinstance(manual_bias, str):
            print(f'...Applying manual bias: {manual_bias}')

            name_bias = manual_bias
            # Use cleaner np.max(y_true, axis=0) or np.ptp(y_true, axis=0) for scaling
            y_max_over_time = np.max(y_true, axis=0)
            t_expanded = t_true[:, np.newaxis, np.newaxis]  # Expand t_true for broadcasting
            
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
            print('...Applying user-defined manual bias')
            # The manual bias is a function of state and/or time
            b_true, name_bias = manual_bias(y_true, t_true)

        # Update true data to include bias
        self.y_true += b_true
        self.b_true, self.name_bias = b_true, name_bias


    def _apply_noise(self):
        """Adds noise to the biased truth data.
            Options:
            - if add_noise is False, returns clean data
            - noise_level: standard deviation of the noise (as fraction of max signal)
            - noise_type: 'gauss' or 'coloured' for Gaussian or coloured noise
            - 'add' or 'mult' for additive or multiplicative noise
            Returns: noisy data.
        """


        if self.add_noise:
            print(f'Adding noise: {self.noise_type}')

            
            if self.y_raw is None:
                y_clean = self.y_true.copy()
            else:
                y_clean = self.y_raw.copy()


            Nt, q, L = y_clean.shape

            # Initialize raw data with clean data
            self.y_raw = np.atleast_3d(y_clean)  
            for ll in range(L):
                # Type/color of the noise
                if 'gauss' in self.noise_type.lower():
                    noise = rng.multivariate_normal(np.zeros(q), np.eye(q) * self.noise_level ** 2, Nt)
                else:
                    i0 = Nt % 2 != 0  # Add extra step if odd
                    noise = np.zeros([Nt, q])
                    for ii in range(q):
                        noise_white = np.fft.rfft(rng.standard_normal(Nt + i0) * self.noise_level)
                        S = colour_noise(Nt + i0, noise_colour=self.noise_type)
                        S = noise_white * S  # Normalize S
                        noise[:, ii] = np.fft.irfft(S)[i0:]  # transform back into time domain
                # Additive or multiplicative noise
                if 'add' in self.noise_type.lower():
                    self.y_raw[:, :, ll] += noise * np.max(abs(y_clean[:, :, ll]), axis=0)
                else:
                    self.y_raw[:, :, ll] += noise * y_clean[:, :, ll]
        else:
            self.y_raw = np.atleast_3d(self.y_raw)  # Ensure y_raw is at least 3D
        


    def _create_observations(self, model, **kwargs):
        """Creates or loads truth data from a model or file.
            - kwargs: parameters to instantiate the model if a class is provided
            Returns:
            - y_raw: raw data (None if loaded from file)
            - y_true: true observable data
            - t_true: time array
            - name_truth: string description of truth data
        """

        y_raw = None  # Initialize y_raw to None. Only set if loaded from file (i.e., real data).


        def object_is_a_Model(obj):
            if isinstance(obj, type):
                return issubclass(obj, Model)
            else:
                return isinstance(obj, Model)


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
            name_truth = f'Truth_{model.filename}'

            if self.results_folder is not None:
                full_path = os.path.join(self.results_folder, name_truth) 
                
                if os.path.isfile(full_path):
                    true_model = load_from_pickle_file(full_path)
                    print('Loaded true data model: ' + name_truth, true_model)
                else:
                    true_model = model.copy()
            else:
                true_model = model.copy()

            # Forecast to t_max if necessary and save file
            if true_model.hist_t[-1] < self.t_max:
                
                Nt_forecast = int((self.t_max - true_model.hist_t[-1]) / true_model.dt) + 1
                psi, t = true_model.time_integrate(Nt_forecast)
                true_model.update_history(psi, t)

                true_model.close()
                
                if self.results_folder is not None:
                    save_to_pickle_file(full_path, true_model)

            # ============================================================
            # Retrieve observables
            y_true = true_model.get_observable_hist()
            t_true = true_model.hist_t
            name_truth = name_truth
            self.true_parameters = true_model.alpha0
            

        elif isinstance(model, str):            
            # Load Data from File
            full_path = os.path.join(self.results_folder, model) if self.results_folder else model
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
                raise FileNotFoundError(f'File {model} not defined in folder {self.results_folder}.')
            
            name_truth = 'Truth_Exp_' + model.split('/')[-1]
            
        else:
            raise ValueError("Model must be either a Model instance or a string filename.")

        # Ensure y_true is at least 3D: (Nt, Nq, L)
        y_true = np.atleast_3d(y_true)

        return y_raw, y_true, t_true, name_truth

#  ====================================================================================================================================================================================
    # Plotting methods
#  ====================================================================================================================================================================================
    @staticmethod
    def plot_truth(case, Nq=None, fig_width=12, window=None, f_max=None):
        """
        Method to plot raw, true, difference time series, PDF, and PSD.
        Assumes Nq=4 based on the example image.
        """
        
        # 1. Data Extraction and Setup
        y_raw, y_true, t_true, y_obs, t_obs, b = [
            getattr(case, key).squeeze() for key in ['y_raw', 'y_true', 't_true', 'y_obs', 't_obs', 'b_true']
        ]

        if y_true.ndim == 1:
            y_true = y_true[:, np.newaxis]
            y_raw = y_raw[:, np.newaxis]
            if y_obs is not None:
                y_obs = y_obs[:, np.newaxis]


        dt = t_true[1] - t_true[0]
        # Calculate noise: Difference between raw and true signal
        noise = y_raw - y_true


        if Nq is None:
            Nq = y_true.shape[1]

        # Compute PSDs
        # find first index for t_obs
        t0_idx = np.argmin(np.abs(case.t_true - case.t_obs[0]))

        nt_PSD = int((len(t_true) - t0_idx) // 2)
        f_raw, PSD_raw = fun_PSD(dt, y_raw[t0_idx:nt_PSD + t0_idx])
        _, PSD_true = fun_PSD(dt, y_true[t0_idx:nt_PSD + t0_idx])
        
        # Determine plotting time window (simplified: use the first X data points if no window is given)
        # Using a simplified window selection for demonstration:
        if window is None:
            # Use a fixed fraction of the data for the time plots, e.g., 20%
            t1_idx = len(t_true) // 5 if len(t_true) > 100 else len(t_true)
        else:
            # Index corresponding to the window time
            t1_idx = int(window // dt)
        
        
        # Trim data for time-domain plots
        t_plot = t_true[t0_idx:t0_idx+t1_idx]
        y_raw_plot = y_raw[t0_idx:t0_idx+t1_idx]
        y_true_plot = y_true[t0_idx:t0_idx+t1_idx]
        bias_plot = b[t0_idx:t0_idx+t1_idx]
        if np.sum(abs(bias_plot)) < 1e-10:
            bias_plot = None
        else:
            if bias_plot.ndim == 1:
                bias_plot = bias_plot[:, np.newaxis]
            _, PSD_bias = fun_PSD(dt, b[t0_idx:nt_PSD + t0_idx])
        


        # X-limits for time plots
        xlim_time = [t_plot[0], t_plot[-1]]
        max_y = np.max(np.abs(y_raw))

        
        # 2. Figure Setup
        fig, axes = plt.subplots(
            Nq, 5, 
            figsize=(fig_width, 2. * Nq), 
            layout='constrained',
            gridspec_kw={'width_ratios': [1, 1, 0.5, 1, 1], 'wspace': 0.1, 'hspace': 0.1}
        )
        
        # If Nq=1, axes will be a 1D array; ensure it's 2D for consistent indexing
        if Nq == 1:
            axes = axes.reshape(1, 5)

        titles = ['Raw', 'Truth', 'PDF', 'PSD', 'Difference']
        xlabels = ['$t$', '$t$', '$p$', '$f$', '$t$']

        
        c_raw = '#20b2aae5'
        c_true = '#000080ff'
        c_unbiased = "#8362caff"
        c_diff = "#6b256fff"
        c_bias = "#db76deff"
        
        
        # 3. Plotting Loop
        
        # Plotting column by column (more readable than the original's structure)
        for q_i in range(Nq):
            # Column 0: Raw Time Series (y_raw)
            ax = axes[q_i, 0]
            ax.plot(t_plot, y_raw_plot[:, q_i], color=c_raw, label=f'$y_{q_i}$')
            if y_obs is not None:
                ax.plot(t_obs, y_obs[:, q_i], 'ro', ms=3, mec='k', lw=.1)
            ax.legend(fontsize='x-small', )
            ax.set(xlim=xlim_time)
            y_lim_base = ax.get_ylim()


            # Column 1: True Time Series (y_true)
            ax = axes[q_i, 1]
            ax.plot(t_plot, y_true_plot[:, q_i], color=c_true, label=f'$y^t_{q_i}$')
            if bias_plot is not None:
                ax.plot(t_plot, y_true_plot[:, q_i]-bias_plot[:, q_i], color=c_unbiased, label=f'$y^t_{q_i}-b^t_{q_i}$')
            # if q_i == 0:
            ax.legend(fontsize='x-small', ncol=2)

            y_lim_2 = ax.get_ylim()
            y_lim_base = [min(y_lim_base[0], y_lim_2[0]), 
                          max(y_lim_base[1], y_lim_2[1])]
            # reset ax0 if changed 
            axes[q_i, 0].set_ylim(y_lim_base)
            ax.set(xlim=xlim_time, ylim=y_lim_base)

            # Column 2: PDF (uses full data)
            ax = axes[q_i, 2]
            # Raw and true PDF
            for ds, c in zip([y_true, y_raw], [c_true, c_raw]):
                ax.hist(ds[:, q_i], bins=20, density=True, orientation='horizontal', color=c, histtype='stepfilled', alpha = 0.7)

            if bias_plot is not None:
                ax.hist(y_true_plot[:, q_i]-bias_plot[:, q_i], bins=20, density=True, orientation='horizontal', alpha = 0.7,
                        color=c_unbiased, label=f'$y^t_{q_i}-b^t_{q_i}$')

            if y_obs is not None:
                ax.hist(y_obs[:, q_i], bins=20, color='r', lw=1, histtype='step', density=True, orientation='horizontal')
            ax.set(ylim=y_lim_base)
            
            # Column 3: PSD (uses full data)
            ax = axes[q_i, 3]
            for ds, c, a in zip([PSD_true, PSD_raw], [c_true, c_raw], [1., .8]):
                ax.semilogy(f_raw, ds[q_i], color=c, alpha=a)
            if bias_plot is not None:
                ax.semilogy(f_raw, PSD_bias[q_i], color=c_unbiased, alpha=.8)
            
            if q_i == 0:
                ylims_PSD = [np.min(PSD_raw) * 0.1, np.max(PSD_raw) * 10]
            ax.set_xlim([0, f_max])
            ax.set_ylim(ylims_PSD)
            
            # Column 4: Difference Time Series (Noise)
            ax = axes[q_i, 4]

            noise = y_true_plot[:, q_i] - y_raw_plot[:, q_i]
            ax.plot(t_plot, noise, color=c_diff, label=f'$y^t - y_{q_i}$')
            ax.axhline(np.mean(noise), color='k', lw=.5, ls='--')

            if bias_plot is not None:
                bias = bias_plot[:, q_i]
                ax.plot(t_plot, bias, color=c_bias, label=f'$b^t_{q_i}$')

                # if q_i == 0:
            ax.legend(fontsize='x-small', ncol=2)

            ax.set(xlim=xlim_time)

            if q_i < Nq:
                for jj, ax in enumerate(axes[q_i, :]):
                    if q_i != Nq-1:
                        ax.set_xticklabels([]) # No x-axis labels
                    if jj in [1,2]:
                        ax.set_yticklabels([]) # No y-axis labels
                
        
        # Set titles and xlabesl
        for i, (title, xlbl) in enumerate(zip(titles, xlabels)):
            axes[0, i].set_title(title)
            axes[-1, i].set_xlabel(xlbl)
    




# %%
if __name__ == "__main__":

    from models_physical import Lorenz63
    truth = Observations(model=Lorenz63, t_start=10.0, t_stop=40.0, Nt_obs=10, noise_level=0.5, noise_type='gauss, add', manual_bias=None)


# %%
    truth.plot_truth(truth)

# %%
