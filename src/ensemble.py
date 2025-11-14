import numpy as np
from copy import deepcopy
from typing import List, Union, Dict, Type
from bias import Bias, NoBias
from model import Model
from utils_plotting import Palette

from observations import Observations
from utils import allowed_kwargs_for_func, interpolate
import matplotlib.pyplot as plt


class Ensemble(object):
    """
    Manages ensemble-related properties and methods for a Model,
    primarily for ensemble forecasting and data assimilation (e.g., EnKF).

    This class handles configuration, initialization, and uncertainty generation.
    """


    m: int = 10  # Number of ensemble members
    
    # Data assiomilation specific parameters
    filter: str = 'EnKF'

    est_phi: bool = True                        # Estimate model state?
    est_alpha: Union[bool, List[str]] = False   # Estimate parameters? If a List, includes the names of model parameters to estimate
    est_bias: bool = False                      # Estimate bias?
    
    ensemble_psi0 : np.ndarray = None  # Precomputed ensemble of initial states (overrides std_phi, std_alpha if provided)


    bias_bayesian_update: bool = False          # Only used if est_bias == True
    regularization_factor: float = 1.0          # Only used if filter == rEnKF

    num_DA_blind: int = 0
    num_SE_only: int = 0
    start_ensemble_forecast: float = 0.0

    inflation: float = 1.00
    reject_inflation: float = 1.002 # Inflation after rejecting an analysis
    

    # Ensemble initialization parameters 
    std_phi: float = 0.001                                  # Std for initial state uncertainty (as a fraction of mean)
    std_alpha: Union[float, 
                     Dict[str, Union[float, 
                                    List[float]]]] = 0.001  # Std/range for initial parameter uncertainty

    distribution_alpha: str = 'uniform' # Distribution for parameter (alpha) uncertainty
    distribution_phi: str = 'normal'    # Distribution for state (psi) uncertainty
    ensure_mean_at_init: bool = False    # Force one ensemble member to be the mean
    


    def __init__(self, parent_model: Type[Model], parent_bias: Type[Bias] = NoBias, **kwargs):
        """
        Initializes the Ensemble and links it back to the parent Model instance.
        """
        # 1. Link back to parent model
        self.model = parent_model.copy()
        self.rng = self.model.rng
        
        # 2. Apply configuration overrides and ensure consistency
        ensemble_dict = kwargs.copy()
        
        # Apply only attributes that exist on the Ensemble class
        for key in kwargs.keys():
            if hasattr(Ensemble, key):
                setattr(self, key, ensemble_dict.pop(key))


        # Ensure est_alpha is a list of parameter names if not provided
        if 'est_alpha' not in kwargs.keys():
            if isinstance(self.std_alpha, dict):
                self.est_alpha = list(self.std_alpha.keys())
            else:
                self.est_alpha = []

        # Push the new configuration snapshot to the Model immediately
        self.update_model_settings()

        # 3. Initialize ensemble state and history in the parent model
        self._init_ensemble_model()
        
        # 4. Initialize bias
        self.bias = deepcopy(parent_bias)
        self._init_bias(**ensemble_dict)
        


    @property
    def Na(self):
        """
        int: The number of estimated parameters.
        """
        return len(self.est_alpha)
    
        
    # ------------------ CONFIGURATION AND INITIALIZATION METHODS ------------------ ##

    def config(self):
        
        return dict(m=self.m,
                    est_phi=self.est_phi,
                    est_alpha=self.est_alpha,
                    est_bias=self.est_bias,
                    Na=self.Na)

    def update_model_settings(self):
        """
        Updates the parent model's settings based on the ensemble configuration.
        And re-syncs the ensemble config if needed.
        """
        # Initialize ensemble config into the model and update model settings
        self.model.ensemble = self.config()
        self.model.modify_settings() 

        # Re-sync ensemble config from the model if discrepancies exist
        current_config = self.config()
        if current_config != self.model._ensemble_config:
            # Update config if there are discrepancies
            for key, val in self.model._ensemble_config.items():
                if getattr(self, key) != val:
                    try:
                        setattr(self, key, val)
                    except AttributeError:
                        print(f"Warning: Could not set attribute {key} on Ensemble instance.")
            # Re-apply model ensemble settings after sync
            self.model.ensemble = self.config()


    @property
    def assimilated_data(self):
        """
        tuple of np.ndarray: The assimilated observations and their assimilation times.
        """
        if not hasattr(self, '_assimilated_data'):
            self._assimilated_data = []
            self._assimilated_times = []

        return self._assimilated_data, self._assimilated_times
    

    @assimilated_data.setter
    def assimilated_data(self, y_obs):
        """
        Appends new assimilated observation data and the current time to the stored lists.
        Parameters
        ----------
        y_obs : np.ndarray
            The observation data to append.
        """
        self._assimilated_data.append(y_obs)
        self._assimilated_times.append(self.current_time)    


    def _init_ensemble_model(self):
        """
        Initializes ensemble members.
        This method creates an ensemble of model states (phi) and, optionally,
        augments that ensemble with uncertain model parameters (alpha). The
        resulting augmented ensemble (psi) is stored in the parent model's
        history and the parent model's settings/filename are updated.
        Parameters
        ----------
        ensemble_psi0 : numpy.ndarray or None, optional
                Precomputed ensemble of model states to use as the initial ensemble. 
                Expected shape: (Nphi+Na, m) where Nphi is the state size, Na number of uncertain parameters,
                and m is the ensemble size (self.m).
        Side effects
        ------------
        - Calls self.add_uncertainty to generate ensembles for state and (optionally) parameters.
        - If self.est_alpha is truthy, reads parameter names from self.est_alpha and
            obtains their nominal values from pm (via getattr) to form mean_a, then
            creates ensemble_alpha0 using self.add_uncertainty with self.std_alpha and
            self.distribution_alpha.
        - Forms the augmented ensemble ensemble_psi0 by vstacking state and parameter
            ensembles when applicable.
        - Calls pm.update_history(psi=ensemble_psi0, reset=True) which resets the
            model's stored initial condition/history to the new ensemble.
        - Appends "_{ModelName}_ensemble_m{m}" to pm.filename (uses getattr(pm, 'name', 'Model'))
            and calls pm.modify_settings() to apply/update configuration derived from the
            new filename or ensemble settings.
        Raises
        ------
        - AssertionError: If provided ensemble_psi0 does not match expected shape (Nphi+Na, m).
        """
        
        pm = self.model

        # print(f'Initializing ensemble for model {getattr(pm, "name", "Model")} with ensemble size {self.m}')
        # print('current state shape:', pm.current_state.shape)

        

        if self.ensemble_psi0 is None:
            # 1. Generate initial state (phi) ensemble
            mean_phi0 = np.mean(pm.current_state, axis=-1)

            # print('Generating ensemble for state with mean shape', mean_phi0.shape,
            #       f'pm.current_state shape {pm.current_state.shape} and m={self.m}')

            ensemble_psi0 = self.add_uncertainty(pm.rng, 
                                                 mean_vec=mean_phi0, 
                                                 std=self.std_phi,
                                                 m=self.m, 
                                                 method=self.distribution_phi,
                                                 ensure_mean_at_init=self.ensure_mean_at_init)
        
            # 2. Augment ensemble with estimated parameters (alpha)
            if self.est_alpha:  
                assert self.Na == len(self.est_alpha)
                mean_a = np.array([getattr(pm, a) for a in self.est_alpha])
                ensemble_alpha0 = self.add_uncertainty(pm.rng, 
                                                       mean_vec=mean_a, 
                                                       std=self.std_alpha, 
                                                       m=self.m,
                                                       method=self.distribution_alpha, 
                                                       ensure_mean_at_init=self.ensure_mean_at_init)
                # print(f'Generated ensemble for parameters {self.est_alpha} with shape {ensemble_alpha0.shape}'
                #       f'ensemble_phi0 shape {ensemble_psi0.shape}')
                ensemble_psi0 = np.concatenate((ensemble_psi0, ensemble_alpha0), axis=0)

            # Store the generated ensemble
            self.ensemble_psi0 = ensemble_psi0
        else:
            assert self.ensemble_psi0.shape[1] == self.m, \
                f'Provided ensemble_psi0 has {self.ensemble_psi0.shape[1]} members, expected {self.m}.'
            assert self.ensemble_psi0.shape[0] == pm.Nphi + self.Na, \
                f'Provided ensemble_psi0 has state size {ensemble_psi0.shape[0]}, expected {pm.Nphi + self.Na}.'

        # 3. Update the parent model's history (resets initial condition)
        print('resetting model history with ensemble initial state of shape:', self.ensemble_psi0.shape)
        pm.update_history(psi=self.ensemble_psi0, 
                          t=pm.hist_t[[0]], 
                          reset=True)
        
        # 4. Update parent model settings/filename
        pm.filename += '_{}_ensemble_m{}'.format(getattr(pm, 'name', 'Model'), self.m)

        print(f'Init ensemble history with shape: {pm.hist.shape} and {pm.hist_t}')


    def _init_bias(self, **Bdict):
        """Initializes the bias instance for the ensemble. If the bias is provided as a class, 
        it instantiates it using the model's current state as the mean observation.
        Parameters
        ----------
        Bdict : dict
            Additional keyword arguments to pass to the bias constructor.
        """
        
        pb = self.bias

        if isinstance(pb, type):
            pm = self.model
            try:
                # Get observable for one member to determine dimension
                y0_all = pm.get_observables()
                y0 = np.mean(y0_all, axis=-1)
                if y0.ndim > 2:
                    y0 = y0.squeeze(axis=-1)
            except (AttributeError, IndexError):
                # Fallback if the model cannot yet produce observables
                y0 = np.zeros(pm.Nq) 
                
            self.bias = pb(
                y=y0, 
                t=pm.current_time, 
                dt=pm.dt, 
                **Bdict
            )


    @staticmethod
    def add_uncertainty(rng: np.random.Generator, 
                        mean_vec: np.ndarray, 
                        std: Union[float, Dict[str, Union[float, List[float]]]], 
                        m: int, 
                        method: str = 'uniform', 
                        ensure_mean_at_init: bool = False) -> np.ndarray:
        """
        Adds uncertainty to a mean state vector/value for ensemble generation.
        Returns an array of shape (state_dim, m).
        
        """
        if method not in ['uniform', 'normal']:
            raise ValueError(f'Distribution "{method}" not supported. Choose "uniform" or "normal".')
            
        mean_vec = np.asarray(mean_vec).flatten()
        
        # Case 1: std is a dictionary (for estimated parameters 'alpha')
        if isinstance(std, dict):
            ensemble_ = []
            for sa in std.values():
                if method == 'uniform':
                    # For uniform, std values are [min_val, max_val]
                    ensemble_.append(rng.uniform(low=sa[0], high=sa[1], size=m))
                else: # normal
                    # Use mean of bounds as location, and half the range as a heuristic scale (std)
                    loc = np.mean(sa)
                    if isinstance(sa, list) and len(sa) == 2:
                        scale = (sa[1] - sa[0]) / 4.0
                    else:
                        scale = loc * 0.5
                    ensemble_.append(rng.normal(loc=loc, scale=scale, size=m))
            ensemble_ = np.array(ensemble_) # Shape: (num_params, m)

        # Case 2: std is a single float (relative standard deviation for state or parameters)
        elif isinstance(std, float):
            if method == 'uniform':
                # Multiplicative uniform perturbation: mean * (1 +/- std)
                perturbation = 1.0 + rng.uniform(-std, std, size=(mean_vec.size, m))
                ensemble_ = mean_vec[:, np.newaxis] * perturbation
            
            else: # normal (using multivariate normal for state vector)
                if np.iscomplexobj(mean_vec):
                    # Handle complex state by perturbing real and imaginary parts independently
                    cov_real = np.diag((mean_vec.real * std) ** 2)
                    cov_imag = np.diag((mean_vec.imag * std) ** 2)
                    real_part = rng.multivariate_normal(mean_vec.real, cov_real, size=m).T
                    imag_part = rng.multivariate_normal(mean_vec.imag, cov_imag, size=m).T
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




    # ------------------ ENSEMBLE GENERATION METHODS ------------------ ##


    def reshape_ensemble(self, m: int = None, reset: bool = True) -> None:
        """
        Reshapes the ensemble state (resampling/re-perturbing) and returns a 
        new Model instance with the updated ensemble.
        """

        # Use deepcopy to ensure a clean, independent Model instance
        pm = self.model

        if m is None:
            m = self.m
            
        # Get the current ensemble state from the ORIGINAL model
        current_psi = pm.current_state # (state_dim, current_m)
        
        # Calculate the mean state across current ensemble members
        mean_psi = np.mean(current_psi, axis=-1) # (state_dim,)
        
        if m == 1:
            raise ValueError('Ensemble size m must be greater than 1 to reshape ensemble.')
        else:
            # Calculate standard deviation for re-perturbation
            std_psi = np.std(current_psi, axis=-1) # (state_dim,)
            new_ensemble = self.add_uncertainty(pm.rng, mean_psi, std_psi, m, method='normal')
            

        # Update the new model's history
        pm.update_history(psi=new_ensemble, t=pm.current_time, reset=reset)
        
    
    
    def time_integrate(self, reset=False, **kwargs) -> None: 
        """
        Advances the ensemble in time for Nt steps using the model's integrator.
        Both, the ensemble model and bias are forecasted:
        - Model is advanced using the integrator's advance method.
        - Bias is advanced using its own time_integrate method.
        Their corresponding histories are updated.
        """
        pm = self.model

        # print(f'Advancing ensemble: current_state shape {pm.current_state.shape}, \
        #       is_ensemble {pm.integrator.is_ensemble}', pm.hist.shape, pm.hist_t.shape)
        print(f'Advancing ensemble: current_time {pm.current_time}, with kwargs {kwargs.keys()}')

        psi, t = pm.time_integrate(**kwargs)

        pm.update_history(psi, t, reset=reset) # add the forecast to the model history

        # Advance bias model
        if hasattr(self, 'bias'):
            pb = self.bias
            bias_psi, bias_t = pb.time_integrate(t=t, **kwargs)
            pb.update_history(bias_psi, bias_t, reset=reset)


    def current_unbiased_state(self) -> np.ndarray:
        """
        Returns the current bias-corrected ensemble state.
        """
        pm = self.model
        pb = self.bias

        y_model = pm.get_observables()  # Shape: (obs_dim, m)
        t_model = pm.current_time # Scalar

        t_b = pb.current_time # Scalar

        if t_model == t_b:
            b = pb.current_bias()  # Shape: (obs_dim,) or (obs_dim, 1)
            # Ensure shape is (obs_dim,1 or m)
            if b.ndim == 1:
                b = b[:, np.newaxis]
            return y_model + b  # Shape: (obs_dim, m)
        else:
            # Expand to allow interpolation function to work correctly
            y_model = y_model[np.newaxis, :]
            t_model = t_model[np.newaxis]
            # Interpolate bias to model time point
            y_unbiased = self._recover_unbiased_solution(pb.hist_t, pb.hist, 
                                                        t_model, y_model)
        
            return y_unbiased.squeeze(axis=0)  # Shape: (obs_dim, m)


    def unbiased_hist(self, Nt=0) -> np.ndarray:
        """
        Returns the bias-corrected ensemble history.
            y_unbiased = self._recover_unbiased_solution(pb.hist_t, pb.hist, pm.hist_t, y_model)
        Parameters
        ----------
        Nt : int, optional
            Time index to retrieve the bias-corrected ensemble history for. Default is 0 (i.e., All history).
        Returns
        -------
        np.ndarray
            Bias-corrected ensemble history.
        Raises
        ------
        ValueError
            If Nt is 1, which is not a valid value for this parameter.
        """
        if Nt == 1:
            raise ValueError('Nt must be 0 (all history) or >1 (number of time steps).')
        
        pm = self.model
        pb = self.bias

        y_model = pm.get_observable_hist(Nt)
        t_model = pm.hist_t[-Nt:]
        y_unbiased = self._recover_unbiased_solution(pb.hist_t, pb.hist, t_model, y_model)

        return y_unbiased


    @staticmethod
    def _recover_unbiased_solution(t_b, b, t, y):
        """
        Returns the bias-corrected solution by interpolating bias to match y's time points.
        We may need to interpolate because the bias and model histories may have different time grids.

        Parameters
        ----------
        t_b : array-like, shape (T_b,)
            Time points corresponding to the bias history.
        b : array-like, shape (T_b, ...) or (T_b,)
            Bias values at each time in t_b.
        t : array-like, shape (T,)
            Time points corresponding to the model history.
        y : array-like, shape (T, ...) or (T,)
            Model observable history at each time in t.

        Returns
        -------
        y_unbiased : array-like, shape (T, ...)
            Bias-corrected observable history at each time in t.
        """
        if b.ndim < y.ndim:
            b = np.expand_dims(b, axis=-1)
        elif b.shape[-1] > 1:
            b = np.mean(b, axis=-1, keepdims=True)

        if b.shape[-1] == 1 and y.shape[-1] > 1:
            b = np.repeat(b, y.shape[-1], axis=-1)

        if len(t_b) != len(t):
            b = interpolate(t_b, b, t, fill_values=None) # Interpolate bias to model time points
        return y + b
    


    # ------------------ ENSEMBLE VISUALIZATION METHODS ------------------ ##

    def print_parameters(self) -> None:
        """
        Prints the ensemble configuration parameters in a readable format.
        """
        print("Ensemble Configuration Parameters:")
        for attr in dir(self):
            if not attr.startswith('_') and not callable(getattr(self, attr)):
                value = getattr(self, attr)
                print(f"  {attr}: {value}")

        self.model.print_model_parameters()
        self.bias.print_bias_parameters()

        

    def visualize_state(self, **kwargs) -> None:
        """
        Visualizes the ensemble state and parameters at a specific time.
        Parameters
        ----------
        time_indices : list of int, optional
            List of time indices to visualize from the model history. Defaults to [0].
        Side effects
        ------------
        - Calls plot_ensemble_model to generate and display plots of the ensemble
          distributions at the specified time indices.
        """
        
        kwargs_state = allowed_kwargs_for_func(plot_model_state, kwargs)
        
        plot_model_state(self.model, **kwargs_state)

        #TODO: plot_ensemble_  



    def visualize_history(self, **kwargs
                          ) -> None:
        """
        Visualize summary plots for the stored model and bias histories.

        This method is a lightweight wrapper that:
        - Calls plot_ensemble_model to produce ensemble distribution snapshots for those times.
        - Provides a placeholder where bias-history plotting should be implemented
          (if a bias instance with history exists).

        Implementation notes 
        """
        #separate kwargs for the different plots 
        kwargs_obs = allowed_kwargs_for_func(plot_observable_history, kwargs)
        plot_observable_history(ensemble=self, **kwargs_obs)

        if self.Na > 0:
            kwargs_alpha = allowed_kwargs_for_func(plot_alpha_hist, kwargs)
            plot_alpha_hist(ensemble=self, **kwargs_alpha)





# ===== AUXILIARY PLOTTING FUNCTIONS ===== #




def plot_alpha_hist(ensemble : Ensemble, 
                    plot_members: bool = False,
                    reference_a=1., 
                    reference_t=1.) -> None:
    """
    Plot the time evolution of the parameters in a object of class model
    """
    pm = ensemble.model

    C = Palette()
    colors = C.get_color_params(n=ensemble.Na, alpha=1)
    colors_alpha = C.get_color_params(n=ensemble.Na, alpha=.2)

    t = pm.hist_t
    t_zoom = int(pm.t_CR / pm.dt)



    if reference_t == 1.:
        t_label = '$t$'
    else:
        t_label = f'$t/{reference_t}$'  
        t = t / reference_t 


    xlims = [[t[0], t[-1]], [t[-t_zoom], t[-1]]]

    hist_alpha = pm.hist[:, -pm.Na:]
        
    reference_alpha = {key: 1. for key in ensemble.est_alpha}
    alpha_lbls = pm.alpha_labels.copy()
    if isinstance(reference_a, dict):
        for key, val in reference_a.items():
            if key in reference_alpha.keys():
                reference_alpha[key] = val
                alpha_lbls[key] += f' / {reference_alpha[key]}'
        


    mean_alpha = np.mean(hist_alpha, axis=-1)
    std_alpha = np.std(hist_alpha, axis=-1)

    fig = plt.figure(figsize=(10, 2*ensemble.Na), layout="constrained")

    axs = fig.subplots(ensemble.Na, 2, sharex='col', sharey='row', width_ratios=[2, 1])
    
    if ensemble.Na == 1:
        axs = [axs]

    for row_i, axs_row, p in zip(range(ensemble.Na), axs, ensemble.est_alpha):
        for col_i, ax in enumerate(axs_row):
                
            avg, s, all_h = [xx[:, row_i] / reference_alpha[p] for xx in [mean_alpha, std_alpha, hist_alpha]
                             ]
            
            if plot_members:
                ax.plot(t, all_h, color=colors_alpha[row_i], lw=1.)

            ax.fill_between(t, avg + 2 * abs(s), avg - 2 * abs(s), alpha=0.2, color=colors_alpha[row_i], label='2 std')
            ax.plot(t, avg, color=colors[row_i], label='mean', lw=2, dashes=(5,1))

            if col_i == 0:
                ax.set(ylabel=alpha_lbls[p], 
                        ylim=[min(avg) - 3 * max(s), max(avg) + 3 * max(s)])
            if row_i == ensemble.Na - 1:
                ax.set(xlabel=t_label, xlim=xlims[col_i])



def plot_observable_history(ensemble : Ensemble,
                            truth : Union[Observations, None] = None, 
                            plot_members : bool = False,
                            reference_y=1., 
                            reference_t=1., 
                            max_time=None, 
                            dims='all') -> None:
    """
    Plots time series of the ensemble mean and individual members for each state variable
    and estimated parameter in the model's history.

    Parameters
    ----------
    model : Model
        The model instance containing the history to plot.

    Side effects
    ------------
    - Generates and displays time series plots of the ensemble mean and individual members
      for each state variable and estimated parameter in the model's history.
    """


    C = Palette()

    t_obs, y_obs = ensemble.assimilated_data

    pm, pb = ensemble.model, ensemble.bias    

    y_unbiased = ensemble.unbiased_hist()

    y_model, t = pm.get_observable_hist(), pm.hist_t

    b, t_b = pb.get_bias_hist(), pb.hist_t
    if b.ndim < 3:
        b = np.expand_dims(b, axis=-1)

    Nq = pm.Nq
    if dims == 'all':
        dims = range(Nq)
    elif isinstance(dims, int):
        dims = [dims]
    



    # cut signals to interval of interest -----
    N_CR = int(pm.t_CR // pm.dt)  # Length of interval to compute correlation and RMS

    if len(t_obs) > 0:
        if max_time is None:
            max_time = min(t_obs[-1] + pm.t_CR, t[-1])

        if len(t_obs) > 0:
            min_time = t_obs[0] - 0.25 * pm.t_CR
        else:
            min_time = t[0]

        i0, i1 = [np.argmin(abs(t - ttt)) for ttt in [min_time, max_time]]  # start/end of assimilation
        y_model, y_unbiased, t = (yy[i0 - N_CR:i1 + N_CR] for yy in [y_model, y_unbiased, t])


        # Nomalize time and observations ----

        if reference_t == 1.:
            t_label = '$t$'
        else:
            t_label = f'$t/{reference_t}$'
            t, t_b, t_obs, max_time = [tt / reference_t for tt in [t, t_b, t_obs, max_time]]
            

        x_lims = [[t_obs[0] - .25 * pm.t_CR, t_obs[0] + pm.t_CR],
                [t_obs[-1] - pm.t_CR, max_time],
                [t[0], max_time]]
        width_ratios = [1, 1, 2]
    else:
        max_time = t[-1]

        if reference_t == 1.:
            t_label = '$t$'
        else:
            t_label = f'$t/{reference_t}$'
            t, t_b, max_time = [tt / reference_t for tt in [t, t_b, max_time]]

        x_lims = [[t[0], t[-1]],
                  [t[-N_CR], t[-1]]]
        width_ratios = [2, 1]
    


    # Get truth if available ---- 

    if truth is not None:
        y_raw = interpolate(truth.t_true, truth.y_raw, t)
        y_true = interpolate(truth.t_true, truth.y_true, t)
    else:
        # Set to nan if no truth is provided
        y_raw = np.full_like(y_model, np.nan)
        y_true = np.full_like(y_model, np.nan)


    # Nomalize time and observations ----
    
    # ensure reference_y is an array
    if isinstance(reference_y, (int, float)):
        if reference_y == 1.:
            normalize_y = False
        else:
            normalize_y = True
        reference_y = reference_y * np.ones(pm.Nq)
    else:
        normalize_y = True
        if len(reference_y) != pm.Nq:
            raise ValueError('reference_y must be a scalar or an array of length Nq.')
    
    
    if not normalize_y:

        y_labels = [pm.obs_labels[qi] for qi in dims]
    else:
        reference_y = reference_y[np.newaxis, :, np.newaxis]
        y_unbiased, y_model = [yy / reference_y for yy in [y_unbiased, y_model]]
        if len(t_obs) > 0:
            y_obs = y_obs / reference_y
        if truth is not None:
            y_raw, y_true = [yy / reference_y for yy in [y_raw, y_true]]
        
        y_labels = [f'{pm.obs_labels[qi]} / ${reference_y[0, qi, 0]}$' for qi in dims]

    
    # % PLOT time series ------------------------------------------------------------------------------------------

    margin = 0.15 * np.mean(abs(y_model), axis=(0, 2))
    max_y = np.max(y_model, axis=(0, 2), keepdims=False)
    min_y = np.min(y_model, axis=(0, 2), keepdims=False)


    fig1 = plt.figure(figsize=(10, 2 * len(dims)), layout="constrained")
    ax_all = fig1.subplots(nrows=len(dims), ncols=len(x_lims), sharey='row', sharex='col', width_ratios=width_ratios)
    if len(dims) == 1:
        ax_all = ax_all[np.newaxis, :]


    for row_i, qi in enumerate(dims):
        yl = [min_y[qi] - margin[qi], max_y[qi] + margin[qi]]

        for col_i, (ax, xl) in enumerate(zip(ax_all[row_i], x_lims)):
            ax.plot(t, y_true[:, qi], label='truth', **C.true_props)
            if pb.name != 'NoBias':
                ax.plot(t, y_unbiased[:, qi], label='bias-corrected estimate', **C.y_unbias_props)

            m = np.mean(y_model[:, qi], axis=-1)
            ax.plot(t, m, **C.y_biased_mean_props, label='model estimate')
            if plot_members:
                for mi in range(y_model.shape[-1]):
                    ax.plot(t, y_model[:, qi, mi], **C.y_biased_props)
            else:
                s = np.std(y_model[:, qi], axis=-1)
                ax.fill_between(t, m + s, m - s, color=C.get_color('BIASED', 0.5))

            if len(t_obs) > 0:
                ax.plot(t_obs, y_obs[:, qi], label='data', **C.obs_props)

            if col_i == 0:
                ax.set(ylabel=y_labels[row_i])

            ax.set(ylim=yl, xlim=xl)
            if row_i == len(dims) - 1:
                ax.set(xlabel=t_label)



def plot_model_state(model: Model, time_indices=[-1], 
                     max_modes=10, reference_params=None, nbins=6) -> None:
    """
    Plots histograms of the state variables and parameters at specified time indices.
    
    Parameters
    ----------
    model : Model
        The model instance containing the history to plot.
    time_indices : list of int, optional (default [-1] i.e., current time)
        List of time indices from the model history to plot. 
    max_modes : int or None, optional (default None)
        Maximum number of state variables (modes) to plot. If None, plots all modes
        in the model.
    reference_params : dict or None, optional (default None)
        Reference parameter values for normalizing parameter histograms. If None,
        uses 1.0 for all parameters.
    nbins : int, optional  (default 6)
        Number of bins to use in the histograms. 

    Side effects
    ------------
    - Generates and displays histograms of the state variables and parameters
      at the specified time indices.
    """


    if max_modes is None:
        max_modes = model.Nphi
    

    # find the number of subplots needed
    ncols_phi = min(max_modes, 4)
    nrows_phi = int(np.ceil(max_modes / ncols_phi))
    if model.Na == 0:
        nrows_alpha, ncols_alpha = 0, 0
        est_alpha = None
    else:
        ncols_alpha = min(model.Na, 4)
        nrows_alpha = int(np.ceil(model.Na / ncols_alpha))
    
        est_alpha = model.ensemble.get('est_alpha')

        reference_alpha = {key: 1. for key in est_alpha}
        if isinstance(reference_params, dict):
            for param, val in reference_params.items():
                reference_alpha[param] = val


    def add_stats_text(_ax, yy):
        # add text with mean and std relative
        mean_yy = np.mean(yy)
        std_yy = np.std(yy) / mean_yy if mean_yy != 0 else 0
        textstr = '\n'.join((f'Mean: {mean_yy:.4f}', f'Std: {std_yy:.4f}'))
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        _ax.text(0.95, 0.95, textstr, transform=_ax.transAxes, fontsize='x-small',
                verticalalignment='top', horizontalalignment='right', bbox=props)

    
    

    for ti in time_indices:
        phi = model.hist[ti, :model.Nphi, :]

        # if complex, plot real and imag parts separately
        if np.iscomplexobj(phi):

            # Combine real and imaginary parts for histogram plotting (stacked as real, img, real, img...)
            phi_complex = phi.copy()
            phi = np.zeros((2 * model.Nphi, phi_complex.shape[1]))
            for i in range(model.Nphi):
                phi[2 * i, :] = np.real(phi_complex[i, :].real)
                phi[2 * i + 1, :] = np.real(phi_complex[i, :].imag)


            if ti == time_indices[0]:  # only need to adjust nrows once
                # Update labels accordingly
                state_labels = []
                for lbl in model.state_labels:
                    state_labels.append(f'{lbl} (real)')
                    state_labels.append(f'{lbl} (imag)')
                nrows_phi = int(np.ceil((2 * max_modes) / ncols_phi))
            
        else:
            state_labels = model.state_labels

        alpha = model.hist[ti, -model.Na:, :]

        # Plot all in one mosaic
        fig = plt.figure(figsize=(10, 2 * (nrows_phi + nrows_alpha)), layout='constrained')
        plt.suptitle(f'Ensemble distributions at time t={model.hist_t[ti]:.3f}')

        if model.Na == 0:
            sf = [fig.subfigures(nrows=1, ncols=1)]
        else:
            sf = fig.subfigures(nrows=2, ncols=1, 
                                height_ratios=[nrows_phi, nrows_alpha], wspace=0.07, hspace=0.15)

        axs = sf[0].subplots( ncols=ncols_phi, nrows=nrows_phi, sharey=True)

        axs = axs.ravel() if ncols_phi * nrows_phi > 1 else [axs]

        for ax, ph, lbl in zip(axs.ravel(), phi, state_labels):
            ax.hist(ph, bins=nbins, color='tab:green')
            ax.set(xlabel=lbl)
            add_stats_text(ax, ph)
        
        if model.Nphi < len(axs):
            for ax in axs[model.Nphi:]:
                ax.axis('off')

        if est_alpha is not None:
            axs = sf[1].subplots(ncols=ncols_alpha, nrows=nrows_alpha, sharey=True)
            axs = axs.ravel() if ncols_alpha * nrows_alpha > 1 else [axs]

            for ax, a, param in zip(axs, alpha, est_alpha):
                ax.hist(a / reference_alpha[param], bins=nbins)
                ax.set(xlabel=model.alpha_labels[param])
                add_stats_text(ax, a)
        
            if model.Na < len(axs):
                for ax in axs[model.Na:]:
                    ax.axis('off')