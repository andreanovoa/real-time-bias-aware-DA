from collections import namedtuple
import numpy as np
from copy import deepcopy
from typing import List, Optional, Tuple, Union, Dict, Type, Sequence
from bias_estimators import Bias
from models import Model
from plotting import Palette

from observations import Observations
from matplotlib.figure import SubFigure
from utils import allowed_kwargs_for_func, interpolate, mean_vector_to_ensemble
import matplotlib.pyplot as plt

from typeguard import typechecked

from data_assimilation import Filter, EnKF


class Ensemble(object):
    """
    Manages ensemble-related properties and methods for a Model,
    primarily for ensemble forecasting and data assimilation (e.g., EnKF).

    This class handles configuration, initialization, and uncertainty generation.

    Properties
    ----------
    assimilated_data : property
        Getter returns a tuple of assimilated observations and their times.
        Setter appends new observation data and time to the stored lists.
    """


    m: int = 10  # Number of ensemble members
    
    # Data assiomilation specific parameters

    est_phi: bool = True                        # Estimate model state?
    est_alpha: List[str] = []                   # Estimate parameters? If a List, includes the names of model parameters to estimate
    est_bias: bool = False                      # Estimate bias?
    
    ensemble_psi0 : Optional[np.ndarray] = None  # Precomputed ensemble of initial states (overrides std_phi, std_alpha if provided)


    bias_bayesian_update: bool = False          # Only used if est_bias == True
    regularization_factor: float = 1.0          # Only used if filter == rEnKF

    num_DA_blind: int = 0
    num_SE_only: int = 0
    start_ensemble_forecast: float = 0.0

    inflation_factor: float = 1.00
    inflation_factor_rejection: float = 1.002 # Inflation after rejecting an analysis

    # Ensemble initialization parameters 
    std_phi: float = 0.001                                  # Std for initial state uncertainty (as a fraction of mean)
    std_alpha: Union[float, 
                     Dict[str, Union[float, 
                                    List[float]]]] = 0.001  # Std/range for initial parameter uncertainty

    distribution_alpha: str = 'uniform' # Distribution for parameter (alpha) uncertainty
    distribution_phi: str = 'normal'    # Distribution for state (psi) uncertainty
    ensure_mean_at_init: bool = False    # Force one ensemble member to be the mean
    
    activate_parameter_estimation: bool = True  # Whether to include parameter estimation in the analysis step

    results_folder: Optional[str] = None

    keys_to_print = ['m', 'est_phi', 'est_alpha', 'est_bias', 'Na',
                     'regularization_factor', 'inflation_factor', 'inflation_factor_rejection',
                     ]

    @typechecked
    def __init__(self, 
                 parent_model: Union[Model, Type[Model]], 
                 parent_bias: Optional[Union[Bias, Type[Bias]]] = None, 
                 da_method: Optional[Union[Filter, Type[Filter]]] = None, 
                 **kwargs #type: ignore
                 ):
        """
        Initializes the Ensemble and links it back to the parent Model instance.
        """

        # 1. Apply configuration overrides and ensure consistency
        ensemble_dict = kwargs.copy()
        
        # Apply only attributes that exist on the Ensemble class
        for key in kwargs.keys():
            if hasattr(Ensemble, key):
                try:
                    setattr(self, key, ensemble_dict.pop(key))
                except:
                    raise AttributeError(f"Could not set attribute {key} on Ensemble instance.")

        # Ensure est_alpha is a list of parameter names if not provided
        if 'est_alpha' not in kwargs.keys():
            if isinstance(self.std_alpha, dict):
                self.est_alpha = list(self.std_alpha.keys())
            else:
                self.est_alpha = []

        # 3. Initialize ensemble state and history in the parent model
        self._init_ensemble_model(parent_model, **ensemble_dict)
        
        # 4. Initialize bias instance if provided (if not, self.bias will be None)
        self._init_bias(parent_bias, **ensemble_dict)

        # 5. Set up data assimilation filter if provided
        if da_method is not None:
            if isinstance(da_method, Filter):
                self._filter = da_method
            elif isinstance(da_method, type) and issubclass(da_method, Filter):                
                self._filter = da_method(M=self.model.M, gamma=self.regularization_factor)


    
    @property
    def model(self) -> Model:
        """
        The parent model instance associated with the ensemble.
        """
        return self._model

    @property
    def bias(self) -> Optional[Bias]:
        """
        The bias instance associated with the ensemble.
        """
        if hasattr(self, '_bias'):
            return self._bias
        else:
            return None

    @bias.setter
    def bias(self, value: Optional[Bias]):
        """Sets (or replaces) the bias estimator instance of the ensemble."""
        assert value is None or isinstance(value, Bias), \
            f"bias must be a Bias instance or None, got {type(value)}"
        self._bias = value

    @property
    def filter(self) -> Optional[Filter]:
        """
        The data assimilation filter instance associated with the ensemble.
        """
        if hasattr(self, '_filter'):
            return self._filter
        else:
            return None


    @property
    def Na(self):
        """
        int: The number of estimated parameters.
        """
        return len(self.est_alpha)
    
    @property
    def Nphi(self):
        """
        int: The size of the model state vector.
        """
        return self.model.Nphi
    
    def copy(self):
        return deepcopy(self)
        
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
        # Invalidate cached M so it's rebuilt with the updated N (Nphi + Na + Nq)
        if hasattr(self.model, '_M'):
            del self.model._M

        # Re-sync ensemble config from the model if discrepancies exist
        current_config = self.config()
        if current_config != self.model._ensemble_config:
            # Update config if there are discrepancies
            for key, val in self.model._ensemble_config.items():
                if getattr(self, key) != val:
                    try:
                        setattr(self, key, val)
                    except AttributeError:
                        print(f"! Warning: Could not set attribute {key}={val} on Ensemble instance.")
            # Re-apply model ensemble settings after sync
            self.model.ensemble = self.config()


    @property
    def assimilated_data(self):
        """
        Property for managing assimilated observations and their times.

        Returns
        -------
        tuple of np.ndarray
            The assimilated observations and their assimilation times.

        Notes
        -----
        This property supports both getting and setting:
        - Getter returns the stored assimilated observations and times.
        - Setter appends new observation data and time to the lists.
        """

        if not hasattr(self, '_assimilated_data'):
            self._assimilated_data = []
            self._assimilated_times = []

        AssimilatedData = namedtuple('AssimilatedData', ['data', 'times'])
        return AssimilatedData(data=self._assimilated_data, times=self._assimilated_times)
    

    @assimilated_data.setter
    def assimilated_data(self, value: tuple):
        """
        Appends new assimilated observation data and the current time to the stored lists.

        Parameters
        ----------
        value : tuple
            A tuple containing the observation data (np.ndarray) and the observation time (float or None).
        """
        if not hasattr(self, '_assimilated_data'):
            self._assimilated_data = []
            self._assimilated_times = []

        y_obs, t_obs = value

        self._assimilated_data.append(y_obs)
        self._assimilated_times.append(t_obs)
    

    @typechecked
    def _init_ensemble_model(self, parent_model: Union[Model, Type[Model]], **kwargs):
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


        if isinstance(parent_model, Model):
            self._model = parent_model.copy()
        else:
            self._model = parent_model(**kwargs)

        # Push the new configuration snapshot to the Model immediately
        self.update_model_settings()
        self.rng = self.model.rng
        
        
        pm = self.model
        

        if self.ensemble_psi0 is None:
            # 1. Generate initial state (phi) ensemble
            mean_phi0 = np.mean(pm.current_state, axis=-1)

            ensemble_psi0 = mean_vector_to_ensemble(pm.rng, 
                                                 mean_vec=mean_phi0, 
                                                 std=self.std_phi,
                                                 m=self.m, 
                                                 method=self.distribution_phi,
                                                 ensure_mean_at_init=self.ensure_mean_at_init)
        
            # 2. Augment ensemble with estimated parameters (alpha)
            if self.est_alpha:  
                assert self.Na == len(self.est_alpha), f"Number of parameters to estimate (Na={self.Na}) must match length of est_alpha list ({len(self.est_alpha)})."
                mean_a = np.array([getattr(pm, a) for a in self.est_alpha])

                ensemble_alpha0 = mean_vector_to_ensemble(pm.rng, 
                                                          mean_vec=mean_a, 
                                                          std=self.std_alpha, 
                                                          m=self.m,
                                                          method=self.distribution_alpha, 
                                                          ensure_mean_at_init=self.ensure_mean_at_init)
                
                ensemble_psi0 = np.concatenate((ensemble_psi0, ensemble_alpha0), axis=0)

            # Store the generated ensemble
            self.ensemble_psi0 = ensemble_psi0[np.newaxis, :, :]  # Shape (1, Nphi+Na, m)

        else:
            if self.ensemble_psi0.ndim == 2:
                self.ensemble_psi0 = self.ensemble_psi0[np.newaxis, :, :]

            assert self.ensemble_psi0.shape[-1] == self.m, \
                f'Provided ensemble_psi0 has {self.ensemble_psi0.shape[-1]} members, expected {self.m}.'
            assert self.ensemble_psi0.shape[1] == pm.Nphi + self.Na, \
                f'Provided ensemble_psi0 has state size {self.ensemble_psi0.shape[1]}, expected {pm.Nphi + self.Na}.'

        # 3. Update the parent model's history (resets initial condition)
        pm.update_history(psi=self.ensemble_psi0, 
                          t=pm.hist_t[[0]], 
                          reset=True)
        
        # 4. Update parent model settings/filename
        pm.filename += '_ensemble_m{}'.format(self.m)

        print(f'OK: Initialized {pm.filename} history with shape: {pm.hist.shape} and {pm.hist_t}')


    @typechecked
    def _init_bias(self, parent_bias: Optional[Union[Bias, Type[Bias]]] = None, **Bdict):
        """Initializes the bias instance for the ensemble. If the bias is provided as a class, 
        it instantiates it using the model's current state as the mean observation.
        Parameters
        ----------
        parent_bias : Union[Bias, Type[Bias], None], optional
            The class of the bias model to instantiate or an existing Bias instance. If None, uses the current bias instance.
        Bdict : dict, optional
            Additional keyword arguments to pass to the bias constructor.
        """

        if isinstance(parent_bias, Bias):
            self._bias = parent_bias.copy()
        elif parent_bias is None:
            self._bias = None
        else:
            assert isinstance(parent_bias, type) and issubclass(parent_bias, Bias), "parent_bias must be a subclass of Bias"
            pm = self.model
            try:
                # Get observable for one member to determine dimension
                y0_all = pm.get_observables()
                y0 = np.mean(y0_all, axis=-1, keepdims=True)  # Shape (Nq, 1)

            except (AttributeError, IndexError):
                # Fallback if the model cannot yet produce observables
                y0 = np.zeros((1, pm.Nq, 1)) 
            
            # remove dt, y, t from Bdict if they exist to avoid duplication
            [Bdict.pop(key, None) for key in ['y', 't', 'dt']]            

            print(f"Initializing bias model {parent_bias.__name__} with initial state shape {y0.shape} at time {pm.current_time}")
            

            self._bias = parent_bias(innovation=y0, 
                                    t=pm.current_time, 
                                    dt=pm.dt, 
                                    initial_capacity=pm.history._initial_capacity,
                                    rom=pm,
                                    **Bdict
                                    )
        



    # ------------------ ENSEMBLE GENERATION METHODS ------------------ ##


    def reshape_ensemble(self, m: Optional[int] = None, reset: bool = True) -> None:
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
            new_ensemble = mean_vector_to_ensemble(pm.rng, mean_psi, std_psi, m, method='normal')
            


        # Update the new model's history
        pm.update_history(psi=new_ensemble, t=pm.current_time, reset=reset)
        
    
    
    def forecast_step(self, t_end=None, reset=False, close=False, **kwargs) -> None: 
        """
        Advances the ensemble in time for Nt steps using the model's integrator.
        Both, the ensemble model and bias are forecasted:
        - Model is advanced using the integrator's advance method.
        - Bias is advanced using its own time_integrate method.
        Their corresponding histories are updated.
        """
        pm = self.model
        
        if t_end is not None:
            t_end = round(t_end, pm.precision_t)
            Nt = int((t_end - pm.current_time).round(pm.precision_t) / pm.dt)
            kwargs_local = kwargs.copy()
            kwargs_local['Nt'] = Nt
        else:
            kwargs_local = kwargs

        psi, t = pm.time_integrate(**kwargs_local)
        if t_end is not None:
            assert abs(t[-1] - t_end) < pm.dt, f"Final time {t[-1]} does not match requested t_end {t_end}."


        # print('Forecasted ensemble shape:', psi.shape)
        # print('Forecasted time shape:', t.shape, 't0 =', t[0], 't_end =', t[-1], 'current_time =', pm.current_time)

        try:
            pm.update_history(psi, t, reset=reset) # add the forecast to the model history
        except ValueError as e:
            print(f"Solver didn't return a homogeneous psi. Check initial conditions and input_parameters")
            raise e


        # Advance bias model the same number of output steps as the model
        if self.bias is not None:
            pb = self.bias
            b, t_b = pb.time_integrate(Nt=psi.shape[0])

            pb.update_history(b, t_b, reset=reset)
            if abs(pm.current_time - pb.current_time) > pm.dt / 2:
                raise AssertionError('t assertion', pm.current_time, pb.current_time)

        if close:
            pm.close()
    

    @property
    def current_state(self) -> np.ndarray:
        """
        Returns the current ensemble state from the model.
        """
        return self.model.current_state
    
    @property
    def current_time(self) -> float:
        """
        Returns the current time from the model.
        """
        return self.model.current_time

    

    def get_observables(self, Nt=1, **kwargs):
        """
        Returns the ensemble observables from the model.
        Parameters
        ----------
        Nt : int, optional
            Time index to retrieve the ensemble observables for. Default is 1 (i.e., current time).
        Returns
        -------
        np.ndarray
            Ensemble observables at the specified time index.
        """
        y_model = self.model.get_observables(Nt=Nt, **kwargs)  # Shape: (T, obs_dim, m) or (obs_dim, m) if Nt=1

        if self.bias is None or self.bias.__class__.__name__ == 'NoBias':
            return y_model  # No bias correction needed
        elif Nt == 1:
            b = self.bias.current_bias  # Shape: (obs_dim, N_ens)

            if b.shape[-1] != y_model.shape[-1]:
                b = np.mean(b, axis=-1, keepdims=True)

            return y_model + b  # Shape: (obs_dim, m)
        else:
            t_model = self.model.hist_t[-Nt:]
            bias = self.bias.get_bias(self.bias.hist)
            bias_t = self.bias.hist_t

            y_unbiased = self._recover_unbiased_solution(bias_t, bias, t_model, y_model)
            return y_unbiased



    def update_history(self, 
                       psi: np.ndarray, 
                       t = None, 
                       b = None,
                       update_last_state: bool = False,
                       reset: bool = False) -> None:
        """
        Updates the model's history with a new ensemble state at time t.
        Parameters
        ----------
        psi : np.ndarray
            New ensemble state to add to the history.
        t : float or np.ndarray, optional
            Time corresponding to the new ensemble state. Default is None.
        update_last_state : bool, optional
            If True, updates the last stored state instead of appending a new one.
            Default is False.
        reset : bool, optional
            If True, resets the history before adding the new state.
            Default is False.
        Side effects
        ------------
        - Calls self.model.update_history to add the new state and time to the model's history.
        """
        self.model.update_history(psi, t, 
                                  reset=reset,
                                  update_last_state=update_last_state)
        if self.bias is not None:
            self.bias.update_history(b, t, 
                                     reset=reset,
                                     update_last_state=update_last_state)



    def get_observable_hist(self, Nt=0) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Returns the bias-corrected ensemble history.
            y_unbiased = self._recover_unbiased_solution(pb.hist_t, pb.hist, pm.hist_t, y_model)
        Parameters
        ----------
        Nt : int, optional
            Time index to retrieve the bias-corrected ensemble history for. Default is 0 (i.e., All history).
        Returns
        -------
        Tuple[Optional[np.ndarray], np.ndarray]
            A tuple containing the bias-corrected ensemble history and the original ensemble history.
        Raises
        ------
        ValueError
            If Nt is 1, which is not a valid value for this parameter.
        """
        
        pb = self.bias

        y_model = self.model.get_observable_hist(Nt=Nt)  # Shape: (T, obs_dim, m) or (obs_dim, m) if Nt=0

        

        if pb is None or pb.__class__.__name__ == 'NoBias':
            return None, y_model  # No bias correction needed
        else:
            t_model = self.model.hist_t[-Nt:]
            b_hist = pb.get_bias(pb.hist)  # Only the bias components of the full bias-estimator state
            y_unbiased = self._recover_unbiased_solution(pb.hist_t, b_hist, t_model, y_model)
            return y_unbiased, y_model


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

        if b.shape[-1] == 1 and y.shape[-1] > 1:
            b = np.repeat(b, y.shape[-1], axis=-1)

        if len(t_b) != len(t):
            print('Interpolating bias to match model time points. this may be slow if histories are long.')
            b = interpolate(t_b, b, t, fill_values=None) # Interpolate bias to model time points

        return y + b
    



# _______________________________________________________________________________________________________________
# Data assimilation methods
# _______________________________________________________________________________________________________________

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
            kwargs_alpha = allowed_kwargs_for_func(plot_alpha_history, kwargs)
            plot_alpha_history(ensemble=self, **kwargs_alpha)



    def analysis_step(self, d: np.ndarray, Cdd: np.ndarray, **kwargs) -> None:
        """
        Performs the analysis step of the data assimilation algorithm.
        This method updates the ensemble state based on observations and their error covariance.
        After the state update, the bias estimator (if any) is updated with the analysis innovation.
        Parameters
        ----------
        d : np.ndarray
            Observation vector at the current time.
        Cdd : np.ndarray
            Observation error covariance matrix.

        Side effects
        ------------
        - Updates the model's history with the analyzed ensemble state.
        - Updates the bias estimator's state with the analysis innovation (if a bias is set).
        """
        assert self.filter is not None, "Data assimilation filter is not initialized. Please set self.filter before calling analysis_step."

        d = np.asarray(d).squeeze()
        if d.ndim == 0:
            d = d[np.newaxis]

        # Number of analysis steps performed so far (used for num_DA_blind / num_SE_only)
        n_analysis = len(self.assimilated_data.times)
        if self.num_SE_only > 0:
            self.activate_parameter_estimation = n_analysis >= self.num_SE_only
        activate_bias_aware = n_analysis >= self.num_DA_blind

        Af = self.current_state.copy()  # state matrix [Nphi + Na] x m

        if self.Na > 0 and not self.activate_parameter_estimation:
            Af_params = Af[-self.Na:, :].copy()  # store forecast parameters to re-append after analysis
            Af = Af[:-self.Na, :]
        else:
            Af_params = None

        # ================== DEFINE AUGMENTED STATE VECTOR =================== #
        y = self.model.get_observables()
        Af = np.vstack((Af, y))


        # ======================== APPLY SELECTED FILTER ======================== #
        if self.filter.is_bias_aware and activate_bias_aware:
            assert self.bias is not None, "Bias-aware filter selected but no bias instance found. Please initialize self.bias with a Bias instance before calling analysis_step."

            # ----------------- Retrieve bias and its Jacobian ----------------- #
            b = self.bias.current_bias
            J = self.bias.state_derivative()

            if b.ndim == 2 and b.shape[-1] not in (1, Af.shape[-1]):
                # Bias-estimator ensemble size differs from the model ensemble size:
                # use the mean bias (the bias is defined on the ensemble mean).
                b = np.mean(b, axis=-1, keepdims=True)

            if self.bias.biased_observations:
                # Adjust observations if they are biased
                obs_bias = np.mean(b - self.bias.current_innovations, axis=-1)
                d = d + obs_bias


            # -------------- Define bias Covariance and the weight -------------- #
            Cbb = Cdd.copy()  # Bias covariance matrix same as obs cov matrix for now

            Aa = self.filter(Af, d, Cdd, Cbb, b, J)

        elif self.filter.is_bias_aware:
            # Bias-aware filter selected but still within the bias-blind window: apply a plain EnKF
            if not hasattr(self, '_bias_blind_filter'):
                self._bias_blind_filter = EnKF(M=self.filter._M)
            Aa = self._bias_blind_filter(Af, d, Cdd)

        else:
            Aa = self.filter(Af, d, Cdd)

        # Inflate the analysis
        if self.inflation_factor > 1.0:
            Aa = self.inflate(Aa, self.inflation_factor, d=d, additive=True)

        # =========== CHECK SPREAD AND PARAMETERS  ========== #
        if not self.has_valid_spread(Aa[:self.model.Nphi, :]):
            self.rejected_analysis = (self.current_time,  'Invalid analysis spread')

        if self.Na > 0 and self.activate_parameter_estimation and self.alpha_limits_matrix is not None:
            Aa_alpha = Aa[self.Nphi:self.Nphi+self.Na, :]
            is_physical, idx_alpha, _ = self.has_valid_params(Aa_alpha, self.alpha_limits_matrix, get_deltas=False)
            if not is_physical:
                # reject analysis and inflate forecast with (higher) factor
                self.rejected_analysis = (self.current_time, f'Non-physical parameters {idx_alpha}')
                Aa = self.inflate(Af, self.inflation_factor_rejection, d=d, additive=True)

        # =========== UPDATE MODEL HISTORY ========== #
        if Af_params is not None:
            # Parameter estimation deactivated: re-append the (unchanged) forecast parameters
            Aa_psi = np.vstack((Aa[:self.model.Nphi, :], Af_params))
        else:
            Aa_psi = Aa[:self.model.Nphi + self.Na, :]

        self.update_history_analysis(Aa_psi, d)
        self.assimilated_data = (d, self.current_time)


    def update_history_analysis(self, Aa_psi: np.ndarray, d: np.ndarray) -> None:
        """
        Stores the analysis state in the model history and updates the bias estimator
        with the analysis innovation i^a = d - y^a.

        Parameters
        ----------
        Aa_psi : np.ndarray
            Analysis state (Nphi + Na, m), i.e., without the augmented observables.
        d : np.ndarray
            (Bias-corrected) observation vector assimilated at the current time.
        """
        self.model.update_history(Aa_psi, self.current_time, update_last_state=True)

        if self.bias is not None:
            # Innovation of the analysis: difference between data and analysis observables
            Ya = self.model.get_observables()                       # (Nq, m)
            ia = d[:, np.newaxis] - Ya                              # (Nq, m)
            updated_state = self.bias.update_state_from_innovation(ia)
            self.bias.update_history(updated_state, t=self.bias.current_time,
                                     update_last_state=True)

    

    @property
    def rejected_analysis(self):
        """
        Property for managing rejected analysis steps during data assimilation.

        Returns
        -------
        namedtuple
            Contains lists of times and reasons for each rejected analysis.
        """
        return self._rejected_analysis
    
    @rejected_analysis.setter
    def rejected_analysis(self, value: tuple):
        
        if not hasattr(self, '_rejected_analysis'):
            RejectedData = namedtuple('RejectedData', ['times', 'reasons'])
            self._rejected_analysis =  RejectedData(times=[], reasons=[])

        time, reason = value
        self._rejected_analysis.times.append(time)
        self._rejected_analysis.reasons.append(reason)
        

        print(f'Number of non-physical analysis = {len(self._rejected_analysis.times)}/{len(self.assimilated_data.times)+1}')

        

    @staticmethod
    def inflate(A: np.ndarray, rho: float, d=None, additive=True) -> np.ndarray:
        """
        Inflates the ensemble around its mean by a factor rho.

        Parameters
        ----------
        A : np.ndarray
            Ensemble state array of shape (state_dim, m).
        rho : float
            Inflation factor.
        d : np.ndarray, optional
            Observation vector used for additive inflation. Should be a 1D array of shape (obs_dim,).
            If None, inflation is applied only to the ensemble state.
        additive : bool, optional
            If True, perform additive inflation (adds scaled difference to ensemble mean).
            If False, perform multiplicative inflation (scales deviations from the mean).
            Default is True.

        Side effects
        ------------
        - Updates the model's history with the inflated ensemble state.

        """

        if d is not None and additive is False:
            raise NotImplementedError('Non-additive inflation with observation vector not implemented yet.')
            # d = np.asarray(d).reshape(-1)  # Ensure d is 1D
            # A[:len(d)] += (d * (rho - 1))[:, np.newaxis]  # Broadcast correctly

        A_m = np.mean(A, -1, keepdims=True)
        return A_m + rho * (A - A_m)





    @staticmethod
    def has_valid_spread(A: np.ndarray, tol=1e-6) -> bool:
        """
        Checks if the ensemble spread is valid (not too large).
        Parameters
        ----------
        A : np.ndarray
            Ensemble state array of shape (state_dim, m).
        Returns
        -------
        bool
            True if the spread is valid, False if too large.
        """
        return True  # Temporarily disable spread check
        # val = np.var(A) / (np.mean(A, axis=-1)**2 + tol)
        # condition = val < 1.0

        # print('Spread check condition per state variable:', condition, val, np.mean(A, axis=-1), np.std(A, axis=-1))

        # return np.all(condition)



    @property
    def alpha_limits_matrix(self) -> Optional[np.ndarray]:
        if not hasattr(self, '_alpha_lims'):
            # Only the limits of the estimated parameters, in est_alpha order
            lows, highs = [], []
            for key in self.est_alpha:
                lo, hi = self.model.alpha_lims.get(key, (None, None))
                lows.append(-np.inf if lo is None else float(lo))
                highs.append(np.inf if hi is None else float(hi))

            alpha_lims = np.array([lows, highs])  # Shape: (2, Na)

            # If all limits are unbounded, skip the check entirely
            if alpha_lims.size == 0 or not np.any(np.isfinite(alpha_lims)):
                self._alpha_lims = None
            else:
                self._alpha_lims = alpha_lims[:, :, np.newaxis]  # Shape: (2, Na, 1)

        return self._alpha_lims
    


    @staticmethod
    def has_valid_params(A_alpha: np.ndarray, alpha_limits_matrix: np.ndarray, get_deltas=False) -> Tuple[bool, List[int], np.ndarray]:
        """
        Checks if the ensemble parameters are within physical bounds.
        Parameters
        ----------
        A_alpha : np.ndarray
            Ensemble parameter array of shape (Na, m).
        alpha_limits : dict
            Dictionary containing parameter bounds.
        Returns
        -------
        Tuple[bool, List[int], np.ndarray]
            - True if all parameters are within bounds, False otherwise.
            - List of indices of parameters that are out of bounds.
            - Array of maximum allowed values for out-of-bounds parameters (if get_deltas is True).
        """

        if alpha_limits_matrix is None:
            return True, None, None

        is_physical, idx_alpha, d_alpha = True, [], []  

        # Masks for out-of-bounds
        low_limits, high_limits = alpha_limits_matrix  # Shape: ((Na, 1), (Na, 1))

        below = A_alpha < low_limits
        above = A_alpha > high_limits

        oob = np.any(below | above, axis=1)
        is_physical = not np.any(oob)
        idx_alpha = np.where(oob)[0].tolist()

        if get_deltas and not is_physical:
            if np.any(above) and np.any(below):
                raise ValueError('both above and below limits detected simultaneously, check alpha_limits and A_alpha')
            
            allowed = A_alpha[~above & ~below]
            # compute deltas only where needed to avoid inf arithmetic warnings
            if np.any(below):
                #append the maximum value of the ensemble within limits
                if allowed.size > 0:
                    max_allowed = np.max(allowed, axis=1)
                else:
                    max_allowed = low_limits[:,0]  # use lower limit if no allowed values  
                d_alpha.append(max_allowed)

            elif np.any(above):
                #append the minimum value of the ensemble within limits
                if allowed.size > 0:
                    min_allowed = np.min(allowed, axis=1)
                else:
                    min_allowed = high_limits[:,0]  # use upper limit if no allowed values  
                d_alpha.append(min_allowed)
                

        return is_physical, idx_alpha, np.array(d_alpha)
    

    # _______________________________________________________________________________________________________________
    # 
    # VISUALIZATION METHODS ##
    # _______________________________________________________________________________________________________________



    def print_parameters(self) -> None:
        """
        Prints the ensemble configuration parameters in a readable format.
        Side effects
        ------------    
        - Outputs ensemble configuration and model/bias parameters to the console.

        """
        print("Ensemble Configuration Parameters:")
        for key, val in self.config().items():
            print(f"  {key}: {val}")    

        self.model.print_parameters()
        if self.filter is not None:
            self.filter.print_parameters()
        if self.bias is not None:
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
        
        kwargs_state = allowed_kwargs_for_func(plot_state_distribution, kwargs)
        
        plot_state_distribution(self.model, **kwargs_state)

        #TODO: plot_ensemble_  





# 
# ===== AUXILIARY PLOTTING FUNCTIONS ===== #


def normalized_time(reference_t: float, *times) -> Tuple[Sequence[Optional[np.ndarray]], str]:
    # Normalize all inputs to ndarray (or None) to keep a stable return type

    if reference_t == 1.0:
        t_label = '$t$'
        normalized_times = [t if t is not None else t for t in times]
    else:
        t_label = f'$t/{reference_t}$'
        normalized_times = [
            t / reference_t if t is not None else t for t in times
        ]

    return times, t_label



def normalized_y(reference_y: Union[float, np.ndarray], y_lables, *ys) -> Tuple[Sequence[Optional[np.ndarray]], Sequence[Optional[str]]]:

    # if ys is only one array, make it a list
    if not isinstance(ys, (list, tuple)):
        ys = [ys]
    # check that all ys have same Ny unless they are None
    Nys = [y.shape[1] for y in ys if y is not None]
    assert len(Nys) > 0, 'At least one y must be provided.'
    assert all(n == Nys[0] for n in Nys), 'All ys must have the same Ny dimension.'
    Ny = Nys[0]
    # ensure reference_y is an array
    if isinstance(reference_y, (int, float)):
        if reference_y == 1.:
            normalize_y = False
        else:
            normalize_y = True
        reference_y = reference_y * np.ones(Ny)
    else:
        normalize_y = True
        if len(reference_y) != Ny:
            raise ValueError('reference_y must be a scalar or an array of length Ny.')
        
    if not normalize_y:
        return ys, y_lables
    else:

        reference_y = reference_y[np.newaxis, :, np.newaxis]

        ys = [y.copy() / reference_y if y is not None else y for y in ys]     
        y_lables = [f'{y_lables[qi]} / ${reference_y[0, qi, 0]}$' for qi in range(Ny)]

        return ys, y_lables


def normalized_alpha(alpha, alpha_keys, alpha_labels, reference_a=None) -> Tuple[np.ndarray, 
                                                                                 Dict[str, str]]:
    
    reference_alpha = {key: 1. for key in alpha_keys}
    alpha_lbls = alpha_labels.copy()
    alpha = alpha.copy()

    if reference_a is not None and isinstance(reference_a, dict):
        for ai, key in enumerate(alpha_keys):
            if key not in reference_a.keys():
                reference_a[key] = 1.
            else:
                reference_alpha[key] = reference_a[key]
                alpha_lbls[key] += f' / {reference_alpha[key]}'
            
            alpha[:, ai] = alpha[:, ai] / reference_alpha[key]

    return alpha, alpha_lbls


def cut_signals(t, *signals, min_time=None, max_time=None):
    """ Cut signals to interval of interest.
    """

    if min_time is None:
        i0 = 0
    else:
        i0 = np.argmin(abs(t - min_time))

    if max_time is None:
        i1 = len(t) - 1
    else:
        i1 = np.argmin(abs(t - max_time))

    t_cut = t[i0:i1]
    signals = [sig[i0:i1].copy() if sig is not None else None for sig in signals]

    # Nomalize time and observations ----           
    return t_cut, signals



def plot_alpha_history(ensemble : Ensemble, 
                    plot_members: bool = False,
                    reference_a=None, 
                    reference_t=1.,
                    max_time=None) -> None:
    """
    Plot the time evolution of the parameters in a object of class model
    """
    pm = ensemble.model

    C = Palette()
    c1 = C.get_color_params(n=ensemble.Na, alpha=1)
    c2 = C.get_color_params(n=ensemble.Na, alpha=.2)

    (t,), t_lbl = normalized_time(reference_t, pm.hist_t)
    assert isinstance(t, np.ndarray), "Model history time array is required."

    hist_alpha, alpha_lbls = normalized_alpha(pm.hist[:, -pm.Na:], 
                                              alpha_keys=ensemble.est_alpha, 
                                              alpha_labels=pm.alpha_labels,
                                              reference_a=reference_a)

    mean_alpha = np.mean(hist_alpha, axis=-1)
    std_alpha = np.std(hist_alpha, axis=-1)

    t_margin = pm.t_CR 
    t_obs = np.array(ensemble.assimilated_data.times)

    if len(t_obs) > 0:
        if max_time is None:
            max_time = min(t_obs[-1] + t_margin, t[-1])
        min_time = t_obs[0] - 0.25 * t_margin       
    else:
        min_time, max_time = t[0], t[-1]

    x_lims = [[min_time, min_time + t_margin], [max_time - t_margin, max_time], [min_time, max_time]]    


    fig = plt.figure(figsize=(12, 2*ensemble.Na), layout="constrained")
    axs = fig.subplots(ensemble.Na, 3, sharex='col', sharey='row', width_ratios=[1, 1, 3])
    
    if ensemble.Na == 1:
        axs = [axs]

    for row_i, axs_row, p in zip(range(ensemble.Na), axs, ensemble.est_alpha):

        avg, s, all_h = [xx[:, row_i] for xx in [mean_alpha, std_alpha, hist_alpha]]

        for col_i, ax in enumerate(axs_row):
                
            if plot_members:
                ax.plot(t, all_h, color=c2[row_i], lw=1.)

            ax.fill_between(t, avg + 2 * abs(s), avg - 2 * abs(s), alpha=0.2, color=c2[row_i], label='2 std')
            ax.plot(t, avg, color=c1[row_i], label='mean', lw=2, dashes=(5,1))

            if col_i == 0:
                ax.set(ylabel=alpha_lbls[p], 
                        ylim=[min(avg) - 3 * max(s), max(avg) + 3 * max(s)])
            if row_i == ensemble.Na - 1:
                ax.set(xlabel=t_lbl, xlim=x_lims[col_i])



def plot_observable_history(ensemble : Ensemble,
                            truth : Optional[Observations] = None, 
                            plot_members : bool = False,
                            reference_y=1., 
                            reference_t: float = 1., 
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

    pm, pb = ensemble.model, ensemble.bias    
    
    y_unbiased, y_model = ensemble.get_observable_hist()

    # Normalize and cut signals to interval of interest -----
    (y_unbiased, y_model), y_labels = normalized_y(reference_y, pm.obs_labels, 
                                                    y_unbiased, y_model)
    
    (t, t_margin), t_label = normalized_time(reference_t, 
                                             pm.hist_t.copy(),  np.array(pm.t_CR))
    assert isinstance(t, np.ndarray), "Model history time array is required for plotting observable history."

    if  len(ensemble.assimilated_data.times) > 0: 
        t_obs = np.array(ensemble.assimilated_data.times)
        if max_time is None:
            max_time = min(t_obs[-2] + t_margin, t[-1]) 
        min_time = t_obs[0] - 0.25 * t_margin  #type: ignore
    else:         
        t_obs = None
        min_time, max_time = t[0], t[-1]

    t, (y_model, y_unbiased) = cut_signals(t, y_model, y_unbiased, min_time=min_time, max_time=max_time)
    assert y_model is not None, "Model history is required for plotting observable history."
    
    # Get truth if available ----
    if truth is not None:
        y_raw  =  np.asarray(truth.y_raw.copy()) 
        y_true = np.asarray(truth.y_true.copy()) 
        t_true = np.asarray(truth.t_true.copy()) 
        
        t_true = normalized_time(reference_t,  t_true)[0][0]
        (y_raw, y_true), y_labels = normalized_y(reference_y, pm.obs_labels, 
                                                  y_raw, y_true)

        t_true, (y_raw, y_true) = cut_signals(t_true, y_raw, y_true,  min_time=min_time, max_time=max_time)
        if len(t) != len(t_true):
            y_raw = interpolate(t_true, y_raw, t)
            y_true = interpolate(t_true, y_true, t) 
        
        assert y_true is not None, "True history is required for plotting observable history when truth is provided."
        y_margin = 0.15 * np.mean(abs(y_true), axis=(0, 2)) 
        max_y = np.max(y_true, axis=(0, 2), keepdims=False) 
        min_y = np.min(y_true, axis=(0, 2), keepdims=False)
    else:
        assert y_model is not None, "Model history is required for plotting observable history when truth is not provided."
        y_margin = 0.15 * np.mean(abs(y_model), axis=(0, 2))
        max_y = np.max(y_model, axis=(0, 2), keepdims=False)
        min_y = np.min(y_model, axis=(0, 2), keepdims=False)
        y_true, y_raw = None, None

    # Get observations if available ----
    if  t_obs is not None:
        t_obs = np.array(ensemble.assimilated_data.times)
        y_obs = np.array(ensemble.assimilated_data.data)[..., np.newaxis]
        t_obs = normalized_time(reference_t, t_obs)[0][0]  # Normalize observation time using the same reference time as the model history
        
        assert y_obs is not None, "Observation history is required for plotting observable history."
        assert t_obs is not None and t_obs.shape[0] == y_obs.shape[0], "Time dimension of observations does not match time array for plotting observable history."

        y_obs = normalized_y(reference_y, pm.obs_labels, y_obs)[0][0] 


    # % PLOT time series ------------------------------------------------------------------------------------------
    
    Nq = pm.Nq
    if dims == 'all':
        dims = np.arange(Nq, dtype=int)
    elif isinstance(dims, int):
        dims = [dims]
    else:
        dims = [int(d) for d in dims]
        
    fig1 = plt.figure(figsize=(12, 2 * len(dims)), layout="constrained")
    
    ax_all = fig1.subplots(nrows=len(dims), ncols=3, sharey='row', sharex='col', width_ratios=[1,1,3])
    if len(dims) == 1:
        ax_all = ax_all[np.newaxis, :]
        
    x_lims = [[min_time, min_time + t_margin], [max_time - t_margin, max_time], [t[0], max_time]]    

    for row_i, qi in enumerate(dims):
        
        yl = [min_y[qi] - y_margin[qi], max_y[qi] + y_margin[qi]]

        for col_i, (ax, xl) in enumerate(zip(ax_all[row_i], x_lims)):
            if y_true is not None:
                ax.plot(t, y_true[:, qi, :], label='truth', **C.true_props)
                if y_raw is not None:
                    ax.plot(t, y_raw[:, qi], label='raw truth', **C.true_noisy_props)

            if isinstance(y_unbiased, np.ndarray) and y_unbiased.ndim == 3:
                if plot_members:
                    first_member = True
                    for mi in range(y_unbiased.shape[-1]):
                        if first_member:
                            props = C.y_unbias_props.copy()
                            props['label'] = 'bias-corrected members' 
                            first_member = False
                            ax.plot(t, y_unbiased[:, qi, mi], **props)
                        else:
                            ax.plot(t, y_unbiased[:, qi, mi], **C.y_unbias_props)
                else:
                    ax.plot(t, np.mean(y_unbiased[:, qi], axis=-1), label='bias-corrected estimate', **C.y_unbias_props)

            m = np.mean(y_model[:, qi], axis=-1)
            ax.plot(t, m, **C.y_biased_mean_props, label='model estimate')

            if plot_members:
                first_member = True
                for mi in range(y_model.shape[-1]):
                    if first_member:
                        props = C.y_biased_props.copy()
                        props['label'] = 'ensemble members' 
                        first_member = False
                        ax.plot(t, y_model[:, qi, mi], **props)
                    else:
                        ax.plot(t, y_model[:, qi, mi], **C.y_biased_props)
            else:
                s = np.std(y_model[:, qi], axis=-1)
                ax.fill_between(t, m + s, m - s, color=C.get_color('BIASED', 0.5))

            if t_obs is not None:
                ax.plot(t_obs, y_obs[:, qi], label='data', **C.obs_props) # type: ignore 

            if col_i == 0:
                ax.set(ylabel=y_labels[row_i])
                if row_i == 0:
                    fig1.legend(loc="center", bbox_to_anchor=(.5, 1.05), ncol=6, frameon=False)

            ax.set(ylim=yl, xlim=xl)
            if row_i == len(dims) - 1:
                ax.set(xlabel=t_label)
                
                

def plot_state_distribution(model: Model, 
                            time_indices=[-1], 
                            max_modes=None, 
                            reference_a=None, 
                            nbins=6) -> None:
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
    reference_a : dict or None, optional (default None)
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
        alpha_hist, alpha_labels, est_alpha = None, None, None
    else:
        ncols_alpha = min(model.Na, 4)
        nrows_alpha = int(np.ceil(model.Na / ncols_alpha))

        est_alpha = model.est_alpha

        alpha = model.hist[:, -model.Na:, :] # Shape: (T, Na, m)
        alpha_hist, alpha_labels = normalized_alpha(alpha, 
                                               est_alpha, 
                                               model.alpha_labels, 
                                               reference_a=reference_a)


    def add_stats_text(_ax, yy):
        # add text with mean and std relative
        mean_yy = np.mean(yy)
        std_yy = np.std(yy) / mean_yy if mean_yy != 0 else 0
        textstr = '\n'.join((f'Mean: {mean_yy:.4f}', f'Std: {std_yy:.4f}'))
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        _ax.text(0.95, 0.95, textstr, transform=_ax.transAxes, fontsize='x-small',
                verticalalignment='top', horizontalalignment='right', bbox=props)

    state_labels = model.state_labels

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
            


        # Plot all in one mosaic
        fig = plt.figure(figsize=(2*max(ncols_phi, ncols_alpha), 2 * (nrows_phi + nrows_alpha)), layout='constrained')
        plt.suptitle(f'Ensemble distributions at time t={model.hist_t[ti]:.3f}')

        if model.Na == 0:
            sf: List[SubFigure] = [fig.subfigures(nrows=1, ncols=1)] 
        else:
            sf: List[SubFigure] = fig.subfigures(nrows=2, ncols=1, height_ratios=[nrows_phi, nrows_alpha], wspace=0.07, hspace=0.15)

        axs = sf[0].subplots( ncols=ncols_phi, nrows=nrows_phi, sharey=True) 
        axs = axs.ravel() if ncols_phi * nrows_phi > 1 else [axs]

        for ax, ph, lbl in zip(axs, phi, state_labels):
            ax.hist(ph, bins=nbins, color='tab:green')
            ax.set(xlabel=lbl)
            add_stats_text(ax, ph)
        
        if model.Nphi < len(axs):
            [ax.axis('off') for ax in axs[model.Nphi:]]

        if model.Na > 0:
            # assert 
            # assert all the alpha_hist, alpha_labels, est_alpha are defined
            assert alpha_hist is not None and alpha_labels is not None and est_alpha is not None, "Alpha history, labels, and keys must be defined for plotting alpha history."

            alpha = alpha_hist[ti, :, :]
            axs = sf[1].subplots(ncols=ncols_alpha, nrows=nrows_alpha, sharey=True)
            axs = axs.ravel() if ncols_alpha * nrows_alpha > 1 else [axs]

            for ax, a, param in zip(axs, alpha, est_alpha):
                ax.hist(a, bins=nbins)
                ax.set(xlabel=alpha_labels[param])
                add_stats_text(ax, a)
        
            if model.Na < len(axs):
                for ax in axs[model.Na:]:
                    ax.axis('off')