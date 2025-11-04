import numpy as np
from copy import deepcopy
from typing import List, Union, Dict, Any, Type
from bias import Bias, NoBias
from model import Model



class Ensemble(object):
    """
    Manages ensemble-related properties and methods for a Model,
    primarily for ensemble forecasting and data assimilation (e.g., EnKF).

    This class handles configuration, initialization, and uncertainty generation.
    """

    # --- 1. Class-Level Default Settings (Attributes) ---
    m: int = 10  # Number of ensemble members
    

    # Data assiomilation specific parameters
    filter: str = 'EnKF'

    est_phi: bool = True                        # Estimate model state?
    est_alpha: Union[bool, List[str]] = False   # Estimate parameters? If a List, includes the names of parent_model parameters to estimate
    est_bias: bool = False                      # Estimate bias?
    

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
        
        self.rng = parent_model.rng 
        self = parent_model
        self.parent_bias = parent_bias
        
        # 2. Apply configuration overrides and ensure consistency
        self._apply_config(**kwargs)
        
        # 3. Initialize ensemble state and history in the parent model
        self._init_ensemble_state(**kwargs)
        
        # 4. Initialize bias
        self._init_bias(parent_bias, **kwargs)
        
    # ------------------ CONFIGURATION AND INITIALIZATION METHODS ------------------ ##

    def _apply_config(self, **kwargs):
        """Applies kwargs to the instance, ensuring est_alpha and std_a are consistent."""
        
        DAdict = kwargs.copy()
        
        # Apply only attributes that exist on the Ensemble class
        for key, val in kwargs.items():
            if hasattr(Ensemble, key):
                setattr(self, key, DAdict.pop(key))
            
        # Ensure est_alpha is correctly set up from potential dict input in std_a
        # This is a critical step for parameter estimation setup
        if isinstance(self.std_a, dict):
            # If std_a is a dict, est_alpha must be the list of its keys
            self.est_alpha = list(self.std_a.keys())
        elif not self.est_alpha:
            self.est_alpha = []
        
        return DAdict




    def _init_ensemble_state(self, seed: int = None, ensemble_phi0: np.ndarray = None):
        """Initializes the ensemble state and augments with parameter uncertainty."""
        pm = self.parent_model
        
        if seed is not None:
            pm.rng = seed # Triggers the RNG setter in Model
        
        # 1. Generate initial state (psi) ensemble
        if ensemble_phi0 is None:
            # Assumes pm.get_current_state returns the mean initial condition (Nphi, 1)
            mean_psi0 = np.mean(pm.get_current_state, axis=-1)
            ensemble_phi0 = self.add_uncertainty(pm.rng, mean_psi0, self.std_phi,
                                                 self.m, method=self.distribution_phi, 
                                                 ensure_mean_at_init=self.ensure_mean_at_init)
        
        # 2. Augment ensemble with estimated parameters (alpha)
        if self.est_alpha:  
            mean_a = np.array([getattr(pm, pp) for pp in self.est_alpha])
            new_alpha0 = self.add_uncertainty(pm.rng, mean_a, self.std_a, self.m,
                                               method=self.distribution_alpha, 
                                               ensure_mean_at_init=self.ensure_mean_at_init)
            # Stack state (psi) and parameters (alpha) -> new state vector
            ensemble_psi0 = np.vstack((ensemble_phi0, new_alpha0))
        else:
            ensemble_psi0 = ensemble_phi0
        
        # 3. Update the parent model's history (resets initial condition)
        pm.update_history(psi=ensemble_psi0, reset=True)
        
        # 4. Update parent model settings/filename
        pm.filename += '_{}_ensemble_m{}'.format(getattr(pm, 'name', 'Model'), self.m)
        pm.modify_settings()



    def _init_bias(self, parent_bias: Bias, bias_model: Type[Bias] = NoBias, **Bdict):
        """Initializes the bias model instance for the ensemble."""

        if 'bias_type' in Bdict or 'model' in Bdict:
            raise ValueError("Please use 'bias_model' to specify the bias class.")

        # If a bias model is explicitly passed via kwargs, use that, otherwise use parent_bias.
        bias_class = Bdict.pop('bias_model', type(parent_bias) if isinstance(parent_bias, Bias) else NoBias)
        
        if isinstance(parent_bias, bias_class) and not isinstance(parent_bias, NoBias):
            # Use the existing, configured bias instance if it matches the requested type
            self.bias = parent_bias
        else:
            # Determine initial observation vector (y0) for bias initialization
            try:
                # Get observable for one member to determine dimension
                y0_all = self.parent_model.get_observables(psi=self.parent_model.get_current_state[:, :1])
                y0 = np.mean(y0_all, axis=-1)
                if y0.ndim > 2:
                    y0 = y0.squeeze(axis=-1)
            except (AttributeError, IndexError):
                # Fallback if the model cannot yet produce observables
                y0 = np.zeros(self.parent_model.Nq) 
                
            # Initialize the new bias model instance
            # We assume self.parent_model has 'dt' and 'get_current_time' properties
            self.bias = bias_class(
                y=y0, 
                t=self.parent_model.get_current_time, 
                dt=self.parent_model.dt, 
                **Bdict
            )

    # ------------------ ENSEMBLE GENERATION METHODS ------------------ ##


    @staticmethod
    def add_uncertainty(rng: np.random.Generator, 
                        mean: np.ndarray, 
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
            
        mean = np.asarray(mean).flatten()
        
        # Case 1: std is a dictionary (for estimated parameters 'alpha')
        if isinstance(std, dict):
            param_ensembles = []
            for sa in std.values():
                if method == 'uniform':
                    # For uniform, std values are [min_val, max_val]
                    param_ensembles.append(rng.uniform(low=sa[0], high=sa[1], size=m))
                else: # normal
                    # Use mean of bounds as location, and half the range as a heuristic scale (std)
                    loc = np.mean(sa)
                    scale = (sa[1] - sa[0]) / 4.0 if isinstance(sa, list) and len(sa) == 2 else loc * 0.5
                    param_ensembles.append(rng.normal(loc=loc, scale=scale, size=m))
            ensemble_ = np.array(param_ensembles) # Shape: (num_params, m)

        # Case 2: std is a single float (relative standard deviation for state 'psi')
        elif isinstance(std, float):
            if method == 'uniform':
                # Multiplicative uniform perturbation: mean * (1 +/- std)
                perturbation = 1.0 + rng.uniform(-std, std, size=(mean.size, m))
                ensemble_ = mean[:, np.newaxis] * perturbation
            
            else: # normal (using multivariate normal for state vector)
                if np.iscomplexobj(mean):
                    # Handle complex state by perturbing real and imaginary parts independently
                    cov_real = np.diag((mean.real * std) ** 2)
                    cov_imag = np.diag((mean.imag * std) ** 2)
                    real_part = rng.multivariate_normal(mean.real, cov_real, size=m).T
                    imag_part = rng.multivariate_normal(mean.imag, cov_imag, size=m).T
                    ensemble_ = real_part + 1j * imag_part
                else:
                    # Covariance matrix is diagonal, perturbation scaled by mean and relative std
                    cov = np.diag((mean * std) ** 2)
                    ensemble_ = rng.multivariate_normal(mean, cov, size=m).T
            
        else:
            raise TypeError(f'Initial std must be a float or a dict, not {type(std)}')

        if ensure_mean_at_init and ensemble_ is not None:
            # Replace the first member with the unperturbed mean
            ensemble_[:, 0] = mean

        return ensemble_

    def reshape_ensemble(self, m: int = None, reset: bool = True) -> Model:
        """
        Reshapes the ensemble state (resampling/re-perturbing) and returns a 
        new Model instance with the updated ensemble.
        """
        # Use deepcopy to ensure a clean, independent Model instance
        model = deepcopy(self.parent_model)
        # Ensure the copied Ensemble links back to the copied model
        model.ensemble.parent_model = model

        if m is None:
            m = self.m
            
        # Get the current ensemble state from the ORIGINAL model
        current_psi = self.parent_model.get_current_state # (state_dim, current_m)
        
        # Calculate the mean state across current ensemble members
        mean_psi = np.mean(current_psi, axis=-1) # (state_dim,)
        
        if m == 1:
            # Reduce to a single mean member
            new_psi = mean_psi[:, np.newaxis]
            model.ensemble.ensemble = False
        else:
            # Calculate standard deviation for re-perturbation
            std_phi = np.std(current_psi, axis=-1) # (state_dim,)
            
            # Simple re-perturbation: Mean + (Std * random_normal)
            perturbation = self.rng.normal(0, 1, size=(mean_psi.size, m))
            new_psi = mean_psi[:, np.newaxis] + std_phi[:, np.newaxis] * perturbation
            
            model.ensemble.ensemble = True

        # Update the new model's history
        model.update_history(psi=new_psi, t=self.parent_model.get_current_time, reset=reset)
        
        # Update the ensemble size property on the new model's ensemble object
        model.ensemble.m = m 
        
        return model