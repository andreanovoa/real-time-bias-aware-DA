
import os

import numpy as np
from romda.utils import interpolate

__all__ = [
    'Integrator',
    'ConstantIntegrator',
    'DiscreteIntegrator',
    'IVPIntegrator',
    'ivp_forecast_helper',
]

from scipy.integrate import solve_ivp
from functools import partial
from copy import deepcopy

from typeguard import typechecked

from typing import Dict, Tuple, Any

import numpy as np

from sys import platform

if platform == "darwin" or platform == "ios":
    import multiprocess as mp
else:
    import multiprocessing as mp




# %% =================================== INTEGRATOR BASE CLASS ============================================= %% #
class Integrator:
    """
    Abstract Base Class for all time integration strategies.
    Defines the interface for advancing the model state.
    Child classes must implement advance_single and advance_ensemble methods.
    ------
    Implemented Integrator Strategies:
        IVPIntegrator - for continuous, variable-step integration using scipy's solve_ivp
            Governing equations: d(psi)/dt = f(t, psi, alpha). 
            * The time_derivative method must be defined by the model
        DiscreteIntegrator - for fixed, discrete-step integration (e.g., ETDRK4, ESN)
            Solving psi_{t+dt} = F(psi_{t}, alpha)
            * The time_step method must be defined by the model
        ConstantIntegrator - holds state constant (for testing or NoBias)
            returns psi(t) = psi(0)
    """

    @typechecked
    def __init__(self, model_instance: object):
        """
        Initialize the integrator with a model instance (not necessarily a Model, but must have the time_derivative/time_step method).
        Parameters:
            model_instance: A pointer instance to access self.time_derivative, self.dt, etc.
        """        

        self.model = model_instance

    @property
    def is_ensemble(self):
        current_state = self.model.current_state
        if current_state.ndim >= 2 and current_state.shape[-1] > 1:
            return True
        else:
            return False

    def close(self):
        """ Close resources held by the integrator (e.g., multiprocessing pools). """
        pass    

    def advance(self, averaged=False, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        The common interface for all integrators.
        Must return: (psi_forecasted[1:], t_forecasted[1:])
        """
        if not self.is_ensemble:
            return self.advance_single(**kwargs)
        else:
            return self.advance_ensemble(averaged=averaged, **kwargs)
    

    def advance_single(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        The common interface for all integrators.
        Must return: (psi_forecasted[1:], t_forecasted[1:])
        """
        raise NotImplementedError("Child Integrator class must implement the advance_single() method.")
    

    def advance_ensemble(self, Nt: int = 100, averaged: bool = False, alpha: Dict[str, Any] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        The common interface for all integrators.
        Must return: (psi_forecasted[1:], t_forecasted[1:])
        """
        raise NotImplementedError("Child Integrator class must implement the advance_ensemble() method.")


class ConstantIntegrator(Integrator):
    """
    Integrator that holds the state constant over time.
    Useful for testing or as a placeholder.
    """

    def __init__(self, model_instance):
        super().__init__(model_instance)

    def advance_single(self, Nt: int = 100, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        model = self.model
        t_out = np.round(model.current_time + np.arange(Nt + 1) * model.dt, model.precision_t)
        psi = np.repeat(model.current_state[:, :, np.newaxis], Nt + 1, axis=2)
        # return psi, t_out, psi shoud have dimensions Nt x N x m
        psi = psi.transpose((2, 0, 1))  # Nt+1 x N x m

        return psi[1:], t_out[1:]

    def advance_ensemble(self, Nt: int = 100, averaged: bool = False, alpha: Dict[str, Any] = None) -> Tuple[np.ndarray, np.ndarray]:
        return self.advance_single(Nt=Nt, averaged=averaged, alpha=alpha)



# %% =================================== CONCRETE STRATEGY 2: DISCRETE STEP ============================================= %% #
class DiscreteIntegrator(Integrator):
    """
    Integrator for models using a fixed, discrete map or scheme (single member). (e.g., ETDRK4 in KS, or ESN).
    """

    def __init__(self, model_instance):
        super().__init__(model_instance)
        
        self.dt_output = getattr(self.model, 'dt')
        self.dt_integrator = getattr(self.model, 'dt_step')

        if self.dt_output != self.dt_integrator:
            self.relation_integrator_output = self.dt_output / self.dt_integrator
            self.relation_integrator_output = round(self.relation_integrator_output, self.model.precision_t)
        else:
            self.relation_integrator_output = 1.0


    def advance_single(self, Nt: int = 100, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        model = self.model
        
        t_out = np.round(model.current_time + np.arange(Nt + 1) * self.dt_output, model.precision_t)

        Nt_step = int(np.ceil(Nt * self.relation_integrator_output))

        psi, t = model.time_step(Nt=Nt_step)

        if len(t_out) == len(t):
            return psi[1:], t[1:]
        else:
            # Interpolate
            assert t[-1] >= t_out[-1], "do not extrapolate beyond the integrator time range, {} vs {}".format(t[-1], t_out[-1])
            
            psi_interp = interpolate(t, psi, t_eval=t_out, fill_values='extrapolate')
            # model.reset_last_state(psi_interp[-1], t_out[-1])
            return psi_interp[1:], t_out[1:]
        
    def advance_ensemble(self, Nt = 100, averaged = False, alpha = None):

        return self.advance_single(Nt, averaged=averaged, alpha=alpha)





# %% ===================================  IVP SOLVER ============================================= %% #
class IVPIntegrator(Integrator):
    """ Integrator using Scipy's solve_ivp for continuous, variable-step integration. 
         Governing equations: d(psi)/dt = f(t, psi, alpha). 
            The time_derivative method must be defined by the model
    
    """
    def __init__(self, model_instance, method: str = 'RK45'):
        super().__init__(model_instance)
        self.method = method

    @property
    def __pool(self):
        
        if not hasattr(self, '_pool'):
            self._pool = None

        if self._pool is None and self.model.m > 1:
            # Initialize multiprocessing pool
            N_pools = min(self.model.m, mp.cpu_count())
            print(f'Initializing multiprocessing pool for IVPIntegrator with m={self.model.m} and {N_pools} pools.')
            self._pool = mp.Pool(N_pools)
        return self._pool


    def close(self):
        if hasattr(self, '_pool'):
            self.__pool.close()
            self.__pool.join()
            delattr(self, "_pool")
        else:
            # print("No multiprocessing pool to close.")
            pass

    def advance_single(self, Nt = 100, averaged=False, alpha = None):
        # print('Using IVPIntegrator advance_single')
        pm = self.model
        
        t_all = np.round(pm.current_time + np.arange(0, Nt + 1) * pm.dt, pm.precision_t)
        
        psi0 = pm.current_state
        args = pm.governing_eqns_params
        
        # --- IVP Logic 
        psi = [ivp_forecast_helper(y0=psi0[:, 0], 
                                    fun=pm.time_derivative, 
                                    t=t_all, 
                                    params={**pm.alpha0, **args})]
            
        try:
            psi = np.array(psi).transpose((1, 2, 0))
        except ValueError as e:
            print(f"Error during final array construction: {e}")
            psi = np.array(psi).T.reshape(-1, psi0.shape[0], pm.m)
            
        return psi[1:], t_all[1:]
        

    def advance_ensemble(self, Nt=100, averaged=False, alpha=None):
        pm = self.model
        
        t_all = np.round(pm.current_time + np.arange(0, Nt + 1) * pm.dt, pm.precision_t)
        
        psi0 = pm.current_state
        args = pm.governing_eqns_params
        
        # --- IVP Logic (Similar to previous Model.time_integrate) ---
    
        if not averaged:
            # Ensemble run (using multiprocessing pool)
            alpha_list = pm.get_alpha()
            forecast_part = partial(ivp_forecast_helper, 
                                    fun=pm.time_derivative, t=t_all, method=self.method)

            sol = [self.__pool.apply_async(forecast_part,
                                            kwds={'y0': psi0[:, mi].T, 'params': {**args, **alpha_list[mi]}})
                    for mi in range(pm.m)]
            
            psi = [s.get() for s in sol]

        else:
            # Averaged forecast
            psi_mean0 = np.mean(psi0, axis=1, keepdims=True)
            psi_deviation = psi0 - psi_mean0
            alpha = pm.get_alpha(psi_mean0)[0]
                
            psi_mean = ivp_forecast_helper(y0=psi_mean0[:, 0], 
                                            fun=pm.time_derivative,
                                            t=t_all,
                                            params={**alpha, **args},
                                            method=self.method)
            
            psi = [psi_mean + psi_deviation[:, ii] for ii in range(pm.m)]


        # Rearrange dimensions to be Nt+1 x N x m and remove initial condition
        try:
            psi = np.array(psi).transpose((1, 2, 0))
        except ValueError as e:
            print(f"Error during final array construction: {e}")
            psi = np.array(psi).T.reshape(-1, psi0.shape[0], pm.m)
            
        return psi[1:], t_all[1:]



def ivp_forecast_helper(y0, fun, t, params, method='RK45'):
    """Generic ODE solver using scipy's solve_ivp, defined globally for pickling."""
    assert len(t) > 1
    part_fun = partial(fun, **params)

    out = solve_ivp(part_fun, t_span=(t[0], t[-1]), y0=y0, t_eval=t, method=method)
    return out.y.T
# %%
