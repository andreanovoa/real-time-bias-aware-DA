
import os

import numpy as np
from utils import *
from bias import *
# from model import Model

from scipy.integrate import solve_ivp
from functools import partial
from copy import deepcopy

import numpy as np

from bias import NoBias

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
    """
    def __init__(self, model_instance):
        # A pointer to the Model instance to access self.time_derivative, self.dt, etc.
        self.model = model_instance


    def close(self):
        """ Close resources held by the integrator (e.g., multiprocessing pools). """
        pass    

    def advance(self, Nt=100, averaged=False, alpha=None):
        """
        The common interface for all integrators.
        Must return: (psi_forecasted[1:], t_forecasted[1:])
        """
        raise NotImplementedError("Child Integrator class must implement the advance() method.")
    


# %% =================================== CONCRETE STRATEGY 2: DISCRETE STEP ============================================= %% #
class DiscreteIntegrator(Integrator):
    """
    Base class for fixed-time-step integrators (e.g., ETDRK4 in KS, or ESN).
    It delegates the actual stepping logic to a method on the Model instance.
    """

    def time_step(self):
        """ Governing equations: psi(t+dt) = f(psi(t), alpha). 
            The time_step method must be defined by the child model
        """
        raise NotImplementedError("Child model must implement time_derivative(t, psi, **params).")
    

    def advance(self, Nt=100, averaged=False, alpha=None):
        """
        Delegates to the model's high-performance discrete step implementation.
        """
        t = np.round(self.get_current_time + np.arange(Nt + 1) * self.dt, self.precision_t)
        
            
        # The model's custom method returns the full forecasted array and time
        return self.model.time_step(Nt=Nt, averaged=averaged, alpha=alpha)


class DiscreteIntegrator(Integrator):
    """
    Handles time integration for models using a fixed, discrete map or scheme.
    It manages two time scales: physical_dt (output) and integrator_dt (step size).
    """
    
    def __init__(self, model_instance):
        super().__init__(model_instance)
        # Assuming the model instance defines these two time steps
        self.physical_dt = getattr(model_instance, 'physical_dt', model_instance.dt)
        self.integrator_dt = getattr(model_instance, 'integrator_dt', model_instance.dt)

        # Check for matching steps and calculate the upsampling factor
        if self.physical_dt < self.integrator_dt:
            # The physical step must be a multiple of the integration step
            # i.e., integrator_dt = self.upsample * physical_dt
            self.upsample = int(np.round(self.integrator_dt / self.physical_dt))
            if abs(self.integrator_dt - self.upsample * self.physical_dt) > 1e-9:
                 raise ValueError("Integrator dt must be a multiple of physical dt.")
            self.interpolate_flag = True
        elif self.physical_dt == self.integrator_dt:
             self.upsample = 1
             self.interpolate_flag = False
        else:
             raise ValueError("Integrator dt cannot be smaller than the physical dt.")


    def advance(self, Nt=100, averaged=False, alpha=None):

        model = self.model
        
        # --- 1. Determine Loop Parameters ---
        
        # Calculate the number of actual integration steps required
        Nt_loop = Nt * self.upsample 
        
        psi0 = model.get_current_state # (N x m)
        
        # Determine the alpha parameters for the loop (same as prior logic)
        if averaged:
            psi_initial = np.mean(psi0, axis=-1, keepdims=True)
            psi_deviation = psi0 - psi_initial
            alpha_list = [model.get_alpha(psi_initial)[0]]
        else:
            psi_initial = psi0
            psi_deviation = 0
            alpha_list = model.get_alpha(psi0)

        m_loop = psi_initial.shape[-1]
        
        # Array to store the raw, un-interpolated history (Nt_loop+1 x N x m_loop)
        psi_raw_history = np.zeros((Nt_loop + 1, psi_initial.shape[0], m_loop), dtype=psi_initial.dtype)
        psi_raw_history[0] = psi_initial
        
        t_raw = np.round(model.get_current_time + np.arange(0, Nt_loop + 1) * self.physical_dt, model.precision_t)
        
        # --- 2. Run the Discrete Forecast Loop ---

        for mi in range(m_loop):
            psi_current = psi_initial[:, mi] # (N,)
            alpha_current = alpha_list[mi]

            for i in range(Nt_loop):
                # Core Step: Advance one step using the model's implementation
                # This step always uses the fixed self.integrator_dt
                psi_next = model.single_step_advance(psi_current, alpha_current) 
                
                # Store and update for next step
                psi_raw_history[i + 1, :, mi] = psi_next
                psi_current = psi_next

        # --- 3. Final Processing and Interpolation ---
        
        psi_forecasted = psi_raw_history
        t_physical = t_raw # Start with the raw time array

        # Interpolate if the integrator step is smaller than the physical step
        if self.interpolate_flag:
            # Generate the target time array for physical output
            Nt_physical = Nt + 1
            t_physical = np.round(model.get_current_time + np.arange(Nt_physical) * self.physical_dt, model.precision_t)
            
            # NOTE: Interpolation must be done column-wise for the state vector and ensemble members.
            # This is complex and depends heavily on the interpolation method.
            # For this example, we'll use a simplified slicing, but in a real system, 
            # you would call i
            
            # Simplified: Just sample the points that match the physical_dt grid
            psi_forecasted = interpolate(t_raw, psi_raw_history, t_physical)
            
        # If averaged, re-add the ensemble deviation
        if averaged:
            # psi_forecasted is (Nt+1 x N x 1). Add deviation (N x m)
            psi_forecasted = psi_forecasted + psi_deviation.T

        # Return forecast (excluding initial condition) and time array (physical_dt)
        return psi_forecasted[1:], t_physical[1:]   


# %% ===================================  IVP SOLVER ============================================= %% #
class IVPIntegrator(Integrator):
    """ Integrator using Scipy's solve_ivp for continuous, variable-step integration. """

    @property
    def __pool(self):
        if not hasattr(self, '_pool'):
            N_pools = min(self.model.m, mp.cpu_count())
            self._pool = mp.Pool(N_pools)
        return self._pool

    def close(self):
        if hasattr(self, '_pool'):
            self.__pool.close()
            self.__pool.join()
            delattr(self, "_pool")
        else:
            pass


    def time_derivative(self, t, psi, **params):
        """ Governing equations: d(psi)/dt = f(t, psi, alpha). 
            The time_derivative method must still be defined by the child model
            """
        raise NotImplementedError("Child model must implement time_derivative(t, psi, **params).")


    def advance(self, Nt=100, averaged=False, alpha=None):
        model = self.model
        
        t_all = np.round(model.get_current_time + np.arange(0, Nt + 1) * model.dt, model.precision_t)
        t_steps = t_all[1:]
        
        psi0 = model.get_current_state
        args = model.governing_eqns_params
        
        # --- IVP Logic (Similar to previous Model.time_integrate) ---
        if not model.ensemble:
            psi = [ivp_forecast_helper(y0=psi0[:, 0], 
                                                fun=model.time_derivative, 
                                                t=t_all, 
                                                params={**model.alpha0, **args})]
        else:
            if not averaged:
                # Ensemble run (using multiprocessing pool)
                alpha_list = model.get_alpha()
                forecast_part = partial(ivp_forecast_helper, 
                                        fun=model.time_derivative, t=t_all)
                
                sol = [self.__pool.apply_async(forecast_part,
                                               kwds={'y0': psi0[:, mi].T, 'params': {**args, **alpha_list[mi]}})
                       for mi in range(model.m)]
                psi = [s.get() for s in sol]
            else:
                # Averaged forecast
                psi_mean0 = np.mean(psi0, axis=1, keepdims=True)
                psi_deviation = psi0 - psi_mean0
                if alpha is None: alpha = model.get_alpha(psi_mean0)[0]
                    
                psi_mean = ivp_forecast_helper(y0=psi_mean0[:, 0], 
                                                        fun=model.time_derivative,
                                                        t=t_all,
                                                        params={**alpha, **args})
                
                psi = [psi_mean + psi_deviation[:, ii] for ii in range(model.m)]


        # Rearrange dimensions to be Nt+1 x N x m and remove initial condition
        try:
            psi = np.array(psi).transpose((1, 2, 0))
        except ValueError as e:
            print(f"Error during final array construction: {e}")
            psi = np.array(psi).T.reshape(-1, psi0.shape[0], psi0.shape[-1])
            
        return psi[1:], t_steps



def ivp_forecast_helper(y0, fun, t, params):
    """Generic ODE solver using scipy's solve_ivp, defined globally for pickling."""
    assert len(t) > 1
    part_fun = partial(fun, **params)
    out = solve_ivp(part_fun, t_span=(t[0], t[-1]), y0=y0, t_eval=t, method='RK45')
    return out.y.T