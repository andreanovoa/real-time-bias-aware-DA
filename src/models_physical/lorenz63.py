from model import *

import numpy as np
from typing import Dict, Any, List

# Assuming imports from your project:
# from model import Model
# from integrator import IVPIntegrator
# from ensemble import Ensemble, NoBias # Assuming Ensemble and NoBias are defined

class Lorenz63(Model):
    """ Lorenz 63 Class: Continuous System Model
    """
    name: str = 'Lorenz63'

    # --- Core Physics Parameters ---
    t_lyap = 0.9056 ** (-1)
    t_transient = 10 * t_lyap
    t_CR = 4 * t_lyap
    Nq = 3
    dt = 0.02
    rho = 28.
    sigma = 10.
    beta = 8. / 3.
    observe_dims = range(3)

    # --- Ensemble/Augmentation Configuration Placeholders (Required by Model Base Class) ---
    # These are populated later, but required for property calculations in Model
    est_a: List[str] = []

    # --- Parameter and State Labels ---
    alpha_labels = dict(rho='$\\rho$', sigma='$\\sigma$', beta='$\\beta$')
    alpha_lims = dict(rho=(None, None), sigma=(None, None), beta=(None, None))
    extra_print_params = ['observe_dims', 'Nq', 't_lyap']
    state_labels = ['$x$', '$y$', '$z$']

    # __________________________ Init method ___________________________ #
    def __init__(self, ensemble_class=None, **kwargs):
        
        # 1. Handle Model-specific default initialization (psi0, observe_dims)
        model_dict = kwargs.copy()
        
        if 'psi0' not in model_dict.keys():
            model_dict['psi0'] = np.array([1.0, 1.0, 1.0])

        if 'observe_dims' in model_dict:
            self.observe_dims = model_dict['observe_dims']
        
        self.Nq = len(self.observe_dims)
        

        # 2. Call Model Base Class Init (which handles integrator and ensemble instantiation)
        # We pass the ensemble class here, which Model.__init__ will use.
        super().__init__(integrator_class=IVPIntegrator, 
                         ensemble_class=ensemble_class, 
                         **model_dict)


    # --- New required method for the Model base class ---
    def get_ensemble_config(self):
        """Returns the dictionary of ensemble configuration parameters."""
        return getattr(self, '_ensemble_config', {})

    # _______________ Lorenz63 specific properties and methods ________________ #

    @property
    def obs_labels(self):
        return [self.state_labels[kk] for kk in self.observe_dims]

    def get_observables(self, Nt=1, **kwargs):
        # Assumes state is always [x, y, z, alpha1, alpha2, ...]
        if Nt == 1:
            return self.hist[-1, self.observe_dims, :]
        else:
            return self.hist[-Nt:, self.observe_dims, :]

    @staticmethod
    def time_derivative(t, psi, sigma, rho, beta):
        """
        Calculates the time derivative of the Lorenz 63 system.
        Note: This derivative must handle the augmented state vector (psi).
        The augmented parameters are stored after the core state (x, y, z).
        """
        # Ensure only the core state (x, y, z) is used for the physics calculation
        x1, x2, x3 = psi[:3]
        
        # The parameter values used for the current derivative calculation
        # These come from the 'params' dict passed by the integrator
        dx1 = sigma * (x2 - x1)
        dx2 = x1 * (rho - x3) - x2
        dx3 = x1 * x2 - beta * x3
        
        # The augmented parameters (if present) are constant or have their own governing eq. (e.g., d(alpha)/dt = 0)
        # We assume the last (len(psi) - 3) elements are parameters with zero derivative.
        
        # Return a tuple of derivatives: (dx1, dx2, dx3, d(alpha1)/dt, d(alpha2)/dt, ...)
        return (dx1, dx2, dx3) + (0,) * (len(psi) - 3)

if __name__ == "__main__":
    # test Lorenz63 model
    model = Lorenz63()
    model.time_integrate(Nt=1000)

    model.close()
    print(model.get_observables(Nt=5))
    