from ..model import Model
from ..integrator import IVPIntegrator
import numpy as np



# %% =================================== VAN DER POL MODEL ============================================== %% #
class VdP(Model):
    r"""Van der Pol oscillator — low-order model of a longitudinal thermoacoustic mode.

    The acoustic pressure mode $\eta$ evolves as

    $$
    \ddot{\eta} + \omega^2 \eta = \dot{\eta} \left( \beta - \zeta - \kappa
    \, g(\eta) \right),
    $$

    with a cubic ($g = \eta^2$, ``law='cubic'``) or arctangent-saturated
    ($g = \eta^2 / (1 + \kappa \eta^2 / \beta)$, ``law='tan'``) heat-release law.
    The estimable parameters are the linear growth rate $\beta$, the damping
    $\zeta$ and the nonlinear saturation $\kappa$.

    References
    ----------
    Nóvoa & Magri (2022). Real-time thermoacoustic data assimilation.
    *J. Fluid Mech.*, 948, A35. [DOI: 10.1017/jfm.2022.653](https://doi.org/10.1017/jfm.2022.653).
    """

    t_transient = 1.5
    t_CR = 0.04

    Nq = 1

    beta = 70.                  # Linear growth rate [1/s]
    kappa = 4.0                 # Nonlinear saturation coefficient [1/s]
    zeta = 60.0                 # Damping coefficient [1/s]
    gamma = 1.7                 # Higher order nonlinearity coefficient (used only if cubic law)
    omega = 2 * np.pi * 120.    # Natural frequency [rad/s]
    law = 'tan'                 # 'cubic' or 'tan' heat release law

    # --- Parameters ---
    params = ['beta', 'zeta', 'kappa']      # Parameters that can be varied for sensitivity analysis or parameter estimation
    fixed_params = ['law', 'omega']         # Parameters that are fixed, but needed for the model equations
    extra_print_params = ['law', 'omega']   

    # __________________________ Init method ___________________________ #
    def __init__(self, **model_dict):

        psi0 = model_dict.pop('psi0', np.array([0.1, 0.1]))
        dt = model_dict.pop('dt', 1e-4)

        super().__init__(psi0=psi0, dt=dt, integrator_class=IVPIntegrator, **model_dict)

        #  Add fixed input_parameters
        self.alpha_labels = dict(beta='$\\beta$', zeta='$\\zeta$', kappa='$\\kappa$')
        self.alpha_lims = dict(zeta=(5, 120), kappa=(0.1, 20), beta=(5, 120))

    @property
    def state_labels(self):
        return  ['$\\eta$', '$\\mu$']

    # _______________ VdP specific properties and methods ________________ #
    @property
    def obs_labels(self):
        if self.Nq == 1: 
            return ["$\\eta$"]
        elif self.Nq == 2: 
            return ['$\\eta$', '$\\mu$']

    @staticmethod
    def time_derivative(t, psi, beta, zeta, kappa, law, omega):
        eta, mu = psi[:2]
        dmu_dt = - omega ** 2 * eta + mu * (beta - zeta)
        # Add nonlinear term
        if law == 'cubic':  # Cubic law
            dmu_dt -= mu * kappa * eta ** 2
        elif law == 'tan':  # arc tan model
            dmu_dt -= mu * (kappa * eta ** 2) / (1. + kappa / beta * eta ** 2)

        return (mu, dmu_dt) + (0,) * (len(psi) - 2)

