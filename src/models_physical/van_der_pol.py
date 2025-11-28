from model import *



# %% =================================== VAN DER POL MODEL ============================================== %% #
class VdP(Model):
    """ Van der Pol Oscillator Class
        - cubic heat release law
        - atan heat release law
            Note: gamma appears only in the higher order polynomial which is currently commented out
    """

    # name: str = 'VdP'
    t_transient = 1.5
    t_CR = 0.04

    Nq = 1
    dt = 1e-4
    law = 'tan'

    beta = 70.
    kappa = 4.0
    zeta = 60.
    gamma = 1.7
    omega = 2 * np.pi * 120.

    alpha_labels = dict(beta='$\\beta$', zeta='$\\zeta$', kappa='$\\kappa$')
    alpha_lims = dict(zeta=(5, 120), kappa=(0.1, 20), beta=(5, 120))

    state_labels: list = ['$\\eta$', '$\\mu$']

    fixed_params = ['law', 'omega']
    extra_print_params = ['law', 'omega']

    # __________________________ Init method ___________________________ #
    def __init__(self, **model_dict):

        if 'psi0' not in model_dict.keys():
            model_dict['psi0'] = np.array([0.1, 0.1])  # initialise eta and mu

        super().__init__(integrator_class=IVPIntegrator, **model_dict)

        #  Add fixed input_parameters
        self.set_fixed_params()

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

