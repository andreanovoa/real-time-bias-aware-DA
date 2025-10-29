

from model import *




class Annular(Model):
    """ Annular combustor model, which consists of two coupled oscillators
    """

    name: str = 'Annular'

    t_transient = 0.5
    t_CR = 0.01

    ER = 0.5
    nu_1, nu_2 = 633.77, -331.39
    c2b_1, c2b_2 = 258.3, -108.27  # values in Matlab codes

    # defaults: dict = dict(Nq=4, n=1., ER=ER_0, dt=1. / 51200,
    #                       theta_b=0.63, theta_e=0.66, omega=1090 * 2 * np.pi, epsilon=2.3E-3,
    #                       nu=nu_1 * ER_0 + nu_2, c2beta=c2b_1 * ER_0 + c2b_2, kappa=1.2E-4)

    Nq = 4
    theta_mic = np.radians([0, 60, 120, 240])

    dt = 1. / 51200
    theta_b = 0.63
    theta_e = 0.66
    omega = 1090 * 2 * np.pi
    epsilon = 2.3E-3

    nu = nu_1 * ER + nu_2
    c2beta = c2b_1 * ER + c2b_2
    kappa = 1.2E-4

    # defaults['nu'], defaults['c2beta'] = 30., 5.  # spin
    # defaults['nu'], defaults['c2beta'] = 1., 25.  # stand
    # defaults['nu'], defaults['c2beta'] = 20., 18.  # mix

    alpha_labels = dict(omega='$\\omega$', nu='$\\nu$', c2beta='$c_2\\beta $', kappa='$\\kappa$',
                        epsilon='$\\epsilon$', theta_b='$\\Theta_\\beta$', theta_e='$\\Theta_\\epsilon$')
    alpha_lims = dict(omega=(1000 * 2 * np.pi, 1300 * 2 * np.pi),
                      nu=(-60., 100.), c2beta=(0., 100.), kappa=(None, None),
                      epsilon=(None, None), theta_b=(0, 2 * np.pi), theta_e=(0, 2 * np.pi))

    state_labels = ['$\\eta_{a}$', '$\\dot{\\eta}_{a}$', '$\\eta_{b}$', '$\\dot{\\eta}_{b}$']

    # __________________________ Init method ___________________________ #
    def __init__(self, **model_dict):
        if 'psi0' not in model_dict.keys():
            C0, X0, th0, ph0 = 10, 0, 0.63, 0  # %initial values
            # Conversion of the initial conditions from the quaternion formalism to the AB formalism
            Ai = C0 * np.sqrt(np.cos(th0) ** 2 * np.cos(X0) ** 2 + np.sin(th0) ** 2 * np.sin(X0) ** 2)
            Bi = C0 * np.sqrt(np.sin(th0) ** 2 * np.cos(X0) ** 2 + np.cos(th0) ** 2 * np.sin(X0) ** 2)
            phai = ph0 + np.arctan2(np.sin(th0) * np.sin(X0), np.cos(th0) * np.cos(X0))
            phbi = ph0 - np.arctan2(np.cos(th0) * np.sin(X0), np.sin(th0) * np.cos(X0))

            # %initial conditions for the fast oscillator equations
            psi0 = [Ai * np.cos(phai),
                    -self.omega * Ai * np.sin(phai),
                    Bi * np.cos(phbi),
                    -self.omega * Bi * np.sin(phbi)]

            model_dict['psi0'] = np.array(psi0)  # initialise \eta_a, \dot{\eta_a}, \eta_b, \dot{\eta_b}

        super().__init__(integrator_class=IVPIntegrator, **model_dict)

    # _______________  Specific properties and methods ________________ #
    @property
    def obs_labels(self, loc=None, measure_modes=False):
        if measure_modes:
            return ["$\\eta_1$", '$\\eta_2$']
        else:
            if loc is None:
                loc = self.theta_mic
            return ["$p(\\theta={}^\\circ)$".format(int(np.round(np.degrees(th)))) for th in np.array(loc)]

    @staticmethod
    def nu_from_ER(ER):
        return Annular.nu_1 * ER + Annular.nu_2

    @staticmethod
    def c2beta_from_ER(ER):
        return Annular.c2b_1 * ER + Annular.c2b_2

    def get_observables(self, Nt=1, loc=None, measure_modes=False, **kwargs):
        """
        pressure measurements at theta = [0º, 60º, 120º, 240º`]
        p(θ, t) = η1(t) * cos(nθ) + η2(t) * sin(nθ).
        """
        if loc is None:
            loc = self.theta_mic

        if measure_modes:
            return self.hist[-Nt:, [0, 2], :]
        else:
            eta1, eta2 = self.hist[-Nt:, 0, :], self.hist[-Nt:, 2, :]
            if max(loc) > 2 * np.pi:
                raise ValueError('Theta must be in radians')

            p_mics = np.array([eta1 * np.cos(th) + eta2 * np.sin(th) for th in np.array(loc)])
            p_mics = p_mics.transpose(1, 0, 2)
            if Nt == 1:
                return p_mics.squeeze(axis=0)
            else:
                return p_mics

    @staticmethod
    def time_derivative(t, psi, nu, kappa, c2beta, theta_b, omega, epsilon, theta_e):
        y_a, z_a, y_b, z_b = psi[:4]  # y = η, and z = dη/dt

        def k1(y1, y2, sign):
            return (2 * nu - 3. / 4 * kappa * (3 * y1 ** 2 + y2 ** 2) +
                    sign * c2beta / 2. * np.cos(2. * theta_b))

        k2 = c2beta / 2. * np.sin(2. * theta_b) - 3. / 2 * kappa * y_a * y_b

        def k3(y1, y2, sign):
            return omega ** 2 * (y1 * (1 + sign * epsilon / 2. * np.cos(2. * theta_e)) +
                                 y2 * epsilon / 2. * np.sin(2. * theta_e))

        dz_a = z_a * k1(y_a, y_b, sign=1) + z_b * k2 - k3(y_a, y_b, sign=1)
        dz_b = z_b * k1(y_b, y_a, sign=-1) + z_a * k2 - k3(y_b, y_a, sign=-1)

        return (z_a, dz_a, z_b, dz_b) + (0,) * (len(psi) - 4)
    


