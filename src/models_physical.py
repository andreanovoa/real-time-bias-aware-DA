from src.model import *

from scipy.interpolate import splrep, splev
from src.Util import Cheb


# %% =================================== VAN DER POL MODEL ============================================== %% #
class VdP(Model):
    """ Van der Pol Oscillator Class
        - cubic heat release law
        - atan heat release law
            Note: gamma appears only in the higher order polynomial which is currently commented out
    """

    name: str = 'VdP'
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

        super().__init__(**model_dict)

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


# %% ==================================== RIJKE TUBE MODEL ============================================== %% #
class Rijke(Model):
    """
        Rijke tube model with Galerkin discretization and gain-delay sqrt heat release law.
    """

    name: str = 'Rijke'
    t_transient = 1.
    t_CR = 0.02

    Nm = 10
    Nc = 10
    Nq = 6
    dt = 1e-4

    beta, tau = 4.0, 1.5E-3
    C1, C2 = 0.05, 0.01
    kappa = 1E5
    xf, L = 0.2, 1.
    law = 'sqrt'

    alpha_labels = dict(beta='$\\beta$', tau='$\\tau$', C1='$C_1$', C2='$C_2$', kappa='$\\kappa$')
    alpha_lims = dict(beta=(0.01, 5), tau=[1E-6, None], C1=(0., 1.), C2=(0., 1.), kappa=(1E3, 1E8))

    fixed_params = ['cosomjxf', 'Dc', 'gc', 'jpiL', 'L',
                    'law', 'meanFlow', 'Nc', 'Nm', 'tau_adv', 'sinomjxf']

    extra_print_params = ['law', 'Nm', 'Nc', 'xf', 'L']

    def __init__(self, **model_dict):

        if 'psi0' not in model_dict.keys():
            if 'Nm' in model_dict.keys():
                Nm = model_dict['Nm']
            else:
                Nm = self.Nm
            if 'Nc' in model_dict.keys():
                Nc = model_dict['Nc']
            else:
                Nc = self.Nc
            model_dict['psi0'] = .05 * np.hstack([np.ones(2 * Nm), np.zeros(Nc)])

        super().__init__(**model_dict)

        self.tau_adv = self.tau
        self.alpha_lims['tau'][-1] = self.tau_adv

        # Chebyshev modes
        self.Dc, self.gc = Cheb(self.Nc, getg=True)

        # Microphone locations
        self.x_mic = np.linspace(self.xf, self.L, self.Nq + 1)[:-1]

        # Define modes frequency of each mode and sin cos etc
        jj = np.arange(1, self.Nm + 1)
        self.jpiL = jj * np.pi / self.L
        self.sinomjxf = np.sin(self.jpiL * self.xf)
        self.cosomjxf = np.cos(self.jpiL * self.xf)

        # Mean Flow Properties
        def weight_avg(y1, y2):
            return self.xf / self.L * y1 + (1. - self.xf / self.L) * y2

        self.meanFlow = dict(u=weight_avg(10, 11.1643), p=101300.,
                             gamma=1.4, T=weight_avg(300, 446.5282), R=287.1)
        self.meanFlow['rho'] = self.meanFlow['p'] / (self.meanFlow['R'] * self.meanFlow['T'])
        self.meanFlow['c'] = np.sqrt(self.meanFlow['gamma'] * self.meanFlow['R'] * self.meanFlow['T'])

        self.set_fixed_params()

        # Wave input_parameters ############################################################################################
        # c1: 347.2492    p1: 1.0131e+05      rho1: 1.1762    u1: 10          M1: 0.0288          T1: 300
        # c2: 423.6479    p2: 101300          rho2: 0.7902    u2: 11.1643     M2: 0.0264          T2: 446.5282
        # Tau: 0.0320     Td: 0.0038          Tu: 0.0012      R_in: -0.9970   R_out: -0.9970      Su: 0.9000
        # Q_bar: 5000     R_gas: 287.1000     gamma: 1.4000
        ##############################################################################################################

    def modify_settings(self):
        if 'tau' in self.est_a:
            extra_Nc = 50 - self.Nc
            self.tau_adv, self.Nc = 1E-2, 50
            self.alpha_lims['tau'][-1] = self.tau_adv
            psi = self.get_current_state
            self.psi0 = np.hstack([np.mean(psi, -1),
                                   np.zeros(extra_Nc)])
            self.Dc, self.gc = Cheb(self.Nc, getg=True)
            self.update_history(reset=True)
            self.set_fixed_params()

    # _______________ Rijke specific properties and methods ________________ #
    @property
    def obs_labels(self, loc=None):
        if loc is None:
            loc = np.expand_dims(self.x_mic, axis=1)
        return ["$p'(x = {:.2f})$".format(x) for x in loc[:, 0]]

    @property
    def state_labels(self):
        lbls0 = [f"$\\eta_{j}$" for j in np.arange(self.Nm)]
        lbls1 = ["$\\dot{\\eta}$" + f"$_{j}$" for j in np.arange(self.Nm)]
        lbls2 = [f"$\\nu_{j}$" for j in np.arange(self.Nc)]
        return lbls0 + lbls1 + lbls2

    def get_observables(self, Nt=1, loc=None, **kwargs):
        if loc is None:
            loc = self.x_mic
        loc = np.expand_dims(loc, axis=1)
        om = np.array([self.jpiL])
        mu = self.hist[-Nt:, self.Nm:2 * self.Nm, :]

        # Compute acoustic pressure and velocity at locations
        p_mic = -np.dot(np.sin(np.dot(loc, om)), mu)
        p_mic = p_mic.transpose(1, 0, 2)
        if Nt == 1:
            p_mic = p_mic[0]
        return p_mic

    @staticmethod
    def time_derivative(t, psi,
                        C1, C2, beta, kappa, tau,
                        cosomjxf, Dc, gc, jpiL, L, law, meanFlow, Nc, Nm, tau_adv, sinomjxf):
        """
            Governing equations of the model.
            Args:
                psi: current state vector
                t: current time
                C1, C2, beta, kappa, tau: Possibly-inferred input_parameters
                cosomjxf, Dc, gc, jpiL, L, law, meanFlow, Nc, Nm, tau_adv, sinomjxf:  fixed input_parameters
            Returns:
                concatenation of the state vector time derivative
        """
        eta, mu, v = psi[:Nm], psi[Nm: 2 * Nm], psi[2 * Nm: 2 * Nm + Nc]

        # Advection equation boundary conditions
        v2 = np.hstack((np.dot(eta, cosomjxf), v))

        # Evaluate u(t_interp-tau) i.e. velocity at the flame at t_interp - tau
        x_tau = tau / tau_adv
        if x_tau < 1:
            f = splrep(gc, v2)
            u_tau = splev(x_tau, f)
        elif x_tau == 1:  # if no tau estimation, bypass interpolation to speed up code
            u_tau = v2[-1]
        else:
            raise Exception("tau = {} can't_interp be larger than tau_adv = {}".format(tau, tau_adv))

        # Compute damping and heat release law
        zeta = C1 * (jpiL * L / np.pi) ** 2 + C2 * (jpiL * L / np.pi) ** .5

        MF = meanFlow.copy()  # Physical properties
        if law == 'sqrt':
            q_dot = MF['p'] * MF['u'] * beta * (
                    np.sqrt(abs(1. / 3 + u_tau / MF['u'])) - np.sqrt(1. / 3))  # [W/m2]=[m/s3]
        elif law == 'tan':
            q_dot = beta * np.sqrt(beta / kappa) * np.arctan(np.sqrt(beta / kappa) * u_tau)  # [m / s3]
        else:
            raise ValueError('Law "{}" not defined'.format(law))
        q_dot *= -2. * (MF['gamma'] - 1.) / L * sinomjxf  # [Pa/s]

        # governing equations
        deta_dt = jpiL / MF['rho'] * mu
        dmu_dt = - jpiL * MF['gamma'] * MF['p'] * eta - MF['c'] / L * zeta * mu + q_dot
        dv_dt = - 2. / tau_adv * np.dot(Dc, v2)

        return np.concatenate((deta_dt, dmu_dt, dv_dt[1:], np.zeros(len(psi) - (2 * Nm + Nc))))


# %% =================================== LORENZ 63 MODEL ============================================== %% #
class Lorenz63(Model):
    """ Lorenz 63 Class
    """
    name: str = 'Lorenz63'

    t_lyap = 0.9056 ** (-1)
    t_transient = 10 * t_lyap
    t_CR = 4 * t_lyap

    Nq = 3
    dt = 0.02
    rho = 28.
    sigma = 10.
    beta = 8. / 3.

    observe_dims = range(3)

    alpha_labels = dict(rho='$\\rho$', sigma='$\\sigma$', beta='$\\beta$')
    alpha_lims = dict(rho=(None, None), sigma=(None, None), beta=(None, None))

    extra_print_params = ['observe_dims', 'Nq', 't_lyap']

    state_labels = ['$x$', '$y$', '$z$']

    # __________________________ Init method ___________________________ #
    def __init__(self, **model_dict):
        if 'psi0' not in model_dict.keys():
            model_dict['psi0'] = np.array([1.0, 1.0, 1.0])  # initialise x, y, z

        if 'observe_dims' in model_dict:
            self.observe_dims = model_dict['observe_dims']

        self.Nq = len(self.observe_dims)

        super().__init__(**model_dict)

    # _______________ Lorenz63 specific properties and methods ________________ #
    @property
    def obs_labels(self):
        return [self.state_labels[kk] for kk in self.observe_dims]

    def get_observables(self, Nt=1, **kwargs):
        if Nt == 1:
            return self.hist[-1, self.observe_dims, :]
        else:
            return self.hist[-Nt:, self.observe_dims, :]

    @staticmethod
    def time_derivative(t, psi, sigma, rho, beta):
        x1, x2, x3 = psi[:3]
        dx1 = sigma * (x2 - x1)
        dx2 = x1 * (rho - x3) - x2
        dx3 = x1 * x2 - beta * x3
        return (dx1, dx2, dx3) + (0,) * (len(psi) - 3)


# %% =================================== 2X VAN DER POL MODEL ============================================== %% #
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

        super().__init__(**model_dict)

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