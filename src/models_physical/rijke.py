from model import *

from scipy.interpolate import splrep, splev
from utils import Cheb


# %% ==================================== RIJKE TUBE MODEL ============================================== %% #
class Rijke(Model):
    """
        Rijke tube model with Galerkin discretization and gain-delay sqrt heat release law.
    """

    name: str = 'Rijke'
    t_transient = .25
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

        super().__init__(integrator_class=IVPIntegrator, **model_dict)

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
        if 'tau' in self.ensemble.get('est_alpha', []):
            extra_Nc = 50 - self.Nc
            self.tau_adv, self.Nc = 1E-2, 50
            self.alpha_lims['tau'][-1] = self.tau_adv
            psi = self.current_state
            
            new_psi = np.concatenate([psi,
                                      np.zeros((extra_Nc, psi.shape[-1]))], axis=0)
            
            self.psi0 = np.mean(new_psi, axis=1, keepdims=True)

            self.Dc, self.gc = Cheb(self.Nc, getg=True)

            self.update_history(t=0., psi=self.psi0, reset=True)

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

