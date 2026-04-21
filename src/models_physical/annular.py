

from model import Model
from integrator import IVPIntegrator
import numpy as np




class Annular(Model):
    """
        Annular combustor model with two coupled oscillators representing the first azimuthal acoustic modes.
        Model used in: 
            Nóvoa A, Noiray N, Dawson JR, Magri L. A real-time digital twin of azimuthal thermoacoustic instabilities. 
            Journal of Fluid Mechanics. 2024;1001:A49. doi:10.1017/jfm.2024.1052
        ------------------------------------------------------------------------------
        Physical governing equations:

            d²p/dt² + ζ dp/dt - [1 + ε cos(2(θ - Θ_ε))] c²/r² d²p/dθ² = (γ-1) dq̇/dt
            
        where
            (γ-1) dq̇/dt = β[1 + c₂ cos(2(θ - Θ_β))] p - κ p³
    
        The equations are transformed into a set of four first-order ODEs for the two coupled oscillators by 
        decomposing the pressure field p(θ,t) into its two azimuthal modes

                p(θ, t) = η_a(t) cos(nθ) + η_b(t) sin(nθ)

        with n=1 (first azimuthal mode). The resulting system of equations is:
        
            dη_a/dt     =   η̇_a
            d²η_a/dt²   =   - ω²[η_a(1 + ε/2 cos(2Θ_ε)) + η_b ε/2 sin(2Θ_ε)]
                            + η̇_a[2ν + c₂β/2 cos(2Θ_β) - 3κ/4(3η_a² + η_b²)]
                            + η̇_b[c₂β/2 sin(2Θ_β) - 3κ/2 η_a η_b]
            dη_b/dt     =   η̇_b
            d²η_b/dt²   =   - ω²[η_b(1 - ε/2 cos(2Θ_ε)) + η_a ε/2 sin(2Θ_ε)]
                            + η̇_b[2ν - c₂β/2 cos(2Θ_β) - 3κ/4(3η_b² + η_a²)]
                            + η̇_a[c₂β/2 sin(2Θ_β) - 3κ/2 η_a η_b]
        
        with
            - η_a, η_b: Amplitudes of the two coupled oscillators (first azimuthal acoustic modes)
            - θ: Azimuthal angle
            - n: Azimuthal mode number (n=1)
            - ω: Angular frequency of the acoustic mode
            - ν: Growth rate parameter
            - κ: Saturation parameter (flame response)
            - c₂β: Resistive asymmetry intensity
            - Θ_β: Direction of maximum root-mean-square (r.m.s.) acoustic pressure
            - ε: Amplitude of the reactive asymmetry
            - Θ_ε: Phase of the reactive asymmetry
            - ζ: Acoustic damping
            - c: Speed of sound
            - r: Mean radius of the annulus
            - γ: Heat capacity ratio
            - q̇: Coherent component of heat release rate fluctuations
            - β: Heat release strength

        ------------------------------------------------------------------------------

        Dynamical Regimes (example parameters):
            - Purely spinning mode:  (ν, c₂β) = (30., 5.)
            - Purely standing mode:  (ν, c₂β) = (0., 50.)
            - Mixed mode:            (ν, c₂β) = (20., 18.)
        """

    t_transient = 0.5
    t_CR = 0.01

    ER = 0.5
    nu_1, nu_2 = 633.77, -331.39
    c2b_1, c2b_2 = 258.3, -108.27  # values in Matlab codes

    Nq = 4
    theta_mic = np.radians([0, 60, 120, 240])

    
    theta_b = 0.63
    theta_e = 0.66
    omega = 1090 * 2 * np.pi
    epsilon = 2.3E-3

    nu = nu_1 * ER + nu_2
    c2beta = c2b_1 * ER + c2b_2
    kappa = 1.2E-4
    params = ['omega', 'nu', 'c2beta', 'kappa', 'epsilon', 'theta_b', 'theta_e'] 

    # __________________________ Init method ___________________________ #
    def __init__(self, **model_dict):

        dt = model_dict.pop('dt', 1. / 51200)
        psi0 = model_dict.pop('psi0', None)
        if psi0 is None:
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

            psi0 = np.array(psi0)  # initialise \eta_a, \dot{\eta_a}, \eta_b, \dot{\eta_b}
            
        super().__init__(psi0=psi0, dt=dt, integrator_class=IVPIntegrator, **model_dict)
        
        self.alpha_labels = dict(omega='$\\omega$', nu='$\\nu$', c2beta='$c_2\\beta $', kappa='$\\kappa$',
                                 epsilon='$\\epsilon$', theta_b='$\\Theta_\\beta$', theta_e='$\\Theta_\\epsilon$')
        
        self.alpha_lims =  dict(omega=(1000 * 2 * np.pi, 1300 * 2 * np.pi), 
                                nu=(-60., 100.), c2beta=(0., 100.), 
                                theta_b=(0, 2 * np.pi), theta_e=(0, 2 * np.pi))

    # _______________  Specific properties and methods ________________ #
    @property
    def obs_labels(self, loc=None, measure_modes=False):
        if measure_modes:
            return ["$\\eta_1$", '$\\eta_2$']
        else:
            if loc is None:
                loc = self.theta_mic
            return ["$p(\\theta={}^\\circ)$".format(int(np.round(np.degrees(th)))) for th in np.array(loc)]
    @property
    def state_labels(self):
        return  ['$\\eta_{a}$', '$\\dot{\\eta}_{a}$', '$\\eta_{b}$', '$\\dot{\\eta}_{b}$']


    @staticmethod
    def nu_from_ER(ER):
        return Annular.nu_1 * ER + Annular.nu_2

    @staticmethod
    def c2beta_from_ER(ER):
        return Annular.c2b_1 * ER + Annular.c2b_2

    def get_observables(self, Nt=1, loc=None, measure_modes=False, **kwargs):
        """
        pressure measurements at theta = [0º, 60º, 120º, 240º]
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
    


