

# %%


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as cm
import warnings

from model import *



class KS(Model):

    """
    This class models the Kuramoto-Sivashinsky  equation

	u_t + u_xx + nu u_xxxx + u*u_x = 0 or u_t = - u_xx - nu u_xxxx - u*u_x
	B.C.s : u(t,0) = u(t,L)
	        u_x(t,0) = u_x(t,L)

	on the domain x in (0,L], where nu = (2pi/L)^2
    """

    name: str = 'KS'
    t_transient = 300.
    t_CR = 50.
    dt = 0.25

    Nq = 1
    Nx = 256             # Spatial discretization
    nu = None            # 'Viscosity' parameter of the KS equation.
    L = None             # Domain length (0, L]
    
    seed = 0
    initial_amplitude = 0.01


    # alpha_labels = dict(nu='$\\nu$')
    # alpha_lims = dict(nu=(0., None))

    extra_print_params = ['Nx']
    sensor_placement_method = 'grid'


    def __init__(self, **model_dict):
        """
        Initialize the KS model.

        Parameters
        ----------
        model_dict : dict
            Dictionary of model parameters. Supported keys include:
            - Nx: int, number of spatial grid points (must be even)
            - nu: float, viscosity parameter
            - dt: float, time step size
            - initial_amplitude: float, amplitude of initial condition
            - Nq: int, number of sensors
            - sensor_placement_method: str, 'grid' or 'random'
            - seed: int, random seed for sensor placement
            - psi0: np.ndarray, initial state in Fourier space (optional)
        The method sets up the spatial grid, wavenumbers, sensor locations, and initial state.
        """


        for key in list(model_dict.keys()):
            if key in vars(KS):
                setattr(self, key, model_dict.pop(key))

        
        if self.Nx % 2 != 0:
            raise ValueError("Nx must be even.")
		
        if self.L is None and self.nu is None:
            raise ValueError("Either L or nu must be specified.")
        elif self.L is None:
            self.L = 2 * np.pi / np.sqrt(self.nu)
        elif self.nu is None:
            # self.nu = (2 * np.pi / self.L)**2
            self.nu = 1

        
        # Define Fourier wavenumbers k on the nondimensional domain
        self.k = 2 * np.pi * np.fft.rfftfreq(self.Nx, d=self.L / self.Nx)

        self.ETDRK4_f_terms = None  # This simply trigers the setter method.
        self.rng = np.random.default_rng(self.seed)


        #  Select sensors ___________________________ #
        if self.sensor_placement_method not in ['grid', 'random']:
            raise NotImplementedError(f"sensor_placement_method '{self.sensor_placement_method}' not recognized.")

        if self.sensor_placement_method == 'grid':
            # Place sensors evenly spaced across the domain
            self.sensor_locations = np.linspace(0, self.Nx-1, self.Nq, endpoint=True, dtype=int)
        elif self.sensor_placement_method == 'random':
            # Place sensors at random locations in the domain
            self.sensor_locations = self.rng.integers(0, self.Nx-1, self.Nq)


        #   Init Model  #

        # Generate random initial condition in physical space (DEFAULT)
        if 'psi0' not in model_dict.keys():
            # Initialize state in physical space and transform to spectral space
            # u0 = self.initial_amplitude * np.cos(self.x/2/np.pi)  # initial condition
            u0 = self.initial_amplitude * self.rng.standard_normal(self.Nx)
            u0 -= np.mean(u0)

            u_hat = KS.physical_to_fourier(u0)[:, None]     # Transform to Fourier space
            model_dict['psi0'] = np.array(u_hat)  
            


        super().__init__(integrator_class=DiscreteIntegrator, **model_dict)

        

    # _______________ Modified Model methods ________________ #


    @property
    def obs_labels(self):
        return [f"$u(x_{{{j+1}}})$" for j in np.arange(self.Nq)]


    @property
    def state_labels(self):
        return [f"$\hat{{u}}_{{{j+1}}}$" for j in np.arange(self.Nphi)]


    def get_observables(self, Nt=1, loc=None, **kwargs):
        """
        """
        if loc is None:
            loc = self.sensor_locations
        elif loc.lower() == 'all':
            loc = np.arange(self.Nx)

        if Nt == 1:
            return KS.fourier_to_physical(self.hist[-1, :self.Nk])[loc]
        else:
            return KS.fourier_to_physical(self.hist[-Nt:, :self.Nk])[:, loc]


    # _______________ KS specific properties and methods ________________ #

    @property
    def x(self):
        return  np.arange(self.Nx) *  self.L / self.Nx


    @property
    def Nk(self):
        return self.k.shape[0]


    def first_derivative_x(self, u_hat):
        if u_hat.ndim == 2:
            assert u_hat.shape[0] == self.Nk, f'u_hat.shape[0] == {u_hat.shape[0]} != {self.Nk}'
            return 1.j * self.k[:, None] * u_hat
        elif u_hat.ndim == 3:
            
            assert u_hat.shape[1] == self.Nk, f'u_hat.shape[1] == {u_hat.shape[1]} != {self.Nk}'
            return 1.j * self.k[None, :, None] * u_hat
        else:
            raise AssertionError(f'u_hat should be shape (Nk, m)=({self.Nk, self.m}) or (Nt, Nk, m). Got {u_hat.shape} instead.')


    def __nonlinear_operator(self, u_hat):
        """
            Compute the nonlinear term N(u) = -u * u_x in Fourier space. 
            F[-u * u_x] = F[1/2(u * u)_x]  
            Input: (N_x, m)
            # rfft outputs the positive frequencies n/2+1 if even, (n+1)/2 if odds
        """
        assert u_hat.shape[0] == self.Nk, f'u_hat.shape[0] == {u_hat.shape[0]} != {self.Nk}'


        # Dealias using 2/3 rule
        cutoff = int(self.Nx * 2/3)
        dealias = np.ones_like(self.k, dtype=bool)
        dealias[cutoff:-cutoff] = False

        # Apply filter to input
        u_hat_filtered = u_hat.copy()
        u_hat_filtered[~dealias] = 0.0

        # Option A-----
        # Square in thw physical space and transform back
        u = KS.fourier_to_physical(u_hat_filtered)
        u2_hat = KS.physical_to_fourier(u**2)

        N_hat =  - 0.5 * self.first_derivative_x(u2_hat) 
        #-----  Option A


        # # Option B-----(numerically equivalenrt. A is faster.)
        # # Transform to physical space
        # u = KS.fourier_to_physical(u_hat_filtered)
        
        # # Compute derivative explicitly
        # # u_x = np.real(ifft(1j * self.k * u_hat_filtered))
        # u_x = KS.fourier_to_physical(self.first_derivative_x(u_hat_filtered) )
        # nonlinear = -u * u_x
        
        # # Transform back to Fourier space and apply filter
        # N_hat = KS.physical_to_fourier(nonlinear)
        # #-----  Option B 

        N_hat[~dealias] = 0.0
        
        return N_hat
    
    @property
    def __linear_operator(self):
        # ÷ Fourier multipliers for linear term
        # return (self.k**2 - self.nu * self.k**4)[:, None] 
        # else:
        return (self.k**2 - self.k**4)[:, None] 
        

    @property
    def ETDRK4_f_terms(self):
        return self._ETDRK4_f_terms



    @ETDRK4_f_terms.setter
    def ETDRK4_f_terms(self, _):
        # Avoid division by zero for k=0 mode
        L = self.__linear_operator

        # Initialize coefficient arrays
        terms = dict(
            f1 = np.zeros((self.Nk, 1), dtype=complex),
            f2 = np.zeros((self.Nk, 1), dtype=complex),
            f3 = np.zeros((self.Nk, 1), dtype=complex),
            E = np.exp(self.dt * L),
            E2 = np.exp(self.dt * L / 2),
            nonlinear_operator = self.__nonlinear_operator
        )

        # Use vectorized computation with error handling
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            
            # Use Taylor series for small arguments to avoid numerical issues
            M = 16  # Number of points for contour integral approximation 
            r = np.exp(1j * np.pi * (np.arange(1, M+1) - 0.5) / M)
            
            LR = self.dt * L + r[np.newaxis, :]
        
            f1 = (-4 - LR + np.exp(LR) * (4 - 3*LR + LR**2)) / LR**3
            f2 = (2 + LR + np.exp(LR) * (-2 + LR)) / LR**3
            f3 = (-4 - 3*LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3

                
        # # Check for NaNs or Infs and replace with Taylor limits if needed
        zero_mask = (L[:,0] <= 1e-10)
        for key, val in zip(['f1', 'f2', 'f3'], [f1, f2, f3]):
            f_raw = val.astype(complex)
            
            if key in ['f1', 'f2']:
                terms[key][zero_mask] = self.dt * 0.5  # For intermediate steps
            elif key == 'f3':
                terms[key][zero_mask] = self.dt * (1/6) # For intermediate steps
                
            for kk in range(self.Nk):
                f = val[kk]
                valid_mask = np.isfinite(f)
                if np.any(valid_mask):
                    terms[key][kk] = self.dt * np.real(np.mean(f))
                else:
                    warnings.warn(f"NaN or Inf detected in ETDRK4 coefficient {key}, replacing with Taylor limits")
                    terms[key][kk] = self.dt / 6

        # Store coefficients in the object
        self._ETDRK4_f_terms = terms.copy()



    @staticmethod
    def ETDRK4_step(u_hat, nonlinear_operator, E, E2, f1, f2, f3):
        """
        ETDRK4 time-stepping scheme:

        a_n = exp(L * dt/2) * u_n + L^{-1} * (exp(L * dt/2) - I) * N(u_n)
        b_n = exp(L * dt/2) * u_n + L^{-1} * (exp(L * dt/2) - I) * N(a_n)
        c_n = exp(L * dt/2) * a_n + L^{-1} * (exp(L * dt/2) - I) * (2 * N(b_n) - N(u_n))

        u_{n+1} = exp(L * dt) * u_n 
                + dt^{-2} * L^{-3} * [
                    (-4 - L h + exp(L h) * (4 - 3 L h + (L h)^2)) * N(u_n)
                    + 2 * (2 + L h + exp(L h) * (-2 + L h)) * (N(a_n) + N(b_n))
                    + (-4 - 3 L h - (L h)^2 + exp(L h) * (4 - L h)) * N(c_n)
                    ]

        Where:
        - u_n: solution at timestep n
        - h: timestep size
        - L: linear operator (diagonalized in Fourier space)
        - N(u): nonlinear operator
        - I: identity operator
        - a_n, b_n, c_n: intermediate Runge-Kutta stages

        This scheme integrates the linear part exactly using exponentials and uses RK4 for nonlinear terms.
        """

        N1 = nonlinear_operator(u_hat)
        a = E2 * u_hat + f1 * N1
        N2 = nonlinear_operator(a)
        b = E2 * u_hat + f2 * N2
        N3 = nonlinear_operator(b)
        c = E * u_hat + f3 * N3
        N4 = nonlinear_operator(c)

        return E * u_hat + (f1 * N1 + 2 * f2 * (N2 + N3) + f3 * N4) / 6.




    def time_step(self, Nt=10, averaged=False, alpha=None):
        """
        Integrator for the KS model that supports ensembles and averaged ensemble propagation.
        Matches interface conventions of other models.

        Parameters
        ----------
        Nt : int
            Number of time steps to integrate.
        averaged : bool, optional
            If True, integrates the mean state and broadcasts ensemble deviations.
        alpha : optional
            Additional model parameters.

        Returns
        -------
        psi : np.ndarray
            Forecasted state array of shape (Nt, Nphi, m).
        t : np.ndarray
            Time vector corresponding to each forecasted state.
        """
        
        u0_hat = self.current_state

        if u0_hat.ndim == 1:  # reshape for non-ensemble
            u0_hat = u0_hat[:, None]
        
        t = np.round(self.current_time + np.arange(Nt + 1) * self.dt, self.precision_t)
        

        if averaged and self.ensemble:
            u0_hat_mean = np.mean(u0_hat, axis=1, keepdims=True)
            psi_deviation = u0_hat - u0_hat_mean

            psi_mean_arr = [u0_hat_mean[:, 0]]
            for _ in range(Nt):
                psi_mean_arr.append(KS.ETDRK4_step(psi_mean_arr[-1][:, None], **self.ETDRK4_f_terms)[:, 0])
            psi_mean_arr = np.stack(psi_mean_arr, axis=0)  # (Nt+1, N_x)

            # Broadcast deviations
            psi = np.array([psi_mean_arr[ii][:, None] + psi_deviation for ii in range(psi_mean_arr.shape[0])])  # (Nt+1, N_x, m)

        else:
            # Single member integration
            psi = [u0_hat]
            for _ in range(Nt):
                psi.append(KS.ETDRK4_step(psi[-1], **self.ETDRK4_f_terms))

            psi = np.stack(psi, axis=0)
        

        return psi[1:], t[1:]

     

    def get_energy(self, Nt=0, u=None):
        """
        Compute the L2 energy of the solution: E = (1/L) * integral(u^2)dx
        
        Returns:
        --------
        float
            L2 energy
        """

        if u is None:
            u = self.get_observable_hist(Nt=Nt, loc="all")


        if u.ndim == 2:
            u = u[np.newaxis, :]

        assert u.shape[1] == self.Nx

        return np.mean(u**2, axis=1) 
    


    def get_enstrophy(self, Nt=0, u_hat=None):
        """
        Compute the enstrophy (integral of (u_x)^2).
        
        Returns:
        --------
        float
            Enstrophy
        """     

        if u_hat is None:
            if Nt != 1:
                u_hat = self.hist[-Nt:]
            else:
                u_hat = self.current_state[np.newaxis, :]
        else:
            if u_hat.ndim == 2:
                u_hat = u_hat[np.newaxis, :]

        assert u_hat.shape[1] == self.k.shape[0]

        u_x_hat = self.first_derivative_x(u_hat)
        u_x = self.fourier_to_physical(u_x_hat)
        
        return np.mean(u_x**2, axis=1) 
    

    @staticmethod
    def fourier_to_physical(u_hat):
        # Inverse real FFT
        if u_hat.ndim > 2:
            ax = 1
        else:
            ax = 0
        return np.fft.irfft(u_hat, axis=ax, n=None)
    
    @staticmethod
    def physical_to_fourier(u):
        # Real-to-complex FFT along spatial dimension
        if u.ndim > 2:
            ax = 1
        else:
            ax = 0
        return np.fft.rfft(u, axis=ax)
    

    
    def visualize_spatiotemporal(self, y_hist=None, t=None, nrows=None, averaged=False):
        """
        Visualize the spatiotemporal evolution of the KS model in the physical space.
        """

        if y_hist is None:
            y_hist = self.get_observable_hist(loc="all")

        if t is None:
            t = self.hist_t

        if not averaged:
            if nrows is None:
                nrows = min(10, y_hist.shape[-1])

            fig = plt.figure(figsize=(10, 1.5 * nrows))
            axs = fig.subplots(nrows=nrows, sharey=True, sharex=True)
            if nrows == 1:
                axs = [axs]

            lim = np.max(abs(y_hist))

            for mi, ax in enumerate(axs):
                im = ax.imshow(y_hist[:, :, mi].T, 
                            aspect='auto', origin='lower', 
                            cmap='RdBu_r', vmin=-lim, vmax=lim,
                            extent=[t[0], t[-1], self.x[0], self.x[-1]])  # TRANSPOSE
                    
                
            axs[0].set(title=rf"KS spatiotemporal evolution. $L={self.L/np.pi:.2f}\pi, \nu={self.nu}$")
            axs[-1].set(xlabel="$t$")

            fig.colorbar(im, ax=axs, orientation='vertical', shrink=1/nrows) 
        else:
            # Averaged ensemble visualization
            y_mean_hist = np.mean(y_hist, axis=-1)

            fig, axs = plt.subplots(nrows=2, figsize=(10, 6), sharex=True)

            # Mean evolution
            lim_mean = np.max(abs(y_mean_hist))
            im0 = axs[0].imshow(y_mean_hist.T, 
                                aspect='auto', origin='lower', 
                                cmap='RdBu_r', vmin=-lim_mean, vmax=lim_mean,
                                extent=[t[0], t[-1], self.x[0], self.x[-1]])
            axs[0].set(title=rf"KS averaged spatiotemporal evolution (mean and std). $L={self.L/np.pi:.2f}\pi, \nu={self.nu}$")
            fig.colorbar(im0, ax=axs[0], orientation='vertical') 

            # Deviation covariance evolution

            var_ensemble = np.var(y_hist, axis=-1, ddof=1).T            # (Nt, Nx)
            var_ensemble = np.sqrt(var_ensemble)                     # Standard deviation

            lim_dev = np.max(abs(var_ensemble))
            im1 = axs[1].imshow(var_ensemble,  # Plot covariance of deviations
                                aspect='auto', origin='lower', 
                                cmap='magma', vmin=0, vmax=lim_dev,
                                extent=[t[0], t[-1], self.x[0], self.x[-1]])
                                
            fig.colorbar(im1, ax=axs[1], orientation='vertical')
        # add the ticks and labels

        # Set spatial ticks as multiples of L
        ticks = (np.arange(4) + 1)* self.L/4
        tick_labels = [r"$L/4$", r"$L/2$", r"$3L/4$",r"$L$"]
        for ax in axs:
            ax.set(ylabel="$x$", yticks=ticks, yticklabels=tick_labels)
        

    @staticmethod
    def plot_temporal_E(model, Nt=0, max_lines=10):
        
        energy = model.get_energy(Nt=Nt)
        enstrophy = model.get_enstrophy(Nt=Nt)

        max_lines = min(model.m, max_lines)
        plot_m = np.arange(max_lines)


        c_energy = plt.get_cmap('tab20b', max_lines)  
        c_enstrophy = plt.get_cmap('tab20b', max_lines)


        _, axs = plt.subplots(ncols=1, nrows=2) 
        for ax, E, ttl, cmap in zip(axs, [energy, enstrophy], ['Energy', 'Enstrophy'], [c_energy, c_enstrophy]):
            for mi in plot_m:
                ax.plot(model.hist_t, E[:, mi], c=cmap(mi / max_lines))

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_lines-1))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, 
                                orientation='vertical', ticks=plot_m)
            cbar.ax.set_yticklabels([str(t) for t in plot_m])
            ax.set_ylabel(ttl)
            cbar.set_label('Member index', fontsize=12)

        axs[-1].set_xlabel("t")

    

# %%


if __name__ == "__main__":

    nu = .08
    dt = 0.25
    Nt = int(100 / dt)
    Nx = 256
    # L = 36

    import time

    t1 = time.time()
    
    for seed in [0]:


        model = KS( Nx=Nx,
                    dt=dt,
                    nu=nu,
                    seed=seed,
                    initial_amplitude=1.)   
        
        # model.init_ensemble(std_psi=0.1,
        #                             m=10)


        print(f"Domain size: L = {model.L  }") 
        print(f"Grid points: N = {model.Nx}")
        print(f"Time step: dt = {model.dt:.6f}")
        print(f"Viscosity: nu = {model.nu:.6f}")

        print(len(model.k))

        print(f"Domain size: L = {model.L  }") 
        print(f"Grid points: N = {model.Nx}")
        print(f"Time step: dt = {model.dt:.6f}")
        print(f"Viscosity: nu = {model.nu:.6f}")



        solution, times = model.time_integrate(Nt=Nt)
        model.update_history(psi=solution, t=times)

        KS.plot_spatiotemporal(model=model)
        KS.plot_temporal_E(model=model)



    print('time =', time.time()-t1)
    plt.show()




    Nt_transient = int(10 / dt)
    for T in [25, 1000]:

        model = KS( Nx=Nx,
                    dt=dt,
                    seed=seed,
                    initial_amplitude=1.,
                    L=48*np.pi)   

        Nt = int(T / model.dt)

        # remove transient
        solution, times = model.time_integrate(Nt=Nt_transient)
        model.update_history(psi=solution[[-1]], t=times[[0]], reset=True)

        comp_time = []
        n_runs = 10
        for _ in range(n_runs):
            t1 = time.time()
            solution, times = model.time_integrate(Nt=Nt)
            t2 = time.time()
            comp_time.append(t2 - t1)

        print(f'Computation time for T={T} over {n_runs} runs: \n ')
        print(f'\t Mean: {np.mean(comp_time):.4f} s')
        print(f'\t Std: {np.std(comp_time):.4f} s')
        print(f'\t Min: {np.min(comp_time):.4f} s')
        print(f'\t Max: {np.max(comp_time):.4f} s')
        

        model.update_history(psi=solution, t=times)
        
        KS.plot_spatiotemporal(model=model)
        KS.plot_temporal_E(model=model)

    plt.show()


# %%
