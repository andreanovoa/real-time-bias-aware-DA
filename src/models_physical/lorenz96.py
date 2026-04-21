# %%

from model import Model
from integrator import IVPIntegrator
import matplotlib.pyplot as plt

import numpy as np
from typing import List


class Lorenz96(Model):
    """ Lorenz 95 Model: Chaotic system with Nx state variables.
        
        Equations: 
            dx_i/dt = x_{i-2} * x_{i-1} - x_{i-1}*x_{i+1} - x_i + F, for i = 0,...,39

        where indices are cyclic (i.e., x_{-2} = x_{Nx-2}, x_{-1} = x_{Nx-1} and x_{Nx} = x_0)
    """

    # --- Core Physics Parameters ---
    t_lyap = 1.67 ** (-1)
    t_transient = 10 * t_lyap
    t_CR = 4 * t_lyap
    Nq = 3
    
    F = 8.0
    Nx = 40

    # --- Ensemble/Augmentation Configuration Placeholders (Required by Model Base Class) ---
    # These are populated later, but required for property calculations in Model
    est_a: List[str] = []

    # --- Parameter and State Labels ---
    extra_print_params = ['observed_idx', 'Nq', 't_lyap', 'Nx']
    fixed_params = ['Nx']
    params = ['F']

    # __________________________ Init method ___________________________ #
    def __init__(self, **model_dict):
        
        self.Nx = model_dict.pop('Nx', 40)
        psi0 = model_dict.pop('psi0', np.array([1.6] + [1.0] * (self.Nx - 1)))
        dt = model_dict.pop('dt', 0.01)

        self.observed_idx = model_dict.pop('observed_idx', [0, self.Nx//2, self.Nx]) # Default to observing three dimensions if not specified 
        self.Nq = len(self.observed_idx)
        
        super().__init__(psi0=psi0, dt=dt, integrator_class=IVPIntegrator, **model_dict)
        
        self.alpha_labels = dict(F='$F$')
        

    # _______________ Lorenz63 specific properties and methods ________________ #

    @property
    def state_labels(self):
        return [f'$x_{{{kk}}}$' for kk in range(self.Nx)]
    

    @property
    def obs_labels(self):
        return [self.state_labels[kk] for kk in self.observed_idx]

    def get_observables(self, Nt=1, **kwargs):
        if Nt == 1:
            return self.hist[-1, self.observed_idx, :]
        else:
            return self.hist[-Nt:, self.observed_idx, :]


    @staticmethod
    def time_derivative(t, psi, Nx, F):
        """
        Calculates the time derivative of the Lorenz 95 system.
        """
       
        x = psi[:Nx]

        dx = (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F 
    
        return dx


    
    def visualize_spatiotemporal_hist(self, y_hist=None, t=None, nrows=None, averaged=False, **kwargs):

        if y_hist is None:
            y_hist = self.hist[:, :self.Nx]

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
                            extent=[t[0], t[-1], 0, self.Nx])  # TRANSPOSE
                    
                
            axs[0].set(title=rf"Lorenz96 spatiotemporal evolution. $F={self.F:.2f}, N_x={self.Nx}$")
            axs[-1].set(xlabel="$t$")

            fig.colorbar(im, ax=axs, orientation='vertical', shrink=1/nrows) #type: ignore
        else:
            # Averaged ensemble visualization
            y_mean_hist = np.mean(y_hist, axis=-1)

            fig, axs = plt.subplots(nrows=2, figsize=(10, 6), sharex=True)

            # Mean evolution
            lim_mean = np.max(abs(y_mean_hist))
            im0 = axs[0].imshow(y_mean_hist.T, 
                                aspect='auto', origin='lower', 
                                cmap='RdBu_r', vmin=-lim_mean, vmax=lim_mean,
                                extent=[t[0], t[-1], 0, self.Nx])  # TRANSPOSE
            axs[0].set(title=rf"Lorenz96 averaged spatiotemporal evolution (mean and std). $F={self.F:.2f}, N_x={self.Nx}$")
            fig.colorbar(im0, ax=axs[0], orientation='vertical') 

            # Deviation covariance evolution

            var_ensemble = np.var(y_hist, axis=-1, ddof=1).T            # (Nt, Nx)
            var_ensemble = np.sqrt(var_ensemble)                     # Standard deviation

            lim_dev = np.max(abs(var_ensemble))
            im1 = axs[1].imshow(var_ensemble,  # Plot covariance of deviations
                                aspect='auto', origin='lower', 
                                cmap='magma', vmin=0, vmax=lim_dev,
                                extent=[t[0], t[-1], 0, self.Nx])  # TRANSPOSE
                                
            fig.colorbar(im1, ax=axs[1], orientation='vertical')
        # add the ticks and labels

        # Set spatial ticks as multiples of L
        ticks = (np.arange(4) + 1)* self.Nx/4
        tick_labels = [r"$N_x/4$", r"$N_x/2$", r"$3N_x/4$",r"$N_x$"]
        for ax in axs:
            ax.set(ylabel="$x$", yticks=ticks, yticklabels=tick_labels)



def plot_attractor(psi_cases, color=None, figsize=(8, 6)):
    


    if type(psi_cases) is not list:
        if psi_cases.ndim == 2:
            psi_cases = [psi_cases[:, :, np.newaxis]]
        else:
            psi_cases = [psi_cases]

    else:
        psi_cases = [p[:, :, np.newaxis] if p.ndim == 2 else p for p in psi_cases]

    if color is None:
        color = plt.colormaps['viridis'](np.linspace(0, 1, len(psi_cases)))
    elif type(color) is str:
        color = [color] * len(psi_cases)
    
    # Check for 3D state dimension
    if psi_cases[0].shape[1] == 3:
        mosaic = [('A', 'ax_xy'),
                ('A', 'ax_xz'),
                ('A', 'ax_yz')]
        
        # Create figure and axes with ratios for larger 3D plot
        fig, axes = plt.subplot_mosaic(mosaic, figsize=figsize, layout='tight', width_ratios=[2,1]) # type: ignore
        lbl = ['$x$', '$y$', '$z$']
        projections = [
            (axes['ax_xy'], 0, 1), # XY
            (axes['ax_xz'], 0, 2), # XZ
            (axes['ax_yz'], 1, 2)  # YZ
        ]
        
        # 2. Configure 3D subplot (replace 2D 'A' with a 3D axis in the same SubplotSpec)
        ss = axes['A'].get_subplotspec()
        fig.delaxes(axes['A'])
        ax_3d = fig.add_subplot(ss, projection='3d', xticks=[], yticks=[], zticks=[], xlabel='', ylabel='', zlabel='')
        axes['A'] = ax_3d
        ax_3d.axis('off')
        ax_3d.set_box_aspect((1,1,1.5))  

        # 4. Plotting Logic (Combined loop for 3D and 2D)
        for psi_, c in zip(psi_cases, color):
            psi_proc = psi_
            if psi_.shape[2] > 10:
                psi_proc = np.mean(psi_, axis=-1)
            
            ax_3d.plot(psi_proc[:, 0], psi_proc[:, 1], psi_proc[:, 2], c=c, alpha=.8)
                
            for ax, i, j in projections:
                ax.plot(psi_proc[:, i], psi_proc[:, j], c=c, alpha=.8, lw=.5)
                ax.set(xlabel=lbl[i], ylabel=lbl[j])




# %%


if __name__ == "__main__":
    # test Lorenz96 model
    model = Lorenz96()

    for _ in range(5):
        psi, t = model.time_integrate(Nt=1000)


        model.update_history(psi, t)


    model.visualize_spatiotemporal_hist()

    plot_attractor([model.hist[:, :3]], color='blue')
    plt.show()

    

# %%
