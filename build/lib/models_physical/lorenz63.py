from model import Model
from integrator import IVPIntegrator
import matplotlib.pyplot as plt

import numpy as np
from typing import List


class Lorenz63(Model):
    """ Lorenz 63 Model: Chaotic system with three state variables.
        
        Equations: 
            dx/dt = sigma * (y - x)
            dy/dt = x * (rho - z) - y
            dz/dt = x * y - beta * z
    """

    # --- Core Physics Parameters ---
    t_lyap = 0.9056 ** (-1)
    t_transient = 10 * t_lyap
    t_CR = 4 * t_lyap
    Nq = 3
    
    rho = 28.
    sigma = 10.
    beta = 8. / 3.

    # --- Ensemble/Augmentation Configuration Placeholders (Required by Model Base Class) ---
    # These are populated later, but required for property calculations in Model
    est_a: List[str] = []

    # --- Parameter and State Labels ---
    alpha_labels = dict(rho='$\\rho$', sigma='$\\sigma$', beta='$\\beta$')
    alpha_lims = dict(rho=(None, None), sigma=(None, None), beta=(None, None))
    extra_print_params = ['observe_dims', 'Nq', 't_lyap']
    state_labels = ['$x$', '$y$', '$z$']

    # __________________________ Init method ___________________________ #
    def __init__(self, **model_dict):
        
        
        psi0 = model_dict.pop('psi0', np.array([1.0, 1.0, 1.0]))
        dt = model_dict.pop('dt', 0.02)

        self.observe_dims = model_dict.pop('observe_dims', [0, 1, 2]) # Default to observing all dimensions if not specified
        self.Nq = len(self.observe_dims)
        
        super().__init__(psi0=psi0, dt=dt, integrator_class=IVPIntegrator, **model_dict)



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
        """
        Calculates the time derivative of the Lorenz 63 system.
        Note: This derivative must handle the augmented state vector (psi).
        The augmented parameters are stored after the core state (x, y, z).
        """
       
        x1, x2, x3 = psi[:3]
        
        # The parameter values used for the current derivative calculation
        # These come from the 'params' dict passed by the integrator
        dx1 = sigma * (x2 - x1)
        dx2 = x1 * (rho - x3) - x2
        dx3 = x1 * x2 - beta * x3
        
        return (dx1, dx2, dx3) + (0,) * (len(psi) - 3)


    def visualize_attractor(self, psi_cases=None, **kwargs):
        """
        Visualizes the Lorenz attractor for given state trajectories.
        Args:
            psi_cases: List of state trajectories to plot. Each trajectory should be of shape (Nt, 3) or (Nt, 3, Ne).
            color: List of colors for each trajectory. If None, a colormap is used.
            figsize: Figure size for the plot.
            ensemble_mean: If True and psi_cases contain ensembles, plot the mean trajectory.
        """
        if psi_cases is None:
            psi_cases = [self.hist[:, :3, :]]

        plot_attractor(psi_cases, **kwargs)






def plot_attractor(psi_cases, color=None, figsize=(8, 6)):
    


    if type(psi_cases) is not list:
        if psi_cases.ndim == 2:
            psi_cases = [psi_cases[:, :, np.newaxis]]
        else:
            psi_cases = [psi_cases]

    else:
        psi_cases = [p[:, :, np.newaxis] if p.ndim == 2 else p for p in psi_cases]

    if color is None:
        color = plt.cm.viridis(np.linspace(0, 1, len(psi_cases)))
    elif type(color) is str:
        color = [color] * len(psi_cases)
    
    # Check for 3D state dimension
    if psi_cases[0].shape[1] == 3:
        mosaic = [['A', 'ax_xy'],
                ['A', 'ax_xz'],
                    ['A', 'ax_yz',]]
        
        # Create figure and axes with ratios for larger 3D plot
        fig, axes = plt.subplot_mosaic(mosaic, figsize=figsize, layout='tight', width_ratios=[2,1])
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



if __name__ == "__main__":
    # test Lorenz63 model
    model = Lorenz63()
    model.time_integrate(Nt=1000)

    model.close()
    print(model.get_observables(Nt=5))
    