from model import *



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

        super().__init__(integrator_class=IVPIntegrator, **model_dict)

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
    


if __name__ == "__main__":
    # test Lorenz63 model
    model = Lorenz63()
    model.time_integrate(Nt=1000)

    model.close()
    print(model.get_observables(Nt=5))
    