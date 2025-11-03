
import os.path

import numpy as np
from utils import *
from bias import *
from model import Model

rng = np.random.default_rng(0)





class Ensemble():

    filter='EnKF'
    constrained_filter=False
    bias_bayesian_update=False
    regularization_factor=1.
    m=10
    dt_obs=None
    est_a=[]
    est_s=True
    est_b=False
    inflation=1.002
    reject_inflation=1.002
    std_psi=0.001
    std_a=0.001
    alpha_distr='uniform'
    phi_distr='normal'
    ensure_mean=False
    num_DA_blind=0
    num_SE_only=0
    start_ensemble_forecast=0.
    

    def __init__(self, base_model, **kwargs):


            if alpha0 is not None:
                for alpha, lims in alpha0.items():
                    forecast_params[alpha] = 0.5 * (lims[0] + lims[1])

            # ==============================  INITIALISE MODEL  ================================= #
            def is_model_subclass(obj_or_class):
                if isinstance(obj_or_class, type):
                    return issubclass(obj_or_class, Model)
                else:
                    return issubclass(obj_or_class.__class__, Model)

            if is_model_subclass(model):
                if not model.initialized:
                    ensemble = model(**forecast_params)
                elif model.initialized:
                    ensemble = model.copy()
            else:
                raise ValueError('Model must be a Model object subclass, got {}'.format(type(model)))

            # Forecast model case to steady state initial condition before initialising ensemble
            Nt = filter_params.get('Nt_transient', int(ensemble.t_CR / ensemble.dt))
            state = ensemble.time_integrate(Nt)[0]
            ensemble.update_history(state[-1], reset=True)

            # =========================  INITIALISE ENSEMBLE & BIAS  =========================== #
            ensemble.init_ensemble(**filter_params)
            # Forecast model case to steady state initial condition before initialising ensemble
            Nt = filter_params.get('Nt_transient', int(ensemble.t_CR / ensemble.dt))
            state = ensemble.time_integrate(Nt)[0]
            ensemble.update_history(state[-1], reset=True)
            ensemble.close()

            return ensemble
        

    pass

