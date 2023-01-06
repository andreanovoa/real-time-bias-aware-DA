import numpy as np
import TAModels
import Bias
from run import main, createESNbias, createEnsemble
from plotResults import *

rng = np.random.default_rng(0)

# %% ========================== SELECT LOOP PARAMETERS ================================= #
folder = 'results/VdP_12.13_newArch'
Ls = [1, 10, 50, 100]
stds = [0.01, 0.1, 0.25]
ks = np.linspace(0., 50., 26)

save_ = True

# %% ============================= SELECT TRUE AND FORECAST MODELS ================================= #
true_params = {'model': TAModels.VdP,
               'manual_bias': True,
               'law': 'tan',
               'beta': 80.,  # forcing
               'zeta': 60.,  # damping
               'kappa': 3.4,  # nonlinearity
               }

# forecast_params = true_params.copy()
forecast_params = {'model': TAModels.VdP,
                   'beta': 70,
                   'zeta': 55,
                   'kappa': 4.2
                   }

# ==================================== SELECT FILTER PARAMETERS =================================== #
filter_params = {'filt': 'EnKFbias',  # 'EnKFbias' 'EnKF' 'EnSRKF'
                 'm': 10,  # Dictionary of DA parameters
                 'est_p': ['beta', 'zeta', 'kappa'],
                 'biasType': Bias.ESN,  # Bias.ESN  # None
                 # Define the observation timewindow
                 't_start': 2.0,
                 't_stop': 4.5,
                 'kmeas': 25,
                 # Inflation and optional parameters
                 'inflation': 1.002,
                 'num_DA_blind': 0,  # int(0.0 / t_obs[1] - t_obs[0]),  # num of obs to start accounting for the bias
                 'num_SE_only': 0  # int(0.0 / t_obs[1] - t_obs[0])  # num of obs to start parameter estimation
                 }

if filter_params['biasType'] is not None and filter_params['biasType'].name == 'ESN':
    train_params = forecast_params.copy()
    # train_params = {'std_a': 0.5,
    #                 'std_psi': 0.5,
    #                 'est_p': filter_params['est_p'],
    #                 'alpha_distr': 'uniform'  # training reference data created with uniform distributions
    #                 }
    train_params['std_a'] = 0.5
    train_params['std_psi'] = 0.5
    train_params['est_p'] = filter_params['est_p']
    train_params['alpha_distr'] = 'uniform'  # training reference data created with uniform distributions


    bias_params = {'N_wash': 50,
                   'upsample': 5,
                   'N_units': 200,
                   't_train': 1.0,
                   'augment_data': True,
                   'train_params': train_params
                   }
else:
    bias_params = None

folder += '_{}PE_{}kmeas/'.format(len(filter_params['est_p']), filter_params['kmeas'])

# ========================================================================================

ensemble, truth, b_args = createEnsemble(true_params, forecast_params, filter_params, bias_params, folder=folder)

figs_folder = folder + 'figs/'
flag = False
for L in Ls:
    blank_ens = ensemble.copy()
    # Reset ESN
    if b_args is not None:
        bias_p = createESNbias(*b_args, L=L, bias_param=bias_params)
        filter_params['Bdict'] = bias_p
        blank_ens.initBias(bias_p)
    flag = True
    for std in stds:
        # Reset stdt
        psi_mean = np.mean(blank_ens.psi, 1)
        # if len(blank_ens.est_p) > 0:
        #     psi_mean[-len(blank_ens.est_p):] = np.array([getattr(blank_ens, p) for p in blank_ens.est_p])
        blank_ens.psi = blank_ens.addUncertainty(psi_mean, std, blank_ens.m, method='normal')
        blank_ens.hist[-1] = blank_ens.psi
        blank_ens.std_psi = std
        blank_ens.std_psi = std

        print(np.mean(blank_ens.psi[-len(blank_ens.est_p):], 1), psi_mean[-len(blank_ens.est_p):])
        results_folder = folder + 'std{}/L{}/'.format(std, L)

        for k in ks:  # Reset gamma value
            filter_ens = blank_ens.copy()
            filter_ens.bias.k = k

            out = main(filter_ens, truth, filter_params,
                       results_folder=results_folder, figs_folder=figs_folder, save_=True)

            if k in (0, 10, 50):
                filename = '{}L{}_std{}_k{}_time'.format(figs_folder, L, std, k)
                post_process_single_SE_Zooms(*out[:2], filename=filename)
        filename = '{}CR_L{}_std{}_results'.format(figs_folder, L, std)
        post_process_multiple(results_folder, filename)
        plt.close()

# out = main(true_params, forecast_params, filter_params, bias_params, Ls, stds, ks, folder, save_)[:2]
# plotResults(folder, stds, Ls, k_plot=(0, 10, 50))

# if len(stds) == 1 and len(Ls) == 1:
#   post_process_single_SE_Zooms(*out)
#   post_process_multiple(folder + 'std{}/L{}/'.format(stds[-1], Ls[-1]))
