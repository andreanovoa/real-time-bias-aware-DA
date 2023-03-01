import TAModels
import Bias
from run import main, createESNbias, createEnsemble
from plotResults import *

# %% ========================== SELECT LOOP PARAMETERS ================================= #
folder = 'results/VdP/'
figs_folder = folder + 'figs/'

run_whyAugment, run_loopParams = False, False

# %% ============================= SELECT TRUE AND FORECAST MODELS ================================= #
true_params = {'model': TAModels.VdP,
               'manual_bias': 'cosine',
               'law': 'tan',
               'beta': 75.,  # forcing
               'zeta': 55.,  # damping
               'kappa': 3.4,  # nonlinearity
               'std_obs': 0.01,
               }

forecast_params = {'model': TAModels.VdP
                   }

# ==================================== SELECT FILTER PARAMETERS =================================== #
filter_params = {'filt': 'EnKFbias',  # 'EnKFbias' 'EnKF' 'EnSRKF'
                 'm': 10,
                 'est_p': ['beta', 'zeta', 'kappa'],
                 'biasType': Bias.ESN,  # Bias.ESN  # None
                 # Define the observation timewindow
                 't_start': 2.0,
                 't_stop': 4.5,
                 'kmeas': 25,
                 # Inflation
                 'inflation': 1.002,
                 'start_ensemble_forecast': 2
                 }

if filter_params['biasType'] is not None and filter_params['biasType'].name == 'ESN':
    # using default TA parameters for ESN training
    train_params = {'model': TAModels.VdP,
                    'std_a': 0.3,
                    'std_psi': 0.3,
                    'est_p': filter_params['est_p'],
                    'alpha_distr': 'uniform',
                    'ensure_mean': True
                    }

    bias_params = {'N_wash': 50,
                   'upsample': 5,
                   'L': 10,
                   'augment_data': True,
                   'train_params': train_params
                   }

else:
    bias_params = None
# ======================= CREATE REFERENCE ENSEMBLE =================================

ensemble, truth, args = createEnsemble(true_params, forecast_params,
                                       filter_params, bias_params, folder=folder)

# ================================================================================== #
# ================================================================================== #

if __name__ == '__main__':

    if run_whyAugment:
        results_folder: str = folder + 'results_whyAugment/'
        flag = True

        # Add standard deviation to the state
        blank_ens = ensemble.copy()
        std = 0.10
        blank_ens.psi = blank_ens.addUncertainty(np.mean(blank_ens.psi, 1), std,
                                                 blank_ens.m, method='normal')
        blank_ens.hist[-1] = blank_ens.psi
        blank_ens.std_psi, blank_ens.std_a = std, std

        # barData = [[], [], [], []]
        for L, augment in [(1, False), (1, True), (10, True)]:
            ks = [0., 6.]
            for ii, k in enumerate(ks):

                filter_ens = blank_ens.copy()

                # Reset ESN
                bias_params['augment_data'] = augment
                bias_params['L'] = L

                filter_params['Bdict'] = createESNbias(*args, bias_param=bias_params)  # reset bias
                filter_ens.initBias(filter_params['Bdict'])

                filter_ens.bias.k = k
                # ======================= RUN DATA ASSIMILATION  =================================
                name = results_folder + 'L{}_Augment{}/'.format(L, augment)
                filter_ens, truth, parameters = main(filter_ens, truth, filter_params,
                                                     results_dir=name, figs_dir=figs_folder, save_=True)

        post_process_WhyAugment(results_folder + 'results_whyAugment/', figs_folder)

    if run_loopParams:

        Ls = [1, 10, 50, 100]
        stds = [.1, 0.25]
        ks = np.linspace(0., 50., 26)

        for L in Ls:
            blank_ens = ensemble.copy()
            # Reset ESN
            bias_params['L'] = L
            filter_params['Bdict'] = createESNbias(*args, bias_param=bias_params)
            blank_ens.initBias(filter_params['Bdict'])
            for std in stds:
                # Reset std
                blank_ens.psi = blank_ens.addUncertainty(np.mean(blank_ens.psi, 1),
                                                         std, blank_ens.m, method='normal')
                blank_ens.hist[-1] = blank_ens.psi
                blank_ens.std_psi, blank_ens.std_a = std, std

                results_folder = folder + 'results_loopParams/std{}/L{}/'.format(std, L)
                for k in ks:  # Reset gamma value
                    filter_ens = blank_ens.copy()
                    filter_ens.bias.k = k

                    out = main(filter_ens, truth, filter_params,
                               results_dir=results_folder, figs_dir=figs_folder, save_=True)

                    if k in (0, 10, 50):
                        filename = '{}L{}_std{}_k{}_time'.format(figs_folder, L, std, k)
                        post_process_single_SE_Zooms(*out[:2], filename=filename)

                filename = '{}CR_L{}_std{}_results'.format(figs_folder, L, std)
                post_process_multiple(results_folder, filename)
                plt.close('all')

        get_CR_values(results_folder)
        fig2(folder + 'results_loopParams/', Ls, stds, figs_folder)
