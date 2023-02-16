import TAModels
import Bias
from run import main, createESNbias, createEnsemble
from plotResults import *

# %% ========================== SELECT LOOP PARAMETERS ================================= #
folder = 'results/VdP_final_.3/'
figs_folder = folder + 'figs/'

run_whyAugment, run_loopParams = True, True

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
                 'start_ensemble_forecast': 1
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
                   'L': 1,
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
        flag: bool = True
        k0_U, k0_B, k1_U, k1_B = [], [], [], []

        # Add standard deviation to the state
        blank_ens = ensemble.copy()
        std = 0.10
        blank_ens.psi = blank_ens.addUncertainty(np.mean(blank_ens.psi, 1), std,
                                                 blank_ens.m, method='normal')
        blank_ens.hist[-1] = blank_ens.psi
        blank_ens.std_psi, blank_ens.std_a = std, std

        barData = [[], [], [], []]
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
                filter_ens, truth, parameters = main(filter_ens, truth, filter_params,
                                                     results_folder=results_folder, figs_folder=figs_folder, save_=True)

                # GET CORRELATION AND RMS FOR SOLUTION ================================================
                # Observable bias
                y_filter, t_filter = filter_ens.getObservableHist(), filter_ens.hist_t
                y_truth, b_truth = truth['y'][:len(y_filter)], truth['b_true'][:len(y_filter)]

                # ESN estimated bias
                b, t_b = filter_ens.bias.hist, filter_ens.bias.hist_t

                # Ubiased signal recovered through interpolation
                y_unbiased = y_filter[::filter_ens.bias.upsample] + np.expand_dims(b, -1)
                y_unbiased = interpolate(t_b, y_unbiased, t_filter)

                if flag:
                    N_CR = int(filter_ens.t_CR / filter_ens.dt)  # Length of interval to compute correlation and RMS
                    istop = np.argmin(abs(t_filter - truth['t_obs'][parameters['num_DA'] - 1]))  # end of assimilation
                    istart = np.argmin(abs(t_filter - truth['t_obs'][0]))  # start of assimilation
                    flag = False

                # GET CORRELATION AND RMS ERROR =====================================================================
                CB, RB = CR(y_truth[istop - N_CR:istop], np.mean(y_filter, -1)[istop - N_CR:istop])  # biased
                CU, RU = CR(y_truth[istop - N_CR:istop], np.mean(y_unbiased, -1)[istop - N_CR:istop])  # unbiased

                barData[2*ii].append((CU, RU))
                barData[2*ii + 1].append((CB, RB))

                # if k == ks[0]:
                #     k0_U.append((CU, RU))
                #     k0_B.append((CB, RB))
                # elif k == ks[1]:
                #     k1_U.append((CU, RU))
                #     k1_B.append((CB, RB))


                filename = '{}WhyAugment_L{}_augment{}_k{}'.format(figs_folder, L, augment, k)
                post_process_single_SE_Zooms(filter_ens, truth, filename=filename)

        # barData = [k0_U, k0_B, k1_U, k1_B]

        # Plot results -------------------------------------------------------------------------
        y_truth_u = y_truth - b_truth
        Ct, Rt = CR(y_truth[-N_CR:], y_truth_u[-N_CR:])
        Cpre, Rpre = CR(y_truth[istart - N_CR:istart + 1:], np.mean(y_filter, -1)[istart - N_CR:istart + 1:])

        barPlot(barData, Ct, Rt, Cpre, Rpre, ks, figs_folder)

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
                               results_folder=results_folder, figs_folder=figs_folder, save_=True)

                    if k in (0, 10, 50):
                        filename = '{}L{}_std{}_k{}_time'.format(figs_folder, L, std, k)
                        post_process_single_SE_Zooms(*out[:2], filename=filename)

                filename = '{}CR_L{}_std{}_results'.format(figs_folder, L, std)
                post_process_multiple(results_folder, filename)
                plt.close('all')

        fig2(folder + 'results_loopParams/', Ls, stds, figs_folder)
