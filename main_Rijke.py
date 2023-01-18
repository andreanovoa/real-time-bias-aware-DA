import TAModels
import Bias
from run import main, createESNbias, createEnsemble
from plotResults import *

# %% ========================== SELECT LOOP PARAMETERS ================================= #
folder = 'results/Rijke_short_zeta/'
figs_folder = folder + 'figs/'

run_whyAugment, run_loopParams = False, True

# %% ============================= SELECT TRUE AND FORECAST MODELS ================================= #
true_params = {'model': 'wave',
               't_max': 4.
               }

forecast_params = {'model': TAModels.Rijke,
                   'beta': 0.6,
                   'tau': 2.E-3,
                   'C1': 0.02,
                   'C2': 0.01,
                   }

# ==================================== SELECT FILTER PARAMETERS =================================== #
filter_params = {'filt': 'EnKFbias',  # 'EnKFbias' 'EnKF' 'EnSRKF'
                 'm': 10,
                 'est_p': ['beta', 'tau', 'C1', 'C2'],
                 'biasType': Bias.ESN,  # Bias.ESN  # None
                 # Define the observation timewindow
                 't_start': 2.0,  # ensure SS
                 't_stop': 2.5,
                 'kmeas': 25,
                 # Inflation
                 'inflation': 1.002
                 }

if filter_params['biasType'] is not None and filter_params['biasType'].name == 'ESN':
    # using default TA parameters for ESN training
    train_params = {'model': TAModels.Rijke,
                    'beta': 0.7,
                    'tau': 2.1E-3,
                    'std_a': 0.5,
                   'C1': 0.03,
                   'C2': 0.01,
                    'std_psi': 0.5,
                    'est_p': filter_params['est_p'],
                    'alpha_distr': 'uniform'
                    }

    bias_params = {'N_wash': 50,
                   'upsample': 5,
                   'N_units': 100,
                   't_train': 1.0,
                   'L': 1,
                   'augment_data': True,
                   'train_params': train_params
                   }
else:
    bias_params = None
# ================================== CREATE REFERENCE ENSEMBLE =================================

ensemble, truth, b_args = createEnsemble(true_params, forecast_params,
                                         filter_params, bias_params, folder=folder)

# ================================================================================== #
# ================================================================================== #

if run_whyAugment:
    results_folder: str = folder + 'results_whyAugment/'
    flag: bool = True
    k0_U, k0_B, k10_U, k10_B = [], [], [], []

    for L, augment in [(1, False), (1, True), (10, True)]:
        # Reset ESN
        blank_ens = ensemble.copy()
        bias_p = bias_params.copy()
        bias_p['augment_data'] = augment
        bias_p['L'] = L
        bias_p = createESNbias(*b_args, L=L, bias_param=bias_p)
        filter_params['Bdict'] = bias_p
        blank_ens.initBias(bias_p)

        # Add standard deviation to the state
        std = 0.10
        blank_ens.psi = blank_ens.addUncertainty(np.mean(blank_ens.psi, 1), std, blank_ens.m, method='normal')
        blank_ens.hist[-1] = blank_ens.psi
        blank_ens.std_psi, blank_ens.std_a = std, std

        # Run simulation for each gamma --------------------------------------------------------------------------
        ks = [0, 10]
        for k in ks:
            filter_ens = blank_ens.copy()
            filter_ens.bias.k = k

            filter_ens, truth, parameters = main(filter_ens, truth, filter_params,
                                                 results_folder=results_folder, figs_folder=figs_folder, save_=True)

            # GET CORRELATION AND RMS FOR SOLUTION ================================================
            # Observable bias
            y_filter, t_filter = filter_ens.getObservableHist()[0], filter_ens.hist_t
            y_truth = truth['y'][:len(y_filter)]
            b_truth = truth['b_true'][:len(y_filter)]

            # ESN estimated bias
            b, t_b = filter_ens.bias.hist, filter_ens.bias.hist_t

            # Ubiased signal recovered through interpolation
            y_unbiased = y_filter[::filter_ens.bias.upsample] + np.expand_dims(b, -1)
            y_unbiased = interpolate(t_b, y_unbiased, t_filter)

            if flag:
                N_CR = int(.1 / filter_ens.dt)  # Length of interval to compute correlation and RMS
                istop = np.argmin(abs(t_filter - truth['t_obs'][parameters['num_DA'] - 1]))  # end of assimilation
                istart = np.argmin(abs(t_filter - truth['t_obs'][0]))  # start of assimilation
                flag = False

            # PLOT CORRELATION AND RMS ERROR =====================================================================
            CB, RB = CR(y_truth[istop - N_CR:istop], np.mean(y_filter, -1)[istop - N_CR:istop])  # biased
            CU, RU = CR(y_truth[istop - N_CR:istop], np.mean(y_unbiased, -1)[istop - N_CR:istop])  # unbiased

            if k == ks[0]:
                k0_U.append((CU, RU))
                k0_B.append((CB, RB))
            elif k == ks[1]:
                k10_U.append((CU, RU))
                k10_B.append((CB, RB))

    # Plot results -------------------------------------------------------------------------
    y_truth_u = y_truth - b_truth
    Ct, Rt = CR(y_truth[-N_CR:], y_truth_u[-N_CR:])
    Cpre, Rpre = CR(y_truth[istart - N_CR:istart + 1:], np.mean(y_filter, -1)[istart - N_CR:istart + 1:])
    args = (k0_U, k0_B, k10_U, k10_B, Ct, Rt, Cpre, Rpre)

    np.save(results_folder + 'barPlot_args', args)
    barPlot(*args, figs_folder)

if run_loopParams:

    Ls = [100]
    stds = [0.1, 0.25]
    # stds = [0.1]
    ks = np.linspace(0., 3., 4)

    for L in Ls:
        blank_ens = ensemble.copy()
        # Reset ESN
        bias_p = createESNbias(*b_args, L=L, bias_param=bias_params)
        filter_params['Bdict'] = bias_p
        blank_ens.initBias(bias_p)
        for std in stds:
            # Reset std
            blank_ens.psi = blank_ens.addUncertainty(np.mean(blank_ens.psi, 1), std, blank_ens.m, method='normal')
            blank_ens.hist[-1] = blank_ens.psi
            blank_ens.std_psi, blank_ens.std_a = std, std

            results_folder = folder + 'results_loopParams/std{}/L{}/'.format(std, L)
            for k in ks:  # Reset gamma value
                filter_ens = blank_ens.copy()
                filter_ens.bias.k = k

                out = main(filter_ens, truth, filter_params,
                           results_folder=results_folder, figs_folder=figs_folder, save_=True)

                if k in ks:
                    # if k in (0, 10, 50):
                    filename = '{}L{}_std{}_k{}'.format(figs_folder, L, std, k)
                    post_process_single_SE_Zooms(*out[:2], filename=filename+'_time')
                    post_process_single(*out, filename=filename+'_J')
            filename = '{}CR_L{}_std{}_results'.format(figs_folder, L, std)
            post_process_multiple(results_folder, filename)
            plt.close('all')

    fig2(folder + 'results_loopParams/', Ls, stds, figs_folder)
