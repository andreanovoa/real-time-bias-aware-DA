import sys
import os as os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt


path = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(path, '..')))


from src.Util import load_from_pickle_file, set_working_directories, interpolate, save_figs_to_pdf

results_folder = set_working_directories('circle')[1]  


from src.plotResults import *


def plot_timeseries(filter_ens, truth, plot_states=True, plot_bias=False, plot_ensemble_members=False,
                    filename=None, reference_y=1., reference_t=1., max_time=None, dims='all'):
    def interpolate_obs(truth, t):
        return interpolate(truth['t'], truth['y_raw'], t), interpolate(truth['t'], truth['y_true'], t)
    
    def normalize(*arrays, factor):
        return [arr / factor for arr in arrays]

    
    t_obs, obs = truth['t_obs'], truth['y_obs']
    y_filter, t = filter_ens.get_observable_hist(), filter_ens.hist_t
    y_mean = np.mean(y_filter, -1, keepdims=True)
    Nq = filter_ens.Nq
    dims = range(Nq) if dims == 'all' else dims
    
    N_CR = int(filter_ens.t_CR // filter_ens.dt)
    max_time = min(truth['t_obs'][-1] + filter_ens.t_CR, t[-1]) if max_time is None else max_time
    i0, i1 = [np.argmin(abs(t - ttt)) for ttt in [truth['t_obs'][0], max_time]]
    y_filter, y_mean, t = (yy[i0 - N_CR:i1 + N_CR] for yy in [y_filter, y_mean, t])
    
    if filter_ens.bias is not None:
        b, t_b = filter_ens.bias.get_bias(state=filter_ens.bias.hist), filter_ens.bias.hist_t
        y_unbiased = recover_unbiased_solution(t_b, b, t, y_mean, upsample=hasattr(filter_ens.bias, 'upsample'))
    else:
        b, t_b, y_unbiased = None, None, None
    
    y_raw, y_true = interpolate_obs(truth, t)
    
    t_wash, wash = truth.get('wash_t', 0.), truth.get('wash_obs', np.zeros_like(obs))
    t_label = '$t$ [s]' if reference_t == 1. else '$t/T$'
    t, t_obs, max_time, t_wash = normalize(t, t_obs, max_time, t_wash, factor=reference_t)
    if t_b is not None:
        t_b = t_b / reference_t
    
    y_raw, y_filter, y_mean, obs, y_true = normalize(y_raw, y_filter, y_mean, obs, y_true, factor=reference_y)
    if y_unbiased is not None:
        y_unbiased = y_unbiased / reference_y
    
    margin, max_y, min_y = 0.15 * np.mean(abs(y_raw)), np.max(y_raw), np.min(y_raw)
    y_lims, x_lims = [min_y - margin, max_y + margin], [[t[0], max_time], [t_obs[-1] - filter_ens.t_CR, max_time]]
    
    def plot_series(true, ref, axs, xlim, lbls):
        for qi, ax in enumerate(axs):
            ax.plot(t, true[:, qi], label=lbls[0], **true_props)
            
            m = np.mean(ref[:, qi], axis=-1)
            ax.plot(t, m, **y_biased_mean_props, label=lbls[1])
            if plot_ensemble_members:
                ax.plot(t, ref[:, qi], **y_biased_props)
            else:
                s = np.std(ref[:, qi], axis=-1)
                ax.fill_between(t, m + s, m - s, alpha=0.5, color=y_biased_props['color'])

            ax.plot(t_obs, obs[:, qi], label='data', **obs_props)

            if 'wash_t' in truth.keys():
                ax.plot(t_wash, wash[:, qi], **obs_props)

            plot_DA_window(t_obs, ax, ens=filter_ens)
            ax.set(ylim=y_lims, xlim=xlim)

            if qi == 0:
                ax.legend(loc='lower left', bbox_to_anchor=(0.1, 1.1), ncol=5, fontsize='small')
        axs[-1].set(xlabel=t_label) 
    
    if plot_states:
        fig1, fig1_axs = plt.subplots(len(dims), 2, figsize=(8, 1.5 * len(dims)), layout="constrained", sharex='col', sharey='row')
        if Nq == 0:
            fig1_axs = fig1_axs[np.newaxis, :]

        for axx, xl in zip(fig1_axs.T, x_lims):
            plot_series(axs=axx, true=y_raw, ref=y_filter, xlim=xl, lbls=['True', 'Model estimate'])

        if filename:
            plt.savefig(filename + '.svg', dpi=350)
            plt.close()







def load_test_data(folder):
    file = f'{folder}/test_data'
    return load_from_pickle_file(file)


def plot_m_ur_loop(results_folder,
                   m_loop = [5, 10, 50],
                   data_name = 'data_noise0.1gauss_smoothing0.1/',
                   run_name = 'run_250213-1455/'):

    dir = f'{results_folder}{data_name}{run_name}'

    X_test_raw, X_test_true = load_test_data(folder=dir)

    def filename(_m, _ur):
        return f'{dir}m_{_m}/results_update_r_{_ur}'

    def set_label(_m, _ur):
        return f'm={_m}, update_reservoir={_ur}'


    # Load one case to define the truth
    file = f'{dir}m_10/results_update_r_False'
    filter_ens, truth = load_from_pickle_file(file)

    # Plot Observables results
    for m in m_loop:
        for ur in [True, False]:
            _ens = load_from_pickle_file(filename(m, ur))[0]
            plot_observables(_ens, truth=truth, plot_states=True, plot_ensemble_members=1)
            plt.suptitle(set_label(m, ur))
            if _ens.Na > 0:
                plot_parameters(_ens, truth=truth)
                plt.suptitle(set_label(m, ur))

    save_figs_to_pdf(f'{dir}Obs.pdf')

    
    # PLOT COVSRIANCE


    tixs = [f'$\\Phi_{j+1}$' for j in np.arange(_ens.N_modes)]
    tixs += [f'$r_{j+1}$' for j in np.arange(_ens.N_units)]
    tixs += _ens.alpha_labels.copy()
    tixs += _ens.obs_labels.copy()

    for m in m_loop:
        for ur in [True, False]:
            _ens = load_from_pickle_file(filename(m, ur))[0]
            plot_covariance(_ens, idx=-1, tixs=tixs)
            plt.suptitle(set_label(m, ur))
    save_figs_to_pdf(f'{dir}Cpps.pdf')

    # RUN MSE OF THE RECONSTRUCTION
    for ur in [True, False]:

        all_reconstructed_data = []
        labels = []


        for m in m_loop:

            labels.append(set_label(m, ur))

            _ens = load_from_pickle_file(filename(m, ur))[0]

            forecast_coefficients = _ens.get_POD_coefficients(Nt=0)
            reconstructed_data = _ens.reconstruct(Phi=np.mean(forecast_coefficients, axis=-1))

            all_reconstructed_data.append(reconstructed_data)

            # Plot single case
            plot_MSE_evolution(_ens, X_test_raw, X_test_true, truth)
            
            plt.suptitle(set_label(m, ur))

        # Plot MSE comparison for all m-ur cases
        plot_MSE_evolution(filter_ens,
                            X_test_raw,
                            X_test_true,
                            truth,
                            reconstructed_data=all_reconstructed_data,
                            tiks=labels)
        
        plt.suptitle(f'Update_reservoir = {ur}')
    save_figs_to_pdf(f'{dir}MSEs.pdf')
        



def plot_loop_Nq(results_folder,
                 data_name = 'data_noise0.1gauss_smoothing0.1/',
                 run_name = 'run_250211-1823/'):

    dir = f'{results_folder}{data_name}{run_name}'

    X_test_raw, X_test_true = load_test_data(folder=dir)


    for nq in range(5):
        file = f'{dir}Nq_{nq}/results'
        _ens, _truth = load_from_pickle_file(file)

        plot_observables(_ens, truth=_truth, plot_states=True, plot_ensemble_members=1)

        plt.suptitle(f'N_sensors={_ens.N_sensors}')

        if _ens.Na > 0:
            plot_parameters(_ens, truth=_truth)
            plt.suptitle(f'N_sensors={_ens.N_sensors}')


    save_figs_to_pdf(f'{dir}Obs.pdf')


    for nq in range(5):
        file = f'{dir}Nq_{nq}/results'
        _ens = load_from_pickle_file(file)[0]

        plot_covariance(_ens, idx=-1)

        plt.suptitle(f'N_sensors={_ens.N_sensors}')



    save_figs_to_pdf(f'{dir}Cpps.pdf')



    # RUN MSE OF THE RECONSTRUCTION


    all_reconstructed_data = []
    labels = []

    for nq in range(5):

        file = f'{dir}Nq_{nq}/results'
        _ens = load_from_pickle_file(file)[0]

        forecast_coefficients = _ens.get_POD_coefficients(Nt=0)
        reconstructed_data = _ens.reconstruct(Phi=np.mean(forecast_coefficients, axis=-1))

        all_reconstructed_data.append(reconstructed_data)
        labels.append(f'N_sensors = {_ens.N_sensors}')

        # # Plot single case
        # plot_MSE_evolution(_ens,
        #                     X_test_raw,
        #                     X_test_true,
        #                     _truth)
        # plt.suptitle(f'N_sensors={_ens.N_sensors}')
        # plt.tight_layout()


    plot_MSE_evolution(_ens,
                        X_test_raw,
                        X_test_true,
                        _truth,
                        reconstructed_data=all_reconstructed_data,
                        tiks=labels)
    

    save_figs_to_pdf(f'{dir}MSEs.pdf')





def plot_loop_Nt_obs(results_folder,
                     Nt_obs_loop=[10, 20, 30],
                     m_loop=[10, 20, 50],
                     m_fixed=50,
                     data_name = 'data_noise0.1gauss_smoothing0.1/',
                     run_name = 'run_250211-1823/',
                     _plot_times=True,
                     _plot_MSE = True):

    dir = f'{results_folder}{data_name}{run_name}'

    X_test_raw, X_test_true = load_test_data(folder=dir)

    def set_filename(_nt, _ns, _m):
        return f'{dir}Nt_obs_{_nt}/results_urTrue_Nsensors{_ns}_m{_m}'

    def store_fig(fig_list, title):
        _fig = plt.gcf()
        plt.suptitle(title)
        fig_list.append(_fig)
        plt.close(_fig)

    def get_rc(_nt, _ns, _m):
        file = set_filename(_nt, _ns, _m)
        rc_ens, rc_truth = load_from_pickle_file(file)
        forecast_coefficients = rc_ens.get_POD_coefficients(Nt=0)
        rc = rc_ens.reconstruct(Phi=np.mean(forecast_coefficients, axis=-1))

        return rc, rc_truth, rc_ens
    

    if _plot_times:

        for nt in Nt_obs_loop:
            for ns in range(5):
                figs_obs, figs_alpha = [], []
                for m in m_loop:
                    file = set_filename(nt, ns, m)
                    _ens, _truth = load_from_pickle_file(file)

                    plot_observables(_ens, truth=_truth, plot_states=True, plot_ensemble_members=1)
                    store_fig(figs_obs, title=f'Nt_obs={nt}, N_sensors={ns}, m={m}')

                    if _ens.Na > 0:
                        plot_parameters(_ens, truth=_truth)
                        store_fig(figs_alpha, title=f'Nt_obs={nt}, N_sensors={ns}, m={m}')

                save_figs_to_pdf(figs=figs_obs, pdf_name=f'{dir}Obs_Nt{nt}_Ns{ns}.pdf')
                save_figs_to_pdf(figs=figs_alpha, pdf_name=f'{dir}Alpha_Nt{nt}_Ns{ns}.pdf')


    if _plot_MSE:

        for nt in Nt_obs_loop:

            mse_figs = []

            ns_rec, ns_lbl = [], []

            for ns in range(5):
                m_rec, m_lbl = [], []
                for m in m_loop:

                    print(f'Nt{nt}, Ns{ns}, m{m}')
                    reconstructed_data = get_rc(nt, ns, m)[0]
                    m_rec.append(reconstructed_data)
                    m_lbl.append(f'm={m}')

                _rc, _truth, _ens = get_rc(nt, ns, m_fixed)
                ns_rec.append(_rc)
                ns_lbl.append(f'N_s={ns}')

                if len(m_loop) > 1:
                    #  plot mse as funciton of m 
                    plot_MSE_evolution(_ens,
                                        X_test_raw,
                                        X_test_true,
                                        _truth,
                                        reconstructed_data=m_rec,
                                        tiks=m_lbl)
                    

                    store_fig(mse_figs, title=f'MSE_Nt{nt}_Ns{ns}')
                    # save_figs_to_pdf(pdf_name=f'{dir}MSE_Nt{nt}_Ns{ns}.pdf')
                
                # store_fig(mse_figs, title=f'Nt_obs={nt}, N_sensors={ns}')


            # #  plot mse as funciton of ns, m=10
            plot_MSE_evolution(_ens,
                                X_test_raw,
                                X_test_true,
                                _truth,
                                reconstructed_data=ns_rec,
                                tiks=ns_lbl)
            
            # plt.suptitle(f'MSE_Nt{nt}_m{m_fixed}')
            # save_figs_to_pdf(pdf_name=f'{dir}MSE_Nt{nt}_m{m_fixed}.pdf')
            
            store_fig(mse_figs, title=f'MSE_Nt{nt}_m{m_fixed}')
            


            save_figs_to_pdf(pdf_name=f'{dir}MSE_Nt{nt}.pdf', 
                             figs=mse_figs)


def plot_converged_parameters(results_folder,
                     Nt_obs_loop=[10, 20, 30],
                     Ns_loop=range(5),
                     m_loop=[10, 20, 50], 
                     data_name = 'data_noise0.1gauss_smoothing0.1/',
                     run_name = 'run_250211-1823/'):

    dir = f'{results_folder}{data_name}{run_name}'

    def set_filename(_nt, _ns, _m):
        return f'{dir}Nt_obs_{_nt}/results_urTrue_Nsensors{_ns}_m{_m}'

    def get_alpha(_nt, _ns, _m):
        file = set_filename(_nt, _ns, _m)
        a_ens = load_from_pickle_file(file)[0]
        
        alpha_indices = [a_ens.est_a.index(f'svd_{key}') for key in range(4)]

        _a = [a_ens.hist[-1, a_ens.Nphi + idx] for idx in alpha_indices]
        _a0 = [a_ens.hist[0, a_ens.Nphi + idx] for idx in alpha_indices]

        return _a, _a0
    
    def get_alpha0(_nt, _ns, _m):
        file = set_filename(_nt, _ns, _m)
        a_ens = load_from_pickle_file(file)[0]
        
        alpha_indices = [a_ens.est_a.index(f'svd_{key}') for key in range(4)]

        return [a_ens.get_current_state[a_ens.Nphi + idx] for idx in alpha_indices]
    

    Ns_loop = [*Ns_loop[1:], Ns_loop[0]]

    xlabels = ['Init'] + [f'{ns}' for ns in Ns_loop]
    xlabels[-1] = 'Full'

    markers = ['^', 'o', 's', 'd']  # Different markers for each parameter
    x_offset = [.2, .4, .6, .8]

    _, axs = plt.subplots(nrows=len(Nt_obs_loop), ncols=len(m_loop), figsize=(12, 10), sharex=True, sharey=True)

    for row_i, nt in enumerate(Nt_obs_loop):

        for col_i, m in enumerate(m_loop):

            ax = axs[row_i, col_i]
 
            for ni, ns in enumerate(Ns_loop):  

                alpha, alpha0 = get_alpha(nt, ns, m)

                # if ns == 0:
                #     for ai, param in enumerate(alpha0):
                #         xval = -1 + x_offset[ai] 
                #         ax.errorbar(xval, 1, 2*np.std(param)/np.mean(param), marker=markers[ai], alpha=0.7, c= 'k', ms=4, capsize=2)
                #         # ax.errorbar(xval, np.mean(param), 2*np.std(param), marker=markers[ai], alpha=0.7, c= 'k', ms=4, capsize=2)


                for ai, (param, param0) in enumerate(zip(alpha, alpha0)):
                    xval = ni + x_offset[ai]
                    yval = np.mean(param) / np.mean(param0)
                    stdd = 2*np.std(param) / np.mean(param0)
                    ax.errorbar(xval, yval, stdd , marker=markers[ai], alpha=0.7, c= f'C{ni+1}', ms=4, capsize=2)

                    if row_i == 0 and col_i == 0 and ai ==0:
                        ax.legend([f'$\\sigma_{k+1}$' for k in range(4)])

            ax.fill_betweenx([0.9, 1.1], -1,  0, alpha=.1, color='k', zorder=-1) 
            ax.plot([-1, 0], [1, 1], 'k')

            ax.set(title=f'Nt={nt}, m={m}', xlim=[-.5, len(Ns_loop)])
            ax.set_xticks([ns + .5 for ns in range(-1, len(Ns_loop))], xlabels)



    #  not notmalized
    _, axs = plt.subplots(nrows=len(Nt_obs_loop), ncols=len(m_loop), figsize=(12, 10), sharex=True, sharey=True)

    for row_i, nt in enumerate(Nt_obs_loop):

        for col_i, m in enumerate(m_loop):

            ax = axs[row_i, col_i]
 
            for ni, ns in enumerate(Ns_loop):  

                alpha, alpha0 = get_alpha(nt, ns, m)

                if ns == 0:
                    for ai, param in enumerate(alpha0):
                        xval = -1 + x_offset[ai] 
                        ax.errorbar(xval, np.mean(param), 2*np.std(param), marker=markers[ai], alpha=0.3, c= 'k', ms=4, capsize=2)


                for ai, (param, param0) in enumerate(zip(alpha, alpha0)):
                    xval = ni + x_offset[ai]
                    yval = np.mean(param) 
                    stdd = 2*np.std(param) 
                    ax.errorbar(xval, yval, stdd , marker=markers[ai], alpha=0.7, c= f'C{ni+1}', ms=4, capsize=2)

                    if row_i == 0 and col_i == 0 and ai ==0:
                        ax.legend([f'$\\sigma_{k+1}$' for k in range(4)])

            ax.set(title=f'Nt={nt}, m={m}', xlim=[-1, len(Ns_loop)])
            ax.set_xticks([ns + .5 for ns in range(-1, len(Ns_loop))], xlabels)



    _, axs = plt.subplots(nrows=4, ncols=len(Nt_obs_loop), figsize=(12, 10), sharex=True, sharey='row')

    for col_i, nt in enumerate(Nt_obs_loop):
 
        for ni, ns in enumerate(Ns_loop):  

            for mi, m in enumerate(m_loop):

                alpha, alpha0 = get_alpha(nt, ns, m)

                for row_i, (param, param0) in enumerate(zip(alpha, alpha0)):

                    ax = axs[row_i, col_i]
                    xval = ni + x_offset[mi]

                    ax.errorbar(xval,  np.mean(param) , 2*np.std(param), marker=markers[row_i], alpha=0.7, c= f'C{mi+1}', ms=4, capsize=2)
                    if mi == 0:
                        ax.errorbar(-0.5,  np.mean(param0) , 2*np.std(param0), marker=markers[row_i], alpha=0.2, c= f'k', ms=4, capsize=2)
                    

                    if row_i == 0 and col_i == 0 and ai ==0:
                        ax.legend([f'$\\sigma_{k+1}$' for k in range(4)])

                    # ax.fill_betweenx([0.9, 1.1], -1,  0, alpha=.1, color='k', zorder=-1) 
                    # ax.plot([-1, 0], [1, 1], 'k')

                    ax.set(title=f'Nt={nt}', xlim=[-1., len(Ns_loop)])
                    ax.set_xticks([ns + .5 for ns in range(-1, len(Ns_loop))], xlabels)




    save_figs_to_pdf(pdf_name=f'{dir}Converged_parameters.pdf')

if __name__ == "__main__":
    # pass

    print('Loading files from:', results_folder)

    # data_name = 'data_noise0.1gauss_smoothing0.1/'
    # run_name = 'run_250204-1907/'
    run_name='run_online_training_100/'

    plot_converged_parameters(results_folder,
                    run_name=run_name,
                     Nt_obs_loop=[10, 20, 30],
                     Ns_loop=range(5),
                     m_loop=[10, 50, 100, 200])

    # file = f'{results_folder}{data_name}{run_name}results'

    # plot_loop_Nt_obs(results_folder,
    #                 run_name=run_name,
    #                 _plot_MSE=True,
    #                 _plot_times=False,
    #                 m_loop=[10],
    #                 m_fixed=10)
    
    # plot_m_ur_loop(results_folder, 
    #                m_loop=[5, 10, 50],
    #                run_name='run_param_est/')

    # filter_ens, truth = load_from_pickle_file(file)

    # # Plot results
    # plot_timeseries(filter_ens, truth=truth, plot_states=True, plot_ensemble_members=1)

    # # filter_ens.print_model_parameters()

    # plot_covariance(filter_ens)


    # plt.show()