import sys
import os


path = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(path, '..')))

from essentials.plotResults import *

from essentials.Util import load_from_pickle_file, set_working_directories, interpolate, save_figs_to_pdf
results_folder = set_working_directories('circle', current_dir='..')[1]  



def plot_m_ur_loop():

    print('Loading files from:', results_folder)

    data_name = 'data_noise0.1gauss_smoothing0.1/'
    run_name = 'run_250213-1455/'

    def filename(m, ur):
        return f'{results_folder}{data_name}{run_name}m_{m}/results_update_r_{ur}'

    file = f'{results_folder}{data_name}{run_name}/test_data'
    X_test_raw, X_test_true = load_from_pickle_file(file)

    file = f'{results_folder}{data_name}{run_name}m_10/results_update_r_False'
    filter_ens, truth = load_from_pickle_file(file)

    filter_ens.print_model_parameters()


    # Plot Observables results
    for m in [5, 10, 50]:
        for ur in [True, False]:
            _ens = load_from_pickle_file(filename(m, ur))[0]

            plot_observables(_ens, truth=truth, plot_states=True, plot_ensemble_members=1)

            plt.suptitle(f'm={_ens.m}, update_reservoir={ur}')


    save_figs_to_pdf(f'{results_folder}{data_name}{run_name}Obs.pdf')

    # PLOT COVSRIANCE
    t_hist = filter_ens.hist_t
    t_obs = truth['t_obs']
    obs_idx = np.searchsorted(t_hist, t_obs)

    for m in [5, 10, 50]:
        for ur in [True, False]:
            _ens = load_from_pickle_file(filename(m, ur))[0]

            plot_covariance(filter_ens, idx=[obs_idx[0]-1, obs_idx[0], obs_idx[0]+1])

            plt.suptitle(f'm={_ens.m}, update_reservoir={ur}')


    save_figs_to_pdf(f'{results_folder}{data_name}{run_name}Cpps.pdf')

    # RUN MSE OF THE RECONSTRUCTION
    all_reconstructed_data = []
    labels = []

    for m in [5, 10, 50]:
        for ur in [True, False]:
            _ens = load_from_pickle_file(filename(m, ur))[0]

            forecast_coefficients = _ens.get_POD_coefficients(Nt=0)
            reconstructed_data = _ens.reconstruct(Phi=np.mean(forecast_coefficients, axis=-1))

            all_reconstructed_data.append(reconstructed_data)

            labels.append(f'm={_ens.m}, update_reservoir={ur}')

            # Plot single case
            plot_MSE_evolution(_ens,
                                X_test_raw,
                                X_test_true,
                                truth)
            
            plt.suptitle(f'm={_ens.m}, update_reservoir={ur}')
            plt.tight_layout()

    plot_MSE_evolution(filter_ens,
                        X_test_raw,
                        X_test_true,
                        truth,
                        reconstructed_data=all_reconstructed_data,
                        tiks=labels)

    save_figs_to_pdf(f'{results_folder}{data_name}{run_name}MSEs.pdf')
        


if __name__ == "__main__":
    pass