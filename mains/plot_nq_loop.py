import sys
sys.path.append('..')
from essentials.plotResults import *

from essentials.Util import load_from_pickle_file, set_working_directories, interpolate, save_figs_to_pdf
results_folder = set_working_directories('circle', current_dir='..')[1]  


print('Loading files from:', results_folder)

data_name = 'data_noise0.1gauss_smoothing0.1/'
run_name = 'run_250211-1823/'


file = f'{results_folder}{data_name}{run_name}/test_data'
X_test_raw, X_test_true = load_from_pickle_file(file)




file = f'{results_folder}{data_name}{run_name}Nq_0/results'

filter_ens, truth = load_from_pickle_file(file)


filter_ens.print_model_parameters()


# Plot Observables results

def plot_loop_Nq():

    for nq in range(5):
        file = f'{results_folder}{data_name}{run_name}Nq_{nq}/results'
        _ens = load_from_pickle_file(file)[0]

        plot_observables(_ens, truth=truth, plot_states=True, plot_ensemble_members=1)

        plt.suptitle(f'N_sensors={_ens.N_sensors}')


    save_figs_to_pdf(f'{results_folder}{data_name}{run_name}Obs.pdf')



    t_hist = filter_ens.hist_t
    t_obs = truth['t_obs']

    obs_idx = np.searchsorted(t_hist, t_obs)



    for nq in range(5):
        file = f'{results_folder}{data_name}{run_name}Nq_{nq}/results'
        _ens = load_from_pickle_file(file)[0]

        plot_covariance(filter_ens, idx=[obs_idx[0]-1, obs_idx[0], obs_idx[0]+1])

        plt.suptitle(f'N_sensors={_ens.N_sensors}')



    save_figs_to_pdf(f'{results_folder}{data_name}{run_name}Cpps.pdf')




    # RUN MSE OF THE RECONSTRUCTION


    all_reconstructed_data = []
    labels = []

    for nq in range(5):

        file = f'{results_folder}{data_name}{run_name}Nq_{nq}/results'
        _ens = load_from_pickle_file(file)[0]

        forecast_coefficients = _ens.get_POD_coefficients(Nt=0)
        reconstructed_data = _ens.reconstruct(Phi=np.mean(forecast_coefficients, axis=-1))

        all_reconstructed_data.append(reconstructed_data)
        labels.append(f'N_sensors = {_ens.N_sensors}')

        # Plot single case
        plot_MSE_evolution(_ens,
                            X_test_raw,
                            X_test_true,
                            truth)
        plt.suptitle(f'N_sensors={_ens.N_sensors}')


    plot_MSE_evolution(filter_ens,
                        X_test_raw,
                        X_test_true,
                        truth,
                        reconstructed_data=all_reconstructed_data,
                        tiks=labels)

    save_figs_to_pdf(f'{results_folder}{data_name}{run_name}MSEs.pdf')
        


