import sys
import os as os

from src.models_datadriven import POD_ESN
from src.utils import set_working_directories, load_from_mat_file, add_noise_to_flow, save_to_mat_file, save_to_pickle_file, load_from_pickle_file, save_figs_to_pdf
from src.plot_results import plot_truth, plot_ensemble, animate_flowfields
from src.create import create_ensemble
from src.data_assimilation import dataAssimilation_bias_blind

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import display, Image

import time


# Define woking directories
data_folder, results_folder = set_working_directories('wakes')[:2]




def plot_timeseries(*plot_cases, zoom_window=None, add_pdf=False, t_factor=1):
    """
    Plot the time evolution of the observables in a object of class model
    """

    fig = plt.figure(figsize=(10, 4.5), layout="constrained")
    axs = fig.subplots(plot_cases[0].Nq, 2 + add_pdf, sharey='row', sharex='col')
    xlabel = '$t$'
    if t_factor != 1:
        xlabel += '$/T$'

    for plot_case in plot_cases:
        y = plot_case.get_observable_hist()  # history of the model observables
        lbl = plot_case.obs_labels

        t_h = plot_case.hist_t / t_factor
        if zoom_window is None:
            zoom_window = [t_h[-1] - plot_case.t_CR, t_h[-1]]

        for ii, ax in enumerate(axs):
            [ax[jj].plot(t_h, y[:, ii]) for jj in range(2)]
            ax[0].set(ylabel=lbl[ii])
            if add_pdf:
                ax[2].hist(y[:, ii], alpha=0.5, histtype='stepfilled', bins=20, density=True, orientation='horizontal',
                           stacked=False)

        axs[-1][0].set(xlabel=xlabel, xlim=[t_h[0], t_h[-1]])
        axs[-1][1].set(xlabel=xlabel, xlim=zoom_window)
        if plot_case.ensemble:
            plt.gcf().legend([f'$mi={mi}$' for mi in range(plot_case.m)], loc='center left', bbox_to_anchor=(1.0, .75),
                             ncol=1, frameon=False)



def load_cylinder_dataset(noise_type = 'gauss', noise_level = 0.1, smoothing = 0.1, display_flow=False):

    dir = f'/data_noise{noise_level}{noise_type}_smoothing{smoothing}/'


    os.makedirs(results_folder+dir, exist_ok=True)

    data_name = results_folder+f'{dir}00_data.mat'

    if not os.path.exists(data_name):

        # Load dataset
        mat = load_from_mat_file(data_folder + 'circle_re_100.mat')

        U, V = [mat[key] for key in ['ux', 'uy']]  # Nu, Nt, Ny, Nx 
        U[np.isnan(U)] = 0.
        V[np.isnan(V)] = 0.

        U_noisy, V_noisy = add_noise_to_flow(U, V, 
                                            noise_level=noise_level, 
                                            noise_type=noise_type, 
                                            spatial_smooth=smoothing)

        gif_name = results_folder+f'{dir}00_data.gif'

        # Visualize the flow fields
        if not os.path.exists(gif_name):
            anim = animate_flowfields([U, U_noisy, V, V_noisy], 
                                    titles=['$u_x$', '$u_y$', '$\\tilde{u}_x$', '$\\tilde{u}_y$'], n_frames=50)
            anim.save(gif_name)

        # Display in notebook
        if display_flow:
            display(Image(filename=gif_name))

        all_data = np.array([U, V])                     # Nu, Nt, Ny, Nx
        all_data_noisy = np.array([U_noisy, V_noisy])   # Nu, Nt, Ny, Nx

        #  Change order of dimensions
        all_data = all_data.transpose(1, 2, 3, 0)               # Nt, Nx, Ny, Nu
        all_data_noisy = all_data_noisy.transpose(1, 2, 3, 0)   # Nt, Nx, Ny, Nu

        save_to_mat_file(data_name, dict(all_data=all_data,
                                         all_data_noisy=all_data_noisy))
    else:
        dataset = load_from_mat_file(data_name)
        all_data, all_data_noisy = [dataset[key] for key in ['all_data', 'all_data_noisy']]

    return all_data, all_data_noisy, dir


def set_truth(case, X_filter, X_filter_true, Nt_obs = 25, plot=False):


    N_test = X_filter.shape[-1]

    if case.sensor_locations is not None:
        data_obs = X_filter.copy().reshape(-1, N_test)[case.sensor_locations].T
        data_obs_true = X_filter_true.copy().reshape(-1, N_test)[case.sensor_locations].T
    else:
        data_obs = case.project_data_onto_Psi(data=X_filter)
        data_obs_true = case.project_data_onto_Psi(data=X_filter_true)

    dt = case.dt
    t_true = np.arange(0, N_test)  * dt

    t_start = .5
    t_stop = min(5., t_true[-10])
    

    obs_idx = np.arange(t_start // dt, t_stop // dt + 1, Nt_obs, dtype=int) + 1


    Nt_extra = len(t_true[obs_idx[-1]:])

    _truth = dict(y_raw=data_obs,
                  y_true=data_obs_true, 
                  t=t_true, 
                  dt=dt,
                  t_obs=t_true[obs_idx], 
                  y_obs=data_obs[obs_idx], 
                  dt_obs=Nt_obs * dt,
                  Nt_extra=Nt_extra
                  )
    if plot:
        plot_truth(**_truth)
    return _truth


def load_POD_ESN_case(folder, N_modes=4, N_units=40):

    case_filename = results_folder + f'{folder}POD{N_modes}_ESN{N_units}_Ntrain{X_train.shape[-1]}'

    if os.path.isfile(case_filename):
        _case = load_from_pickle_file(case_filename)
        print('Loaded case')
    else:
        _case = POD_ESN(data=X_train,
                        N_modes=N_modes, 
                        N_units=N_units, 
                        domain=[-2, 2, 0, 12],
                        rho_range = (.2, 1.05), 
                        figs_folder=results_folder, 
                        plot_case=True,
                        plot_training=True,
                        skip_sensor_placement=True,
                        pdf_file=case_filename,
                        t_val=t_val,
                        t_train=t_train,
                        dt = dt,
                        t_test=0.
                        )
        _case.name = case_filename

        save_to_pickle_file(case_filename, _case)

    return _case


def plot_initial_case(case, name='case'):
    original_data = X_train[..., -1]
    true_data = X_train_true[..., -1]
    POD_ESN.plot_case(case, 
                      datasets=[original_data, true_data],
                      names=['data', 'truth'],
                      )
    case.plot_Wout()
    plot_ensemble(case, max_modes=10)

    _case = case.copy()
    # Forecast the ensemble
    psi, t = _case.time_integrate(Nt=500)
    _case.update_history(psi, t)

    # Visualize the time evolution of the physical states
    plot_timeseries(_case, zoom_window=[t[-1]-_case.t_CR, t[-1]])

    save_figs_to_pdf(figs_folder+f'{name}.pdf')




def set_sensors(case, n_sensors=0):

    model = case.copy()

    if N_sensors > 0:
        model.reset_sensors(N_sensors=n_sensors,
                            domain_of_measurement=[-1, 1,4,7], 
                            down_sample_measurement=(10, 40),
                            qr_selection=True
                            )
    else:
        model.reset_sensors(measure_modes=True)

    return model




if __name__ == '__main__':

    X_true, X_noisy, data_dir = load_cylinder_dataset(noise_type='gauss', noise_level=0.1, smoothing=0.1)

    # Split the data into POD-ESN training and data to use for assimilation
    # N_train = int(.2 *  X_true.shape[0])
    N_train = 100

    X_train, X_train_true = [yy[:N_train].T for yy in [X_noisy, X_true]]
    X_filter, X_filter_true = [yy[N_train:].T for yy in [X_noisy, X_true]]

    dt = 0.01
    t_val = X_train.shape[-1] *.1 * dt 
    t_train = X_train.shape[-1] * dt - t_val

    #  ======================== Load the POD-ESN objsect ============================  #

    model_og = load_POD_ESN_case(folder=data_dir,
                                 N_modes = 4,  # Number of POD modes
                                 N_units = 40, # Number of neurons in the ESN
                                 )

    # run_name = f'/run_{time.strftime("%y%m%d-%H%M")}/'
    # run_name = 'run_250213-1455/'
    run_name = f'/run_online_training_{X_train.shape[-1]}/'


    loop_folder = results_folder + data_dir + run_name

    save_to_pickle_file(f'{loop_folder}test_data', X_filter, X_filter_true)



    update_state = True
    update_reservoir = True


    def set_filename(case):
        return f'{figs_folder}results_ur{case.update_reservoir}_Nsensors{case.N_sensors}_m{case.m}'


    #  ************************************************************************************************************************  #

    # N_sensors = 0 # Number of sensors. if 0, measure POD coefficients
    for N_sensors in range(5):


        #  ======================== Define the measurements ============================  #

        model = set_sensors(model_og, N_sensors)

        for nt_obs in [10, 20, 30]:

            figs_folder = loop_folder + f'Nt_obs_{nt_obs}/'
            os.makedirs(figs_folder, exist_ok=True)

            truth = set_truth(case=model,  X_filter=X_filter, X_filter_true=X_filter_true, Nt_obs=nt_obs)


            #  ************************************************************************************************************************  #
            # m = 10
            # for m in [5, 10, 50]:
            m_loop = [10, 50, 100, 200]
            # m_loop = [10]
            for m in m_loop:
                # figs_folder = loop_folder + f'm_{m}/'

                # os.makedirs(figs_folder, exist_ok=True)

                # Open a file in write mode
                with open(figs_folder+"log.txt", "w") as f:

                    # sys.stdout = f  # Redirect stdout to the file

                    #  ===================== Define the assimilation type ============================  #


                    # for update_reservoir in [True, False]:
                    for update_reservoir in [True]:
                        # filename = f'{figs_folder}results_update_r_{update_reservoir}'
                        
                        model.update_reservoir = update_reservoir # Important
                        model.t_CR = 0.5
                        
                        # Define ensemble
                        ensemble =  create_ensemble(model=model,
                                                    filter='EnSRKF', 
                                                    m=m,
                                                    model_bias=None,
                                                    Nt_transient = 500, # how long to forecast the initial ensemble to remove transient 
                                                    std_psi=.5,
                                                    est_a=['Wout'],
                                                    std_a=0.1             
                                                    )

                
                        # ***************************************  ********************************************************************************* #

                        filename = set_filename(case=ensemble)
                        plot_initial_case(ensemble, name=f"case_{filename.split('results_')[-1]}")

                        #  ===================== Run simulation ============================  #

                        # Perform assimilation
                        filter_ens = dataAssimilation_bias_blind(ensemble=ensemble,
                                                                std_obs=0.01,
                                                                **truth)
                        
                        # Save simulation
                        save_to_pickle_file(filename, filter_ens, truth)

                        

    # Reset stdout back to the console
    sys.stdout = sys.__stdout__
    print("This text appears in the terminal")

    from src.post_processing.cylinder import plot_loop_Nt_obs

    plot_loop_Nt_obs(results_folder,
                    run_name=run_name,
                    _plot_MSE=True,
                    _plot_times=True,
                    m_loop=m_loop,
                    m_fixed=m_loop[-1])
    
