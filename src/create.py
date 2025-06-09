import os.path

import numpy as np
from src.utils import *
from src.bias import *


rng = np.random.default_rng(0)


def create_ensemble(model=None, forecast_params=None, dt=None, alpha0=None, **filter_params):
    if forecast_params is None:
        forecast_params = filter_params.copy()
    else:
        forecast_params = forecast_params.copy()
    if dt is not None:
        forecast_params['dt'] = dt

    if alpha0 is not None:
        for alpha, lims in alpha0.items():
            forecast_params[alpha] = 0.5 * (lims[0] + lims[1])

    # ==============================  INITIALISE MODEL  ================================= #
    if model is not None:
        forecast_params['model'] = model

    if not model.initialized:
        ensemble = forecast_params['model'](**forecast_params)
    else:
        ensemble = model.copy()

    # Forecast model case to steady state initial condition before initialising ensemble
    if 'Nt_transient' in filter_params.keys():
        Nt = filter_params['Nt_transient']
    else:
        Nt = int(ensemble.t_CR / ensemble.dt)
    state = ensemble.time_integrate(Nt)[0]


    ensemble.update_history(state[-1], reset=True)

    # =========================  INITIALISE ENSEMBLE & BIAS  =========================== #
    ensemble.init_ensemble(**filter_params)



    # Forecast model case to steady state initial condition before initialising ensemble
    if 'Nt_transient' in filter_params.keys():
        Nt = filter_params['Nt_transient']
    else:
        Nt = int(ensemble.t_CR / ensemble.dt)
    state = ensemble.time_integrate(Nt)[0]


    ensemble.update_history(state[-1], reset=True)

    ensemble.close()
    return ensemble

def create_truth(model, t_start=None, t_stop=None, Nt_obs=20, std_obs=0.05, t_max=None, t_min=0.,
                 noise_type='gauss, add', post_processed=False, manual_bias=None, **kwargs):
    # =========================== LOAD DATA OR CREATE TRUTH FROM LOM ================================ #
    if t_start is None:
        t_start = model.t_transient
    if t_stop is None:
        t_stop = t_start + 3 * model.t_CR

    if t_max is None:
        t_max = t_stop + t_start

    if type(model) is str:
        y_raw, y_true, t_true, name_truth = create_observations_from_file(model, t_max=t_max, t_min=t_min, )
        case_name = model.split('/')[-1]
        name_bias = 'Exp_' + case_name
        b_true = np.zeros(1)
        true_case = None
    else:
        y_true, t_true, name_truth, true_case = create_observations(model, t_max=t_max, t_min=t_min, **kwargs)

        #  ADD BIAS TO THE TRUTH #
        if manual_bias is None:
            b_true = y_true * 0.
            name_bias = 'No_bias'
        elif type(manual_bias) is str:
            name_bias = manual_bias
            if manual_bias == 'time':
                b_true = .4 * y_true * np.sin((np.expand_dims(t_true, -1) * np.pi * 2) ** 2)
            elif manual_bias == 'periodic':
                b_true = 0.2 * np.max(y_true, axis=0) * np.cos(2 * y_true / np.max(y_true, axis=0))
            elif manual_bias == 'linear':
                b_true = .1 * np.max(y_true, axis=0) + .3 * y_true
            elif manual_bias == 'cosine':
                b_true = np.cos(y_true)
            else:
                raise ValueError("Bias {} not recognized choose [linear, periodic, time]".format(manual_bias))
        else:
            # The manual bias is a function of state and/or time
            b_true, name_bias = manual_bias(y_true, t_true)
        # Add bias to the reference data
        y_true += b_true

    # =========================== ADD NOISE TO THE TRUTH ================================ #
    if type(model) is not str or post_processed:
        y_raw = create_noisy_signal(y_true, noise_level=std_obs, noise_type=noise_type)
    else:
        if post_processed:
            y_raw = y_true.copy()
        noise_type = name_bias
        std_obs = None

    # =========================== COMPUTE OBSERVATIONS AT DESIRED TIME =========================== #
    if t_min > 0:
        t_start, t_max, t_true = [tt - t_min for tt in [t_start, t_stop, t_true]]

    dt_t = t_true[1] - t_true[0]
    obs_idx = np.arange(t_start // dt_t, t_stop // dt_t + 1, Nt_obs, dtype=int)

    # ================================ SAVE DATA TO DICT ==================================== #
    if '/' in name_truth:
        name_truth = '_'.join(name_truth.split('/'))

    truth = dict(y_raw=y_raw, y_true=y_true, t=t_true, b=b_true, dt=dt_t,
                 t_obs=t_true[obs_idx], y_obs=y_raw[obs_idx], dt_obs=Nt_obs * dt_t,
                 name=name_truth, name_bias=name_bias, noise_type=noise_type,
                 model=model, std_obs=std_obs, true_params=kwargs, case=true_case)
    return truth


def create_observations_from_file(name, t_max, t_min=0.):
    # Wave case: load .mat file ====================================
    try:
        if 'rijke' in name:
            mat = load_from_mat_file(name)
            y_raw, y_true, t_true = [mat[key].transpose() for key in ['p_mic', 'p_mic', 't_mic']]
        elif 'annular' in name:
            mat = load_from_mat_file(name)
            y_raw, y_true, t_true = [mat[key] for key in ['y_raw', 'y_filtered', 't']]
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        raise FileNotFoundError('File ' + name + ' not defined')

    if len(np.shape(t_true)) > 1:
        t_true = np.squeeze(t_true)

    if y_raw.shape[0] != len(t_true):
        y_raw, y_true = [yy.transpose() for yy in [y_raw, y_true]]

    id0, id1 = [np.argmin(abs(t_true - tx)) for tx in [t_min, t_max]]
    y_raw, y_true, t_true = [yy[id0:id1] for yy in [y_raw, y_true, t_true]]

    return y_raw, y_true, t_true, name.split('data/')[-1]


def create_observations(model, t_max, t_min, save=False, data_folder=None, **true_parameters):
    try:
        TA_params = true_parameters.copy()
        model = model
    except AttributeError:
        raise 'true_parameters must be dict'

    # ============================================================
    # Add key input_parameters to filename
    suffix = ''

    for key, val in TA_params.items():
        if key in model.alpha_labels.keys():
            if type(val) is str:
                suffix += val + '_'
            else:
                suffix += key + '{:.2e}'.format(val) + '_'

    if data_folder is None:
        data_folder = os.path.join(os.getcwd() + '/data/')
        os.makedirs(data_folder, exist_ok=True)
    name = data_folder + 'Truth_{}_{}tmax-{:.2}'.format(model.name, suffix, t_max)

    if os.path.isfile(name) and save:
        case = load_from_pickle_file(name)
        print('Load true data: ' + name)
    else:
        case = model(**TA_params)
        psi, t = case.time_integrate(int(t_max / case.dt))
        case.update_history(psi, t)
        case.close()
        if save:
            save_to_pickle_file(name, case)
            print('Save true data: ' + name)

    # Retrieve observables
    y_true = case.get_observable_hist()
    y_true = np.squeeze(y_true, axis=-1)
    t_true = case.hist_t

    if t_min > 0.:
        id0 = np.argmin(abs(t_true - t_min))
        y_true, t_obs = [yy[id0:] for yy in [y_true, t_true]]

    return y_true, t_true, name.split('Truth_')[-1], case



def create_noisy_signal(y_clean, noise_level=0.1, noise_type='gauss, add'):
    if y_clean.ndim == 2:
        y_clean = np.expand_dims(y_clean, -1)

    Nt, q, L = y_clean.shape
    y_noisy = y_clean.copy()

    for ll in range(L):
        if 'gauss' in noise_type.lower():
            noise = rng.multivariate_normal(np.zeros(q), np.eye(q) * noise_level ** 2, Nt)
        else:
            i0 = Nt % 2 != 0  # Add extra step if odd
            noise = np.zeros([Nt, q])
            for ii in range(q):
                noise_white = np.fft.rfft(rng.standard_normal(Nt + i0) * noise_level)
                S = colour_noise(Nt + i0, noise_colour=noise_type)
                S = noise_white * S  # Normalize S
                noise[:, ii] = np.fft.irfft(S)[i0:]  # transform back into time domain
        if 'add' in noise_type.lower():
            y_noisy[:, :, ll] += noise * np.max(abs(y_clean[:, :, ll]), axis=0)
        else:
            y_noisy[:, :, ll] += noise * y_noisy[:, :, ll]

    y_noisy = y_noisy.squeeze()

    if y_noisy.ndim == 1:
        y_noisy = np.expand_dims(y_noisy, axis=-1)

    return y_noisy



def create_bias_model(ensemble, bias_params: dict,
                      training_dataset: dict or list, # type: ignore
                      wash_t=None,
                      wash_obs=None,
                      bias_filename=None,
                      folder=None):
    """
    This function creates the bias model for the input ensemble, truth and input_parameters.
    The bias model is added to the ensemble object as ensemble.bias.
    If the bias model requires washout (e.g. ESN) the washout observations are added to
    the truth dictionary.
    """
    if isinstance(training_dataset, dict):
        training_dataset = [training_dataset]
    elif not isinstance(training_dataset, list):
        raise ValueError('Training dataset must be a list of dicts or a dict')

    y_raw = [_data['y_raw'].copy() for _data in training_dataset]
    y_true = [_data['y_true'].copy() for _data in training_dataset]
    truth = training_dataset[-1]
    bias_params['noise_type'] = truth['noise_type']

    train_ens = ensemble.copy()
    edited_file = False

    bias = None

    if bias_filename is not None:
        # Create bias estimator if the class is not 'NoBias'
        if folder is not None:
            os.makedirs(folder, exist_ok=True)
            bias_filename = folder + bias_filename
        if os.path.isfile(bias_filename):
            load_bias, wash_obs, wash_t = load_from_pickle_file(bias_filename)
            if check_valid_file(load_bias, bias_params) is True:
                bias = load_bias
        else:
            print('Create bias model: ', bias_filename)

    if bias is None:
        edited_file = True
        train_ens.init_bias(**bias_params)
        bias = train_ens.bias.copy()

        # Create training data on a multi-parameter approach
        train_data = create_bias_training_dataset(y_raw=y_raw, y_pp=y_true, ensemble=train_ens,
                                                  **bias_params)

        # Run bias model training
        for key, val in bias_params.items():
            if not hasattr(bias, key):
                train_data[key] = val

        bias.train_bias_model(**train_data)

        # Create washout if needed
        if bias.t_init is None:
            bias.t_init = truth['t_obs'][0]

        if hasattr(bias, 'N_wash') and wash_t is None:
            wash_t, wash_obs = create_washout(bias, truth['t'], truth['y_raw'])

    # Create washout if needed
    if hasattr(bias, 'N_wash') and wash_t[0] > truth['t_obs'][0]:
        if bias.t_init is None or bias.t_init > truth['t_obs'][0]:
            bias.t_init = truth['t_obs'][0]

        wash_t, wash_obs = create_washout(bias, truth['t'], truth['y_raw'])
        edited_file = True

    if edited_file and bias_filename is not None:
        save_to_pickle_file(bias_filename, bias.copy(), wash_obs, wash_t)


    return bias, wash_obs, wash_t


def create_washout(bias_case, t_true, y_raw):
    i1 = np.argmin(abs(bias_case.t_init - t_true))
    i0 = i1 - bias_case.N_wash * bias_case.upsample

    if i0 < 0:
        bias_case.t_init -= (bias_case.N_wash + 1) * bias_case.upsample
        i1 = np.argmin(abs(bias_case.t_init - t_true))
        i0 = i1 - bias_case.N_wash * bias_case.upsample

    wash_obs = y_raw[i0:i1 + 1:bias_case.upsample]
    wash_t = t_true[i0:i1 + 1:bias_case.upsample]
    return wash_t, wash_obs


def create_bias_training_dataset(y_raw: list, y_pp: list, ensemble,
                                 t_train=None,
                                 t_val=None,
                                 t_test=None,
                                 L=10,
                                 N_wash=5,
                                 augment_data=True,
                                 perform_test=True,
                                 bayesian_update=False,
                                 biased_observations=True,
                                 correlation_based_training=True,
                                 add_noise=True,
                                 filename=None,
                                 len_augment_set=2,
                                 plot_train_dataset=True,
                                 **train_params):

    """
    Multi-parameter data generation for ESN training.
    """

    assert len(y_pp) == len(y_raw)

    train_ens = ensemble.copy()

    if t_train is None:
        t_train = train_ens.t_transient
    if t_val is None:
        t_val = train_ens.t_CR

    training_keys = ['t_train', 't_val', 'add_noise', 'augment_data',
                     'bayesian_update', 'biased_observations', 'L']
    requested_dict = dict((k, v) for k, v in locals().items() if k in training_keys)

    Nt_min = t_train + t_val
    if Nt_min > y_raw[0].shape[0] // train_ens.dt:
        raise ValueError('Nt_min < y_raw length')
    if perform_test:
        if t_test is None:
            t_test = t_val * 5
        Nt_min += t_test
    Nt_min = min(int(Nt_min / train_ens.dt) + N_wash, y_raw[0].shape[0])

    def _plot_dataset(plot_data):
        _L, _Nt, _Ndim = plot_data.shape

        if biased_observations:
            _nc = 2
        else:
            _nc = 1
        _Ndim = int(round(_Ndim // _nc))
        _nr = int(min(_Ndim, 10))

        fig, axs = plt.subplots(nrows=_nr, ncols=_nc, figsize=(8*_nc, 2*_nr), sharex=True, layout='constrained')
        if not isinstance(axs, np.ndarray):
            axs = [axs]
        if _nc > 1:
            axs = axs.T.flatten()
            [axs[_ii].set(title=_ttl) for _ii, _ttl in zip([0, _nr], ['Model bias', 'Innovations'])]
        _t_data = np.arange(0, _Nt) * ensemble.dt
        _Ls = np.random.randint(low=0, high=_L, size=len(axs))

        for kk, ax, Li in zip(range(len(axs)), axs, _Ls):
            if kk < _nr:
                ax.plot(_t_data, plot_data[Li, :, kk], lw=1., color='k')
            else:
                ax.plot(_t_data, plot_data[Li, :, kk - _nr + _Ndim], lw=1., color='k')

            times = [0, t_train, t_train + t_val, _t_data[-1]]
            [ax.axvspan(times[_ii], times[_ii+1], facecolor=_c, alpha=0.3, zorder=-100,
                        label=_lbl) for _ii, _c, _lbl in zip(range(3), ['orange', 'red', 'navy'],
                                                             [f'Train, Li{Li}', 'Validate', 'Test'])]
        axs[0].legend(ncols=3, loc='upper center', bbox_to_anchor=(0.5, 1.5))
        plt.show()

    # =========================== Load training data if available ============================== #
    if filename is not None:
        try:
            loaded_train_data = load_from_pickle_file(filename)
            try:
                if check_valid_file(loaded_train_data, dict(**requested_dict, **train_params)):
                    _U = loaded_train_data['data']
                    if _U.shape[1] < Nt_min:
                        print('Re-run multi-parameter training data: Increase the length of the training data')
                    elif augment_data and _U.shape[0] == L:
                        print('Re-run multi-parameter training data: need data augment ')
                    else:
                        print('Loaded multi-parameter training data')
                        if plot_train_dataset:
                            _plot_dataset(plot_data=loaded_train_data['data'])
                        return loaded_train_data
            except TypeError:
                print(f'File {filename} type = {type(loaded_train_data)} is not dict')
        except FileNotFoundError:
            print(f'Run multi-parameter training data: file {filename} not found')
    print('Rerun training data')

    # =================  Create ensemble for multi-parameter training data generation ============================ #

    # Create ensemble of training data
    train_params['m'] = L
    train_ens.init_ensemble(**train_params)

    # Forecast ensemble
    psi, t = train_ens.time_integrate(Nt=int(round(t_train / train_ens.dt)))
    train_ens.update_history(psi, t)
    y_L_model = train_ens.get_observable_hist()  # N_train x Nq x m

    # -------------  Remove and replace fixed points ------------- #
    tol = 1e-1
    N_CR = int(round(train_ens.t_CR / train_ens.dt))
    range_y = np.max(np.max(y_L_model[-N_CR:], axis=0) - np.min(y_L_model[-N_CR:], axis=0), axis=0)
    idx_FP = (range_y < tol)
    psi0 = psi[-1, :, ~idx_FP]  # Nq x (m - #FPs)
    if len(np.flatnonzero(idx_FP)) / len(idx_FP) >= 0.2:
        allowed_FPs = np.flatnonzero(idx_FP)[0:int(0.2 * len(idx_FP)) + 1]
        idx_FP[allowed_FPs] = 0
        psi0 = psi[-1, :, ~idx_FP]  # non-fixed point ICs (keeping one)
        print('There are {}/{} fixed points'.format(len(np.flatnonzero(idx_FP)), len(idx_FP)))
        new_psi0 = rng.multivariate_normal(np.mean(psi0, axis=0), np.cov(psi0.T), len(np.flatnonzero(idx_FP)))
        psi0 = np.concatenate([psi0, new_psi0], axis=0)

    # Reset ensemble with post-transient ICs
    train_ens.update_history(psi=psi0.T, reset=True)

    # -------------  Forecast fixed-point-free ensemble ------------- #
    N_corr = N_CR
    psi, tt = train_ens.time_integrate(Nt=Nt_min + N_corr)
    train_ens.update_history(psi, tt)
    train_ens.close()

    # ========================================  GENERATE TRAINING DATA ========================================= #

    # If the observations are biased, the bias estimator must predict  (1) the difference between the
    # raw data and the model, which are the observable quantities; and (2) the difference between the
    # post-processed data (i.e. the truth) and the model, which is the actual model bias.

    y_L_model = train_ens.get_observable_hist()
    N_datasets = len(y_raw)

    innovations_all, model_bias_all = [], []
    for _y_raw, _y_pp in zip(y_raw, y_pp):
        _y_raw, _y_pp = [yy[-Nt_min:].copy() for yy in [_y_raw, _y_pp]]
        if _y_raw.ndim < 3:
            _y_raw, _y_pp = [np.expand_dims(yy, axis=-1) for yy in [_y_raw, _y_pp]]

        if not correlation_based_training:   # (Nóvoa & Magri 2023 CMAME)
            print('Not correlation_based_training')
            train_data_model = y_L_model[-Nt_min:]

        else:  # -------- Correlate observations and estimates (Nóvoa et al. 2024 JFM) -------- #
            print('Yes correlation_based_training')
            lags = np.linspace(start=0, stop=N_corr, num=N_corr, dtype=int)
            _y_raw_corr = _y_raw[:N_corr, ..., 0]
            train_data_model = np.zeros([Nt_min, train_ens.Nq, L * len_augment_set])

            for ii in range(train_ens.m):
                yy = y_L_model[:, :, ii]

                _RS = [CR(_y_raw_corr, yy[lag:N_corr + lag] / np.max(yy[lag:N_corr + lag]))[1] for lag in lags]
                best_lag = lags[np.argmin(_RS)]  # fully correlated
                worst_lag = lags[np.argmax(_RS)]  # fully uncorrelated
                mid_lag = int(np.mean([best_lag, worst_lag]))  # mid-correlated
                # Store train data
                train_data_model[:, :, len_augment_set * ii] = yy[best_lag:best_lag + Nt_min]
                train_data_model[:, :, len_augment_set * ii + 1] = yy[mid_lag:mid_lag + Nt_min]
                if len_augment_set == 3:
                    train_data_model[:, :, len_augment_set * ii + 2] = yy[worst_lag:worst_lag + Nt_min]

        # ================ Create training biases as (observations - model estimates) ================= #
        innovations = (_y_raw - train_data_model).transpose((2, 0, 1))  # Force shape to be (L x Nt x N_dim). Note: N_dim = Nq

        assert innovations.shape[1] == Nt_min
        assert innovations.shape[2] == y_L_model.shape[1]

        if biased_observations:
            model_bias = _y_pp - train_data_model
            model_bias = model_bias.transpose((2, 0, 1))
            model_bias_all.append(model_bias)
        elif augment_data and not correlation_based_training:
            inn = innovations.copy()
            innovations = np.zeros([L * 3, Nt_min, train_ens.Nq])

            innovations[:L] = inn
            innovations[L:-L] = inn * 1e-1
            innovations[-L:] = inn * -1e-2

        innovations_all.append(innovations)

    # ------------- Combine the innovations (and model biases) --------------- #
    # If training with more than one experimental dataset, the shape is (N_datasets*L x Nt x N_dim).
    # Example: Generalization section in Nóvoa et al. (2024 JFM).
    innovations_all = np.concatenate(innovations_all, axis=0)
    if not biased_observations:
        train_data = dict(data=innovations_all,
                          observed_idx=np.arange(ensemble.Nq))
    else:
        model_bias_all = np.concatenate(model_bias_all, axis=0)
        train_data = dict(data=np.concatenate([model_bias_all,
                                               innovations_all], axis=2),
                          observed_idx=ensemble.Nq + np.arange(ensemble.Nq))

    # =============================== Save train_data dict ================================ #
    # Save key keywords
    for k in training_keys:
        train_data[k] = locals()[k]

    if filename is not None:
        save_to_pickle_file(filename, train_data)

    if plot_train_dataset:
        _plot_dataset(plot_data=train_data['data'])

    return train_data


