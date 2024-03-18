from essentials.Util import *
from essentials.bias_models import *

import scipy.io as sio

rng = np.random.default_rng(0)


def create_ensemble(forecast_params=None, model=None, **filter_params):
    if forecast_params is None:
        forecast_params = dict()

    # ==============================  INITIALISE MODEL  ================================= #
    if 'std_a' in filter_params.keys() and type(filter_params['std_a']) is dict:
        for key, vals in filter_params['std_a'].items():
            forecast_params[key] = .5 * (vals[1] + vals[0])

    if model is not None:
        forecast_params['model'] = model

    ensemble = forecast_params['model'](**forecast_params)

    # Forecast model case to steady state initial condition before initialising ensemble
    state, t_ = ensemble.time_integrate(int(ensemble.t_CR / ensemble.dt))
    ensemble.update_history(state[-1], reset=True)

    # =========================  INITIALISE ENSEMBLE & BIAS  =========================== #
    ensemble.init_ensemble(**filter_params)
    ensemble.init_bias()

    ensemble.close()
    return ensemble


def create_truth(model, t_start=1., t_stop=1.5, Nt_obs=20, std_obs=0.05, t_max=None, t_min=0.,
                 noise_type='gauss, add', post_processed=False, manual_bias=None, **kwargs):
    # =========================== LOAD DATA OR CREATE TRUTH FROM LOM ================================ #
    if t_max is None:
        t_max = t_stop + t_start

    if type(model) is str:
        y_raw, y_true, t_true, name_truth = create_observations_from_file(model, t_max=t_max, t_min=t_min,)
        case_name = model.split('/')[-1]
        name_bias = 'Exp_' + case_name
        b_true = np.zeros(1)
    else:
        y_true, t_true, name_truth = create_observations(model, t_max=t_max, t_min=t_min, **kwargs)

        #  ADD BIAS TO THE TRUTH #
        if manual_bias is None:
            b_true = np.zeros(1)
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
        y_raw = y_true + b_true

    # =========================== ADD NOISE TO THE TRUTH ================================ #
    if type(model) is dict or post_processed:
        y_raw = create_noisy_signal(y_raw, noise_level=std_obs, noise_type=noise_type)
    else:
        noise_type = name_bias
        std_obs = None

    # =========================== COMPUTE OBSERVATIONS AT DESIRED TIME =========================== #
    dt_t = t_true[1] - t_true[0]
    obs_idx = np.arange(t_start // dt_t, t_stop // dt_t + 1, Nt_obs, dtype=int)

    # ================================ SAVE DATA TO DICT ==================================== #
    if '/' in name_truth:
        name_truth = '_'.join(name_truth.split('/'))

    truth = dict(y_raw=y_raw, y_true=y_true, t=t_true, b=b_true, dt=dt_t,
                 t_obs=t_true[obs_idx], y_obs=y_raw[obs_idx], dt_obs=Nt_obs * dt_t,
                 name=name_truth, name_bias=name_bias, noise_type=noise_type,
                 model=model, std_obs=std_obs, true_params=kwargs)

    return truth


def create_observations_from_file(name, t_max, t_min=0.):
    # Wave case: load .mat file ====================================
    try:
        if 'rijke' in name:
            mat = sio.loadmat(name + '.mat')
            y_raw, y_true, t_true = [mat[key].transpose() for key in ['p_mic', 'p_mic', 't_mic']]
        elif 'annular' in name:
            mat = sio.loadmat(name + '.mat')
            y_raw, y_true, t_true = [mat[key] for key in ['y_raw', 'y_filtered', 't']]
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        raise 'File ' + name + ' not defined'

    if len(np.shape(t_true)) > 1:
        t_true = np.squeeze(t_true)

    if y_raw.shape[0] != len(t_true):
        y_raw, y_true = [yy.transpose() for yy in [y_raw, y_true]]

    id0, id1 = [np.argmin(abs(t_true - tx)) for tx in [t_min, t_max]]
    y_raw, y_true, t_true = [yy[id0:id1] for yy in [y_raw, y_true, t_true]]

    return y_raw, y_true, t_true, name.split('data/')[-1]


def create_observations(model, t_max, t_min, **true_parameters):
    try:
        TA_params = true_parameters.copy()
        model = model
    except AttributeError:
        raise 'true_parameters must be dict'


    # ============================================================
    # Add key parameters to filename
    suffix = ''
    key_save = model.params + ['law']

    for key, val in TA_params.items():
        if key in key_save:
            if type(val) is str:
                suffix += val + '_'
            else:
                suffix += key + '{:.2e}'.format(val) + '_'

    name = os.path.join(os.getcwd() + '/data/')
    os.makedirs(name, exist_ok=True)
    name += 'Truth_{}_{}tmax-{:.2}'.format(model.name, suffix, t_max)

    try:
        case = load_from_pickle_file(name)
        print('Load true data: ' + name)
    except ModuleNotFoundError or FileNotFoundError:
        case = model(**TA_params)
        psi, t = case.time_integrate(int(t_max / case.dt))
        case.update_history(psi, t)
        case.close()
        save_to_pickle_file(name, case)
        print('Save true data: ' + name)

    # Retrieve observables
    y_true = case.get_observable_hist()
    if len(np.shape(y_true)) > 2:
        y_true = np.squeeze(y_true, axis=-1)
    t_true = case.hist_t

    if t_min > 0.:
        id0 = np.argmin(abs(t_true - t_min))
        y_true, t_obs = [yy[id0:] for yy in [y_true, t_true]]

    return y_true, t_true, name.split('Truth_')[-1]


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
                # Generate the noise signal
                S = colour_noise(Nt + i0, noise_colour=noise_type)
                S = noise_white * S / np.sqrt(np.mean(S ** 2))  # Normalize S
                noise[:, ii] = np.fft.irfft(S)[i0:]  # transform back into time domain
        if 'add' in noise_type.lower():
            y_noisy[:, :, ll] += noise * np.max(abs(y_clean[:, :, ll]))
        else:
            y_noisy[:, :, ll] += noise * y_noisy[:, :, ll]
    return y_noisy


def create_bias_model(ensemble, truth: dict, bias_params: dict, bias_name: str,
                      bias_model_folder: str):
    """
    This function creates the bias model for the input ensemble, truth and parameters.
    The bias model is added to the ensemble object as ensemble.bias.
    If the bias model requires washout (e.g. ESN) the washout observations are added to the truth dictionary.
    """

    os.makedirs(bias_model_folder, exist_ok=True)

    y_raw = truth['y_raw'].copy()
    y_pp = truth['y_true'].copy()
    bias_params['noise_type'] = truth['noise_type']
    if ensemble.bias_bayesian_update:
        bias_params['m'] = ensemble.m

    run_bias = True
    if bias_params['bias_type'].name == 'None':
        ensemble.init_bias(**bias_params)
        run_bias = False
    else:
        # Create bias estimator if the class is not 'NoBias'
        if os.path.isfile(bias_model_folder + bias_name):
            bias = load_from_pickle_file(bias_model_folder + bias_name)
            if check_valid_file(bias, bias_params) is True:
                ensemble.bias = bias
                run_bias = False

    if run_bias:
        # ======================== SET NECESSARY TRAINING PARAMETERS ======================
        train_params = bias_params.copy()

        # Create training data on a multi-parameter approach
        train_data = create_bias_training_dataset(y_raw, y_pp, ensemble, filename=None, **train_params)

        # train_data_name = bias_model_folder + 'Train_data_L{}_augment{}'.format(train_data['L'],
        #                                                                         train_data['augment_data'])
        # # Save training data
        # save_to_pickle_file(train_data_name, train_data)

        # Run bias model training
        train_params['y0'] = np.zeros((train_data['data'].shape[-1], 1))

        ensemble.init_bias(**train_params)
        ensemble.bias.filename = bias_name
        ensemble.bias.train_bias_model(train_data=train_data,
                                       folder=bias_model_folder,
                                       plot_training=True)

        # Save bias object
        save_to_pickle_file(bias_model_folder + bias_name, ensemble.bias)

    ensemble.bias.print_bias_parameters()

    if ensemble.bias_bayesian_update:
        ensemble.bias.bayesian_update = True
        if ensemble.bias.N_ens != ensemble.m:
            raise AssertionError(ensemble.bias.N_ens, ensemble.m,
                                 ensemble.bias.wash_obs.shape)

    if not hasattr(ensemble.bias, 't_init'):
        ensemble.bias.t_init = truth['t_obs'][0] - 2 * truth['dt_obs']

    # Ensure the truth has washout if needed
    if hasattr(ensemble.bias, 'N_wash') and 'wash_t' not in truth.keys():
        wash_t, wash_obs = create_washout(ensemble.bias.t_init, ensemble.bias.upsample,
                                          ensemble.bias.N_wash, **truth)
        truth['wash_t'], truth['wash_obs'] = wash_t, wash_obs


def create_washout(bias_case, t=False, y_raw=False, **truth):
    i1 = np.argmin(abs(bias_case.t_init - t))
    i0 = i1 - bias_case.N_wash * bias_case.upsample

    if i0 < 0:
        raise ValueError('increase bias.t_init > t_wash + dt_obs')

    wash_obs = y_raw[i0:i1 + 1:bias_case.upsample]
    wash_t = t[i0:i1 + 1:bias_case.upsample]
    return wash_t, wash_obs


def create_bias_training_dataset(y_raw, y_pp, ensemble,
                                 t_train=None,
                                 t_val=None,
                                 L=10,
                                 N_wash=5,
                                 augment_data=True,
                                 perform_test=True,
                                 biased_observations=False,
                                 bayesian_update=False,
                                 add_noise=True,
                                 filename='train_data', **train_params):
    """
    Multi-parameter data generation for ESN training.
    """

    ensemble = ensemble.copy()

    if t_train is None:
        t_train = ensemble.t_transient
    if t_val is None:
        t_val = ensemble.t_CR

    min_train_data = t_train + t_val
    if perform_test:
        min_train_data += t_val * 5
    min_train_data = int(min_train_data / ensemble.dt) + N_wash

    # =========================== Load training data if available ============================== #
    try:
        train_data = load_from_pickle_file(filename)
        rerun = True
        if type(train_data) is dict:
            U = train_data['data']
        else:
            U = train_data.copy()
        if U.shape[1] < min_train_data:
            print('Rerun multi-parameter training data: Increase the length of the training data')
            rerun = True
        if augment_data and U.shape[0] == L:
            print('Rerun multi-parameter training data: need data augment ')
            rerun = True
    except FileNotFoundError:
        rerun = True
        print('Run multi-parameter training data: file not found')

    # Load training data if all the conditions are ok
    if not rerun:
        train_data = load_from_pickle_file(filename)
        print('Loaded multi-parameter training data')
    else:
        # =======================  Create ensemble of initial guesses ============================ #
        # Forecast one realization to t_transient
        train_ens = ensemble.reshape_ensemble(m=1, reset=True)
        psi = train_ens.time_integrate(int(train_ens.t_transient / train_ens.dt))[0]
        train_ens.update_history(psi[-1], reset=True)

        # Create ensemble of training data
        train_params['m'] = L
        train_ens.init_ensemble(**train_params)

        # Forecast ensemble
        psi, t = train_ens.time_integrate(Nt=int(round(t_train / train_ens.dt)))
        train_ens.update_history(psi, t)
        y_L_model = train_ens.get_observable_hist()  # N_train x Nq x m

        # =========================  Remove and replace fixed points =================================== #
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

        # =========================  Forecast fixed-point-free ensemble ============================== #

        Nt = min_train_data
        N_corr = int(round(train_ens.t_CR / train_ens.dt))

        psi, tt = train_ens.time_integrate(Nt=Nt + N_corr)
        train_ens.update_history(psi, tt)
        train_ens.close()

        # ===============  If data augmentation, correlate observations and estimates ================= #
        y_L_model = train_ens.get_observable_hist()

        # Equivalent observation signals (raw and post-processed)
        y_raw = y_raw[-Nt:].copy()
        y_pp = y_pp[-Nt:].copy()
        if len(y_raw.shape) < 3:
            y_raw = np.expand_dims(y_raw, -1)
            y_pp = np.expand_dims(y_pp, -1)

        if not augment_data:
            train_data_model = y_L_model[-Nt:]
        else:
            y_pp_corr = y_pp[:N_corr, :, 0]
            lags = np.linspace(start=0, stop=N_corr, num=N_corr, dtype=int)
            train_data_model = np.zeros([y_pp.shape[0], y_pp.shape[1], train_ens.m * 3])

            for ii in range(train_ens.m):
                yy = y_L_model[:, :, ii]
                RS = []
                for lag in lags:
                    RS.append(CR(y_pp_corr, yy[lag:N_corr + lag] / np.max(yy[lag:N_corr + lag]))[1])
                best_lag = lags[np.argmin(RS)]  # fully correlated
                worst_lag = lags[np.argmax(RS)]  # fully uncorrelated
                mid_lag = int(np.mean([best_lag, worst_lag]))  # mid-correlated
                # Store train data
                train_data_model[:, :, 3 * ii] = yy[worst_lag:worst_lag + Nt]
                train_data_model[:, :, 3 * ii + 1] = yy[mid_lag:mid_lag + Nt]
                train_data_model[:, :, 3 * ii + 2] = yy[best_lag:best_lag + Nt]

        # ================ Create training biases as (observations - model estimates) ================= #
        innovations = y_raw - train_data_model
        innovations = innovations.transpose((2, 0, 1))  # Force shape to be (L x Nt x N_dim). Note: N_dim = Nq

        assert innovations.ndim == 3
        assert innovations.shape[1] == Nt
        assert innovations.shape[2] == y_L_model.shape[1]

        if not biased_observations:
            train_data = dict(data=innovations,
                              observed_idx=np.arange(ensemble.Nq)
                              )
        else:
            # If the observations are biased, the bias estimator must predict  (1) the difference between the
            # raw data and the model, which are the observable quantities; and (2) the difference between the
            # post-processed data (i.e. the truth) and the model, which is the actual model bias.
            actual_bias = y_pp - train_data_model
            actual_bias = actual_bias.transpose((2, 0, 1))

            train_data = dict(data=np.concatenate([actual_bias, innovations], axis=2),
                              observed_idx=ensemble.Nq + np.arange(ensemble.Nq)  # only observe innovations
                              )

        # =============================== Save train_data dict ================================ #
        # Save key keywords
        train_data['t_train'] = t_train
        train_data['t_val'] = t_val

        train_data['add_noise'] = add_noise
        train_data['augment_data'] = augment_data
        train_data['bayesian_update'] = bayesian_update
        train_data['biased_observations'] = biased_observations
        train_data['L'] = L

        if filename is not None:
            save_to_pickle_file(filename, train_data)

    return train_data
