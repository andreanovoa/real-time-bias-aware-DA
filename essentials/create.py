from essentials.Util import *
from essentials.bias_models import *

import scipy.io as sio

rng = np.random.default_rng(6)


def create_truth(true_params: dict, filter_params: dict, post_processed=False):
    name_bias, noise_type = 'None', 'None'
    b_true = np.zeros(1)

    if type(true_params) is not dict or type(filter_params) is not dict:
        raise AttributeError('true_p and filter_p must be dict')

    if type(true_params['model']) is str:
        # Load experimental data
        y_raw, y_true, t_true, name_truth = create_observations_from_file(true_params['model'])
        case_name = true_params['model'].split('/')[-1]
        name_bias = 'Exp_' + case_name
        noise_type = 'Exp_' + case_name
    else:
        # =========================== CREATE TRUTH FROM LOM ================================ #
        y_true, t_true, name_truth = create_observations(true_params)
        # =========================== ADD BIAS TO THE TRUTH ================================ #
        if 'manual_bias' in true_params.keys():
            if type(true_params['manual_bias']) is str:
                name_bias = true_params['manual_bias']
                if name_bias == 'time':
                    b_true = .4 * y_true * np.sin((np.expand_dims(t_true, -1) * np.pi * 2) ** 2)
                elif name_bias == 'periodic':
                    b_true = 0.2 * np.max(y_true, 0) * np.cos(2 * y_true / np.max(y_true, 0))
                elif name_bias == 'linear':
                    b_true = .1 * np.max(y_true, 0) + .3 * y_true
                elif name_bias == 'cosine':
                    b_true = np.cos(y_true)
                else:
                    raise ValueError(
                        "Bias type {} not recognized choose: 'linear', 'periodic', 'time'".format(name_bias))
            else:
                # The manual bias is a function of state and/or time
                b_true, name_bias = true_params['manual_bias'](y_true, t_true)
        y_true += b_true

    # =========================== ADD NOISE TO THE TRUTH ================================ #
    if type(true_params['model']) is not str or post_processed:
        Nt, q = y_true.shape[:2]
        if 'std_obs' not in true_params.keys():
            true_params['std_obs'] = 0.01

        if 'noise_type' not in true_params.keys():
            noise_type = 'gauss, add'
        else:
            noise_type = true_params['noise_type']
        noise_type += ', ' + str(true_params['std_obs'])

        # Create noise to add to the truth
        if 'gauss' in noise_type.lower():
            Cdd = np.eye(q) * true_params['std_obs'] ** 2
            noise = rng.multivariate_normal(np.zeros(q), Cdd, Nt)
        else:
            i0 = Nt % 2 != 0
            noise = np.zeros([Nt+i0, q])
            for ii in range(q):
                noise_white = np.fft.rfft(rng.standard_normal(Nt+i0) * true_params['std_obs'])
                # Generate the noise signal
                S = colour_noise(Nt+i0, noise_colour=noise_type)
                S = noise_white * S / np.sqrt(np.mean(S ** 2))  # Normalize S
                noise[:, ii] = np.fft.irfft(S)[i0:]  # transform back into time domain

        if 'multi' in noise_type.lower():
            y_raw = y_true * (1 + noise)
        else:
            mean_y = np.mean(abs(y_true))
            y_raw = y_true + noise * mean_y

    # =========================== COMPUTE OBSERVATIONS AT DESIRED TIME =========================== #
    dt_t = t_true[1] - t_true[0]
    obs_idx = np.arange(filter_params['t_start'] // dt_t,
                        filter_params['t_stop'] // dt_t + 1, filter_params['dt_obs'], dtype=int)

    # ================================ SAVE DATA TO DICT ==================================== #
    if '/' in name_truth:
        name_truth = '_'.join(name_truth.split('/'))

    truth = dict(y_raw=y_raw, y_true=y_true, t=t_true, b=b_true, dt=dt_t,
                 t_obs=t_true[obs_idx], y_obs=y_raw[obs_idx], dt_obs=filter_params['dt_obs'] * dt_t,
                 true_params=true_params, name=name_truth, name_bias=name_bias, noise_type=noise_type,
                 model=true_params['model'], std_obs=true_params['std_obs'])
    return truth


def create_observations_from_file(name):
    # Wave case: load .mat file ====================================
    if 'rijke' in name:
        try:
            mat = sio.loadmat(name + '.mat')
        except FileNotFoundError:
            raise 'File ' + name + ' not defined'
        y_raw, y_true, t_obs = [mat[key].transpose() for key in ['p_mic', 'p_mic', 't_mic']]
    elif 'annular' in name:
        try:
            mat = sio.loadmat(name + '.mat')
            y_raw, y_true, t_obs = [mat[key] for key in ['y_raw', 'y_filtered', 't']]
        except FileNotFoundError:
            raise 'File ' + name + ' not defined'
    else:
        raise 'File ' + name + ' not defined'

    if len(np.shape(t_obs)) > 1:
        t_obs = np.squeeze(t_obs)

    if y_raw.shape[0] != len(t_obs):
        y_true = y_true.transpose()
    if y_raw.shape[0] != len(t_obs):
        y_raw = y_raw.transpose()

    return y_raw, y_true, t_obs, name.split('data/')[-1]


def create_observations(true_parameters=None):
    try:
        TA_params = true_parameters.copy()
        classType = TA_params['model']
        if 't_max' in TA_params.keys():
            t_max = TA_params['t_max']
        else:
            t_max = 8.
    except AttributeError:
        raise 'true_parameters must be dict'

    # ============================================================
    # Add key parameters to filename
    suffix = ''
    key_save = classType.params + ['law']

    for key, val in TA_params.items():
        if key in key_save:
            if type(val) == str:
                suffix += val + '_'
            else:
                suffix += key + '{:.2e}'.format(val) + '_'

    name = os.path.join(os.getcwd() + '/data/')
    os.makedirs(name, exist_ok=True)
    name += 'Truth_{}_{}tmax-{:.2}'.format(classType.name, suffix, t_max)

    try:
        case = load_from_pickle_file(name)
        print('Load true data: ' + name)
    except ModuleNotFoundError or FileNotFoundError:
        case = classType(**TA_params)
        psi, t = case.timeIntegrate(Nt=int(t_max / case.dt))
        case.updateHistory(psi, t)
        case.close()
        save_to_pickle_file(name, case)
        print('Save true data: ' + name)

    # Retrieve observables
    p_obs = case.getObservableHist()
    if len(np.shape(p_obs)) > 2:
        p_obs = np.squeeze(p_obs, axis=-1)

    return p_obs, case.hist_t, name.split('Truth_')[-1]


def create_ensemble(forecast_params, filter_params):

    # ==============================  INITIALISE MODEL  ================================= #
    if 'std_a' in filter_params.keys() and type(filter_params['std_a']) is dict:
        for key, vals in filter_params['std_a'].items():
            forecast_params[key] = .5 * (vals[1] + vals[0])

    ensemble = forecast_params['model'](**forecast_params)

    # Forecast model case to steady state initial condition before initialising ensemble
    state, t_ = ensemble.timeIntegrate(int(ensemble.t_CR / ensemble.dt))
    ensemble.updateHistory(state, t_)

    # =========================  INITIALISE ENSEMBLE & BIAS  =========================== #
    ensemble.initEnsemble(**filter_params)
    ensemble.initBias()

    ensemble.close()
    return ensemble


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
        bias_params['N_ens'] = ensemble.m

    run_bias = True
    if bias_params['biasType'].name == 'None':
        ensemble.initBias(**bias_params)
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
        train_params = {'L': 10,
                        't_train': ensemble.t_transient,
                        't_val': ensemble.t_CR,
                        't_test': ensemble.t_CR,
                        'alpha_distr': 'uniform',
                        'ensure_mean': True,
                        'plot_training': True,
                        'perform_test': True}  # Default parameters
        for key, val in bias_params.items():  # Combine defaults with prescribed parameters
            train_params[key] = val
        # print('Train and save bias case')
        ensemble.bias.filename = bias_name
        ensemble.initBias(**train_params)

        # Create training data on a multi-parameter approach
        train_data_filename = bias_model_folder
        train_data_filename += 'Train_data_L{}_augment{}'.format(ensemble.bias.L, ensemble.bias.augment_data)
        train_data = create_bias_training_dataset(y_raw, y_pp, ensemble, train_params,
                                                  train_data_filename)

        # Run bias model training
        ensemble.bias.train_bias_model(train_data=train_data,
                                       folder=bias_model_folder,
                                       plot_training=True)
        # Save
        save_to_pickle_file(bias_model_folder + bias_name, ensemble.bias)

    if ensemble.bias_bayesian_update:
        assert ensemble.bias.N_ens == ensemble.m

    if not hasattr(ensemble.bias, 't_init'):
        ensemble.bias.t_init = truth['t_obs'][0] - truth['dt_obs']

    # Ensure the truth has washout if needed
    if hasattr(ensemble.bias, 'N_wash') and 'wash_t' not in truth.keys():

        i1 = np.argmin(abs(ensemble.bias.t_init - truth['t']))
        i0 = i1 - ensemble.bias.N_wash * ensemble.bias.upsample

        if i0 < 0:
            raise ValueError('increase bias.t_init > t_wash + dt_obs')

        # Add washout to the truth
        truth['wash_obs'] = truth['y_raw'][i0:i1 + 1:ensemble.bias.upsample]
        truth['wash_t'] = truth['t'][i0:i1 + 1:ensemble.bias.upsample]


def create_bias_training_dataset(y_raw, y_pp, ensemble, train_params, filename):

    """
    Multi-parameter data generation for ESN training

    :param y_raw:
    :param y_pp:
    :param ensemble:
    :param train_params:
    :param filename:
    :param plot_train_data:
    :return:
    """
    # =========================== Load training data if available ============================== #
    try:
        train_data = load_from_pickle_file(filename)
        rerun = True
        if type(train_data) is dict:
            U = train_data['inputs']
        else:
            U = train_data.copy()
        if U.shape[1] < ensemble.bias.len_train_data:
            print('Rerun multi-parameter training data: Increase the length of the training data')
            rerun = True
        if train_params['augment_data'] and U.shape[0] == train_params['L']:
            print('Rerun multi-parameter training data: need data augment ')
            rerun = True
        # Load training data if all the conditions are ok
        if not rerun:
            print('Loaded multi-parameter training data')
            return train_data
    except FileNotFoundError:
        print('Run multi-parameter training data: file not found')

    # =======================  Create ensemble of initial guesses ============================ #

    # Forecast one realization to t_transient
    train_ens = ensemble.reshape_ensemble(m=1, reset=True)
    out = train_ens.timeIntegrate(int(train_ens.t_transient / train_ens.dt))
    train_ens.updateHistory(*out)

    # Create ensemble of training data
    train_params['m'] = train_params['L']
    train_ens.initEnsemble(**train_params)

    # Forecast ensemble
    Nt_train = int(round(train_params['t_train'] / train_ens.dt))
    psi, t = train_ens.timeIntegrate(Nt=Nt_train)
    train_ens.updateHistory(psi, t)
    y_L_model = train_ens.getObservableHist(Nt=Nt_train)  # Nt x Nq x m

    # =========================  Remove and replace fixed points =================================== #
    tol = 1e-1
    N_CR = int(round(train_ens.t_CR / train_ens.dt))
    range_y = np.max(np.max(y_L_model[-N_CR:], axis=0) - np.min(y_L_model[-N_CR:], axis=0), axis=0)
    idx_FP = (range_y < tol)
    psi0 = psi[-1, :, ~idx_FP]
    if len(np.flatnonzero(idx_FP)) / len(idx_FP) >= 0.2:
        allowed_FPs = np.flatnonzero(idx_FP)[0:int(0.2 * len(idx_FP))+1]
        idx_FP[allowed_FPs] = 0
        psi0 = psi[-1, :, ~idx_FP]  # non-fixed point ICs (keeping one)
        print('There are {}/{} fixed points'.format(len(np.flatnonzero(idx_FP)), len(idx_FP)))
        mean_psi = np.mean(psi0, axis=0)
        cov_psi = np.cov(psi0.T)
        new_psi0 = rng.multivariate_normal(mean_psi, cov_psi, len(np.flatnonzero(idx_FP)))
        psi0 = np.concatenate([psi0, new_psi0], axis=0)

    # Reset ensemble with post-transient ICs
    train_ens.updateHistory(psi=psi0.T, reset=True)

    # =========================  Forcast fixed-point free ensemble ============================== #
    Nt = train_ens.bias.len_train_data * train_ens.bias.upsample
    N_corr = int(round(train_ens.t_CR / train_ens.dt))
    psi, tt = train_ens.timeIntegrate(Nt=Nt + N_corr)
    train_ens.updateHistory(psi, tt)
    train_ens.close()

    # ===============  If data augmentation, correlate observations and estimates ================= #
    y_L_model = train_ens.getObservableHist()
    # Add the mean of batches of the ensemble size as options (might help short washout in DA)   < ======================= Test
    if ensemble.m < train_ens.m:
        batch_size = ensemble.m
    else:
        batch_size = ensemble.m - 1
    # Select randomly m realizations of y_L_model
    y_L_extra = []
    for _ in range(3):
        idx_to_avg = rng.integers(train_ens.m, size=batch_size)
        y_L_extra.append(np.mean(y_L_model[:, :, idx_to_avg], axis=-1))

    y_L_extra = np.array(y_L_extra).transpose(1, 2, 0)
    y_L_model = np.concatenate([y_L_model, y_L_extra], axis=-1)

    # Equivalent observation signals (raw and post-processed)
    y_raw = y_raw[-Nt:].copy()
    y_pp = y_pp[-Nt:].copy()
    if len(y_raw.shape) < 3:
        y_raw = np.expand_dims(y_raw, -1)
        y_pp = np.expand_dims(y_pp, -1)

    if not train_params['augment_data']:
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
            best_lag = lags[np.argmin(RS)]                  # fully correlated
            worst_lag = lags[np.argmax(RS)]                 # fully uncorrelated
            mid_lag = int(np.mean([best_lag, worst_lag]))   # mid-correlated
            # Store train data
            train_data_model[:, :, 3*ii] = yy[worst_lag:worst_lag + Nt]
            train_data_model[:, :, 3*ii+1] = yy[mid_lag:mid_lag + Nt]
            train_data_model[:, :, 3*ii+2] = yy[best_lag:best_lag + Nt]

    # ================ Create training biases as (observations - model estimates) ================= #
    train_data_in = y_raw - train_data_model
    train_data_out = y_pp - train_data_model

    # =======================  Force train_data shape to be (L x Nt x Ndim) ======================= #
    train_data_out = train_data_out.transpose((2, 0, 1))
    train_data_in = train_data_in.transpose((2, 0, 1))

    assert len(train_data_in.shape) == 3
    assert train_data_in.shape[1] == Nt
    assert train_data_in.shape[2] == y_L_model.shape[1]

    # =============================== Store in dictionary and save ================================ #
    train_data = dict(inputs=train_data_in,
                      labels=np.concatenate([train_data_out, train_data_in], axis=2),
                      observed_idx=y_L_model.shape[1] + np.arange(y_L_model.shape[1])
                      )
    save_to_pickle_file(filename, train_data)

    return train_data
