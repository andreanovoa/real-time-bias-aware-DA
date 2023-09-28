import os
import scipy.io as sio
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from Util import load_from_pickle_file, save_to_pickle_file, CR, colour_noise, check_valid_file

rng = np.random.default_rng(6)


def create_observations(classParams=None):
    if type(classParams) is dict:
        TA_params = classParams.copy()
        classType = TA_params['model']
        if 't_max' in TA_params.keys():
            t_max = TA_params['t_max']
        else:
            t_max = 8.
    else:
        raise ValueError('classParams must be dict')

    # Wave case: load .mat file ====================================
    if type(classType) is str:
        try:
            mat = sio.loadmat(classType + '.mat')
        except FileNotFoundError:
            raise ValueError('File ' + classType + ' not defined')
        p_obs = mat['p_mic'].transpose()
        t_obs = mat['t_mic'].transpose()
        if len(np.shape(t_obs)) > 1:
            t_obs = np.squeeze(t_obs, axis=-1)

        if t_obs[-1] > t_max:
            idx = np.argmin(abs(t_obs - t_max))
            t_obs, p_obs = [yy[:idx + 1] for yy in [t_obs, p_obs]]
            print('Data too long. Redefine t_max = ', t_max)

        return p_obs, t_obs, 'Wave'

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

    # Load or create and save file
    case = classType(**TA_params)

    psi, t = case.timeIntegrate(Nt=int(t_max / case.dt))
    case.updateHistory(psi, t)
    case.close()

    save_to_pickle_file(name, case)

    # Retrieve observables
    p_obs = case.getObservableHist()
    if len(np.shape(p_obs)) > 2:
        p_obs = np.squeeze(p_obs, axis=-1)

    return p_obs, case.hist_t, name.split('Truth_')[-1]


def create_truth(true_p, filter_p, noise_type):
    y_true, t_true, name_truth = create_observations(true_p)

    if 'manual_bias' in true_p.keys():
        if type(true_p['manual_bias']) is str:
            if true_p['manual_bias'] == 'time':
                b_true = .4 * y_true * np.sin((np.expand_dims(t_true, -1) * np.pi * 2) ** 2)
            elif true_p['manual_bias'] == 'periodic':
                b_true = 0.2 * np.max(y_true, 0) * np.cos(2 * y_true / np.max(y_true, 0))
            elif true_p['manual_bias'] == 'linear':
                b_true = .1 * np.max(y_true, 0) + .3 * y_true
            elif true_p['manual_bias'] == 'cosine':
                b_true = np.cos(y_true)
            else:
                raise ValueError("Bias type not recognized choose: 'linear', 'periodic', 'time'")
        else:
            b_true = true_p['manual_bias'](y_true, t_true)

    else:
        b_true = np.zeros(1)

    y_true += b_true
    dt_t = t_true[1] - t_true[0]
    obs_idx = np.arange(round(filter_p['t_start'] / dt_t),
                        round(filter_p['t_stop'] / dt_t) + 1, filter_p['dt_obs'])

    Nt, q = y_true.shape[:2]
    if 'std_obs' not in true_p.keys():
        true_p['std_obs'] = 0.01

    # Create noise to add to the truth
    if 'gauss' in noise_type.lower():
        Cdd = np.eye(q) * true_p['std_obs'] ** 2
        noise = rng.multivariate_normal(np.zeros(q), Cdd, Nt)
    else:
        noise = np.zeros([Nt, q])
        for ii in range(q):
            noise_white = np.fft.rfft(rng.standard_normal(Nt + 1) * true_p['std_obs'])
            # Generate the noise signal
            S = colour_noise(Nt + 1, noise_colour=noise_type)
            S = noise_white * S / np.sqrt(np.mean(S ** 2))  # Normalize S
            noise[:, ii] = np.fft.irfft(S)[1:]  # transform back into time domain

    if 'multi' in noise_type.lower():
        y_noise = y_true * (1 + noise)
    else:
        mean_y = np.mean(abs(y_true))
        y_noise = y_true + noise * mean_y

    # Select obs_idx only
    y_obs, t_obs = y_noise[obs_idx], t_true[obs_idx]

    # Compute signal-to-noise ratio
    P_signal = np.mean(y_true ** 2, axis=0)
    P_noise = np.mean((y_noise - y_true) ** 2, axis=0)

    # Save as a dict
    truth = dict(y=y_true, t=t_true, b=b_true, dt=dt_t,
                 t_obs=t_obs, y_obs=y_obs, dt_obs=t_obs[1] - t_obs[0],
                 true_params=true_p, name=name_truth,
                 model=true_p['model'], std_obs=true_p['std_obs'],
                 SNR=P_signal / P_noise, noise=noise, noise_type=noise_type)
    return truth


def create_ensemble(true_p, forecast_p, filter_p, bias_p=None):
    if bias_p is None:
        bias_p = dict()
    # ==============================  INITIALISE MODEL  ================================= #
    if 'std_a' in filter_p.keys() and type(filter_p['std_a']) is dict:
        for key, vals in filter_p['std_a'].items():
            forecast_p[key] = .5 * (vals[1] + vals[0])

    ensemble = forecast_p['model'](**forecast_p)

    # Forecast model case to steady state initial condition before initialising ensemble
    state, t_ = ensemble.timeIntegrate(int(ensemble.t_CR / ensemble.dt))
    ensemble.updateHistory(state, t_)

    # =========================  INITIALISE ENSEMBLE & BIAS  =========================== #
    ensemble.initEnsemble(**filter_p)
    ensemble.initBias(**bias_p)

    # =============================  CREATE OBSERVATIONS ============================== #
    if 'noise_type' not in true_p.keys():
        noise_type = 'gauss, add'
    else:
        noise_type = true_p['noise_type']

    truth = create_truth(true_p, filter_p, noise_type)

    if not hasattr(ensemble, 't_max'):
        ensemble.t_max = truth['t'][-1]

    # Add washout  -----------------------------------------------------------
    if hasattr(ensemble.bias, 'N_wash'):
        truth = create_washout(ensemble.bias, truth)

    return ensemble, truth


def create_bias_model(filter_ens, y_true,
                      bias_params, bias_filename,
                      bias_model_folder, plot_train_data=False):
    ensemble = filter_ens.copy()

    if bias_params['biasType'].name == 'None':
        ensemble.initBias(**bias_params)
    else:
        # Create bias estimator if the class is not 'NoBias'
        if os.path.isfile(bias_model_folder + bias_filename):
            bias = load_from_pickle_file(bias_model_folder + bias_filename)
            if check_valid_file(bias, bias_params) is False:
                filter_ens.bias = bias
                return bias

        # print('Train and save bias case')
        ensemble.filename += '_L{}'.format(bias_params['L'])
        ensemble.initBias(**bias_params)

        # Create training data on a multi-parameter approach
        train_data_filename = bias_model_folder + bias_filename + '_train_data'
        train_data = create_bias_training_dataset(y_true, ensemble, bias_params,
                                                  train_data_filename, plot_=plot_train_data)
        # Run bias model training
        ensemble.bias.train_bias_model(train_data=train_data,
                                       folder=bias_model_folder,
                                       plot_training=True)
        # Save
        save_to_pickle_file(bias_model_folder + bias_filename, ensemble.bias)

    return ensemble.bias


def create_washout(bias_model, true_data):
    truth = dict(true_data)
    if not hasattr(bias_model, 't_init'):
        bias_model.t_init = truth['t_obs'][0] - truth['dt_obs']
    if not hasattr(bias_model, 'upsample'):
        bias_model.upsample = 1

    i1 = np.argmin(abs(bias_model.t_init - truth['t']))
    i0 = i1 - bias_model.N_wash * bias_model.upsample

    if i0 < 0:
        raise ValueError('increase bias.t_init > t_wash + dt_obs')
    truth['wash_obs'] = truth['y'][i0:i1 + 1:bias_model.upsample]
    truth['wash_t'] = truth['t'][i0:i1 + 1:bias_model.upsample]

    return truth


def create_bias_training_dataset(y_truth, ensemble, train_params, filename, plot_=False):
    try:
        train_data = load_from_pickle_file(filename)
        rerun = True
        if train_data.shape[1] < ensemble.bias.len_train_data:
            print('Rerun multi-parameter training data: Increase the length of the training data')
            rerun = True

        if train_params['augment_data'] and train_data.shape[0] == train_params['L']:
            print('Rerun multi-parameter training data: need data augment ')
            rerun = True

        if not rerun:
            print('Loaded multi-parameter training data')
            return train_data
    except FileNotFoundError:
        print('Run multi-parameter training data: file not found')

    # ========================  Multi-parameter training approach ====================
    train_ens = ensemble.reshape_ensemble(m=1, reset=True)
    state, t_ = train_ens.timeIntegrate(int(train_ens.t_transient / train_ens.dt))
    train_ens.updateHistory(state, t_)

    train_ens.initEnsemble(**train_params)

    # Forecast one member to transient ------------------------------------------------------
    psi, t = train_ens.timeIntegrate(Nt=int(round(train_ens.t_transient / train_ens.dt)))
    train_ens.updateHistory(psi, t)

    # Remove fixed points ---------------------------------------------------------
    y_train = train_ens.getObservableHist(Nt=int(round(train_ens.t_transient / train_ens.dt)))  # Nt x Nq x m

    tol = 1e-1
    N_CR = int(round(train_ens.t_CR / train_ens.dt))
    range_y = np.max(np.max(y_train[-N_CR:], axis=0) - np.min(y_train[-N_CR:], axis=0), axis=0)
    idx_FP = (range_y < tol)
    psi0 = psi[-1, :, ~idx_FP]  # non-fixed point ICs (keep 1)
    if len(np.flatnonzero(idx_FP)) > 1:
        idx_FP[np.flatnonzero(idx_FP)[0]] = 0
        psi0 = psi[-1, :, ~idx_FP]  # non-fixed point ICs (keep 1)
        print('There are {}/{} fixed points'.format(len(np.flatnonzero(idx_FP)), len(idx_FP)))
        mean_psi = np.mean(psi0, axis=0)
        cov_psi = np.cov(psi0.T)
        new_psi0 = rng.multivariate_normal(mean_psi, cov_psi, len(np.flatnonzero(idx_FP)))
        psi0 = np.concatenate([psi0, new_psi0], axis=0)

    # Reset ensemble with post-transient ICs -------------------------------------
    train_ens.updateHistory(psi=psi0.T, reset=True)

    # Forcast training ensemble and correlate to the truth -----------------------
    Nt = train_ens.bias.len_train_data * train_ens.bias.upsample
    N_corr = int(round(train_ens.t_CR / train_ens.dt))

    psi, tt = train_ens.timeIntegrate(Nt=Nt + N_corr)
    train_ens.updateHistory(psi, tt)

    y_train = train_ens.getObservableHist()
    y_true = y_truth[-Nt:].copy()
    if len(y_true.shape) < 3:
        y_true = np.expand_dims(y_true, -1)

    # Create the synthetic bias as innovations ------------------------------------
    # train_data = y_true - y_train[-Nt:]

    if train_params['augment_data']:
        # Compute correlated signals --------------
        yy_true = y_true[:N_corr, :, 0]
        lags = np.linspace(0, N_corr, N_corr // 2, dtype=int)
        y_corr = np.zeros([Nt, y_train.shape[1], y_train.shape[2]])

        for ii in range(train_ens.m):
            yy = y_train[:, :, ii]
            RS = [CR(yy_true, yy[:N_corr])[1]]
            for lag in lags[1:]:
                RS.append(CR(yy_true[:N_corr], yy[lag:N_corr + lag])[1])
            best_lag = lags[np.argmin(RS)]
            y_corr[:, :, ii] = yy[best_lag:Nt + best_lag]

        # Augment train data with correlated signals and a fraction --------------
        train_data = y_true - np.concatenate([y_train[-Nt:],
                                              y_train[-Nt:] * -.5,
                                              y_corr],
                                             axis=-1)
    else:
        train_data = y_true - y_train[-Nt:]

    # Force train_data shape to be (L, Nt, Ndim)
    train_data = train_data.transpose((2, 0, 1))

    assert len(train_data.shape) == 3
    assert train_data.shape[1] == Nt
    assert train_data.shape[2] == y_train.shape[1]

    if plot_:
        # Compute the correlation of raw and processed data --------------------------
        RS, RS_corr = [], []
        for ii in range(y_train.shape[-1]):
            if train_params['augment_data']:
                RS_corr.append(CR(y_true[:, :, 0], y_corr[:, :, ii])[1])
            RS.append(CR(y_true[:, :, 0], y_train[:Nt, :, ii])[1])

        # Create the synthetic bias as innovations ------------------------------------

        fig1 = plt.figure(figsize=[12, 6], layout="constrained")
        sub_figs = fig1.subfigures(1, 2, width_ratios=[1.2, 1])
        norm = mpl.colors.Normalize(vmin=0, vmax=max(RS))
        cmap = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis)
        ax1, ax2 = [sf.subplots(3, 1) for sf in sub_figs]

        for yy, RR, axs, title in zip([y_train, y_corr], [RS, RS_corr],
                                      [ax1, ax2], ['Original', 'Correlated']):
            axs[0].set_title(title)
            for ll in range(yy.shape[-1]):
                clr = cmap.to_rgba(RR[ll])
                axs[0].plot(tt[:N_corr], yy[:N_corr, 0, ll], color=clr)
                axs[-1].plot(tt[:N_corr], yy[:N_corr, 0, ll] - y_true[:N_corr, 0, 0], color=clr)
            axs[0].plot(tt[:N_corr], y_true[:N_corr, 0], 'darkgrey')
            for ll in range(yy.shape[-1]):
                clr = cmap.to_rgba(RR[ll])
                axs[1].plot(ll, RR[ll], 'o', color=clr)

        plt.show()

    # Save training data ------------------------------------
    save_to_pickle_file(filename, train_data)

    return train_data
