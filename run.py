import os
import numpy as np
import pylab as plt
import pickle
from Util import createObservations, CR
from DA import dataAssimilation
from plotResults import plot_train_data

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=12)
plt.rc('legend', facecolor='white', framealpha=1, edgecolor='white')
rng = np.random.default_rng(6)


def main(filter_ens, truth, filter_p, results_dir="results/", figs_dir='results/figs/', save_=False):
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    if not os.path.isdir(figs_dir):
        os.makedirs(figs_dir)

    # =========================  PERFORM DATA ASSIMILATION ========================== #
    filter_ens = dataAssimilation(filter_ens, truth['p_obs'], truth['t_obs'],
                                  std_obs=truth['std_obs'], method=filter_p['filt'])
    # Integrate further without assimilation as ensemble mean (if truth very long, integrate only .2s more)
    Nt_extra = 0
    if filter_ens.hist_t[-1] < truth['t'][-1]:
        Nt_extra = int(min((truth['t'][-1] - filter_ens.hist_t[-1]), filter_ens.t_CR) / filter_ens.dt) + 1
        psi, t = filter_ens.timeIntegrate(Nt_extra, averaged=True)
        filter_ens.updateHistory(psi, t)
        if filter_ens.bias is not None:
            y = filter_ens.getObservableHist(Nt_extra)
            b, t_b = filter_ens.bias.timeIntegrate(t=t, y=y)
            filter_ens.bias.updateHistory(b, t_b)
    filter_ens.close()

    # ================================== SAVE DATA  ================================== #
    parameters = dict(biasType=filter_p['biasType'], forecast_model=filter_ens.name,
                      true_model=truth['model'], num_DA=len(truth['t_obs']), Nt_extra=Nt_extra)
    if save_:
        filename = '{}{}-{}_F-{}_B-{}_k{}'.format(results_dir, filter_p['filt'], truth['name'],
                                                  filter_ens.name, filter_p['biasType'].name, filter_ens.bias.k)
        with open(filename, 'wb') as f:
            pickle.dump(parameters, f)
            pickle.dump(truth, f)
            pickle.dump(filter_ens, f)
    return filter_ens, truth, parameters


# ====================================================================================================
def createEnsemble(true_p, forecast_p, filter_p, bias_p,
                   working_dir="results", filename='reference_Ensemble', results_dir=None):
    if results_dir is None:
        results_dir = working_dir

    if not os.path.isdir(results_dir + 'figs/'):
        os.makedirs(results_dir + 'figs/')
        print('created dir', results_dir + 'figs/')
    if os.path.isfile(results_dir + filename):
        with open(results_dir + filename, 'rb') as f:
            ensemble = pickle.load(f)
            truth = pickle.load(f)
            b_args = pickle.load(f)
        reinit = False
        # check that true and forecast model parameters
        for key, val in filter_p.items():
            if hasattr(ensemble, key) and getattr(ensemble, key) != val:
                reinit = True
                print('Re-initialise ensemble as ensemble {}={} != {}'.format(key, getattr(ensemble, key), val))
                break
            elif b_args is not None and type(b_args) is dict and key in b_args[0].keys() and val != b_args[0][key]:
                reinit = True
                print('Re-initialise ensemble as filter_p {}={} != {}'.format(key, b_args[0][key], val))
                break
        if truth['t_obs'][-1] < filter_p['t_stop']:
            reinit = True

        if not reinit and bias_p is not None:
            # check that bias and assimilation parameters are the same
            for key, val in bias_p.items():
                if key in b_args[0]['Bdict'].keys():
                    try:
                        if val != b_args[0]['Bdict'][key]:
                            reinit = True
                            print('Re-init ensemble as {} = {} != {}'.format(key, b_args[0]['Bdict'][key], val))
                            break
                    except:
                        for v1, v2 in zip(val, b_args[0]['Bdict'][key]):
                            if v1 != v2:
                                reinit = True
                                print('Re-init ensemble as {} = {} != {}'.format(key, b_args[0]['Bdict'][key], val))
                                break
        if not reinit:
            return ensemble, truth, b_args

    # =============================  CREATE OBSERVATIONS ============================== #
    y_true, t_true, name_truth = createObservations(true_p)
    if 'manual_bias' in true_p.keys():
        if true_p['manual_bias'] == 'time_func':
            # Time dependent bias ------------------
            b_true = .5 * y_true * np.cos(np.expand_dims(t_true, -1)*np.pi/2)
            name_truth += '_timefuncBias'
        elif true_p['manual_bias'] == 'cosine':
            # Nonlinear bias ------------------
            b_true = 0.2 * np.max(y_true, 0) * np.cos(2*y_true/np.max(y_true, 0))
            name_truth += '_cosBias'
        elif true_p['manual_bias'] == 'linear':
            # Linear bias ------------------
            b_true = 2. + .3 * y_true
            name_truth += '_linearBias'
    else:
        b_true = np.zeros(1)

    # plt.figure()
    # plt.plot(b_true[:, 0])
    #
    # plt.figure()
    # plt.plot(y_true[:, 0])
    # plt.plot(y_true[:, 0] + b_true[:, 0])
    # plt.show()


    y_true += b_true

    dt_t = t_true[1] - t_true[0]
    obs_idx = np.arange(round(filter_p['t_start'] / dt_t), 
                        round(filter_p['t_stop'] / dt_t) + 1, filter_p['kmeas'])
    t_obs = t_true[obs_idx]

    q = np.shape(y_true)[1]
    if 'std_obs' not in true_p.keys():
        true_p['std_obs'] = 0.01
    Cdd = np.eye(q) * true_p['std_obs'] ** 2

    noise = rng.multivariate_normal(np.zeros(q), Cdd, len(obs_idx))
    obs = y_true[obs_idx] * (1. + noise)

    truth = dict(y=y_true, t=t_true, b=b_true, dt=dt_t,
                 t_obs=t_obs, p_obs=obs, dt_obs=t_obs[1]-t_obs[0],
                 true_params=true_p, name=name_truth,
                 model=true_p['model'], std_obs=true_p['std_obs'])

    # %% =============================  DEFINE BIAS ======================================== #
    if filter_p['biasType'].name == 'ESN':
        args = (filter_p, forecast_p['model'], truth, working_dir)
        filter_p['Bdict'] = createESNbias(*args, bias_param=bias_p)
    else:
        args = (None,)
    # ===============================  INITIALISE ENSEMBLE  =============================== #
    ensemble = forecast_p['model'](forecast_p, filter_p)
    with open(results_dir + filename, 'wb') as f:
        pickle.dump(ensemble, f)
        pickle.dump(truth, f)
        pickle.dump(args, f)

    return ensemble, truth, args


def createESNbias(filter_p, model, truth, folder, bias_param=None):
    if bias_param is None:
        raise ValueError('Provide bias parameters dictionary')

    if 'L' not in bias_param.keys():
        bias_param['L'] = 10

    bias_p = bias_param.copy()
    train_params = bias_p['train_params'].copy()
    train_params['m'] = bias_p['L']
    
    # Compute reference bias. Create an ensemble of training data
    ref_ens = model(train_params, train_params)
    try:
        name_train = folder + 'Truth_{}_{}'.format(ref_ens.name, ref_ens.law)
    except:
        name_train = folder + 'Truth_{}'.format(ref_ens.name)
        
    for k in ref_ens.params:
        name_train += '_{}{}'.format(k, getattr(ref_ens, k))
    name_train += '_std{:.2}_m{}_{}'.format(ref_ens.std_a, ref_ens.m, ref_ens.alpha_distr)
    # Load or create reference ensemble (multi-parameter solution)
    rerun = True

    print(name_train)

    if os.path.isfile(name_train):
        with open(name_train, 'rb') as f:
            load_ens = pickle.load(f)
        if len(truth['t']) <= len(load_ens.hist_t):
            ref_ens = load_ens.copy()
            rerun = False

    if rerun:
        print('Creating Reference solution(s)')
        psi, t = ref_ens.timeIntegrate(Nt=len(truth['t']) - 1)
        ref_ens.updateHistory(psi, t)
        ref_ens.close()
        with open(name_train, 'wb') as f:
            pickle.dump(ref_ens, f)

    y_ref, lbl = ref_ens.getObservableHist(Nt=len(truth['t'])), ref_ens.obsLabels
    t = ref_ens.hist_t[:len(truth['t'])]

    if len(truth['y'].shape) < len(y_ref.shape):
        biasData = np.expand_dims(truth['y'], -1) - y_ref   # [Nt x Nmic x L]
    else:
        biasData = truth['y'] - y_ref   # [Nt x Nmic x L]

    #
    # print('reashefd point')
    # print(truth['y'].shape, y_ref.shape)


    # phase_shift = 100
    # shiftedData = np.append(np.zeros([phase_shift, y_true.shape[1]]), y_true[phase_shift:], axis=0)
    # biasData2 = np.expand_dims(y_true - shiftedData, -1)
    # biasData2 = 2 * np.sin(y_ref/2.)   # [Nt x Nmic x L]
    # biasData2 = 1. + .1 * y_ref
    # biasData = np.append(biasData, biasData2, axis=-1)
    # biasData3 = np.expand_dims(np.expand_dims(np.sin(2*np.pi*t), -1) * 0.2*np.max(y_true, axis=0, keepdims=True), -1)
    # biasData = np.concatenate([biasData, biasData2, biasData3], axis=-1)
    # biasData = np.append(biasData, ref_ens.hist[:, -len(ref_ens.est_p):], axis=1)

    # provide data for washout before first observation
    if 'start_ensemble_forecast' not in filter_p.keys():
        filter_p['start_ensemble_forecast'] = 2

    tol = 1e-5

    i1 = truth['t_obs'][0] - truth['dt_obs'] * filter_p['start_ensemble_forecast']

    i1 = int(np.where(abs(truth['t'] - i1) < tol)[0])
    i0 = i1 - bias_p['N_wash'] * bias_p['upsample']
    if i0 < 0:
        min_t = (bias_p['N_wash'] * bias_p['upsample'] + filter_p['kmeas']) * (t[1] - t[0])
        raise ValueError('increase t_start to > t_wash + dt_a = {}'.format(min_t))

    # create bias dictionary
    bias_p['trainData'] = biasData[int(ref_ens.t_transient / truth['dt']):]
    bias_p['washout_obs'] = truth['y'][i0:i1 + 1]
    bias_p['washout_t'] = truth['t'][i0:i1 + 1]
    bias_p['filename'] = folder + truth['name'] + '_' + name_train.split('Truth_')[-1] + '_bias'


    plot_train_data(truth, y_ref, ref_ens.t_CR, folder)

    
    return bias_p


def get_error_metrics(results_folder):

    out = dict(Ls=[], ks=[])
    keys = ['R_biased_DA', 'R_biased_post',
            'C_biased_DA', 'C_biased_post',
            'R_unbiased_DA', 'R_unbiased_post',
            'C_unbiased_DA', 'C_unbiased_post']

    L_dirs = []
    LLL = os.listdir(results_folder)
    LLL.sort()
    for Ldir in LLL:
        if os.path.isdir(results_folder + Ldir + '/') and Ldir[0] == 'L':
            L_dirs.append(results_folder + Ldir + '/')
            out['Ls'].append(float(Ldir.split('L')[-1]))

    out['L_dirs'] = L_dirs
    for ff in os.listdir(L_dirs[0]):
        k = float(ff.split('_k')[-1])
        out['ks'].append(k)
    out['ks'].sort()

    print(out['ks'])
    print('Number of L folders = ', len(out['Ls']))
    print('Number of k files = ', len(out['ks']))

    for key in keys:
        out[key] = np.empty([len(out['Ls']), len(out['ks'])])


    # ==================================================================================================================
    ii = -1
    for Ldir in L_dirs:
        ii += 1
        print('L = ', out['Ls'][ii])
        k_files = os.listdir(Ldir)
        k_files.sort()
        for ff in k_files:
            # Read file
            with open(Ldir + ff, 'rb') as f:
                _ = pickle.load(f)
                truth = pickle.load(f)
                filter_ens = pickle.load(f)
            jj = np.argmin(abs(out['ks'] - filter_ens.bias.k))

            print('\t k = ', out['ks'][jj])
            # Compute biased and unbiased signals
            y, t = filter_ens.getObservableHist(), filter_ens.hist_t
            b, t_b = filter_ens.bias.hist, filter_ens.bias.hist_t
            y_mean = np.mean(y, -1)

            # Unbiased signal error
            if hasattr(filter_ens.bias, 'upsample'):
                y_unbiased = y_mean[::filter_ens.bias.upsample] + b
                y_unbiased = interpolate(t_b, y_unbiased, t)
            else:
                y_unbiased = y_mean + b

            if ii == 0 and ff == k_files[0]:
                N_CR = int(filter_ens.t_CR / filter_ens.dt)  # Length of interval to compute correlation and RMS
                i0 = np.argmin(abs(t - truth['t_obs'][0]))  # start of assimilation
                i1 = np.argmin(abs(t - truth['t_obs'][-1]))  # end of assimilation
                for key, val in zip(['i0', 'i1', 'N_CR'], [i0, i1, N_CR]):
                    out[key] = val
                # true and pre-DA R
                y_truth_u = truth['y'] - truth['b']
                out['C_true'], out['R_true'] = CR(truth['y'][-N_CR:], y_truth_u[-N_CR:])
                out['C_pre'], out['R_pre'] = CR(truth['y'][i0 - N_CR:i0 + 1:], y_mean[i0 - N_CR:i0 + 1:])
                out['t_interp'] = truth['t'][i0-N_CR:len(y_mean):N_CR]
                scale = np.max(truth['y'], axis=0)

                for key in ['error_biased', 'error_unbiased']:
                    out[key] = np.empty([len(out['Ls']), len(out['ks']), len(out['t_interp']), y_mean.shape[-1]])

            # End of assimilation
            for yy, key in zip([y_mean, y_unbiased], ['_biased_DA', '_unbiased_DA']):
                C, R = CR(truth['y'][i1 - N_CR:i1], yy[i1-N_CR:i1])
                out['C'+key][ii, jj] = C
                out['R'+key][ii, jj] = R

            # After Assimilaiton
            for yy, key in zip([y_mean, y_unbiased], ['_biased_post', '_unbiased_post']):
                C, R = CR(truth['y'][i1:i1+N_CR], yy[i1:i1+N_CR])
                out['C'+key][ii, jj] = C
                out['R'+key][ii, jj] = R

            # Compute mean errors
            b_obs = truth['y'][i0-N_CR:len(y_mean)] - y_mean[i0-N_CR:]
            b_obs_u = truth['y'][i0-N_CR:len(y_mean)] - y_unbiased[i0-N_CR:]

            ei, a = -N_CR, -1
            while ei < len(b_obs) - N_CR:
                a += 1
                ei += N_CR
                out['error_biased'][ii, jj, a, :] = np.mean(abs(b_obs[ei:ei + N_CR]), axis=0) / scale
                out['error_unbiased'][ii, jj, a, :] = np.mean(abs(b_obs_u[ei:ei + N_CR]), axis=0) / scale

    with open(results_folder + 'CR_data', 'wb') as f:
        pickle.dump(out, f)