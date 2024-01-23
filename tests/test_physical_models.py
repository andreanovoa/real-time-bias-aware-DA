

if __name__ == '__main__':

    from essentials.physical_models import *


    # os.environ["OMP_NUM_THREADS"] = '1'
    num_proc = os.cpu_count()
    if num_proc > 1:
        num_proc = int(num_proc)

    plt.rc('text', usetex=True)
    plt.rc('font', family='times', size=14, serif='Times New Roman')
    plt.rc('mathtext', rm='times', bf='times:bold')
    plt.rc('legend', facecolor='white', framealpha=1, edgecolor='white')


    t0 = time.time()

    # Initialise model
    case = Annular()

    # Forecast model and update
    state, t_ = case.time_integrate(int(case.t_transient / case.dt))
    case.update_history(state, t_)

    # Generate ensemble
    DA_params = dict(m=10, est_a=case.params[:2], std_psi=0.1, std_a=0.4, alpha_distr='uniform')
    case.init_ensemble(**DA_params)

    t1 = time.time()
    for _ in range(2):
        state, t_ = case.time_integrate(int(case.t_transient / case.dt))
        case.update_history(state, t_)

    print('Elapsed time = ', str(time.time() - t1))

    t_h = case.hist_t
    t_zoom = min([len(t_h) - 1, int(case.t_CR / case.dt)])

    fig = plt.figure(figsize=(12, 7.5), layout="constrained")
    subfigs = fig.subfigures(1, 2, width_ratios=[2, 1])

    axs = subfigs[0].subplots(2, 1, sharey='all')

    # State evolution
    y, lbl = case.get_observable_hist(), case.obs_labels[0]

    axs[0].plot(t_h, y[:, 0], color='blue', label=lbl)
    axs[1].plot(t_h[-t_zoom:], y[-t_zoom:, 0], color='blue')

    axs[0].set(xlabel='t', ylabel=lbl, xlim=[t_h[0], t_h[-1]])
    axs[1].set(xlabel='t', xlim=[t_h[-t_zoom], t_h[-1]])

    # Params
    ai = -case.Na
    max_p, min_p = -1000, 1000
    c = ['g', 'sandybrown', 'mediumpurple', 'cyan']
    mean = np.mean(case.hist, -1, keepdims=True)

    ii = case.Nphi
    mean_p, hist_p, std_p, labels_p = [], [], [], []
    for p in case.est_a:
        m_ = mean[:, ii].squeeze()
        s = abs(np.std(case.hist[:, ii], axis=1))
        labels_p.append(case.params_labels[p])
        mean_p.append(m_)
        std_p.append(s)
        hist_p.append(case.hist[:, ii, :])
        ii += 1

    colors_alpha = ['green', 'sandybrown', [0.7, 0.7, 0.87], 'blue', 'red', 'gold', 'deepskyblue']

    axs = subfigs[1].subplots(len(case.est_a), 1, sharex='col')
    if type(axs) is not np.ndarray:
        axs = [axs]
    for ax, p, h, avg, s, c, lbl in zip(axs, case.est_a, hist_p, mean_p, std_p, colors_alpha, labels_p):
        max_p = np.max(h)
        min_p = np.min(h)
        ax.plot(case.hist_t, avg, color=c, label='mean ' + lbl)
        ax.plot(case.hist_t, h, color=c, lw=0.5)
        ax.fill_between(case.hist_t, avg + abs(s), avg - abs(s), alpha=0.2, color=c, label='std ' + lbl)
        ax.legend(loc='upper right', fontsize='small', ncol=2)
        ax.set(ylabel='', ylim=[min_p, max_p])

    # print('time={}, size={}'.format(time.time() - t0, pympler.asizeof.asizeof(case)))

    plt.show()
