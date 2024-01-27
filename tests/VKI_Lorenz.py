
# %% 1. EMPLOY LORENZ63 MODEL

from default_parameters.lorenz63 import *

observe_idx = np.array([0, 1, 2])  # Number of dimensions the ESN prediction
update_reservoir = 1
plot_training_data = 0
plot_timeseries_flag = 1


t_ref = t_lyap


def plot_data(axs, type_plot='train'):
    if type_plot == 'train':
        ci = -1
        for yyy, ttt in out:
            ci += 1
            for axj, axx in enumerate(axs):
                axx.plot(ttt / t_ref, yyy[:, axj], '-', c=cs[ci])
                axx.set(ylabel=truth.obsLabels[axj])
            if ci == 2:
                axs[0].set(xlim=[0, ttt[-1] / t_ref])
                axs[0].legend(['Transient', 'Train+val', 'Tests'], ncols=3, loc='lower center', bbox_to_anchor=(0.5, 1.0))
    else:
        if jj == 0:
            for ii, idx in enumerate(ESN_case.observed_idx):
                axs[idx].plot(t_wash_model / t_ref, wash_model[:, ii], '.-', c=cs[-1])
                axs[idx].plot(t_wash / t_ref, wash_data[:, ii], 'x-', c='tab:green')
        lbl_pred = ['ESN_prediction'] + [None] * (Aa.shape[-1]-1)
        lbl_a = ['Analysis'] + [None] * (Aa.shape[-1]-1)
        lbl_f = ['Forecast'] + [None] * (Aa.shape[-1]-1)

        for axi, ax in enumerate(axs):
            ax.plot(tt / t_ref, yy[:, axi], '-', c='k', lw=2, alpha=.8, label='Truth')
            if axi < ESN_case.N_dim:
                ax.plot(tt_up / t_ref, u_closed[:, axi], '--', c='#20b2aae5', ms=4, label=lbl_pred)
                ax.plot(tt_up[-1] / t_ref, Af[[axi]], '^', c='#20b2aae5', ms=4, label=lbl_f)
                ax.plot(tt_up[-1] / t_ref, Aa[[axi]], '*', c='#0054D8', ms=7, label=lbl_a)
                if axi in observe_idx:
                    for yy_obs in yy[-1, axi]:
                        ax.plot(tt[-1] / t_ref, yy_obs, 'o', c='r', ms=8, alpha=.8, label='Data')
                if jj == 0 and axi == 0:
                    axs[0].legend(ncols=3, loc='lower center', bbox_to_anchor=(0.5, 1.0))

        if jj == num_tests:
            margin = dt_ESN * ESN_case.N_wash
            axs[-1].set(xlabel='$t/T$', xlim=[t_wash[0] / t_ref - margin, tt_up[-1] / t_ref + margin])



if __name__ == '__main__':
    # %% 2. CREATE TRUTH

    forecast_params['model'] = Lorenz63
    forecast_params['dt'] = dt_model

    model = Lorenz63(**forecast_params)



    model.timeIntegrate()





