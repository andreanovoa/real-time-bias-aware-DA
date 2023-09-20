import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import time
import datetime
from physical_models import Annular

if __name__ == '__main__':
    MyModel = Annular
    paramsTA = dict(dt=1 / 51.2E3)

    animate = False
    anim_name = '{}_Annulat_mov_mix_epsilon.gif'.format(datetime.date.today())

    # Non-ensemble case =============================
    t1 = time.time()
    case = MyModel(**paramsTA)
    state, t_ = case.timeIntegrate(int(case.t_transient * 3 / case.dt))
    case.updateHistory(state, t_)

    print(case.dt)
    print('Elapsed time = ', str(time.time() - t1))

    fig1 = plt.figure(figsize=[12, 3], layout="constrained")
    subfigs = fig1.subfigures(1, 2, width_ratios=[1.2, 1])

    ax = subfigs[0].subplots(1, 2)
    ax[0].set_title(MyModel.name)

    t_h = case.hist_t
    t_zoom = min([len(t_h) - 1, int(0.05 / case.dt)])

    # State evolution
    y, lbl = case.getObservableHist(modes=True), case.obsLabels

    ax[0].scatter(t_h, y[:, 0], c=t_h, label=lbl, cmap='Blues', s=10, marker='.')

    ax[0].set(xlabel='$t$', ylabel=lbl[0])
    i, j = [0, 1]

    if len(lbl) > 1:
        # ax[1].scatter(y[:, 0]/np.max(y[:,0]), y[:, 1]/np.max(y[:,1]), c=t_h, s=3, marker='.', cmap='Blues')
        # ax[1].set(xlabel='$'+lbl[0]+'/'+lbl[0]+'^\mathrm{max}$', ylabel='$'+lbl[1]+'/'+lbl[1]+'^\mathrm{max}$')
        ax[1].scatter(y[:, 0], y[:, 1], c=t_h, s=3, marker='.', cmap='Blues')
        ax[1].set(xlabel=lbl[0], ylabel=lbl[1])
    else:
        ax[1].plot(t_h[-t_zoom:], y[-t_zoom:, 0], color='green')

    ax[1].set_aspect(1. / ax[1].get_data_ratio())

    if not animate:
        ax2 = subfigs[1].subplots(2, 1)

        y, lbl = case.getObservableHist(modes=False), case.obsLabels
        y = np.mean(y, axis=-1)
        # print(np.min(y, axis=0))
        # sorted_id = np.argsort(np.max(abs(y[-1000:]), axis=0))
        # print(sorted_id)
        # y = y[:, sorted_id]
        # lbl = [lbl[idx] for idx in sorted_id]

        for ax in ax2:
            ax.plot(t_h, y / 1E3)
        ax2[0].set_title('Acoustic Pressure')
        ax2[0].legend(lbl, bbox_to_anchor=(1., 1.), loc="upper left", ncol=1, fontsize='small')
        ax2[0].set(xlim=[t_h[0], t_h[-1]], xlabel='$t$', ylabel='$p$ [kPa]')
        ax2[1].set(xlim=[t_h[-1] - case.t_CR, t_h[-1]], xlabel='$t$', ylabel='$p$ [kPa]')
    else:
        ax2 = subfigs[1].subplots(1, 1, subplot_kw={'projection': 'polar'})
        angles = np.linspace(0, 2 * np.pi, 200)  # Angles from 0 to 2π
        y, lbl = case.getObservableHist(modes=False, loc=angles), case.obsLabels
        y = np.mean(y, axis=-1)

        radius = [0, 0.5, 1]
        theta, r = np.meshgrid(angles, radius)

        # Remove radial tick labels
        ax2.set_yticklabels([])
        ax2.grid(False)

        # Add a white concentric circle
        circle_radius = 0.5
        ax2.plot(angles, [circle_radius] * len(angles), color='black', lw=1)

        idx_max = np.argmax(y[:, 0])
        polar_mesh = ax2.pcolormesh(theta, r, [y[idx_max].T] * len(radius), shading='auto', cmap='RdBu')

        ax2.set_theta_zero_location('S')  # Set zero angle to the north (top)
        ax2.set_title('Acoustic Pressure')
        ax2.set_theta_direction(1)  # Set clockwise rotation

        start_i = np.argmin(abs(t_h[-1] - (t_h[-1] - case.t_CR)))

        print((t_h[-1]))
        start_i = int((t_h[-1] - .03) // case.dt)

        print(start_i, t_h[start_i])

        dt_gif = 10

        t_gif = t_h[start_i::dt_gif]

        y_gif = y[start_i::dt_gif]


        def update(frame):
            ax2.fill(angles, [circle_radius] * len(angles), color='white')
            polar_mesh.set_array([y_gif[frame].T] * len(radius))
            ax2.set_title('Acoustic Pressure $t$ = {:.3f}'.format(t_gif[frame]))  # , fontsize='small')#, labelpad=50)


        plt.colorbar(polar_mesh, label='Pressure', shrink=0.75)
        anim = FuncAnimation(fig1, update, frames=len(t_gif))
        dt = t_gif[1] - t_gif[0]
        anim.save(anim_name, fps=dt_gif * 10)

    plt.show()