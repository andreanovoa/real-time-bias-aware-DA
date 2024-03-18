import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from essentials.physical_models import Lorenz63

plt.style.use('dark_background')
plt.rcParams['grid.linewidth'] = .1

plt.rc('text', usetex=True)
plt.rc('font', family='times', size=14, serif='Times New Roman')
plt.rc('mathtext', rm='times', bf='times:bold')
plt.rc('legend', facecolor='white', framealpha=1, edgecolor='white')

rnd = np.random.RandomState(6)

run_ergodic_gif = False
run_butterfly_gif = False
run_parameters_gif = True

dt_model = 0.005
t_lyap = 0.906 ** (-1)  # Lyapunov Time (inverse of largest Lyapunov exponent

if __name__ == '__main__':
    # %% 2. CREATE TRUTH

    forecast_params = dict(dt=dt_model,
                           psi0=rnd.random(3),
                           t_transient=10 * t_lyap,
                           t_lyap=t_lyap)

    truth = Lorenz63(**forecast_params)

    # %%

    if run_ergodic_gif:
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        psi0_all = [(40, 20, 60), (40, -60, 20), (-40, -20, -20),
                    (-40, 60, 60), (10, 50, 70), (-10, 20, -70)]

        ax.set(xlabel='$x$', ylabel='$y$', zlabel='$z$')

        colors = ['c', 'r', 'm', 'g', 'y', 'b']


        # Function to update the plot for each framex
        def update(frame):
            Nt = 2 * frame

            ax.clear()  # Clear the previous frame
            ax.plot(truth.hist[:, 0], truth.hist[:, 1], truth.hist[:, 2], lw=.52, alpha=0.5, c='w')

            ax.view_init(elev=20 + np.sin(np.radians(frame)), azim=frame / 2.)

            for psi0, c in zip(psi0_all, colors):

                if Nt == 0:
                    ax.plot(*psi0, c=c, marker='o', markerfacecolor=c)  # Scatter plot points for clarity
                else:
                    forecast_params['psi0'] = np.array(psi0)
                    case = Lorenz63(**forecast_params)
                    psi, t = case.time_integrate(Nt=Nt)
                    ax.plot(psi[:, 0], psi[:, 1],
                            psi[:, 2], lw=1., alpha=0.9, c=c)
                    ax.plot(psi[-1, 0], psi[-1, 1], psi[-1, 2], c=c, marker='o', markerfacecolor=c)
                    ax.set(title='$t={:.4}$ LT'.format(t[-1] / t_lyap))

            ax.set_xlim(-50, 50)  # Adjust as needed
            ax.set_ylim(-40, 40)  # Adjust as needed
            ax.set_zlim(-20, 60)  # Adjust as needed
            ax.set(xlabel='$x$', ylabel='$y$', zlabel='$z$')


        # Set up the figure and axis
        # Create an animation
        animation = FuncAnimation(fig, update, frames=np.arange(400))

        # Save the animation as a video file
        animation.save('Lorenz_ergodic.gif', writer='ffmpeg', fps=10)

        # Show the plot (optional)
        # plt.show()

    if run_butterfly_gif:

        # Run until attractor
        N_lyap = int(t_lyap / truth.dt)
        psi_transient = truth.time_integrate(Nt=30 * N_lyap)[0]

        # Initialise case in the attractor
        case0 = truth.copy()
        psi0 = psi_transient[-1]
        case0.update_history(psi=psi0, reset=True)

        # Initialise second case as perturbation
        case1 = truth.copy()
        psi1 = psi_transient[-1] + 1e-4
        case1.update_history(psi=psi1, reset=True)
        psi_transient = truth.time_integrate(Nt=30 * N_lyap)[0]

        print(psi0, psi1)


        # Forecast both cases
        Nt = 40 * N_lyap
        psi0, t0 = case0.time_integrate(Nt=Nt)
        psi1, t1 = case1.time_integrate(Nt=Nt)



        c0, c1 = 'w', '#FFA500'
        fig = plt.figure(figsize=(12, 5))
        subfigs = fig.subfigures(1, 2, wspace=-0.1, width_ratios=[1, 1])
        ax3d = subfigs[-1].add_subplot(111, projection='3d')
        ax3d.xaxis.pane.fill = False
        ax3d.yaxis.pane.fill = False
        ax3d.zaxis.pane.fill = False
        ax3d.set(xlabel='$x$', ylabel='$y$', zlabel='$z$')
        for psi, c, mfc in zip([psi0, psi1], [c0, c1], [c0, 'none']):
            ax3d.plot(psi[:, 0], psi[:, 1], psi[:, 2], lw=1., c=c)
        for psi, c, mfc in zip([psi0, psi1], [c0, c1], [c0, 'none']):
            ax3d.plot(psi[-1, 0], psi[-1, 1], psi[-1, 2], 'o', lw=1., c=c, mfc=mfc, mew=2)

        axs = subfigs[0].subplots(3, 1, sharex=True)
        ax_labels = ['$x$', '$y$', '$z$']
        for ii, ax in enumerate(axs):
            for t, psi, c, mfc in zip([t0, t1], [psi0, psi1], [c0, c1], [c0, 'none']):
                ax.plot(t/t_lyap, psi[:, ii], c=c)
                ax.plot(t[-1]/t_lyap, psi[-1, ii], 'o', c=c, mfc=mfc, mew=2)
                ax.plot(t[-10]/t_lyap, psi[-10, ii], 'o', c=c, mfc=mfc, mew=2)
                ax.plot(t[0]/t_lyap, psi[0, ii], 'o', c=c, mfc=mfc, mew=2)
            ax.set(ylabel=ax_labels[ii])
        axs[-1].set(xlabel='$t/T$')
        # plt.show()

        xlims = ax3d.get_xlim()
        ylims = ax3d.get_ylim()
        zlims = ax3d.get_zlim()

        YL = []
        for ax in axs:
            YL.append(ax.get_ylim())


        dt_frame = 20

        def update(frame):
            tf = dt_frame * frame

            ax3d.clear()  # Clear the previous frame
            [ax_.clear() for ax_ in axs]

            ax3d.set(xlim=xlims, ylim=ylims, zlim=zlims)
            ax3d.view_init(elev=20 + np.sin(np.radians(frame)), azim=frame / 2.)
            ax3d.set(xlabel='$x$', ylabel='$y$', zlabel='$z$')
            if t0[tf] > 20:
                axs[-1].set_xlim([t0[tf]-23, t0[tf]+3])
            else:
                axs[-1].set_xlim([-3, 23])
            for ax_, yl, lbl in zip(axs, YL, ax_labels):
                ax_.set(ylim=yl, ylabel=lbl)
            axs[-1].set(xlabel='$t/T$')

            if tf > 0:  # Plot trajectories
                for t_, x, c_ in zip([t0, t1], [psi0, psi1], [c0, c1]):
                    ax3d.plot(x[:tf, 0], x[:tf, 1], x[:tf, 2], lw=1., c=c_)
                    for jj, ax_ in enumerate(axs):
                        ax_.plot(t_[:tf] / t_lyap, x[:tf, jj], c=c_)

            # Plot marker at current state
            for t_, x, c_, fc in zip([t0, t1], [psi0, psi1], [c0, c1], [c0, 'none']):
                ax3d.plot(x[tf, 0], x[tf, 1], x[tf, 2], 'o', c=c_, mfc=fc, mew=2)
                for jj, ax_ in enumerate(axs):
                    ax_.plot(t_[tf] / t_lyap, x[tf, jj], 'o', c=c_, mfc=fc, mew=2)



        animation = FuncAnimation(fig, update,  frames=np.arange(Nt//dt_frame), blit=False)
        animation.save('Lorenz_butterfly.gif', fps=5)

    if run_parameters_gif:
        fig = plt.figure(figsize=(15, 10), dpi=60)
        ax = fig.add_subplot(111, projection='3d')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        rho_all = [10, 90, 160, 200, 260]

        ax.set(xlabel='$x$', ylabel='$y$', zlabel='$z$')

        colors = ['w', 'y', 'darkmagenta', 'mediumslateblue', 'darkturquoise', 'tab:red', 'violet']

        colors = plt.cm.plasma(np.linspace(0.1, 1, len(rho_all)))[::-1]
        colors = colors.tolist()
        colors[2] = 'tab:red'

        colors[3] = colors[-1]
        colors[-1] = 'c'

        forecast_params = dict(dt=dt_model,
                               t_transient=10 * t_lyap,
                               sigma=10,
                               beta=8./3.,
                               t_lyap=t_lyap)

        HIST = []
        RHOS = []
        CS = []

        psi0 = np.array([40, 100, 100])
        Nt = Lorenz63.t_transient // dt_model * 2

        for kk in range(len(rho_all)):

            rho, c = rho_all[kk], colors[kk]

            forecast_params['psi0'] = psi0
            forecast_params['rho'] = rho
            case = Lorenz63(**forecast_params)
            psi = case.time_integrate(Nt=Nt)[0]

            HIST.append(psi)
            RHOS.append(rho*np.ones(psi.shape[0]))
            CS.append(kk*np.ones(psi.shape[0]))


            forecast_params['psi0'] = psi[-1]
            case = Lorenz63(**forecast_params)

            psi = case.time_integrate(Nt=Nt)[0]

            ax.plot(psi[:, 0], psi[:, 1], psi[:, 2], c=c, lw=.5)
            ax.plot(psi[-1, 0], psi[-1, 1], psi[-1, 2], c=c, marker='o', markerfacecolor=c)

            HIST.append(psi)
            RHOS.append(rho*np.ones(psi.shape[0]))
            CS.append(kk*np.ones(psi.shape[0]))

            psi0 = psi[-1]


        def update(frame):
            ax.view_init(elev=20 + np.sin(np.radians(frame)), azim=frame / 2.)
            ax.set_xlim(-50, 50)  # Adjust as needed
            ax.set_ylim(-150, 150)  # Adjust as needed
            ax.set_zlim(-20, 350)  # Adjust as needed
            ax.set(xlabel='$x$', ylabel='$y$', zlabel='$z$')

        #
        # # Set up the figure and axis
        # # Create an animation
        animation = FuncAnimation(fig, update, frames=np.arange(300))

        # #
        # # # Save the animation as a video file
        animation.save('Lorenz_bifurcations_attractors_quick.gif', writer='ffmpeg', fps=10)


        raise


        frame_dt = 15
        window_length = 200
        offset = 0
        for kk, H in enumerate(HIST):

            c = colors[kk//2]
            r = rho_all[kk//2]
            if kk > 0:
                offset = len(H) // frame_dt

            print(kk)
            def update(frame):
                t1 = frame * frame_dt
                t0 = max(0, t1 - window_length * frame_dt)

                ax.clear()  # Clear the previous frame
                ax.view_init(elev=20 + np.sin(np.radians(frame + offset)), azim=(frame + offset) / 2.)

                if kk > 0:
                    t0_old = - (t1 - window_length * frame_dt)
                    if t0_old > 0:
                        old_H = HIST[kk-1]
                        ax.plot(old_H[-t0_old:, 0],
                                old_H[-t0_old:, 1], old_H[-t0_old:, 2], lw=1., alpha=0.9, c=colors[(kk - 1) // 2])
                if t1 == 0:
                    ax.plot(H[0, 0], H[0, 1], H[0, 2], c=c, marker='o', markerfacecolor=c)
                else:
                    psi = H[t0:t1]
                    ax.plot(psi[:, 0], psi[:, 1], psi[:, 2], lw=1., alpha=0.9, c=c)
                    ax.plot(psi[-1, 0], psi[-1, 1], psi[-1, 2], c=c, marker='o', markerfacecolor=c)

                ax.set_title('$\\rho={}$'.format(r), c=c)

                ax.set_xlim(-50, 50)  # Adjust as needed
                ax.set_ylim(-150, 150)  # Adjust as needed
                ax.set_zlim(-20, 350)  # Adjust as needed
                ax.set(xlabel='$x$', ylabel='$y$', zlabel='$z$')


            # # # Create an animation
            animation = FuncAnimation(fig, update, frames=np.arange(len(H)//frame_dt))
            animation.save('Lorenz_bifurcations_quik{}.gif'.format(kk), writer='ffmpeg', fps=10)

