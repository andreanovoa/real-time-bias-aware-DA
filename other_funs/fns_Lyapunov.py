import numpy as np
import numpy as np
import os
import matplotlib.pyplot as plt
os.environ["OMP_NUM_THREADS"] = "1"


def QR(M, N_exp):
    """ Compute an orthogonal basis, Q, and the exponential change
        in the norm along each element of the basis, S.
    """

    Q = [None] * N_exp
    S = np.empty(N_exp)

    S[0] = np.linalg.norm(M[0])
    Q[0] = M[0] / S[0]

    for i in range(1, N_exp):

        # orthogonalize
        temp = 0
        for j in range(i):
            temp += np.dot(Q[j], M[i]) * Q[j]
        Q[i] = M[i] - temp

        # normalize
        S[i] = np.linalg.norm(Q[i])  # increase of the perturbation along i-th direction
        Q[i] /= S[i]

    return Q, np.log(S)

def Lorenz(q):
    ''' Right hand side of the Lorenz equations '''
    x, y, z = q
    sigma, beta, rho = 10.0, 8.0/3, 28.0
    dqdt = [sigma*(y-x), x*(rho-z) - y, x*y - beta*z]

    return np.array(dqdt)


def VdP(psi):
    eta, mu = psi[:2]
    omega, nu, kappa, gamma, beta  = 2 * np.pi * 120., 7., 3.4, 1.7, 70.
    law = 'tan'

    deta_dt = mu
    dmu_dt = - omega ** 2 * eta
    if law == 'cubic':  # Cubic law
        dmu_dt += mu * (2. * nu - kappa * eta ** 2)
    elif law == 'tan':  # arc tan model
        dmu_dt += mu * (beta ** 2 / (beta + kappa * eta ** 2) - beta + 2 * nu)
    else:
        raise TypeError("Undefined heat release law. Choose 'cubic' or 'tan'.")
        # dmu_dt  +=  mu * (2.*P['nu'] + P['kappa'] * eta**2 - P['gamma'] * eta**4) # higher order polinomial
    return np.hstack([deta_dt, dmu_dt])


def RK4(q0,dt,N,func):
    ''' 4th order explicit Tunge-Kutta integration method '''

    for i in range(N):

        k1   = dt * func(q0)
        k2   = dt * func(q0 + k1/2)
        k3   = dt * func(q0 + k2/2)
        k4   = dt * func(q0 + k3)

        q0   = q0 + (k1 + 2*k2 + 2*k3 + k4)/6

    return  q0

def FE(q0,dt,N,func):
    ''' 1st order Forward Euler method'''

    for i in range(N):
        q0   = q0 + dt * func(q0)

    return  q0

if __name__ == '__main__':
    ##### Initialize

    N_dim = 2  # Degrees of freedom of the system
    N_exp = 2  # Number of exponents to compute

    dt = 2e-4  # timestep

    N_orth = 100  # number of steps before orthonormalization of the basis
    N = int(40. / dt)  # length of time series
    N_times = N // N_orth  # total number of orthonormalizations

    integrator = RK4  # integration scheme
    system = VdP  # Right hand side of the governing equations

    N_transient = int(3. / dt)  # integration points to reach the attractor


    q0 = np.random.rand(N_dim) #[0.1, 0.1] # random initial condition
    q0 = integrator(q0, dt, N_transient, system)  # unperturbed initial condition on the attractor

    eps = 1.e-9  # multiplication factor to make the orthonormalized perturbation infinitesimal

    SS = np.empty((N_times, N_exp))  # initialize lyapunov exponents
    x0P = []  # initialize perturbations
    for ii in range(N_exp):
        q_per = np.random.rand(N_dim)
        x0P.append(q0 + eps * (q_per / np.linalg.norm(q_per)))  # N_exp randomly perturbed initial conditions


    #### Compute lyapunov exponents

    S = 0
    for jj in range(N_times):

        q0 = integrator(q0, dt, N_orth, system)  # compute unperturbed trajectory
        for ii in range(N_exp):
            x0P[ii] = integrator(x0P[ii], dt, N_orth, system)  # compute perturbed trajectories

        a = [(x - q0) / eps for x in x0P]  # compute the final value of the N_exp perturbations

        aa, S1 = QR(a, N_exp)  # orthornormalize basis and compute exponents

        x0P = [q0 + eps * a for a in aa]  # perturb initial condition with orthonormal basis

        if jj > 0:  # skip the first step, which does not start from the orthonormalized basis
            S += S1
            SS[jj] = S / (jj * dt * N_orth)
            if jj % (N_times // 10) == 0:
                print('Lyapunov exponents, completion percentage:', SS[jj], jj / N_times)

    ## Compute Kaplan-Yorke dimension
    Lyap_exp = SS[-1]
    print('Lyapunov exponents      ', Lyap_exp)

    if Lyap_exp.sum() > 0:
        print('Error: not enough exponents have been computed. Increase N_exp to compute KY-dimension')

    else:
        sums = np.cumsum(Lyap_exp)  # cumulative sum of the Lyapunov exponents
        arg = np.argmax(sums < 0)  # index for which the cumulative sum becomes negative

        KY_dim = arg + sums[arg - 1] / np.abs(Lyap_exp[arg])

        print('Kaplan-Yorke dimension  ', KY_dim)


    ### Plot convergence of the exponents
    plt.rcParams["figure.figsize"] = (10, 5)
    plt.rcParams["font.size"] = 25

    plt.plot(np.arange(N_times) * dt * N_orth, SS)
    plt.xlabel('Time')
    plt.ylabel('Lyapunov Exponents')
    plt.tight_layout(pad=0.2)
    plt.show()