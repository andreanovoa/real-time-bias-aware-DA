"""
@defneozan codes on ESN implementation
https://github.com/MagriLab/Adjoint-ESN/blob/main/adjoint_esn/generate_reservoir_weights.py
https://github.com/MagriLab/Adjoint-ESN/blob/main/adjoint_esn/generate_input_weights.py
https://github.com/MagriLab/Adjoint-ESN/blob/main/adjoint_esn/validation.py
"""

import numpy as np

# Reservoir weights generation methods
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs as sparse_eigs

# Input weights generation methods
from scipy.sparse import lil_matrix

# Validation methods
from functools import partial
from itertools import product
import skopt
from skopt.learning import GaussianProcessRegressor as GPR
from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern
from skopt.space import Real


# ==================================== OUTPUT MATRIX ======================================= #
def erdos_renyi1(W_shape, sparseness, W_seeds):
    """Create the reservoir weights matrix according to Erdos-Renyi network
    Args:
        seeds: a list of seeds for the random generators;
            one for the connections, one for the uniform sampling of weights
    Returns:
        W: sparse matrix containing reservoir weights
    """
    # set the seeds
    rnd0 = np.random.RandomState(W_seeds[0])  # connection rng
    rnd1 = np.random.RandomState(W_seeds[1])  # sampling rng

    # initialize with zeros
    W = np.zeros(W_shape)

    # generate a matrix sampled from the uniform distribution (0,1)
    W_connection = rnd0.rand(W_shape[0], W_shape[1])

    # generate the weights from the uniform distribution (-1,1)
    W_weights = rnd1.uniform(-1, 1, W_shape)

    # replace the connections with the weights
    W = np.where(W_connection < (1 - sparseness), W_weights, W)
    # 1-sparseness is the connection probability = p,
    # after sampling from the uniform distribution between (0,1),
    # the probability of being in the region (0,p) is the same as having probability p
    # (this is equivalent to drawing from a Bernoulli distribution with probability p)

    W = csr_matrix(W)

    # find the spectral radius of the generated matrix
    # this is the maximum absolute eigenvalue
    rho_pre = np.abs(sparse_eigs(W, k=1, which="LM", return_eigenvectors=False))[0]

    # first scale W by the spectral radius to get unitary spectral radius
    W = (1 / rho_pre) * W

    return W


def erdos_renyi2(W_shape, sparseness, W_seeds):
    prob = 1 - sparseness

    # set the seeds
    rnd0 = np.random.RandomState(W_seeds[0])  # connection rng
    rnd1 = np.random.RandomState(W_seeds[1])  # sampling rng

    # initialize with zeros
    W = np.zeros(W_shape)
    for i in range(W_shape[0]):
        for j in range(W_shape[1]):
            b = rnd0.random()
            if (i != j) and (b < prob):
                W[i, j] = rnd1.random()

    W = csr_matrix(W)

    # find the spectral radius of the generated matrix
    # this is the maximum absolute eigenvalue
    rho_pre = np.abs(sparse_eigs(W, k=1, which="LM", return_eigenvectors=False))[0]

    # first scale W by the spectral radius to get unitary spectral radius
    W = (1 / rho_pre) * W
    return W


# ==================================== INPUT MATRIX ======================================= #
def sparse_random(W_in_shape, N_param_dim, W_in_seeds):
    """Create the input weights matrix
    Inputs are not connected, except for the parameters

    Args:
        W_in_shape: N_reservoir x (N_inputs + N_input_bias + N_param_dim)
        seeds: a list of seeds for the random generators;
            one for the column index, one for the uniform sampling
    Returns:
        W_in: sparse matrix containing the input weights
    """
    # initialize W_in with zeros
    W_in = lil_matrix(W_in_shape)
    # set the seeds
    rnd0 = np.random.RandomState(W_in_seeds[0])
    rnd1 = np.random.RandomState(W_in_seeds[1])
    rnd2 = np.random.RandomState(W_in_seeds[2])

    # make W_in
    for j in range(W_in_shape[0]):
        rnd_idx = rnd0.randint(0, W_in_shape[1] - N_param_dim)
        # only one element different from zero
        # sample from the uniform distribution
        W_in[j, rnd_idx] = rnd1.uniform(-1, 1)

    # input associated with system's bifurcation parameters are
    # fully connected to the reservoir states
    if N_param_dim > 0:
        W_in[:, -N_param_dim:] = rnd2.uniform(-1, 1, (W_in_shape[0], N_param_dim))

    W_in = W_in.tocsr()

    return W_in


def sparse_grouped(W_in_shape, N_param_dim, W_in_seeds):
    # The inputs are not connected but they are grouped within the matrix

    # initialize W_in with zeros
    W_in = lil_matrix(W_in_shape)
    rnd0 = np.random.RandomState(W_in_seeds[0])
    rnd1 = np.random.RandomState(W_in_seeds[1])

    for i in range(W_in_shape[0]):
        W_in[
            i,
            int(np.floor(i * (W_in_shape[1] - N_param_dim) / W_in_shape[0])),
        ] = rnd0.uniform(-1, 1)

    if N_param_dim > 0:
        W_in[:, -N_param_dim:] = rnd1.uniform(-1, 1, (W_in_shape[0], N_param_dim))

    W_in = W_in.tocsr()
    return W_in


def dense(W_in_shape, W_in_seeds):
    # The inputs are all connected

    rnd0 = np.random.RandomState(W_in_seeds[0])
    W_in = rnd0.uniform(-1, 1, W_in_shape)
    return W_in


def sparse_grouped_rijke(W_in_shape, N_param_dim, W_in_seeds, u_f_order):
    # Sparse input matrix that has the parameter concatenated only with u_f(t-\tau)
    # The different orders of u_f(t-\tau) appear individually (not connected)

    # initialize W_in with zeros
    W_in = lil_matrix(W_in_shape)
    rnd0 = np.random.RandomState(W_in_seeds[0])
    rnd1 = np.random.RandomState(W_in_seeds[1])

    for i in range(W_in_shape[0]):
        W_in[
            i,
            int(np.floor(i * (W_in_shape[1] - N_param_dim) / W_in_shape[0])),
        ] = rnd0.uniform(-1, 1)

    # find the indices of u_f(t-\tau)
    for order in range(u_f_order):
        u_f_idx = np.where(W_in[:, -N_param_dim - (u_f_order - order)].toarray() != 0)[
            0
        ]

        if N_param_dim > 0:
            W_in[u_f_idx, -N_param_dim:] = rnd1.uniform(
                -1, 1, (len(u_f_idx), N_param_dim)
            )

    W_in = W_in.tocsr()
    return W_in


def sparse_grouped_rijke_dense(W_in_shape, N_param_dim, W_in_seeds, u_f_order):
    # Sparse input matrix that has the parameter concatenated only with u_f(t-\tau)
    # The different orders of u_f(t-\tau) are connected

    # initialize W_in with zeros
    W_in = lil_matrix(W_in_shape)
    rnd0 = np.random.RandomState(W_in_seeds[0])
    rnd1 = np.random.RandomState(W_in_seeds[1])

    n_groups = int(
        np.floor(W_in_shape[0] / (W_in_shape[1] - N_param_dim - u_f_order + 1))
    )
    n_sparse_groups = W_in_shape[0] - n_groups
    for i in range(n_sparse_groups):
        W_in[
            i,
            int(
                np.floor(
                    i * (W_in_shape[1] - N_param_dim - u_f_order) / n_sparse_groups
                )
            ),
        ] = rnd0.uniform(-1, 1)

    W_in[n_sparse_groups:, -N_param_dim - u_f_order : -N_param_dim] = rnd0.uniform(
        -1, 1, (n_groups, u_f_order)
    )

    if N_param_dim > 0:
        W_in[n_sparse_groups:, -N_param_dim:] = rnd1.uniform(
            -1, 1, (n_groups, N_param_dim)
        )

    W_in = W_in.tocsr()
    return W_in


# ============================== VALIDATION ======================================= #

def create_search_grid(n_param, n_grid, grid_range):
    """Generates a grid for the given grid ranges and number of grid points
    Works for any number of parameters
    Needs return a list of lists for skopt minimize
    Args:
        n_grid: number of grid points for each parameter
        grid_range: search range for each parameter
    Returns:
        search_grid
    """
    # initialise list of np.arrays with the range of each parameter
    param_grid = [None] * n_param
    for param_idx in range(n_param):
        param_grid[param_idx] = np.linspace(*grid_range[param_idx], n_grid[param_idx])

    # create the search grid from the parameter grids
    search_grid = product(*param_grid, repeat=1)

    # itertools.product returns a list of tuples
    # turn it to a list of lists
    search_grid = [list(search_list) for search_list in search_grid]

    return search_grid


def create_search_space(n_param, grid_range, param_names):
    search_space = [None] * n_param
    for param_idx in range(n_param):
        search_space[param_idx] = Real(
            *grid_range[param_idx], name=param_names[param_idx]
        )
    return search_space


def run_gp_optimization(gp_kernel, val_func, search_space, search_grid, n_total, n_initial):
    # Gaussian Process reconstruction
    b_e = GPR(
        kernel=gp_kernel,
        normalize_y=True,  # if true mean assumed to be equal to the average of the obj function data, otherwise =0
        n_restarts_optimizer=3,  # number of random starts to find the gaussian process hyperparameters
        noise=1e-10,  # only for numerical stability
        random_state=10,
    )  # seed

    # Bayesian Optimization
    res = skopt.gp_minimize(
        val_func,  # the function to minimize
        search_space,  # the bounds on each dimension of params
        base_estimator=b_e,  # GP kernel
        acq_func="EI",  # the acquisition function
        n_calls=n_total,  # total number of evaluations of f
        x0=search_grid,  # initial grid search points to be evaluated at
        n_random_starts=n_initial,  # the number of additional random initialization points
        n_restarts_optimizer=3,  # number of tries for each acquisition
        random_state=10,  # seed
        # acq_optimizer="lbfgs",
        # n_jobs=-1,  # number of cores to use
    )
    return res


def set_ESN(my_ESN, param_names, param_scales, params):
    # set the ESN with the new parameters
    for param_name in set(param_names):
        # get the unique strings in the list with set
        # now the indices of the parameters with that name
        # (because ESN has attributes that are set as arrays and not single scalars)
        param_idx_list = np.where(np.array(param_names) == param_name)[0]

        new_param = np.zeros(len(param_idx_list))
        for new_idx in range(len(param_idx_list)):
            # rescale the parameters according to the given scaling
            param_idx = param_idx_list[new_idx]
            if param_scales[param_idx] == "uniform":
                new_param[new_idx] = params[param_idx]
            elif param_scales[param_idx] == "log10":
                new_param[new_idx] = 10 ** params[param_idx]

        if len(param_idx_list) == 1:
            new_param = new_param[0]

        setattr(my_ESN, param_name, new_param)
    return


def RVC(
    case,
    my_ESN,
    U_washout,
    U_tv,
    Y_tv,
    N_init_steps,
    N_fwd_steps,
    N_wash,
    N_val_steps,
    N_transient_steps=0,
    tikh_range=[1e-12],
    tikh_hist=None,
    print_flag=False
):
    """Recycle cross validation method from
    Racca, A., Magri, L. (2021). Robust Optimization and Validation of Echo State Networks for
    learning chaotic dynamics. Neural Networks 142: 252-268.
    """

    # first train ESN with the complete data
    X_augmented = case.reservoir_for_train(U_washout, U)

    # train for different tikhonov coefficients since the input data will be the same,
    # we don't rerun the open loop multiple times just to train with different tikhonov

    W_out_list = [None] * len(tikh_range)
    for tikh_idx, tikh in enumerate(tikh_range):
        W_out_list[tikh_idx] = my_ESN.solve_ridge(X_augmented, Y, tikh)

    # save the MSE error with each tikhonov coefficient over all folds

    mse_sum = np.zeros(len(tikh_range))
    for fold in range(case.N_folds):
        # select washout and validation
        N_steps = N_init_steps + fold * N_fwd_steps
        U_washout_fold = U[N_steps : N_wash + N_steps].copy()
        Y_val = U[N_wash + N_steps + 1 : N_wash + N_steps + N_val_steps
        ].copy()

        # run washout before closed loop
        x0_fold = my_ESN.run_washout(U_washout_fold)

        for tikh_idx in range(len(tikh_range)):
            # set the output weights
            my_ESN.output_weights = W_out_list[tikh_idx]

            # predict output validation in closed-loop
            _, Y_val_pred = my_ESN.closed_loop(x0_fold, N_val_steps - 1)
            Y_val_pred = Y_val_pred[1:, :]

            # add the mse error with this tikh in log10 scale
            mse_sum[tikh_idx] += np.log10(
                np.mean(
                    (
                        Y_val[N_transient_steps:, :]
                        - Y_val_pred[N_transient_steps:, :]
                    )
                    ** 2
                )
            )

        # find the mean mse over folds
        mse_mean = mse_sum / n_folds

    # select the optimal tikh
    tikh_min_idx = np.argmin(mse_mean)
    tikh_min = tikh_range[tikh_min_idx]
    mse_mean_min = mse_mean[tikh_min_idx]

    # if a tikh hist is provided append to it
    if tikh_hist is not None:
        tikh_hist.append(tikh_min)

    if print_flag:
        print("log10(MSE) = ", mse_mean_min)

    return mse_mean_min


def RVC_Noise(x, case, U_wash, U_tv, Y_tv, tikh_opt, hp_names):

    # Re-set hyperparams as the optimization goes on
    case.reset_hyperparams(x, hp_names)

    # chaotic Recycle Validation
    N_tikh = len(case.tikh_range)
    Mean = np.zeros(N_tikh)

    N_in = 0  # interval before the first fold
    N_fw = (case.N_train - case.N_val) // (case.N_folds - 1)  # num steps forward the validation interval is shifted

    # Train using tv: training+val, Wout is passed with all the combinations of tikh_ and target noise

    # This must result in L-Xa timeseries

    R_train, LHS, RHS = case.computeRRterms(U_wash, U_tv, Y_tv)
    Wout = np.empty((N_tikh, case.N_units + 1, case.N_dim))


    for tik_j in range(N_tikh):
        if tik_j == 0:   # add tikhonov to the diagonal (fast way that requires less memory)
            LHS.ravel()[::LHS.shape[1]+1] += case.tikh_range[tik_j]
        else:
            LHS.ravel()[::LHS.shape[1]+1] += case.tikh_range[tik_j] - case.tikh_range[tik_j-1]

        Wout[tik_j] = np.linalg.solve(LHS, RHS)

    if case.val_k == 0:
        print('\t', end="")
        for kk in case.optimize_hyperparams:
            print('\t {}'.format(kk), end="")
        print('\t MSE val ')

    for kk in range(case.L):
        # Different validation folds
        for fold in range(case.N_folds):
            p = N_in + fold * N_fw

            print(p, U_tv.shape)

            # data to compare the closed-loop prediction with
            U_fold = U_tv[kk][p]
            R_fold = R_train[kk][p]
            Y_val = U_tv[kk, p+case.N_wash: p+case.N_wash + case.N_val].copy()


            for tik_j in range(N_tikh):  # cloop for each tikh_-noise combinatio
                case.Wout = Wout[tik_j]
                Yh_val = case.closedLoop(case.N_val - 1, reservoir_state=(U_fold, R_fold))[0]

                print(Y_val.shape, Yh_val.shape)

                Mean[tik_j] += np.log10(np.mean((Y_val - Yh_val) ** 2) / np.mean(case.norm ** 2))

                # import matplotlib.pyplot as plt
                # plt.figure()
                # plt.plot(Y_val, '-x')
                # plt.plot(Yh_val, '-o')
                # plt.plot(U_fold, '-*')
                # plt.show()


                # prevent from diverging to infinity: put MSE equal to 10^10 (useful for hybrid and similar
                # architectures)
                if np.isnan(Mean[tik_j]) or np.isinf(Mean[tik_j]):
                    Mean[tik_j] = 10 * case.N_folds

    # select and save the optimal tikhonov and noise level in the targets
    a = Mean.argmin()
    tikh_opt[case.val_k] = case.tikh_range[a]

    case.val_k += 1
    print(case.val_k, end="")
    for hp in hp_names:
        print('\t {:.3e}'.format(getattr(case, hp)), end="")
    print('\t {:.4f}'.format(Mean[a] / case.N_folds / case.L))

    return Mean[a] / case.N_folds / case.L


