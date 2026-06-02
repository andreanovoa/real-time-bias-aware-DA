# %% [markdown]
# <script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=default'></script>
# 
# # Tutorial: POD-ESN model of a 2D cylinder flow
# 
# The aim of a `POD-ESN` object is to create a reduced order model of a dataset (2D velocity fields in this case) by first, performing proper orthogonal decomposition (POD) on the data and then train an echo state network (ESN) to learn the temporal evolution of the POD coefficients. 
# 
# The class `POD-ESN` from `src.models_datadriven` combines two parent classes:
# 1. `POD` $\rightarrow$ tutorial `03_Class POD` gives further details on this object. 
# 2. `ESN_model` $\rightarrow$ tutorial `03_Class ESN_model` gives further details on this object.
# 
# <br>
# 
# 
# This notebook is divided in two parts: 
# * [Part I. Apply POD to the dataset with `tools.POD`](#part1)
# * [Part II. Reduced order modelling via POD-ESN from `src.models_datadriven.POD-ESN`](#part2)
# 
# First, let's load and visualize the dataset we will be working with. 
# 

# %%
from copy import deepcopy

from utils import set_working_directories, load_from_mat_file, get_wake_data
import numpy as np
from utils import animate_flowfields


# Define woking directories
data_folder, results_folder, figs_folder = set_working_directories('wakes')

get_wake_data(data_folder)


# Load dataset
mat = load_from_mat_file(data_folder + '/circle_re_100.mat')

ux = mat['ux']   # (N_t, Nx, Ny) — NaN marks cylinder body interior
uy = mat['uy']

N_t, Nx, Ny = ux.shape
print(f'Snapshots  : N_t = {N_t}')
print(f'Grid       : {Nx} × {Ny}')


# %% [markdown]
# ## 1. Data preparation <a name="part1-1"></a>
# - Add noise to the data to mimic PIV measurements.
# - Split the data, i.e., divide the dataset into train and test data
# - Prepare the dataset for dimensionality reduction using 
# 
#         prepare_data():
#             - detects the NaN mask (cylinder interior) from the first snapshot
#             - flattens fluid points to (N_fluid, N_t) and subtracts the temporal mean
#             - returns to_grid() for re-embedding flat vectors onto the 2-D mesh
# 

# %%
from utils import add_noise_to_flow


# Stack the velocity components together and Split into training and test sets
Nt_train = 300
data_true = np.array([ux, uy])

data_noisy = add_noise_to_flow(data_true, noise_level=0.1, noise_type="gauss", spatial_smooth=0.1)

X_test_true = data_true[:, Nt_train:]
X_train_true = data_true[:, :Nt_train]

X_train = data_noisy[:, :Nt_train]
X_test  = data_noisy[:, Nt_train:]

# %%


# %% ── visualize flow fields ────────────────────────────────────────────────────────────────


from IPython.display import HTML

# Visualize the flow fields
datasets = {
    '$u_x$': X_train_true[0].T,
    '$u_x$ noisy': X_train[0].T,
    '$u_y$': X_train_true[1].T,
    '$u_y$ noisy': X_train[1].T,
}
# anim = animate_flowfields(datasets, n_frames=10, step=2, figsize=(8, 6))
# HTML(anim.to_jshtml())


# %% [markdown]
# # Part II. Reduced order modelling via POD-ESN from `src.models_datadriven`<a name="part2"></a>
# 
# 
# We have seen that with 4 modes we reconstruct over 95% of the system's energy, which provides a very small reconstruction RMS. Therefore, we select N_modes = 4 for our reduced order model. 
# 
# 

# %% [markdown]
# 
# ## 2. Initilize POD_ESN instance <a name="part2-2"></a>
# - The data provided is used to perform the POD decomposition as well as the ESN training on the POD coefficients.
# - If the flag ```train_ESN=False```, only the POD is perfromed and the ESN is not trained

# %%
from models.data_driven import POD_ESN

help(POD_ESN)

# %%
X_train.shape

# %%
N_train = X_train.shape[1]
dt = 0.01  # time step size

case_ESN = POD_ESN(data=X_train, 
                   dt=dt,
                   n_modes=4, 
                   domain  = [0, 12, -2, 2],
                   # ====== ESN arguments ======== # 
                   train_ESN=True,
                   t_train=.7* N_train*dt,
                   t_val=.1*N_train*dt,
                   N_wash=10,
                   noise=0.1,
                   N_func_evals=26,
                   rho_range=[0.2, 0.9],
                   upsample=1,
                   perform_test=True,
                   domain_of_measurement=[5, 10, -1, 1],
                   down_sample_measurement=5,
                   # ====== Plotting flags ======== # 
                   qr_selection=0,
                   plot_case=True,
                   Nq=2,
                  )
case_ESN_og = case_ESN.copy()

# %% [markdown]



# %% [markdown]
# ## 3. Verify implementation <a name="part2-3"></a>
# Check that the POD and POD_ESN cases give the same reconstruction.
# The small diferences are due to the lack of convergence of the POD, as the amount of data used is different. 

# %%
import matplotlib.pyplot as plt

Ntest = X_test.shape[1]
case_ESN = case_ESN_og.copy()
# POD-ESN fiorecast of the POD coefficients
state, time = case_ESN.time_integrate(Nt=Ntest)
case_ESN.update_history(state, time)

phi = case_ESN.get_POD_coefficients(Nt=Ntest)

plt.figure(figsize=(5,3))
plt.plot(time, state[:, 0, 0], label='POD-ESN')
plt.plot(time, phi[:, 0, 0], label='POD-ESN')
plt.gca().set(xlabel='time', ylabel='POD coeff 1')
plt.legend()    

# %%
# washout phase

case_ESN = case_ESN_og.copy()

phi_test = case_ESN.encode(X_test)
Nwash = 100

x_wash = phi_test[:, :Nwash]

case_ESN = case_ESN_og.copy()
r_open = np.zeros((case_ESN.N_units,))

for u_in in x_wash.T:
    u_open, r_open = case_ESN._single_step(u_in, r_open)

psi = case_ESN.build_psi(u_open, r_open)
case_ESN.update_history(psi, reset=True)


# %%
# POD coeffs of test data

phi_test = phi_test[:, Nwash:]
phi_test_true = case_ESN.encode(X_test_true)[:, Nwash:]


# POD-ESN fiorecast of the POD coefficients
state, time = case_ESN.time_integrate(Nt=Ntest-Nwash)
case_ESN.update_history(state, time)
phi = case_ESN.get_POD_coefficients(Nt=Ntest-Nwash)


plt.figure(figsize=(5,3))
plt.plot(time, phi[:, 0, 0], label='POD-ESN')
plt.plot(time, phi_test[0], '--', label='POD test data')
plt.plot(time, phi_test_true[0], '-.', label='POD true data')
plt.gca().set(xlabel='time', ylabel='POD coeff 1')
plt.legend()



# %%

# %% [markdown]
# ## 4. Test forecast of the POD-ESN <a name="part2-4"></a>
# We can visualize:
# - The temporal evolution of the POD coefficients
# - The reconstruction of the flow and its MSE 

# %%
X_test.shape

# %%

U_test = X_test.reshape(case_ESN.Q_mean.shape[0], -1) - case_ESN.Q_mean

# Compare the POD coefficients
Phi_true = case_ESN.project_data_onto_Psi(data=U_test)
# Add noise to the test data
Phi_noisy =  Phi_true.copy()
rng_noise = np.random.default_rng(0)
U_std = np.std(Phi_true, axis=0)
for dd in range(Phi_true.shape[1]):
    Phi_noisy[:, dd] += rng_noise.normal(loc=0, scale=0.1 * U_std[dd], size=Phi_true.shape[0])

# _case = case_ESN.copy()

# _case.run_test(U_test=Phi_noisy[::case_ESN.upsample], 
#                   Y_test=Phi_true[::case_ESN.upsample], 
#                   Nt_test=len(Phi_noisy), 
#                   max_L_tests=1,
#                  )

# %%
case_ESN = case_ESN_og.copy()

# === Upsample reference data ===
# t_true = np.arange(X_test.shape[-1]) * case_ESN.dt
# test_data_up = X_test[..., ::case_ESN.upsample]
# test_data_up_true = X_test_true[..., ::case_ESN.upsample]

t_true = np.arange(X_test.shape[-1]) * case_ESN.dt
test_data_up = X_test.copy()
test_data_up_true = X_test_true.copy()


# === Get ESN prediction on the same time ===
# initialize the ESN state with the first test data point
reduced_state = case_ESN.encode(test_data_up[:, 0])
case_ESN.state = reduced_state


state, time = case_ESN.time_integrate(Nt=len(t_true))





# %%

case_ESN.update_history(state, time)

# Get reconstruction
Phi_ESN = case_ESN.get_POD_coefficients(Nt=-0).squeeze()
prediction = case_ESN.decode(Phi_ESN[1:].T)
prediction = case_ESN._to_physical_grid(prediction)


# # Compute the mean square error 
# plt.figure(figsize=(5,3))
# plt.plot(time, np.log10(MSE_evolution))
# plt.plot(time, np.log10(MSE_evolution_true), '--')
# plt.gca().set(xlabel='time', ylabel='log$_{10}$(MSE)')
# plt.legend(['Noisy', 'True'])


#%% plto

datasets = {
    '$u_x$ true': test_data_up_true[:, -10],
    '$u_x$ noisy': test_data_up[:, -10],
    '$u_x$ POD-ESN': prediction[:, -10],
}

from plotting.pod import plot_flows_rms

plot_flows_rms(case_ESN, datasets)

# %%
