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
Nt_train = 100
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
# # Part I. Apply POD to the dataset with `tools.POD` <a name="part1"></a>
# We apply POD to the velocity fielsd $u=[U_x; U_y]$ only because the flow is incompressible. This is because the velocity components carry most of the total kinetic energy (TKE) in a turbulent or unsteady flow.
# The pressure field does not directly contribute to TKE but rather acts as a constraint on the velocity field via the Navier-Stokes equations, specifically $P$ is often computed indirectly from the incompressible constraint ($\nabla\cdot u$) and it exhibits different spatial structures compared to velocity
# 
# 


# %% [markdown]
# ## Initialize POD instance <a name="part1-1"></a>
# 


from tools import POD

case_pod = POD(X = X_train, 
               N_modes=4,
              domain=[0, 12, -2.5, 2.5])

# %%

# %% [markdown]
# ## POD eigenvalues and the POD basis <a name="part1-3"></a> 
# The eigenvalues contain the kinetic energy of the flow. With only 400 snapshots the POD decomposition is not yet converged as the two lines do not match. 

# %%
from plotting.pod import plot_spectrum, plot_modes

plot_spectrum(case=case_pod, max_mode=20)
plot_modes(case=case_pod, num_modes=10)


# %% [markdown]
# ## Flow reconstuction <a name="part1-4"></a>
# 
# We can see that most of the energy is concentrated on the first two modes, reaching over 95% of the TKE of the system. Let's now visualize the reconstruction for different number of modes. 

# %%

_case = case_pod.copy() 

N_modes = _case.N_latent
datasets = [X_train[:, -1]]


names = ['Original data']

for N_modes in [6, 2]:
    # POD case reconstruction
    _case.truncate(n_modes=N_modes)

    Q = _case.reconstruct(X_train[:, -1])
    X = _case._to_physical_grid(Q)

    
    datasets.append(X[:,0])
    names.append(f'{N_modes} modes')
    


# %%
from plotting.pod import plot_flows_rms


reconstructed_data = case_pod.reconstruct(X_train[:, -1])
reconstructed_data = case_pod._to_physical_grid(reconstructed_data)[:,0]

plot_flows_rms(case_pod, 
               reconstructed_data=reconstructed_data,
                   datasets=datasets, 
                   names=names, 
                   display_RMS='all');



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
N_train = X_train.shape[-1]
dt = 0.01  # time step size

case_ESN = POD_ESN(data=X_train, 
                   dt=dt,
                   N_modes=4, 
                   domain  = [-2, 2, 0, 12],
                   # ====== ESN arguments ======== # 
                   train_ESN=True,
                   N_train=.8* N_train,
                   N_val=.2*N_train,
                   N_wash=10,
                   noise=0.1,
                   N_func_evals=26,
                   rho_range=[0.2, 0.9],
                   upsample=2,
                   run_test=False,
                   # ====== Plotting flags ======== # 
                   plot_case=False
                  )
case_ESN_og = case_ESN.copy()

# %% [markdown]
# ## 5. Create Observations from test flowfields
# Project the true and noisy test flow fields onto the trained POD modes,
# then wrap them in an `Observations` object for downstream data assimilation.

# %%
from observations import Observations

Nt_test = X_test.shape[1]
t_test  = np.arange(Nt_test) * dt

# Flatten test flow fields onto fluid points using the same NaN mask as training
# (subtract_mean=False because project_data_onto_Psi subtracts the training mean)
Q_test_true,  _, _ = prepare_data([*X_test_true], subtract_mean=False)
Q_test_noisy, _, _ = prepare_data([*X_test],      subtract_mean=False)

# Project onto POD modes → temporal coefficients (Nt_test, N_modes)
Phi_test_true  = case_ESN.project_data_onto_Psi(Q_test_true,  remove_mean=True)
Phi_test_noisy = case_ESN.project_data_onto_Psi(Q_test_noisy, remove_mean=True)

obs = Observations(
    model=None,
    y_true=Phi_test_true,
    y_raw=Phi_test_noisy,
    t_true=t_test,
    t_start=t_test[0],
    t_stop=t_test[-1],
    Nt_obs=20,
)

print(f'y_obs shape : {obs.y_obs.shape}  ({len(obs.t_obs)} observation times)')


# %% [markdown]
# ## 3. Verify implementation <a name="part2-3"></a>
# Check that the POD and POD_ESN cases give the same reconstruction.
# The small diferences are due to the lack of convergence of the POD, as the amount of data used is different. 

# %%
import matplotlib.pyplot as plt

# POD-ESN reconstruct
Q_esn=case_ESN.reconstruct()


# POD reconstruct
_case = case_pod.copy()
_case.rerun_POD_decomposition(N_modes=case_ESN.N_modes)
Q_ref = _case.reconstruct(Phi=_case.Phi[[len(case_ESN.Phi)-1]])

# Differnce
diff = abs(Q_ref - Q_esn) / np.max(Q_ref, axis=(1,2), keepdims=True) * 100

fig, axs = plt.subplots(2, 3, figsize=(10, 4), sharex=True, sharey=True, layout='constrained')
for D, col, cmap, tlt in zip([Q_esn, Q_ref, diff], axs.T, ['viridis', 'viridis', 'Reds'], ['Q_POD-ESN', 'Q_POD', 'difference [%]']):
    for d, ax in zip(D, col):
        im=ax.imshow(d[...,0], cmap=cmap)  
    fig.colorbar(im, ax=col, shrink=.5, orientation='horizontal')
    col[0].set(title=tlt)




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
reduced_state = case_ESN.project_data_onto_Psi(data=test_data_up[..., 0] - case_ESN.Q_mean)
case_ESN.state = reduced_state


state, time = case_ESN.time_integrate(Nt=len(t_true))





# %%

case_ESN.update_history(state, time)

# Get reconstruction
Phi_ESN = case_ESN.get_POD_coefficients(Nt=-0).squeeze()
prediction = case_ESN.reconstruct(Phi=Phi_ESN[1:])


# Compute the mean square error 
MSE_evolution = POD.compute_MSE(prediction, test_data_up, time_evolution=True)
MSE_evolution_true = POD.compute_MSE(prediction, test_data_up_true, time_evolution=True)
plt.figure(figsize=(5,3))
plt.plot(time, np.log10(MSE_evolution))
plt.plot(time, np.log10(MSE_evolution_true), '--')
plt.gca().set(xlabel='time', ylabel='log$_{10}$(MSE)')
plt.legend(['Noisy', 'True'])

# %%

# Visualize the flow field reconstruction
RMS_noisy = np.array([POD.compute_RMS(test_data_up[...,ti], 
                                          prediction[...,ti]) for ti in range(prediction.shape[-1])])
RMS_true = np.array([POD.compute_RMS(test_data_up_true[...,ti], 
                                      prediction[...,ti]) for ti in range(prediction.shape[-1])])

U_datasets = [test_data_up_true[0].transpose(2,1,0),
              test_data_up[0].transpose(2,1,0),
              prediction[0].transpose(2,1,0), 
              RMS_noisy[:,0].transpose(0,2,1),
              RMS_true[:,0].transpose(0,2,1),]

anim = animate_flowfields(U_datasets, 
                          titles=['Truth', 'Noisy data', 'POD-ESN', 'RMS noisy', 'RMS true'], 
                          n_frames=10, figsize=(10, 6)  )
# anim

# %%
HTML(anim.to_jshtml())

# %% [markdown]
# The POD-ESN recovers the true field from a noisy dataset, effectively acting as a de-noiser. Can we achieve this at higher noise levels? How about spatially smoothed or coloured noise? 


