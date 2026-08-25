"""Integration tests for the POD_ESN reduced-order model, including sensor placement.

These use a small synthetic travelling-wave 'flow' so no external dataset is needed.
"""
import numpy as np
import pytest

from romda.models.data_driven import POD_ESN


def synthetic_flow(N_t=260, Nx=24, Ny=12):
    """Two-field periodic snapshot data (Nu, N_t, Nx, Ny) with a NaN 'body' mask."""
    t = np.linspace(0, 16 * np.pi, N_t)
    x = np.linspace(0, 2 * np.pi, Nx)
    y = np.linspace(0, np.pi, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    u = (np.sin(X)[None] * np.cos(t)[:, None, None]
         + 0.5 * np.sin(2 * X + Y)[None] * np.sin(2 * t)[:, None, None])
    v = np.cos(X + Y)[None] * np.sin(t)[:, None, None]
    u[:, 2:4, 2:4] = np.nan
    v[:, 2:4, 2:4] = np.nan
    return np.array([u, v])


@pytest.fixture(scope='module')
def pod_esn_case():
    X = synthetic_flow()
    dt = 0.05
    return POD_ESN(data=X,
                   domain=[0, 2 * np.pi, 0, np.pi],
                   N_modes=4,
                   skip_sensor_placement=False,
                   qr_selection=True,
                   down_sample_measurement=[3, 2],
                   t_val=20 * dt,
                   t_train=200 * dt,
                   t_test=0.,
                   perform_test=False,
                   N_units=30,
                   train_ESN=True,
                   N_wash=5,
                   noise=1e-4,
                   N_func_evals=4,
                   N_grid=2,
                   rho_range=(0.2, 0.9),
                   upsample=2,
                   Nq=6,
                   dt=dt,
                   plot_case=False), X


class TestPODESN:

    def test_sensor_locations_are_fluid_grid_points(self, pod_esn_case):
        case, X = pod_esn_case
        Nu, Nx, Ny = case.grid_shape
        locs = np.asarray(case.sensor_locations)
        assert case.Nq == len(locs)
        assert np.all(locs >= 0) and np.all(locs < Nu * Nx * Ny)
        # every sensor must sit on a fluid point
        assert np.all(case.fluid_mask_flat[locs % (Nx * Ny)])

    def test_observables_match_data_at_sensors(self, pod_esn_case):
        """Regression test for the sensor-placement bug: the model observables computed
        from the POD coefficients must equal the (POD-reconstructed) flow at the sensor
        locations — indexed on the RAW grid like the tutorials do."""
        case, X = pod_esn_case

        # observables from the training coefficients at the last snapshot
        Phi_last = case.Phi[:, -1][:, np.newaxis]              # (N_modes, 1)
        y_model = case.get_observables(Phi=Phi_last)           # (Nq, 1)

        # ground truth: raw data at the sensor grid locations (variable-block flattening)
        X_flat = X.transpose(1, 0, 2, 3).reshape(X.shape[1], -1)  # (N_t, Nu*Nx*Ny)
        y_data = X_flat[-1, case.sensor_locations]

        # POD truncation error bound: compare against the N_modes reconstruction instead
        rows = case.grid_index_to_flat_rows(case.sensor_locations)
        y_recon = (case.Psi[rows] @ Phi_last + case.Q_mean[rows]).ravel()

        np.testing.assert_allclose(y_model.ravel(), y_recon, atol=1e-8)
        # and the reconstruction itself must be close to the true data at the sensors
        scale = np.max(abs(X_flat[-1][case.fluid_mask_flat[
            np.arange(X_flat.shape[1]) % (case.grid_shape[1] * case.grid_shape[2])]]))
        assert np.max(abs(y_recon - y_data)) < 0.15 * scale

    def test_forecast_and_observables(self, pod_esn_case):
        case, _ = pod_esn_case
        model = case.copy()
        psi, t = model.time_integrate(Nt=100)
        model.update_history(psi, t)
        assert np.isfinite(psi).all()
        y = model.get_observables()
        assert y.shape[0] == model.Nq
        assert np.isfinite(y).all()

    def test_measure_modes_option(self, pod_esn_case):
        case, _ = pod_esn_case
        model = case.copy()
        model.select_sensors(measure_modes=True)
        assert model.Nq == model.N_modes
        y = model.get_observables()
        assert y.shape[0] == model.N_modes
