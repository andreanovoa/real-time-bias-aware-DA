"""Tests for the POD / SPOD decompositions (models.data_driven.autoencoders: pod, pod_utils)."""
import numpy as np
import pytest

from romda.models.data_driven.autoencoders import POD, SPOD, spod_towne
from romda.models.data_driven.autoencoders.pod_utils import snapshot_pod, snapshot_pod_randomized, spod_sieber


def synthetic_flow(N_t=200, Nx=24, Ny=12, noise=0.0, seed=0):
    """Synthetic 2-field snapshot data (Nu, N_t, Nx, Ny) with a NaN 'body' mask
    and two travelling-wave modes."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 8 * np.pi, N_t)
    x = np.linspace(0, 2 * np.pi, Nx)
    y = np.linspace(0, np.pi, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    u = (np.sin(X)[None] * np.cos(t)[:, None, None]
         + 0.5 * np.sin(2 * X + Y)[None] * np.sin(2 * t)[:, None, None])
    v = (np.cos(X + Y)[None] * np.sin(t)[:, None, None])
    if noise > 0:
        u = u + noise * rng.standard_normal(u.shape)
        v = v + noise * rng.standard_normal(v.shape)

    # mask a small solid body
    u[:, 2:4, 2:4] = np.nan
    v[:, 2:4, 2:4] = np.nan
    return np.array([u, v])  # (2, N_t, Nx, Ny)


class TestPOD:

    def test_fit_exact(self):
        X = synthetic_flow(noise=0.02)  # noise makes the data full-rank
        pod = POD(n_modes=10, method='exact').fit(X)
        assert pod.fitted
        assert pod.Psi.shape[1] == 10
        assert pod.Phi.shape == (10, X.shape[1])
        assert pod.Sigma.shape == (10,)
        # orthonormal spatial modes
        np.testing.assert_allclose(pod.Psi.T @ pod.Psi, np.eye(10), atol=1e-6)
        # singular values descending
        assert np.all(np.diff(pod.Sigma) <= 1e-12)

    def test_low_rank_data_is_captured(self):
        """The synthetic flow has 3 coherent structures: 4 modes capture >99% energy."""
        X = synthetic_flow()
        pod = POD(n_modes=8, method='exact').fit(X)
        _, cum = pod.energy_fraction()
        assert cum[3] > 0.99

    def test_encode_decode_roundtrip(self):
        X = synthetic_flow()
        pod = POD(n_modes=20, method='exact').fit(X)
        Q = pod.preprocess_snapshot(X)
        Z = pod.encode(X)
        np.testing.assert_allclose(Z, pod.Phi, atol=1e-8)
        Q_hat = pod.decode(Z) - pod.Q_mean
        # low-rank data: 20 modes should reconstruct nearly exactly
        assert np.mean((Q - Q_hat) ** 2) < 1e-10 * np.mean(Q ** 2) + 1e-12

    def test_randomized_close_to_exact(self):
        X = synthetic_flow(noise=0.01)
        pod_e = POD(n_modes=4, method='exact').fit(X)
        pod_r = POD(n_modes=4, method='randomized', random_state=0).fit(X)
        # the leading (energetic) singular values must agree
        np.testing.assert_allclose(pod_e.Sigma[:3], pod_r.Sigma[:3], rtol=5e-2)

    def test_truncate(self):
        X = synthetic_flow()
        pod = POD(n_modes=10, method='exact').fit(X)
        pod.truncate(3)
        assert pod.N_latent == 3
        assert pod.Psi.shape[1] == 3
        assert pod.Phi.shape[0] == 3

    def test_unfitted_access_raises(self):
        pod = POD(n_modes=3)
        with pytest.raises(AttributeError):
            _ = pod.Psi

    def test_flat_input_fit(self):
        """POD can be fitted directly on a flat (N_x, N_t) data matrix."""
        rng = np.random.default_rng(5)
        Q_raw = rng.normal(size=(30, 80)) + 3.
        pod = POD(n_modes=5, method='exact').fit(Q_raw)
        assert pod.Psi.shape == (30, 5)
        Z = pod.encode(Q_raw)
        np.testing.assert_allclose(Z, pod.Phi, atol=1e-8)

    def test_variable_block_ordering(self):
        """Rows [0:N_fluid] of the flat representation must be the FIRST field:
        regression test for the documented Psi_ux = Psi[:N_fluid] convention."""
        X = synthetic_flow()
        X[1] = 0. * X[1]  # kill the second field (keep its NaN mask)
        X[1, :, 2:4, 2:4] = np.nan
        prep = POD()
        Q = prep.preprocess_snapshot(X, subtract_mean=False)
        N_fluid = int(prep.fluid_mask_flat.sum())
        assert np.any(Q[:N_fluid] != 0.)                       # first block: u field
        np.testing.assert_allclose(Q[N_fluid:], 0.)            # second block: zeroed field

    def test_grid_index_to_flat_rows_roundtrip(self):
        """decode(idx=grid_index_to_flat_rows(...)) must return the data at those points."""
        X = synthetic_flow()
        pod = POD(n_modes=30, method='exact').fit(X)
        Nu, Nx, Ny = pod.grid_shape

        fluid_idx = np.flatnonzero(pod.fluid_mask_flat)
        picks = fluid_idx[[3, 17, 101]]
        grid_idx = np.concatenate([picks, picks + Nx * Ny])   # same points in both fields

        rows = pod.grid_index_to_flat_rows(grid_idx)
        Q = pod.preprocess_snapshot(X)
        rec = pod.decode(pod.Phi, idx=rows) - pod.Q_mean[rows]
        np.testing.assert_allclose(rec, Q[rows], atol=1e-8)

        # non-fluid points must be rejected
        solid = np.flatnonzero(~pod.fluid_mask_flat)[:1]
        with pytest.raises(ValueError):
            pod.grid_index_to_flat_rows(solid)


class TestSPOD:

    def test_nf0_recovers_pod(self):
        X = synthetic_flow(noise=0.02)
        pod = POD(n_modes=6, method='exact').fit(X)
        spod = SPOD(Nf=0, n_modes=6).fit(X)
        np.testing.assert_allclose(pod.Sigma, spod.Sigma, rtol=1e-8)

    def test_filtered_fit(self):
        X = synthetic_flow(noise=0.05)
        spod = SPOD(Nf=10, filter_kind='gaussian', n_modes=6).fit(X)
        assert spod.Psi.shape[1] == 6
        assert np.isfinite(spod.Sigma).all()
        assert hasattr(spod, 'C_tilde')


class TestFunctionalAPI:

    def test_snapshot_pod_reconstruction(self):
        rng = np.random.default_rng(1)
        Q = rng.normal(size=(30, 50))
        Q -= Q.mean(axis=1, keepdims=True)
        Sigma, Psi, Phi, C = snapshot_pod(Q)
        np.testing.assert_allclose(Psi @ Phi, Q, atol=1e-8)

    def test_randomized_pod_shapes(self):
        rng = np.random.default_rng(2)
        Q = rng.normal(size=(40, 60))
        Sigma, Psi, Phi = snapshot_pod_randomized(Q, n_modes=5, random_state=0)
        assert Psi.shape == (40, 5)
        assert Phi.shape == (5, 60)

    def test_spod_sieber_nf0_equals_pod(self):
        rng = np.random.default_rng(3)
        Q = rng.normal(size=(20, 40))
        Q -= Q.mean(axis=1, keepdims=True)
        S0 = snapshot_pod(Q)[0]
        S1 = spod_sieber(Q, Nf=0)[0]
        np.testing.assert_allclose(S0, S1, atol=1e-10)

    def test_spod_towne_runs(self):
        """Regression test: spod_towne previously crashed with NameError
        (missing get_window / gammaincinv imports)."""
        t = np.linspace(0, 40 * np.pi, 512)
        Q = np.array([np.sin(t + phi) for phi in np.linspace(0, 1, 12)])
        Q += 0.01 * np.random.default_rng(4).standard_normal(Q.shape)
        Q -= Q.mean(axis=1, keepdims=True)
        L, Psi, f, Lc, info = spod_towne(Q, dt=t[1] - t[0], n_fft=128)
        assert L.shape[0] == len(f)
        assert np.isfinite(L).all()
        # the dominant frequency should be ~1/(2 pi)
        f_peak = f[np.argmax(L[:, 0])]
        assert f_peak == pytest.approx(1 / (2 * np.pi), rel=0.2)
