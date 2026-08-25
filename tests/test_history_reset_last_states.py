import numpy as np

from romda.models.history import HistoryTracker


def _tracker(Nt=10, N=6, m=5):
    h = HistoryTracker()
    hist = np.arange(Nt * N * m, dtype=float).reshape(Nt, N, m)
    h.update_history(hist, np.arange(Nt) * 0.1, reset=True)
    return h, hist


def test_single_2d_state_only_touches_the_last_timestep():
    """analysis_step passes (N, m); it must not clobber N timesteps."""
    h, hist = _tracker()
    new = np.full((6, 5), -1.0)
    h.update_history(new, t=0.9, modify_saved_states=True)

    assert np.allclose(h.hist[-1], new)
    assert np.allclose(h.hist[:-1], hist[:-1])
    assert np.allclose(h.hist_t, np.arange(10) * 0.1)


def test_window_of_states_overwrites_that_window():
    h, hist = _tracker()
    new = np.full((3, 6, 5), -1.0)
    h.update_history(new, t=None, modify_saved_states=True)

    assert np.allclose(h.hist[-3:], new)
    assert np.allclose(h.hist[:-3], hist[:-3])


if __name__ == "__main__":
    test_single_2d_state_only_touches_the_last_timestep()
    test_window_of_states_overwrites_that_window()
    print("ok")
