"""The romda.models shim: re-exports from dynamodels + pickle path back-compat."""
import io
import pickle
import sys


def test_reexports_are_dynamodels():
    import dynamodels
    from romda.models import HistoryTracker, Model
    from romda.models.physical import Lorenz63

    assert Model is dynamodels.Model
    assert HistoryTracker is dynamodels.HistoryTracker
    assert Lorenz63 is dynamodels.physical.Lorenz63


def test_old_pickle_class_paths_resolve():
    # stock pickle stores 'romda.models.physical.lorenz63' + 'Lorenz63' and resolves
    # them with find_class (import module, getattr) — exactly what the sys.modules
    # aliases in romda.models must serve for pre-split .pkl files
    import romda.models  # noqa: F401  (installs the aliases)
    for mod in ('romda.models.model', 'romda.models.history', 'romda.models.integrator',
                'romda.models.physical.lorenz63', 'romda.models.physical.rijke'):
        assert mod in sys.modules, f'{mod} alias missing'
    cls = pickle.Unpickler(io.BytesIO()).find_class('romda.models.physical.lorenz63', 'Lorenz63')
    import dynamodels
    assert cls is dynamodels.physical.Lorenz63
