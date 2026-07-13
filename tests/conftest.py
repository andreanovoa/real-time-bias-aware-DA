"""
Shared configuration for the test suite.

The tests import the project as ``romda`` (e.g. ``from romda.ensemble import Ensemble``).
The package maps onto the ``src`` directory (see ``[tool.setuptools]`` in pyproject.toml),
so if ``romda`` is not pip-installed we register ``src`` as the ``romda`` package here.
"""
import importlib.util
import os
import sys

import matplotlib

matplotlib.use('Agg')  # never open figure windows during tests

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import romda  # noqa: F401  (installed, e.g. via `pip install -e .`)
except ImportError:
    spec = importlib.util.spec_from_file_location(
        'romda', os.path.join(SRC, '__init__.py'),
        submodule_search_locations=[SRC])
    module = importlib.util.module_from_spec(spec)
    sys.modules['romda'] = module
    spec.loader.exec_module(module)
