"""
Shared configuration for the test suite.

The tests import the project modules directly (e.g. ``from ensemble import Ensemble``),
so the ``src`` directory is prepended to ``sys.path`` in case the package is not
pip-installed in the current environment.
"""
import os
import sys

import matplotlib

matplotlib.use('Agg')  # never open figure windows during tests

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if SRC not in sys.path:
    sys.path.insert(0, SRC)
