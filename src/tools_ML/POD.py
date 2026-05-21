"""
POD.py  —  backward-compatibility shim
=======================================
Re-exports POD and SPOD from linear_rom so that any existing code that does

    from tools_ML.POD import POD

continues to work without modification.

For new code, import directly from the module that contains the class:

    from tools_ML.linear_rom import POD, SPOD
    from tools_ML.rom_base   import ROM, AE, CAE
    from tools_ML.pod_spod   import prepare_data, snapshot_pod, spod_sieber
"""

from .linear_rom import POD, SPOD          # noqa: F401  (re-export)
from .rom_base   import ROM, AE, CAE       # noqa: F401
from .pod_spod   import prepare_data       # noqa: F401

__all__ = ['POD', 'SPOD', 'ROM', 'AE', 'CAE', 'prepare_data']
