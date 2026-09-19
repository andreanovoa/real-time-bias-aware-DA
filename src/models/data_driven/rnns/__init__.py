"""
romda.models.data_driven.rnns
==============================

Recurrent-network building blocks mixed into the data-driven forecast models
(mirrors the ``autoencoders`` subpackage). Currently the ESN wrapper
(`ESN_model`, `phi_to_esn_layout`) and its save/load config (`ESNConfig`);
other RNN architectures (e.g. LSTM, GRU) will live here too.
"""

from .esn import ESN_model, phi_to_esn_layout
from .esn_config import (
    ESN_STORE,
    ESNConfig,
    auto_load_or_create,
    find_matching_config,
    list_saved_configs,
    load_esn_model_from_config,
    save_esn_model_to_config,
)

__all__ = [
    "ESN_model",
    "phi_to_esn_layout",
    "ESN_STORE",
    "ESNConfig",
    "auto_load_or_create",
    "find_matching_config",
    "list_saved_configs",
    "load_esn_model_from_config",
    "save_esn_model_to_config",
]
