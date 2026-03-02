from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple, Union
import numpy as np
import yaml
from pathlib import Path
import hashlib
import json
from utils import get_project_root, convert_to_python_type

from models_data_driven import ESN_model


BASE_CONFIG_DIR = get_project_root() + '/src/config'

INIT_KEYS = [# fixed hyperparameter settings
            'N_units',
            'Win_type', 
            'N_wash', 
            'upsample',
            'bias_in',  
            'bias_out',  
            'norm_method',
            'connect',
            # state parameters
            'dt',
            'observed_idx',
            'update_reservoir', 
            'update_state',
            'Wout_svd',
            'N_dim',
            # training parameters and hyperparameter optimization
            't_train', 
            't_val',
            't_test',
            'N_func_evals',
            'N_grid',
            'N_folds',
            'N_initial_rand',
            'N_split',
            'rho_range',
            'sigma_in_range',
            'tikh_range', 
            'noise', 
            'seed',
            'training_data_filename']

TRAINED_KEYS = ['norm', 'shift', 
                'rho', 'sigma_in', 'tikh', 
                'Win', 'W', 'Wout', 'validation_data', 'reservoir_state']


@dataclass
class ESNConfig:
    """Configuration class for ESN_model to enable easy save/load without retraining."""
    
    # Model dimensions
    N_dim: Optional[int] = None
    observed_idx: Optional[list] = None
    update_reservoir: bool = True
    update_state: bool = True
    Wout_svd: bool = False
    training_data_filename: Optional[str] = None
    plot_training: bool = False
    
    # Time parameters. 
    dt: float = 0.1
    t_train: Optional[float] = None
    t_val: Optional[float] = None
    t_test: Optional[float] = None
    
    # ESN architecture parameters
    N_units: int = 50
    norm_method: str = 'range'
    Win_type: str = 'sparse'
    N_wash: int = 5
    upsample:  int = 1
    bias_in: float = 0.1
    bias_out: float = 1.0
    connect: float = 3
    
    # Training parameters
    noise:  float = 1e-2
    noise_type: str = 'gauss'
    N_func_evals: int = 40
    N_grid:  int = 5
    N_folds: int = 8
    N_split: int = 5
    N_initial_rand: int = 0

    # Hyperparameter optimization ranges
    rho_range: Tuple[float, float] = (0.2, 0.8)
    sigma_in_range: Tuple[float, float] = (-2, 2)
    tikh_range: Tuple[float, ... ] = (1e-6, 1e-9, 1e-12)
    hyperparameters_to_optimize: Tuple[str, ...] = ('rho', 'sigma_in', 'tikh')
    
    # Additional metadata
    seed: Optional[int] = 0
    config_hash: Optional[str] = None  # Hash for matching configs


    # Trained data keys (loaded separately)
    norm: Optional[np.ndarray] = None
    shift: Optional[np.ndarray] = None
    rho: Optional[float] = None
    sigma_in: Optional[float] = None
    tikh: Optional[float] = None
    Win: Optional[np.ndarray] = None
    W: Optional[np.ndarray] = None
    Wout: Optional[np.ndarray] = None
    validation_data: Optional[np.ndarray] = None
    reservoir_state: Optional[np.ndarray] = None
    
    
    @staticmethod
    def _init_config_dict(case):
        return {key: getattr(case, key) for key in INIT_KEYS}
    
    def to_hash(self):
        """
        Compute a hash of the configuration parameters that define the model.
        Uses INITIAL parameters only (before any optimization).
        This allows matching configs even if hyperparameter optimization produces
        slightly different results due to randomness.
        
        Returns:
            str: Hash string
        """
        
        hash_params = ESNConfig._init_config_dict(self)

        # Normalize types to ensure consistent hashing
        # Convert all numpy arrays and lists to consistent format
        for key in ['bias_in', 'bias_out']:
            if hash_params[key] is not None:
                if isinstance(hash_params[key], (float, int)):
                    hash_params[key] = [float(hash_params[key])]
                elif isinstance(hash_params[key], np.ndarray):
                    hash_params[key] = hash_params[key].tolist()
                    
        # Normalize observed_idx to list
        if hash_params.get('observed_idx') is not None:
            if isinstance(hash_params['observed_idx'], np.ndarray):
                hash_params['observed_idx'] = hash_params['observed_idx'].tolist()
            elif not isinstance(hash_params['observed_idx'], list):
                hash_params['observed_idx'] = list(hash_params['observed_idx'])
        
        # Normalize tikh_range to tuple (canonical form)
        if hash_params.get('tikh_range') is not None:
            if isinstance(hash_params['tikh_range'], (list, np.ndarray)):
                hash_params['tikh_range'] = tuple(hash_params['tikh_range'])

        hash_params = convert_to_python_type(hash_params)

        # Convert to JSON string (sorted keys for consistency)
        hash_string = json.dumps(hash_params, sort_keys=True)
        
        # Compute SHA256 hash
        return hashlib.sha256(hash_string.encode()).hexdigest()[:16]
    
    
    @classmethod
    def from_esn_model(cls, 
                       esn_model):
        """
        Create config from an existing ESN_model instance.
        
        Args:
            esn_model: Instance of ESN_model to extract configuration from
        
        Returns:
            ESNConfig instance
        """
        
        
        config_dict = cls._init_config_dict(esn_model)

        
        return cls(**config_dict)
    
    
    @classmethod
    def from_init_params(cls, **kwargs):
        """
        Create a config from initialization parameters (before training).
        Useful for checking if a matching config exists.
        
        Args:
            **kwargs: ESN initialization parameters
        
        Returns:
            ESNConfig instance
        """
        
        config_dict = {key: kwargs[key] for key in INIT_KEYS if key in kwargs}

        # If N_test, N_val, N_train are not provided, compute them from t_test, t_val, t_train and dt
        for key in ['train', 'val', 'test']:
            if f'N_{key}' in kwargs.keys():
                config_dict[f't_{key}'] = kwargs[f'N_{key}'] * kwargs['dt']

        init_config = cls(**config_dict)
        if init_config.N_dim is None:
            assert 'data' in kwargs and kwargs['data'] is not None, "N_dim not provided and data not available to infer it."
            init_config.N_dim = kwargs['data'].shape[1]

        if init_config.observed_idx is None:
            init_config.observed_idx = list(range(init_config.N_dim))

        return init_config
    
    
    def to_esn_model(self, data=None, retrain:  bool = False, **override_kwargs):
        """
        Create an ESN_model instance from this config.
        
        Args:
            data:  Training data (only needed if retraining)
            load_trained_matrices: If True, load saved matrices instead of retraining
            **override_kwargs: Any parameters to override from the config
        
        Returns: 
            ESN_model instance (either loaded or freshly trained)
        """
        
        if not retrain:
            # Load pre-trained model
            config = asdict(self)
            config['y0'] = np.zeros((self.N_dim,self.reservoir_state.shape[-1]))  # Dummy initial state
            # print("Loading pre-trained ESN_model...")
            return ESN_model(**config)
        else:
            # Create new model (requires training data)
            if data is None: 
                raise ValueError("Training data required to create new ESN_model")

            init_config = asdict(self)
            init_config.update(override_kwargs)
            for key in TRAINED_KEYS:
                init_config.pop(key, None)  # Remove trained keys if present

            print("Creating and training new ESN_model...")
            return ESN_model(data=data, **init_config)
    
    
    def save(self, save_dir:  Path):
        """Save configuration to YAML file."""
        filepath = Path(save_dir / "esn_config.yaml")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Get initial parameters only
        config_dict = ESNConfig._init_config_dict(self)  

        # Convert tuples to lists for YAML compatibility and  sort keys for consistent ordering in YAML file
        config_dict = convert_to_python_type(config_dict)
        config_dict = dict(sorted(config_dict.items()))

        
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        # Print the last 3 folders and name of the save directory for confirmation
        print(f"Configuration saved to ...{'/'.join(list(save_dir.parts)[-3:])}/esn_config.yaml")


    @classmethod
    def load(cls, load_dir: Path, verbose=1):
        """Load configuration from YAML file."""
        # if yaml file is given, get its parent directory
        try:
            if load_dir.suffix == '.yaml':
                load_dir = load_dir.parent

            filepath = Path(load_dir / 'esn_config.yaml')
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)

            for key in config_dict.keys():
                if config_dict[key] == "none":
                    config_dict[key] = None

            if verbose:
                print(f"Configuration loaded from ...{'/'.join(list(load_dir.parts)[-3:])}/esn_config.yaml")
            return cls(**config_dict)
        
        except Exception as e:
            print(f"Error loading configuration from {load_dir}: {e}")
            raise e

    

    @staticmethod
    def save_trained_matrices(esn_model, save_dir: Path):
        """
        Save large matrices and other metadata defined after training the network.
        """ 
        data_to_save = {key: getattr(esn_model, key, None) for key in TRAINED_KEYS }
        
        np.savez_compressed(save_dir / 'trained_matrices.npz', **data_to_save)
        print(f"Trained matrices saved to ...{'/'.join(list(save_dir.parts)[-3:])}/trained_matrices.npz")
        
    
    def update(self, update_dict: dict):
        """
        Update configuration parameters from a dictionary.
        
        Args:
            update_dict: Dictionary of parameters to update
        """
        for key, value in update_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @staticmethod
    def load_trained_matrices(save_dir: Path):
        """
        Load saved matrices from a single .npz file.
        
        Args:
            save_dir: Directory containing the .npz file (e.g., 'saved_configs/esn_abc123/')
        
        Returns:
            dict: Loaded matrices
        """
        saved_file = Path(save_dir / 'trained_matrices.npz')
        
        if not saved_file.exists():
            # Return dict with None values if file doesn't exist
            return {name: None for name in TRAINED_KEYS}
            # return None
        
        trained_dict = {}
        with np.load(saved_file, allow_pickle=True) as data:
            for name in TRAINED_KEYS:
                item = data.get(name)
                if isinstance(item, np.ndarray) and item.dtype == object:
                    item = item.item()

                trained_dict[name] = item
                
        return trained_dict


# Convenience functions
def save_esn_model_to_config(esn_model, save_dir: str = None, name: str = None):
    """
    Convenience function to save an ESN model configuration.
    
    Args:
        esn_model: ESN_model instance to save
        save_dir: Directory to save configuration
        name: Base name for saved files (default: auto-generated from hash)
    
    Returns:
        Tuple of (config, save_path)
    """

    if save_dir is None:
        save_dir = Path(BASE_CONFIG_DIR) / "esn_configs"

    save_dir = Path(save_dir)
    

    config = ESNConfig.from_esn_model(esn_model)

    if name is None:
        name = save_dir / f"{config.to_hash()}"

    save_path = save_dir / name 
    config.save(save_path)

    ESNConfig.save_trained_matrices(esn_model, save_path)
    
    return config, save_path



def load_esn_model_from_config(q: Optional[str]=None,
                               config: Optional[ESNConfig]=None, 
                               load_dir: str = Path(BASE_CONFIG_DIR) / "esn_configs"):
    """
    Load an ESN_model instance from a saved configuration.
    Args:
        q: Query string to match against saved config hashes
        config: ESNConfig instance to load (if not None, q is ignored)
    """
    if q is not None:
        matching_path = find_matching_config(load_dir, q)

    elif config is not None:
        if isinstance(config, ESNConfig):
            q = config.to_hash()
        else:
            raise ValueError("Input config must be an instance of ESNConfig")

        matching_path = find_matching_config(load_dir, q)

    else:
        raise ValueError("Either query string q or config instance must be provided to load ESN_model")

    if not matching_path:
        print(f"No matching config {q} found in {load_dir}.")
        return None

    config = ESNConfig.load(matching_path)
    config.update(ESNConfig.load_trained_matrices(matching_path))

    return config.to_esn_model() # Note: data is not needed to load a trained model since matrices are loaded separately
    

def find_matching_config(search_dir: str, query_hash):
    """
    Search for a saved config that matches the given initialization parameters.
    
    Args:
        search_dir: Directory to search for configs
        query_hash: Hash string of the desired configuration
    
    Returns: 
        Path to matching config directory, or None if not found
    """
    
    # Create config from init params to get hash

    search_dir = Path(search_dir)
    if not search_dir.exists():
        print(f"Search directory does not exist: {search_dir}")
        return None
    
    
    # First check for hash-based directory name
    hash_based_path = search_dir / f"{query_hash}"
    if hash_based_path.exists() and (hash_based_path / "esn_config.yaml").exists():
        print(f"✓ Found matching config: {query_hash}")
        return hash_based_path
     
    # # Search for matching hash in all yaml files. [I MAY DELETE THIS LATER, BUT IT'S USEFUL FOR DEBUGGING NOW] 
    # print(f"Searching for matching config with hash {query_hash} in {search_dir}...")
    # for yaml_file in search_dir.glob("*/esn_config.yaml"):
    #     try:
    #         saved_config = ESNConfig.load(yaml_file.parent, verbose=0)
    #         saved_hash = saved_config.to_hash()
    #         print(f"\t {yaml_file.parent}: saved hash = {saved_hash}")
            
    #         if saved_hash == query_hash: 
    #             print(f"✓ Found matching config: {yaml_file.parent}")
    #             return yaml_file.parent
    #     except Exception as e: 
    #         # Skip invalid configs
    #         print(f"  Skipping {yaml_file} due to {type(e).__name__}: {e}")
    #         continue
    
    print("✗ No matching config found")
    return None


def auto_load_or_create(config_dir: str = Path(BASE_CONFIG_DIR) / "esn_configs", 
                        auto_save: bool = True, 
                        force_create: bool = False,
                        query_hash: Optional[str] = None,
                        data: Optional[np.ndarray] = None, 
                        **kwargs):
    """
    Automatically load a matching config or create and train a new model.
    
    Args:
        data: Training data
        config_dir: Directory to search for/save configs
        auto_save: If True, save newly trained models
        **kwargs: ESN initialization parameters
    
    Returns:
        ESN_model instance (loaded or newly trained)
    """
    
    
    # If query_hash is provided, skip config creation and search directly for matching hash
    if query_hash is None:
        initial_params = kwargs.copy()
        initial_params['data'] = data
        query_config = ESNConfig.from_init_params(**initial_params)
        query_hash = query_config.to_hash()

    # print(f"Searching for config with hash: {query_hash}...")

    
    # Try to find matching config
    if force_create:
        print("Force creating new model (skipping search)...")
        matching_path = None
    else:
        # print(f"Searching for matching config... {config_dir} and hash {query_hash}")
        matching_path = find_matching_config(config_dir, query_hash)
    
    if matching_path:
        # Load existing model
        config = ESNConfig.load(matching_path)
        config.update(ESNConfig.load_trained_matrices(matching_path))
        return config.to_esn_model(data=data)
    else:
        # Train new model
        model = ESN_model(data=data, **kwargs)
        if auto_save:
            print(f"Saving new model to {config_dir}")
            save_esn_model_to_config(model, save_dir=config_dir, name=f"{query_hash}")
        return model
    



def list_saved_configs(search_dir: str, verbose=True):
    """
    List all saved ESN configs in a directory.
    
    Args:
        search_dir:  Directory to search
        verbose: If True, print details about each config
    
    Returns:
        List of config information dictionaries
    """
    search_dir = Path(search_dir)
    configs = []
    
    if not search_dir.exists():

        search_dir = Path(BASE_CONFIG_DIR) / search_dir
        if not search_dir.exists():

            print(f"Directory not found: {search_dir}")
            return configs
    
    for yaml_file in sorted(search_dir.glob("*/esn_config.yaml")):
        try:
            config = ESNConfig.load(yaml_file)
            
            config_info = {
                'directory': yaml_file.parent,
                'name': yaml_file.parent.name,
                'N_units': config.N_units,
                'dt': config.dt,
                'training_data_filename': config.training_data_filename,
            }
            
            configs.append(config_info)
            
            if verbose:
                print(f"\n{yaml_file.parent.name}:")
                print(f"  Training data: {config.training_data_filename}")
                print(f"  N_units: {config.N_units}")
                print(f"  N_dim: {config.N_dim}")
                print(f"  dt:  {config.dt}")
                print(f"  etc.: ...")
        
        except Exception as e:
            if verbose:
                print(f"Error reading {yaml_file}: {e}")
    
    return configs




if __name__ == "__main__":
    # Example usage
    from models_data_driven import ESN_model

    # Generate dummy data
    t = np.linspace(0, 10, 1000)
    data = np.sin(t)[np.newaxis, :, np.newaxis]

    # Auto load or create ESN model
    esn = auto_load_or_create(
        data=data,
        config_dir="./esn_configs",
        N_units=10,
        seed=42,
        N_grid=3,
        N_func_evals=10,
    )

    print("ESN model ready.")