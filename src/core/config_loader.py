"""
#############################################################################
### Configuration loader file
###
### @file config_loader.py
### @author Sebastian Russo
### @date 2025
#############################################################################

This module loads configuration data from a JSON file.
It reads the file, parses JSON, and instantiates variables
for other modules to access essential settings.
"""
import json
import sys
from pathlib import Path

def read_data_from_config_json(file_path: str, exit_on_error: bool = True) -> dict:
    """
    Reads data from a JSON configuration file.

    Parameters:
        file_path (str): Path to the JSON config file.
        exit_on_error (bool): Whether to exit on error or return None.

    Returns:
        dict: Parsed JSON configuration data.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            config_data = json.load(file)
        return config_data
    except FileNotFoundError:
        print(f"ERROR: Config file not found: {file_path}")
        if exit_on_error:
            sys.exit(1)
        else:
            return None
    except json.JSONDecodeError:
        print(f"ERROR: Failed to parse JSON config file: {file_path}")
        if exit_on_error:
            sys.exit(1)
        else:
            return None

def setup_directories(config_data: dict, project_root: Path) -> None:
    """
    Create necessary directories based on configuration.
    
    Parameters:
        config_data (dict): Configuration data containing directory names.
        project_root (Path): Root directory of the project.
    """
    dir_names = config_data.get("dir_names", {})
    directories_to_create = [
        project_root / dir_names.get("trained_models", "trained_models"),
        project_root / dir_names.get("logs_dir", "logs"),
        project_root / dir_names.get("results_dir", "results"),
        project_root / dir_names.get("plots_dir", "plots")
    ]
    
    for directory in directories_to_create:
        directory.mkdir(exist_ok=True)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_FILE_PATH = Path(__file__).parent / "config.json"

# Load the entire configuration data
config_loader = read_data_from_config_json(str(CONFIG_FILE_PATH), exit_on_error=True)

# Setup directories
setup_directories(config_loader, PROJECT_ROOT)

# Create convenient path objects for backward compatibility
DATA_DIR = PROJECT_ROOT / config_loader["dir_names"]["data_dir"]
MODELS_DIR = PROJECT_ROOT / config_loader["dir_names"]["trained_models"]
LOGS_DIR = PROJECT_ROOT / config_loader["dir_names"]["logs_dir"]
RESULTS_DIR = PROJECT_ROOT / config_loader["dir_names"]["results_dir"]
PLOTS_DIR = PROJECT_ROOT / config_loader["dir_names"]["plots_dir"]
DATASET_FILE = DATA_DIR / config_loader["dataset"]["dataset_name"]

# Training parameters for backward compatibility
TRAIN_TEST_SPLIT = config_loader["training_params"]["train_test_split"]
CV_FOLDS = config_loader["training_params"]["cv_folds"]
RANDOM_STATE = 42  # Default random state
