"""Helpers for the runtime models/ folder, which may not exist until first use."""
import os

from utils.user_data import data_path


def models_dir():
    """Absolute path of the models folder under the user data root."""
    return str(data_path("models"))


def ensure_models_dir():
    """Create the models folder if needed and return its path."""
    path = models_dir()
    os.makedirs(path, exist_ok=True)
    return path


def list_model_pkls():
    """Absolute paths of .pkl files in the models folder; empty if the folder is missing."""
    path = models_dir()
    if not os.path.isdir(path):
        return []
    return [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith(".pkl")]
