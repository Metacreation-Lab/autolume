"""Helpers for the runtime model folders, which may not exist until first use."""
import os
import re

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


def list_training_run_pkls():
    """(run_folder, snapshots) tuples for training run folders with snapshot .pkl files.

    Snapshots are (filename, path) tuples. Runs and snapshots are ordered
    newest first; empty if the training-runs folder is missing.
    """
    root = str(data_path("training-runs"))
    if not os.path.isdir(root):
        return []
    run_regex = re.compile(r"\d+-.*")
    pkl_regex = re.compile(r"network-snapshot-\d+\.pkl")
    runs = []
    for run in sorted(os.listdir(root), reverse=True):
        run_dir = os.path.join(root, run)
        if not (run_regex.fullmatch(run) and os.path.isdir(run_dir)):
            continue
        snapshots = [(name, os.path.join(run_dir, name))
                     for name in sorted(os.listdir(run_dir), reverse=True)
                     if pkl_regex.fullmatch(name)]
        if snapshots:
            runs.append((run, snapshots))
    return runs
