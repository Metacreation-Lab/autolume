"""Resolve paths to bundled resource files across source and PyInstaller runtimes."""
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent


def resource_root() -> Path:
    """Root directory where bundled data files live.

    Source checkout: the repo root. PyInstaller bundle: ``sys._MEIPASS``.
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return _REPO_ROOT


def resource_path(*parts: str) -> Path:
    """Resolve a resource path relative to :func:`resource_root`."""
    return resource_root().joinpath(*parts)
