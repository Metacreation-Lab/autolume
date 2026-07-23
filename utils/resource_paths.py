"""Resolve paths to bundled resource files across source and PyInstaller runtimes."""
from functools import lru_cache
from pathlib import Path
import sys

import tomllib

_REPO_ROOT = Path(__file__).resolve().parent.parent


def is_frozen() -> bool:
    """True when running from a PyInstaller bundle rather than a source checkout."""
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def resource_root() -> Path:
    """Root directory where bundled data files live.

    Source checkout: the repo root. PyInstaller bundle: ``sys._MEIPASS``.
    """
    if is_frozen():
        return Path(sys._MEIPASS)
    return _REPO_ROOT


def resource_path(*parts: str) -> Path:
    """Resolve a resource path relative to :func:`resource_root`."""
    return resource_root().joinpath(*parts)


@lru_cache(maxsize=1)
def _project_table() -> dict:
    """Parse the ``[project]`` table from the bundled ``pyproject.toml``."""
    with open(resource_path("pyproject.toml"), "rb") as fp:
        return tomllib.load(fp)["project"]


def get_version() -> str:
    """Return the application version declared in ``pyproject.toml``."""
    return _project_table()["version"]
