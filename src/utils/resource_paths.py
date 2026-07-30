"""Resolve paths to bundled resource files across source and PyInstaller runtimes."""
from functools import lru_cache
from pathlib import Path
import sys

import tomllib

_SRC_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _SRC_ROOT.parent


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
    """Resolve a resource path relative to :func:`resource_root`.

    A frozen bundle flattens all data under ``sys._MEIPASS``. A source
    checkout splits it between the repo root (``sr_models/``,
    ``pyproject.toml``) and ``src/`` (``assets/``), so fall back to ``src/``
    when the root candidate does not exist.
    """
    candidate = resource_root().joinpath(*parts)
    if not candidate.exists() and not is_frozen():
        src_candidate = _SRC_ROOT.joinpath(*parts)
        if src_candidate.exists():
            return src_candidate
    return candidate


@lru_cache(maxsize=1)
def _project_table() -> dict:
    """Parse the ``[project]`` table from the bundled ``pyproject.toml``."""
    with open(resource_path("pyproject.toml"), "rb") as fp:
        return tomllib.load(fp)["project"]


def get_version() -> str:
    """Return the application version declared in ``pyproject.toml``."""
    return _project_table()["version"]
