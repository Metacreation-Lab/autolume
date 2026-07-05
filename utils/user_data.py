"""Resolve the writable user-data root and load/save application preferences.

Read-only bundled resources live under :mod:`utils.resource_paths`. This module is
its writable counterpart: it decides where models, presets, recordings, training
runs and other user-generated files are stored, and persists that choice (plus any
future preferences) to a small JSON config file.

Layout:
- Preferences file: ``$XDG_CONFIG_HOME/autolume/config.json`` (Linux), else
  ``~/.config/autolume/config.json`` (macOS, Windows, and Linux without the env var).
- Data root: ``~/autolume`` by default, overridable via the Settings screen and
  stored in the preferences file. Categories (``models``, ``presets``, ...) are
  subfolders of the data root.
"""
import json
import logging
import os
from pathlib import Path

PREFS_VERSION = 1

logger = logging.getLogger(__name__)

_prefs = None
_data_root = None
_failed_data_root = None


def config_dir() -> Path:
    """Directory holding the preferences file (created on first save)."""
    base = os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")
    return Path(base) / "autolume"


def config_file() -> Path:
    """Path to the preferences JSON file."""
    return config_dir() / "config.json"


def default_data_root() -> str:
    """Default writable data root: ``~/autolume``."""
    return str(Path.home() / "autolume")


def load_prefs() -> dict:
    """Load preferences, falling back to defaults if the file is missing or invalid."""
    global _prefs
    if _prefs is None:
        defaults = {"version": PREFS_VERSION, "data_root": default_data_root()}
        try:
            with open(config_file(), "r", encoding="utf-8") as fp:
                loaded = json.load(fp)
            if isinstance(loaded, dict):
                defaults.update(loaded)
        except (OSError, ValueError):
            pass
        _prefs = defaults
    return _prefs


def save_prefs(prefs: dict) -> None:
    """Persist preferences to the config file, creating its directory if needed."""
    global _prefs
    _prefs = prefs
    config_dir().mkdir(parents=True, exist_ok=True)
    with open(config_file(), "w", encoding="utf-8") as fp:
        json.dump(prefs, fp, indent=2)


def ui_font_size(default: int) -> int:
    """Configured DPI-independent UI font size, or ``default`` if unset/invalid."""
    try:
        return int(load_prefs().get("ui_font_size", default))
    except (TypeError, ValueError):
        return default


def set_ui_font_size(size: int) -> None:
    """Persist the UI font size to the preferences file."""
    prefs = load_prefs()
    prefs["ui_font_size"] = int(size)
    save_prefs(prefs)


def data_root() -> str:
    """Configured writable data root (or the default)."""
    global _data_root
    if _data_root is None:
        _data_root = str(load_prefs().get("data_root") or default_data_root())
    return _data_root


def set_data_root(path) -> None:
    """Update the data root in memory and persist it to the preferences file."""
    global _data_root
    _data_root = str(path)
    prefs = load_prefs()
    prefs["data_root"] = _data_root
    save_prefs(prefs)
    logger.info("Data root set to %s; logs move to the new location after restart", _data_root)


def data_path(*parts: str) -> Path:
    """Resolve a path under the data root.

    The first part names a category (e.g. ``"models"``). An optional ``overrides``
    map in the preferences lets a future version redirect individual categories
    elsewhere; absent that, everything resolves under :func:`data_root`.
    """
    overrides = load_prefs().get("overrides") or {}
    if parts and parts[0] in overrides:
        return Path(overrides[parts[0]]).joinpath(*parts[1:])
    return Path(data_root()).joinpath(*parts)


def ensure_data_path(*parts: str) -> Path:
    """Like :func:`data_path`, but create the resolved directory if needed."""
    path = data_path(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def init_data_root() -> Path:
    """Create the data root, falling back to the default for this session only.

    If the configured folder is inaccessible (e.g. an unmounted drive), the
    persisted preference is left untouched and :func:`failed_data_root` reports
    the path that failed.
    """
    global _data_root, _failed_data_root
    try:
        return ensure_data_path()
    except OSError:
        configured = data_root()
        if configured == default_data_root():
            raise
        logger.warning("Data root %s is not accessible; using %s for this session",
                       configured, default_data_root())
        _failed_data_root = configured
        _data_root = default_data_root()
        return ensure_data_path()


def failed_data_root():
    """Configured data root that was inaccessible at startup, or None."""
    return _failed_data_root


def cache_path(*parts: str) -> Path:
    """Resolve a path under ``~/.cache/autolume`` (folder may not exist yet).

    Honors ``XDG_CACHE_HOME``. Used for disposable, re-downloadable data
    such as on-demand super-resolution weights.
    """
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base, "autolume", *parts)
