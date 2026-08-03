"""Disposable state describing what the app was last doing.

Distinct from preferences (:func:`utils.user_data.load_prefs`), which hold
choices the user made deliberately, and from the data root, which holds files
they own. This is throwaway convenience state: losing it costs a retype, so it
lives in the cache directory and every operation is best-effort.
"""
import json
import logging
import os
import tempfile

from utils.user_data import cache_path

logger = logging.getLogger(__name__)

DEFAULT_LIMIT = 10

_state = None


def state_file():
    """Path to the session state file (folder may not exist yet)."""
    return cache_path("session.json")


def load() -> dict:
    """Load session state, falling back to empty if missing or unreadable."""
    global _state
    if _state is None:
        try:
            with open(state_file(), "r", encoding="utf-8") as fp:
                loaded = json.load(fp)
            _state = loaded if isinstance(loaded, dict) else {}
        except (OSError, ValueError):
            _state = {}
    return _state


def save() -> None:
    """Write the state file atomically. Never raises: this data is expendable."""
    path = state_file()
    tmp = None
    try:
        os.makedirs(path.parent, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            json.dump(load(), fp, indent=2)
        os.replace(tmp, path)
        tmp = None
    except (OSError, ValueError) as e:
        logger.debug("Could not save session state: %s", e)
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def get(section: str, key: str, default=None):
    """Read one value from a section."""
    values = load().get(section)
    return values.get(key, default) if isinstance(values, dict) else default


def set(section: str, key: str, value) -> None:
    """Write one value into a section and persist it."""
    load().setdefault(section, {})[key] = value
    save()


def get_recent(section: str, key: str) -> list:
    """Read a most-recent-first list, or [] if absent or malformed."""
    values = get(section, key)
    return [v for v in values if isinstance(v, str)] if isinstance(values, list) else []


def push_recent(section: str, key: str, value: str, limit: int = DEFAULT_LIMIT) -> list:
    """Move a value to the front of a most-recent-first list and persist it.

    Empty values are ignored so a cleared field never displaces real history.
    """
    if not value:
        return get_recent(section, key)
    recent = [v for v in get_recent(section, key) if v != value]
    recent.insert(0, value)
    del recent[limit:]
    set(section, key, recent)
    return recent
