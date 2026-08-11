"""Preset storage and state transfer, free of any UI dependency.

A preset is a folder under the presets data root holding one ``.pkl`` per
widget. A folder counts as a preset only if it contains ``MARKER_FILE``;
empty folders left behind by older versions are ignored, never deleted.
"""

import logging
import os
import re
import shutil

logger = logging.getLogger(__name__)

MARKER_FILE = "latent.pkl"

WIDGET_FILES = [
    ("latent_widget", "latent.pkl"),
    ("trunc_noise_widget", "trunc.pkl"),
    ("layer_widget", "layer.pkl"),
    ("adjuster_widget", "adjuster.pkl"),
    ("looping_widget", "looper.pkl"),
    ("pickle_widget", "pickle.pkl"),
    ("collapsed_widget", "collap.pkl"),
    ("mixing_widget", "mix.pkl"),
]

# Files whose absence or corruption should not abort a load.
_OPTIONAL_FILES = {"pickle.pkl"}


def save_preset(viz, path):
    try:
        os.makedirs(path, exist_ok=True)
        for attr, filename in WIDGET_FILES:
            getattr(viz, attr).save(os.path.join(path, filename))
        return True
    except Exception:
        logger.exception("Failed to save preset to %s", path)
        return False


def load_preset(viz, path):
    try:
        for attr, filename in WIDGET_FILES:
            try:
                getattr(viz, attr).load(os.path.join(path, filename))
            except Exception as e:
                if filename in _OPTIONAL_FILES:
                    logger.warning("Ignored error while loading %s: %s",
                                   filename, e)
                else:
                    raise
        return True
    except Exception:
        logger.exception("Failed to load preset from %s", path)
        return False


_INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = {"CON", "PRN", "AUX", "NUL",
             *(f"COM{i}" for i in range(1, 10)),
             *(f"LPT{i}" for i in range(1, 10))}


def _well_formed(name):
    """Windows filename rules on every platform, so preset folders stay portable."""
    return (bool(name)
            and name == name.strip()
            and not name.startswith(".")
            and not name.endswith(".")
            and not _INVALID_CHARS.search(name)
            and name.split(".")[0].upper() not in _RESERVED)


class PresetStore:
    def __init__(self, root):
        self.root = str(root)
        self._names = None
        self._mtime = None

    def invalidate(self):
        self._names = None
        self._mtime = None

    def names(self):
        try:
            mtime = os.stat(self.root).st_mtime_ns
        except OSError:
            self.invalidate()
            return []
        if self._names is None or mtime != self._mtime:
            self._names = sorted(
                (entry for entry in os.listdir(self.root)
                 if os.path.isfile(os.path.join(self.root, entry, MARKER_FILE))),
                key=str.lower)
            self._mtime = mtime
        return list(self._names)

    def path(self, name):
        return os.path.join(self.root, name)

    def is_valid_name(self, name):
        if not _well_formed(name):
            return False
        return name.lower() not in {taken.lower() for taken in self.names()}

    def create(self, name):
        if not self.is_valid_name(name):
            return None
        path = self.path(name)
        try:
            os.makedirs(path, exist_ok=True)
        except OSError:
            logger.exception("Failed to create preset folder %s", path)
            return None
        self.invalidate()
        return path

    def rename(self, old, new):
        if old not in self.names() or not _well_formed(new):
            return False
        if new == old:
            return True
        case_only = new.lower() == old.lower()
        if not case_only and not self.is_valid_name(new):
            return False
        # A non-preset folder of that name would be silently removed on POSIX
        # and raise on Windows, so refuse it everywhere.
        if not case_only and os.path.exists(self.path(new)):
            return False
        try:
            os.rename(self.path(old), self.path(new))
        except OSError:
            logger.exception("Failed to rename preset %s to %s", old, new)
            return False
        self.invalidate()
        return True

    def delete(self, name):
        if name not in self.names():
            return False
        try:
            shutil.rmtree(self.path(name))
        except OSError:
            logger.exception("Failed to delete preset %s", name)
            return False
        self.invalidate()
        return True
