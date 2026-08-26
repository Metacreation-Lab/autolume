"""Downloadable upscaling weights: registry, cache paths, download."""

import logging
import os
import threading

from utils.downloads import download_file
from utils.user_data import cache_path

logger = logging.getLogger(__name__)

# name -> (filename, url). Filenames are the upstream release names so the
# cache stays valid across versions of this module.
WEIGHTS = {
    "RealPLKSR": ("4xNomosWebPhoto_RealPLKSR.safetensors",
                  "https://huggingface.co/Phips/4xNomosWebPhoto_RealPLKSR/resolve/main/4xNomosWebPhoto_RealPLKSR.safetensors"),
    "DAT2": ("4xRealWebPhoto_v4_dat2.safetensors",
             "https://huggingface.co/Phips/4xRealWebPhoto_v4_dat2/resolve/main/4xRealWebPhoto_v4_dat2.safetensors"),
    "CompactC3": ("4xLSDIRCompactC3.safetensors",
                  "https://huggingface.co/Phips/4xLSDIRCompactC3/resolve/main/4xLSDIRCompactC3.safetensors"),
}


def weight_path(name):
    filename, _ = WEIGHTS[name]
    return str(cache_path("upscale", filename))


def ensure_weight(name, progress_cb=None, cancel_event=None):
    """Return the weight path, downloading into the cache if missing.

    Returns None if the download was cancelled. Headless-safe: works with
    no progress callback.
    """
    path = weight_path(name)
    if not os.path.exists(path):
        _, url = WEIGHTS[name]
        if progress_cb is None:
            logger.info("Downloading %s upscaling weights from %s", name, url)
            def progress_cb(done, total):
                pass
        if cancel_event is None:
            cancel_event = threading.Event()
        if not download_file(url, path, cancel_event, progress_cb):
            return None
    return path
