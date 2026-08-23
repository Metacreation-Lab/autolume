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
    "Balance": ("realesr-general-x4v3.pth",
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"),
    "BalanceWDN": ("realesr-general-wdn-x4v3.pth",
                   "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth"),
    "Quality": ("RealESRGAN_x4plus.pth",
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"),
}


def weight_path(name):
    filename, _ = WEIGHTS[name]
    return str(cache_path("real-esrgan", filename))


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
