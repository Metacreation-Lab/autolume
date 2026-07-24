"""Background feature extraction: one subprocess per job."""
import logging
import multiprocessing as mp

import dnnlib
from torch_utils import legacy
from utils import device_utils
from utils.app_logging import LoggedProcess

from features import store
from features.extraction import ExtractionConfig, extract

logger = logging.getLogger(__name__)


def _default_config():
    return ExtractionConfig()


def _load_generator(model_path, device):
    with dnnlib.util.open_url(model_path, verbose=False) as f:
        data = legacy.load_network_pkl(f, custom=True)
    # G_ema is what the renderer performs with; directions must live in its w space.
    return data["G_ema"].to(device)


def _job_fn(model_path, progress_queue, root=None):
    try:
        device = device_utils.get_device()
        progress_queue.put(("progress", 0.0, "Loading model"))
        G = _load_generator(model_path, device)
        config = _default_config()
        directions = extract(
            G, config,
            progress_cb=lambda f, m: progress_queue.put(("progress", f, m)))
        path = store.save(model_path, directions, config, root=root)
        progress_queue.put(("done", str(path)))
    except Exception as e:
        logger.exception("Feature extraction failed for %s", model_path)
        progress_queue.put(("error", str(e)))


class FeatureExtractionJob:
    def __init__(self, model_path, process_factory=LoggedProcess):
        self.progress = 0.0
        self.message = "Starting"
        self.error = None
        self.result_path = None
        self._queue = mp.Queue()
        self._process = process_factory(
            target=_job_fn, args=(str(model_path), self._queue),
            daemon=True, name="feature-extraction")
        self._process.start()

    @property
    def done(self):
        return self.result_path is not None

    def poll(self):
        while not self._queue.empty():
            message = self._queue.get()
            kind = message[0]
            if kind == "progress":
                _, self.progress, self.message = message
            elif kind == "done":
                self.result_path = message[1]
                self.progress = 1.0
            elif kind == "error":
                self.error = message[1]
        if (self.error is None and self.result_path is None
                and self._process.exitcode not in (None, 0)):
            self.error = "Extraction process exited unexpectedly"
        if (self.done or self.error is not None) and not self._process.is_alive():
            self._process.join(timeout=0)

    def cancel(self):
        self._process.terminate()
        self._process.join(timeout=5)
