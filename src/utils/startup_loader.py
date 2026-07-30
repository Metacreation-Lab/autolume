"""Background loading of the heavy startup imports.

Runs on a daemon thread started before the main window opens so the splash
screen appears immediately and can report progress. Imports performed here
land in ``sys.modules``, making the later imports on the UI thread instant.

Failures are captured rather than raised so they can be re-raised on the main
thread (via :meth:`StartupLoader.finalize`) where the fatal-error logging in
``main.py`` catches them instead of vanishing with the thread.
"""
import logging
import threading

logger = logging.getLogger(__name__)


def _load_torch():
    import torch
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _load_render_modules():
    import modules.renderloop  # noqa: F401
    import modules.visualizer  # noqa: F401


class StartupLoader:
    STAGES = (
        ('Loading PyTorch', _load_torch),
        ('Loading rendering engine', _load_render_modules),
    )

    def __init__(self):
        self.status = self.STAGES[0][0]
        self.started = False
        self.done = False
        self.error = None
        self._thread = threading.Thread(
            target=self._run, name='startup-loader', daemon=True)

    def start(self):
        logger.info("Starting background loading of heavy imports")
        self.started = True
        self._thread.start()
        return self

    def _run(self):
        try:
            for label, stage in self.STAGES:
                self.status = label
                stage()
        except BaseException as e:
            self.error = e
        finally:
            self.done = True

    def finalize(self):
        """Complete loading on the calling (main) thread. Re-raises any error
        captured by the loader thread. Gradient mode is thread-local in torch,
        so it must be disabled here rather than in :meth:`_run`."""
        if self.error is not None:
            raise self.error
        import torch
        torch.set_grad_enabled(False)
