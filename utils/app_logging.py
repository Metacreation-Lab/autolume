"""Central logging for every Autolume process.

The main process owns the only file handler: a daily-rotating
``<data_root>/logs/autolume.log`` drained by a :class:`~logging.handlers.QueueListener`
thread. Every process (main and spawned children alike) logs through a shared
``multiprocessing.Queue`` via :class:`~logging.handlers.QueueHandler`, so rotation
is safe on Windows (a single writer) and disk I/O stays off the GUI thread.

Wiring:
- Main process: call :func:`setup_main_logging` once, inside the ``__main__`` guard.
- Child processes: spawn workers with :class:`LoggedProcess` instead of
  ``multiprocessing.Process`` — the child configures itself before running.
- Pool workers: pass ``initializer=pool_worker_init,
  initargs=(get_log_queue(), "<name>")`` to ``multiprocessing.Pool``.

Every process also redirects ``sys.stdout``/``sys.stderr`` into the log (safety
net for vendored dnnlib/torch_utils prints and third-party libraries — required
in the macOS windowed build where those streams are otherwise gone), installs
``sys.excepthook``/``threading.excepthook``, and enables ``faulthandler`` on an
append-only ``logs/crashes.log`` for native crashes.
"""
import atexit
import datetime
import faulthandler
import logging
import logging.handlers
import multiprocessing
import os
import re
import sys
import tempfile
import threading
import traceback

LOG_DIR_NAME = "logs"
LOG_FILE_NAME = "autolume.log"
CRASH_FILE_NAME = "crashes.log"
BACKUP_DAYS = 14
LOG_FORMAT = "%(asctime)s %(levelname)-7s [%(processName)s] %(name)s: %(message)s"
NOISY_THIRD_PARTY_LOGGERS = ("PIL", "matplotlib", "urllib3", "numba")


class _ISOFormatter(logging.Formatter):
    """Emit local, timezone-aware ISO 8601 timestamps, e.g. 2026-07-22T21:41:52.185+02:00."""

    def formatTime(self, record, datefmt=None):
        dt = datetime.datetime.fromtimestamp(record.created).astimezone()
        return dt.isoformat(timespec="milliseconds")

_log_queue = None
_listener = None
_crash_file = None
_configured = False


def _resolve_log_dir():
    """Directory for log files, or None if nowhere is writable. Never raises."""
    try:
        from utils.user_data import ensure_data_path
        return ensure_data_path(LOG_DIR_NAME)
    except Exception:
        pass
    try:
        from pathlib import Path
        fallback = Path(tempfile.gettempdir()) / "autolume" / LOG_DIR_NAME
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback
    except Exception:
        return None


class _StreamToLogger:
    """File-like object routing writes to a logger, line by line.

    Deliberately has no ``fileno()``: callers that need a real file descriptor
    (e.g. faulthandler) must not be handed this object.
    """

    def __init__(self, logger, level):
        self._logger = logger
        self._level = level
        self._buffer = ""
        self._emitting = False

    def _emit(self, line):
        if self._emitting:
            return
        self._emitting = True
        try:
            self._logger.log(self._level, line)
        finally:
            self._emitting = False

    def write(self, text):
        if not isinstance(text, str):
            text = str(text)
        self._buffer += text
        *lines, self._buffer = re.split(r"[\r\n]", self._buffer)
        for line in lines:
            if line.strip():
                self._emit(line)
        return len(text)

    def flush(self):
        if self._buffer.strip():
            self._emit(self._buffer)
        self._buffer = ""

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def isatty(self):
        return False


def _configure_root(log_queue, level):
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(logging.handlers.QueueHandler(log_queue))
    root.setLevel(level)
    for name in NOISY_THIRD_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def _log_level():
    return os.environ.get("AUTOLUME_LOG_LEVEL", "INFO").upper()


def _install_runtime_capture(log_dir):
    """Redirect std streams, hook uncaught exceptions, enable faulthandler."""
    global _crash_file

    sys.stdout = _StreamToLogger(logging.getLogger("stdout"), logging.INFO)
    sys.stderr = _StreamToLogger(logging.getLogger("stderr"), logging.WARNING)

    def _excepthook(exc_type, exc_value, exc_tb):
        logging.getLogger("autolume").critical(
            "Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))
        if sys.__stderr__ is not None:
            traceback.print_exception(exc_type, exc_value, exc_tb, file=sys.__stderr__)

    def _thread_excepthook(args):
        thread_name = args.thread.name if args.thread else "unknown"
        logging.getLogger("autolume").critical(
            "Uncaught exception in thread %s", thread_name,
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

    sys.excepthook = _excepthook
    threading.excepthook = _thread_excepthook
    logging.captureWarnings(True)

    if log_dir is not None:
        try:
            _crash_file = open(log_dir / CRASH_FILE_NAME, "a", encoding="utf-8")
            _crash_file.write("--- pid %d (%s) started ---\n"
                              % (os.getpid(), multiprocessing.current_process().name))
            _crash_file.flush()
            faulthandler.enable(file=_crash_file, all_threads=True)
        except Exception:
            _crash_file = None


def setup_main_logging(level=None):
    """Configure logging for the main process. Idempotent, never raises."""
    global _log_queue, _listener, _configured
    if _configured:
        return
    _configured = True

    level = (level or _log_level())
    formatter = _ISOFormatter(LOG_FORMAT)
    handlers = []

    log_dir = _resolve_log_dir()
    if log_dir is not None:
        try:
            file_handler = logging.handlers.TimedRotatingFileHandler(
                str(log_dir / LOG_FILE_NAME), when="midnight",
                backupCount=BACKUP_DAYS, encoding="utf-8", delay=True)
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)
        except Exception:
            log_dir = None
    if sys.__stderr__ is not None:
        console_handler = logging.StreamHandler(sys.__stderr__)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)

    _log_queue = multiprocessing.get_context("spawn").Queue(-1)
    _listener = logging.handlers.QueueListener(
        _log_queue, *handlers, respect_handler_level=True)
    _listener.start()

    _configure_root(_log_queue, level)
    _install_runtime_capture(log_dir)
    atexit.register(shutdown_logging)

    logger = logging.getLogger("autolume")
    try:
        from utils.resource_paths import get_version
        version = get_version()
    except Exception:
        version = "unknown"
    try:
        from utils.user_data import data_root
        root = data_root()
    except Exception:
        root = "unknown"
    frozen = bool(getattr(sys, "frozen", False))
    logger.info("Autolume %s starting: platform=%s python=%s frozen=%s "
                "data_root=%s log_dir=%s",
                version, sys.platform, sys.version.split()[0], frozen, root, log_dir)


def shutdown_logging():
    """Stop the listener (draining pending records) and restore std streams."""
    global _listener, _crash_file
    if _listener is not None:
        try:
            _listener.stop()
        except Exception:
            pass
        _listener = None
    if isinstance(sys.stdout, _StreamToLogger) and sys.__stdout__ is not None:
        sys.stdout = sys.__stdout__
    if isinstance(sys.stderr, _StreamToLogger) and sys.__stderr__ is not None:
        sys.stderr = sys.__stderr__
    if _crash_file is not None:
        try:
            faulthandler.disable()
            _crash_file.close()
        except Exception:
            pass
        _crash_file = None
    logging.shutdown()


def get_log_queue():
    """The shared log queue (None if logging was never configured)."""
    return _log_queue


def configure_child_logging(log_queue, name=None):
    """Attach this (child) process to the parent's log queue. Idempotent."""
    global _log_queue, _configured
    if _configured or log_queue is None:
        return
    _configured = True
    _log_queue = log_queue
    if name:
        multiprocessing.current_process().name = name

    _configure_root(log_queue, _log_level())

    log_dir = None
    try:
        from utils.user_data import data_path
        candidate = data_path(LOG_DIR_NAME)
        if candidate.is_dir():
            log_dir = candidate
    except Exception:
        pass
    _install_runtime_capture(log_dir)


def pool_worker_init(log_queue, name_prefix):
    """``multiprocessing.Pool`` initializer wiring workers to the log queue."""
    configure_child_logging(log_queue, "%s-%d" % (name_prefix, os.getpid()))


class LoggedProcess(multiprocessing.Process):
    """Drop-in ``multiprocessing.Process`` that wires logging in the child."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._log_queue = get_log_queue()

    def run(self):
        configure_child_logging(self._log_queue)
        try:
            super().run()
        except Exception:
            logging.getLogger("autolume").exception(
                "Unhandled exception in process %s", self.name)
            raise SystemExit(1)
