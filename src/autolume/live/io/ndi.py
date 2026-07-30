"""The NDI output sink: one sender, fed from the render fan-out.

The old app converted and sent from the UI thread, which put the show's
output rate behind whatever the network was doing (constraints.md legacy bug
8). Here the render loop only drops a frame into a one-slot mailbox and
returns; conversion and `send_send_video_v2` happen on this sink's own
thread, and a receiver that cannot keep up costs frames rather than fps.

`NDIlib` is optional. A machine without it must grey the checkbox, not fail
to start, so the import is guarded and `available()` is what the UI asks.
"""

import dataclasses
import logging
import re
import threading

import numpy as np

from autolume.live.core.store import LatestValueStore
from autolume.live.errors import describe, safe_describe

try:
    import NDIlib
except Exception:  # pragma: no cover - depends on what is installed
    # Not just ImportError: the wheel loads a native runtime, and a missing
    # or mismatched one surfaces as an OSError from the loader.
    NDIlib = None

logger = logging.getLogger(__name__)

STOP_TIMEOUT = 5.0
_IDLE_WAIT = 0.05
# Past this many distinct causes the sink has bigger problems than a missing
# log line (mirrors superres.py's cap, which says why it went quiet).
_LOG_ONCE_CAP = 64
_DIGIT_RUN = re.compile(r"\d+")

_NO_LIBRARY = "NDI is not installed on this machine."
_STILL_STOPPING = "NDI is still stopping. Try again in a moment."


def available() -> bool:
    """Whether NDI output can be offered at all on this machine."""
    return NDIlib is not None


@dataclasses.dataclass(frozen=True)
class NdiStatus:
    """What the NDI sink is doing, published for the performance panel.

    `sending` is the truth to show rather than the `ndi_enabled` parameter: a
    failed send disables the sink and says why here, and the runtime follows
    by putting the parameter back.
    """

    sending: bool = False
    name: str = ""
    error: str | None = None


class NdiSink:
    """Sends rendered frames out over NDI from its own thread.

    One sender, created when the sink is enabled and destroyed when it stops.
    A name change recreates it, because the name is what a receiver finds the
    source by and NDI cannot rename a live one.
    """

    def __init__(self, stop_timeout: float = STOP_TIMEOUT) -> None:
        self._stop_timeout = float(stop_timeout)
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._name = ""
        self._wake = threading.Event()
        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        # Read by the render thread on every frame, so it is a plain flag and
        # not a lock acquisition.
        self._enabled = False
        self._store: LatestValueStore[NdiStatus] = LatestValueStore(NdiStatus())
        self._logged_errors: set[str] = set()
        self._log_cap_warned = False

    def status(self) -> NdiStatus:
        return self._store.snapshot()

    def start(self, name: str) -> None:
        """Begin sending as `name`. Called from the control thread: never blocks."""
        if self._enabled:
            return
        if NDIlib is None:
            logger.info("Ignoring an NDI request, the library is not installed")
            self._store.set(NdiStatus(sending=False, name=str(name), error=_NO_LIBRARY))
            return
        thread = self._thread
        if thread is not None and thread.is_alive():
            # The previous thread still owns a sender advertising this name.
            # Creating a second one behind it would put two sources with the
            # same name on the network, so the request is refused with a
            # reason the panel can show instead.
            logger.info("Ignoring an NDI request while the previous sender is stopping")
            self._store.set(
                dataclasses.replace(self.status(), sending=False, error=_STILL_STOPPING)
            )
            return
        with self._lock:
            self._name = str(name)
            self._frame = None
        self._enabled = True
        self._wake.clear()
        self._running.set()
        self._store.set(NdiStatus(sending=True, name=str(name), error=None))
        self._thread = threading.Thread(target=self._run, name="ndi", daemon=True)
        self._thread.start()

    def set_name(self, name: str) -> None:
        """Rename the source. The sink thread recreates its sender to match."""
        if not self._enabled:
            return
        with self._lock:
            self._name = str(name)
        self._wake.set()
        self._store.set(dataclasses.replace(self.status(), name=str(name)))

    def stop(self, timeout: float | None = None) -> None:
        """Stop sending and destroy the sender.

        `timeout` bounds the join. The control thread passes 0.0: a send that
        has gone slow must never hold the show's heartbeat, and the sender is
        destroyed by the sink thread itself either way. Shutdown passes the
        full timeout so the sender is provably gone before the process ends.
        """
        self._enabled = False
        thread = self._thread
        if thread is None:
            return
        self._running.clear()
        self._wake.set()
        # An error already on the status is why this is being stopped, so it
        # survives; a clean stop has nothing to preserve.
        self._store.set(dataclasses.replace(self.status(), sending=False))
        thread.join(self._stop_timeout if timeout is None else float(timeout))
        if thread.is_alive():
            logger.info("The NDI sender is still stopping in the background")
            return
        self._thread = None

    def on_frame(self, frame: np.ndarray, seq: int) -> None:
        """Post one rendered frame to the mailbox. Runs on the render thread.

        Latest wins: a frame that has not been sent yet is simply replaced.
        """
        if not self._enabled:
            return
        with self._lock:
            self._frame = frame
        self._wake.set()

    def _run(self) -> None:
        ndi = NDIlib
        sender = None
        current_name = None
        reason: str | None = None
        try:
            import cv2

            if not ndi.initialize():
                reason = "The NDI runtime could not be started."
                logger.warning(reason)
                return
            video = ndi.VideoFrameV2()
            # The SDK's send is asynchronous and reads the buffer after it
            # returns, and the frame object holds the array rather than
            # copying it, so the one sent last stays referenced for one more
            # iteration instead of being freed the moment the next assignment
            # replaces it.
            previous_data = None
            while self._running.is_set():
                # Cleared before the mailbox is read, so a frame posted while
                # this thread is busy sending is never mistaken for one that
                # was already picked up.
                self._wake.clear()
                with self._lock:
                    name = self._name
                    frame = self._frame
                    self._frame = None
                if sender is None or name != current_name:
                    if sender is not None:
                        self._destroy(ndi, sender)
                    sender = ndi.send_create(self._settings(ndi, name))
                    current_name = name
                    if sender is None:
                        reason = f"Could not create an NDI sender named {name}."
                        logger.warning(reason)
                        return
                if frame is None:
                    self._wake.wait(_IDLE_WAIT)
                    continue
                try:
                    data = cv2.cvtColor(frame, cv2.COLOR_RGB2BGRA)
                    video.data = data
                    video.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRX
                    ndi.send_send_video_v2(sender, video)
                    previous_data = data
                except Exception as exc:
                    reason = self._record_failure("Sending an NDI frame failed", exc)
                    return
        except Exception as exc:
            logger.exception("The NDI sink thread failed")
            reason = f"NDI stopped. {describe(exc)}"
        finally:
            if sender is not None:
                self._destroy(ndi, sender)
            # Deliberately no `ndi.destroy()` here, and this is not an
            # oversight. `initialize` reference counts, so pairing it per
            # session leaks one count for the life of the process, which
            # looks like the thing to fix. It is not: `destroy` is
            # process-wide and unloads the SDK at zero, and
            # `modules/visualizer.py` creates a sender in this same process
            # without ever calling `initialize`. Pairing them here would
            # take the count to zero and tear the runtime out from under a
            # legacy sender that is still sending, every time this checkbox
            # is unticked. The leak is a refcount only: measured across five
            # initialize calls with no sockets, no descriptors and no growth,
            # and it never blocks a later enable. That is the cheaper of the
            # two costs. Do not add it back without owning the legacy
            # surface's lifetime too.
            self._finish(reason)

    @staticmethod
    def _settings(ndi, name: str):
        settings = ndi.SendCreate()
        settings.ndi_name = name
        return settings

    @staticmethod
    def _destroy(ndi, sender) -> None:
        try:
            ndi.send_destroy(sender)
        except Exception:
            logger.exception("Destroying the NDI sender failed")

    def _finish(self, reason: str | None) -> None:
        """Close the session out from the sink thread, whatever ended it."""
        self._enabled = False
        previous = self.status()
        self._store.set(
            NdiStatus(
                sending=False, name=previous.name, error=reason or previous.error
            )
        )

    def _record_failure(self, context: str, exc: Exception) -> str:
        """Note a failure, logged once per cause rather than once per frame.

        The key normalises digit runs, following `superres.py`: a driver
        message carrying a frame number or a byte count is one cause, and
        keying on the raw text would put a traceback in the log for every
        frame of a failing session.
        """
        text = safe_describe(exc)
        message = f"{context}: {text}"
        key = f"{type(exc).__name__}:{_DIGIT_RUN.sub('N', text)}"
        if key in self._logged_errors:
            return message
        if len(self._logged_errors) >= _LOG_ONCE_CAP:
            if not self._log_cap_warned:
                self._log_cap_warned = True
                logger.warning(
                    "Reached %d distinct NDI failure causes, further distinct "
                    "causes will not be logged",
                    _LOG_ONCE_CAP,
                )
            return message
        self._logged_errors.add(key)
        logger.warning(message)
        return message
