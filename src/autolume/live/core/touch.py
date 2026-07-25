"""Tracks which parameters the performer is holding right now.

While a widget is dragged, and for a short grace after release, binding
writes to that parameter are suppressed so a hand and a modulation source
stop fighting over the same value. Begin and end arrive as control events
from the UI thread while the control thread reads the tracker, so the
state is guarded by its own lock.

A hold also expires on its own after `TOUCH_HOLD_LIMIT`. A panel closed or
a widget destroyed mid drag never sends its end, and without a ceiling that
parameter would ignore its binding for the rest of the show with nothing to
show for it. Lapsing early only means the binding starts fighting the hand
again, which is visible and self correcting.
"""

import logging
import threading

logger = logging.getLogger(__name__)

TOUCH_GRACE = 0.25
TOUCH_HOLD_LIMIT = 60.0

TOUCH_BEGIN = "/touch/begin"
TOUCH_END = "/touch/end"


class TouchTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Name to (begin time, release time), release being None while held.
        self._touches: dict[str, tuple[float, float | None]] = {}

    def begin(self, name: str, now: float) -> None:
        with self._lock:
            self._touches[name] = (now, None)

    def end(self, name: str, now: float) -> None:
        with self._lock:
            touch = self._touches.get(name)
            if touch is not None:
                self._touches[name] = (touch[0], now)

    def is_held(self, name: str, now: float) -> bool:
        with self._lock:
            touch = self._touches.get(name)
            if touch is None:
                return False
            begun, released = touch
            if released is not None:
                return now - released < TOUCH_GRACE
            if now - begun < TOUCH_HOLD_LIMIT:
                return True
            del self._touches[name]
        # Always a UI bug: something began a touch and never ended it. Logged
        # outside the lock, and at most once per lapse since the entry is gone.
        logger.warning(
            "Releasing %s after %.0fs without a touch end", name, now - begun
        )
        return False
