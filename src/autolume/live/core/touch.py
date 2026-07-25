"""Tracks which parameters the performer is holding right now.

While a widget is dragged, and for a short grace after release, binding
writes to that parameter are suppressed so a hand and a modulation source
stop fighting over the same value. Begin and end arrive as control events
from the UI thread while the control thread reads the tracker, so the
state is guarded by its own lock.
"""

import threading

TOUCH_GRACE = 0.25

TOUCH_BEGIN = "/touch/begin"
TOUCH_END = "/touch/end"


class TouchTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Name to release time, or None while the name is still held.
        self._touches: dict[str, float | None] = {}

    def begin(self, name: str, now: float) -> None:
        with self._lock:
            self._touches[name] = None

    def end(self, name: str, now: float) -> None:
        with self._lock:
            if name in self._touches:
                self._touches[name] = now

    def is_held(self, name: str, now: float) -> bool:
        with self._lock:
            if name not in self._touches:
                return False
            released = self._touches[name]
        return released is None or now - released < TOUCH_GRACE
