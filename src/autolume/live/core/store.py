"""Thread-safe latest-value snapshot holder.

Writers swap in a freshly built immutable snapshot under a lock; readers
get the current instance directly. Writes between reads coalesce to the
newest value by construction.
"""

import dataclasses
import threading
from typing import Generic, TypeVar

T = TypeVar("T")


class LatestValueStore(Generic[T]):
    def __init__(self, initial: T) -> None:
        self._lock = threading.Lock()
        self._current = initial

    def snapshot(self) -> T:
        with self._lock:
            return self._current

    def update(self, **changes: object) -> T:
        with self._lock:
            self._current = dataclasses.replace(self._current, **changes)
            return self._current

    def set(self, value: T) -> None:
        with self._lock:
            self._current = value
