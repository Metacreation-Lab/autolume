"""Frame sinks: consumers of rendered frames, fed by the render loop."""

import threading
from typing import Protocol

import numpy as np


class FrameSink(Protocol):
    def on_frame(self, frame: np.ndarray, seq: int) -> None: ...


class PreviewMailbox:
    """Latest-frame mailbox for the UI. A slow reader simply misses frames."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seq = 0
        self._frame: np.ndarray | None = None

    def on_frame(self, frame: np.ndarray, seq: int) -> None:
        with self._lock:
            self._seq = seq
            self._frame = frame

    def latest(self) -> tuple[int, np.ndarray | None]:
        with self._lock:
            return self._seq, self._frame
