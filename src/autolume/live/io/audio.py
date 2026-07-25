"""Audio input transport: analysis on its own thread, features become events.

The engine is owned by this thread and touched by nothing else. UI calls
queue a command that the next tick applies, so opening a device never
blocks the caller, and the UI reads one immutable status snapshot instead
of the engine's live attributes.
"""

import collections
import dataclasses
import logging
import threading
import time
from types import MappingProxyType
from typing import Callable, Mapping, Protocol

import numpy as np

from autolume.audio.features import FEATURE_NAMES, ONSET_SENSITIVITY_DEFAULT
from autolume.live.core.events import ControlEvent
from autolume.live.core.store import LatestValueStore

logger = logging.getLogger(__name__)

ADDRESS_PREFIX = "/audio/"
SOURCE = "audio"

_COMMAND_LIMIT = 64
_GUARD_REPEAT_INTERVAL = 1000
_IDLE_FEATURES = MappingProxyType({name: 0.0 for name in FEATURE_NAMES})
_STUCK_THREAD_ERROR = "Audio thread did not stop. Restart Autolume to use audio again."


class AudioEngineLike(Protocol):
    """The engine surface the audio thread depends on."""

    devices: tuple[tuple[int, str], ...]
    device_pos: int
    features: Mapping[str, float]
    spectrum: np.ndarray | None
    error: str | None
    onset_sensitivity: float

    # Read only on the real engine, so they are declared read only here: an
    # attribute member would not be satisfied by a property.
    @property
    def enabled(self) -> bool: ...

    @property
    def sample_rate(self) -> int: ...

    def enable(self) -> None: ...
    def disable(self) -> None: ...
    def select_device(self, pos: int) -> None: ...
    def set_onset_sensitivity(self, value: float) -> None: ...
    def refresh(self) -> None: ...
    def update(self) -> None: ...


# eq=False: the ndarray field makes a generated __eq__ raise and the mapping
# makes the matching __hash__ raise, so identity comparison is the safe default
# for a snapshot that only ever needs to be read.
@dataclasses.dataclass(frozen=True, eq=False)
class AudioStatus:
    enabled: bool = False
    devices: tuple[tuple[int, str], ...] = ()
    device_pos: int = 0
    features: Mapping[str, float] = _IDLE_FEATURES
    spectrum: np.ndarray | None = None
    error: str | None = None
    onset_sensitivity: float = ONSET_SENSITIVITY_DEFAULT
    sample_rate: int = 0


def _build_default_engine() -> AudioEngineLike:
    # Imported here, not at module scope: importing the engine pulls in
    # sounddevice and constructing it enumerates devices, neither of which
    # may happen just because something imported this module.
    from autolume.audio.engine import AudioEngine

    # The engine's own publish hook stays unused. Features are submitted by
    # the thread after every update, which is the one path a fake engine in a
    # test also exercises.
    return AudioEngine(lambda features: None)


class AudioInput:
    def __init__(
        self,
        submit: Callable[[ControlEvent], None],
        engine: AudioEngineLike | None = None,
        rate_hz: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._submit = submit
        self._engine = engine
        self._period = 1.0 / rate_hz
        self._clock = clock
        self._store: LatestValueStore[AudioStatus] = LatestValueStore(AudioStatus())
        self._commands: collections.deque[tuple[str, tuple]] = collections.deque(
            maxlen=_COMMAND_LIMIT
        )
        self._command_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._engine_failed = False
        self._tick_failures = 0

    def status(self) -> AudioStatus:
        return self._store.snapshot()

    def enable(self) -> None:
        self._command("enable")

    def disable(self) -> None:
        self._command("disable")

    def select_device(self, pos: int) -> None:
        self._command("select_device", int(pos))

    def set_onset_sensitivity(self, value: float) -> None:
        self._command("set_onset_sensitivity", float(value))

    def refresh(self) -> None:
        # Refresh is the one command a user reaches for when audio is missing,
        # so it also lifts a terminal build failure and lets the next tick try
        # again. Without it a failed build would swallow every later command.
        self._engine_failed = False
        self._command("refresh")

    def tick(self) -> None:
        engine = self._engine
        try:
            if engine is None:
                # A build failure means there is no audio subsystem at all, so
                # it is terminal. Retrying would re-enumerate devices 60 times
                # a second for as long as the runtime lives.
                if self._engine_failed:
                    return
                try:
                    engine = self._engine = _build_default_engine()
                except Exception:
                    self._engine_failed = True
                    raise
            for name, args in self._drain():
                getattr(engine, name)(*args)
            engine.update()
            if engine.enabled:
                self._submit_features(engine.features)
            self._store.set(self._snapshot(engine))
        except Exception as exc:
            # Keeps the last good status so the UI still shows its device
            # list, with the failure attached.
            logger.exception("Audio tick failed")
            self._store.set(
                dataclasses.replace(self._store.snapshot(), error=_describe(exc))
            )

    def start(self) -> None:
        if self._thread is not None:
            return
        self._running.set()
        self._thread = threading.Thread(target=self._run, name="audio", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
            if thread.is_alive():
                # The reference stays so a later start cannot put a second
                # thread on the same engine, which also means this instance is
                # done for good. The UI has to be able to say so.
                logger.warning("Audio thread did not stop, leaving the device open")
                self._store.set(
                    dataclasses.replace(
                        self._store.snapshot(), error=_STUCK_THREAD_ERROR
                    )
                )
                return
            self._thread = None
        # The thread is gone, so the engine is unowned and this is the only
        # place another thread may touch it.
        engine = self._engine
        if engine is None:
            return
        try:
            engine.disable()
            self._store.set(self._snapshot(engine))
        except Exception:
            logger.exception("Audio engine shutdown failed")

    def _run(self) -> None:
        deadline = time.monotonic() + self._period
        while self._running.is_set():
            # tick() guards its own body, so reaching this is near impossible,
            # but an unguarded raise here would kill audio silently: the status
            # would freeze, the device would stay open and nothing would
            # restart the thread.
            try:
                self.tick()
            except Exception:
                self._report_tick_failure()
            remaining = deadline - time.monotonic()
            if remaining > 0.0:
                time.sleep(remaining)
            deadline += self._period
            now = time.monotonic()
            if deadline < now:
                deadline = now + self._period

    def _report_tick_failure(self) -> None:
        """Log a guard hit, throttled like the control loop throttles its own.

        Whatever breaks tick()'s own guard is a persistent condition, not a bad
        input, so it recurs on every tick: at 60 Hz an unthrottled traceback
        would bury the log and eat the budget. One call site and no input
        derived keys, so a single counter replaces the control loop's keyed
        bookkeeping. Must be called from an `except` block.
        """
        self._tick_failures += 1
        if self._tick_failures == 1:
            logger.exception("Audio tick failed outside its own guard")
        elif self._tick_failures % _GUARD_REPEAT_INTERVAL == 0:
            logger.error(
                "Audio tick failed outside its own guard (%d more suppressed)",
                _GUARD_REPEAT_INTERVAL - 1,
            )

    def _command(self, name: str, *args: object) -> None:
        with self._command_lock:
            self._commands.append((name, args))

    def _drain(self) -> list[tuple[str, tuple]]:
        with self._command_lock:
            commands = list(self._commands)
            self._commands.clear()
        return commands

    def _submit_features(self, features: Mapping[str, float]) -> None:
        timestamp = self._clock()
        for name, value in features.items():
            self._submit(
                ControlEvent(
                    address=ADDRESS_PREFIX + name,
                    value=float(value),
                    source=SOURCE,
                    timestamp=timestamp,
                )
            )

    def _snapshot(self, engine: AudioEngineLike) -> AudioStatus:
        spectrum = engine.spectrum
        if spectrum is not None:
            spectrum = np.array(spectrum, copy=True)
        return AudioStatus(
            enabled=bool(engine.enabled),
            devices=tuple((int(index), str(label)) for index, label in engine.devices),
            device_pos=int(engine.device_pos),
            features=MappingProxyType(
                {str(name): float(value) for name, value in engine.features.items()}
            ),
            spectrum=spectrum,
            error=engine.error,
            onset_sensitivity=float(engine.onset_sensitivity),
            sample_rate=int(engine.sample_rate),
        )


def _describe(exc: Exception) -> str:
    return str(exc) or type(exc).__name__
