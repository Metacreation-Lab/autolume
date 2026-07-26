"""Assembles and owns the live runtime's threads.

Thread inventory: control (125 Hz), render (gpu paced), audio (60 Hz),
model loader, osc server. The UI is not part of the runtime; it is one
more producer of control events and one consumer of the preview mailbox.
"""

import dataclasses
import logging
from typing import Callable

from autolume.live.core.control import ControlLoop
from autolume.live.core.events import ControlEvent
from autolume.live.core.generator import ModelHost
from autolume.live.core.engine import RenderLoop
from autolume.live.core.params import ControlState, to_render_params
from autolume.live.core.sinks import PreviewMailbox
from autolume.live.core.sources import SourceTable
from autolume.live.core.store import LatestValueStore
from autolume.live.io.audio import AudioEngineLike, AudioInput
from autolume.live.io.osc import OscEmitter, OscInput

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class OscStatus:
    """What the inbound OSC transport is doing, published for the panel.

    `bound_port` is the actual port after the scan-upward behavior, the
    truth a panel must show rather than the requested value on
    `ControlState.osc_port`. `error` is set only when the most recent rebind
    attempt failed; the previous transport keeps serving in that case,
    unaffected, and `bound_port` still names the port it is serving on.
    """

    bound_port: int | None = None
    error: str | None = None


class Runtime:
    def __init__(
        self,
        model_host: ModelHost,
        osc_port: int,
        start_osc: bool,
        start_audio: bool = True,
        audio_engine: AudioEngineLike | None = None,
        emit: Callable[[str, int, str, float], None] | None = None,
        osc_factory: Callable[[int], object] | None = None,
    ) -> None:
        self.control_store = LatestValueStore(ControlState())
        self.render_store = LatestValueStore(to_render_params(ControlState()))
        self.source_store = LatestValueStore(SourceTable())
        self.model_host = model_host
        self.model_info_store = model_host.info_store
        self.preview = PreviewMailbox()
        # `emit` defaults to a real `OscEmitter`'s send, the outbound half of
        # the OSC transport. A test may inject a fake here, the same way
        # `audio_engine` lets a test stand in for the audio device, so no test
        # opens a socket to prove the pulse is wired.
        self.control_loop = _ModelWatchingControlLoop(
            self.control_store,
            self.render_store,
            self.source_store,
            model_host,
            model_info_store=self.model_info_store,
            emit=emit or OscEmitter().send,
            on_osc_port_change=self._restart_osc,
        )
        self.render_loop = RenderLoop(
            self.render_store, self.model_host, [self.preview]
        )
        self.submit = self.control_loop.submit
        self._start_osc = start_osc
        self._start_audio = start_audio
        # `osc_factory` builds the transport for both the initial start and
        # every later rebind, so a test can inject a fake transport once and
        # never open a real socket, the same way `audio_engine` stands in for
        # the audio device.
        self._osc_factory = osc_factory or (
            lambda port: OscInput(self.control_loop.submit, port=port)
        )
        self.osc = self._osc_factory(osc_port)
        self.osc_status_store: LatestValueStore[OscStatus] = LatestValueStore(
            OscStatus()
        )
        # `audio_engine` stays None in the app so the audio thread builds and
        # owns its engine. Only a test passes one, and only a test may hold a
        # reference to an engine another thread touches.
        self.audio = AudioInput(self.control_loop.submit, engine=audio_engine)
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        # Flagged as started before anything is, so that a subsystem failing to
        # come up can be unwound by the same stop() a normal shutdown uses. The
        # OSC transport raises when no port in its range binds, and leaving the
        # audio thread behind for that would hold the input device open with no
        # way left to release it.
        self._started = True
        try:
            self.control_loop.start()
            self.render_loop.start()
            if self._start_audio:
                self.audio.start()
            if self._start_osc:
                bound_port = self.osc.start()
                self.osc_status_store.set(OscStatus(bound_port=bound_port, error=None))
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        self.osc.stop()
        self.audio.stop()
        self.render_loop.stop()
        self.control_loop.stop()
        self.model_host.stop()

    def _restart_osc(self, new_port: int) -> None:
        """Rebind the OSC transport to `new_port`, called from the control
        thread on an `osc_port` transition.

        The replacement is built and started before the previous transport is
        touched, so a bind failure never drops the one currently serving:
        losing OSC mid-performance over a taken port would be a serious
        regression. Never raises; a failure is logged and left on
        `osc_status_store` for the panel instead.

        Guarded by `_started` both before and after the (possibly slow)
        `candidate.start()` call: `stop()` sets `_started = False` as its
        very first action, before it ever touches `self.osc`, so a rebind
        racing a shutdown either bails before opening a socket at all, or
        notices right after and stops the candidate itself rather than
        swapping it in behind `stop()`'s back and leaving a live,
        unstoppable transport (and a concurrent double `.stop()` on the one
        `stop()` is already tearing down).
        """
        if not self._start_osc or not self._started:
            return
        try:
            candidate = self._osc_factory(new_port)
            bound_port = candidate.start()
        except Exception as exc:
            logger.warning("Could not rebind OSC to port %s: %s", new_port, exc)
            previous = self.osc_status_store.snapshot()
            self.osc_status_store.set(dataclasses.replace(previous, error=str(exc)))
            return
        if not self._started:
            try:
                candidate.stop()
            except Exception:
                logger.exception("Failed stopping an orphaned OSC transport")
            return
        previous_osc = self.osc
        self.osc = candidate
        try:
            previous_osc.stop()
        except Exception:
            logger.exception("Failed stopping the previous OSC transport")
        self.osc_status_store.set(OscStatus(bound_port=bound_port, error=None))


class _ModelWatchingControlLoop(ControlLoop):
    """Control loop that forwards pkl_path, device and osc_port changes to
    the model host, the render device, and the OSC transport respectively.

    All three are side effects of a state change, so they trigger on the
    control tick where every state change already flows.
    """

    def __init__(
        self,
        control_store,
        render_store,
        source_store,
        model_host,
        on_osc_port_change: Callable[[int], None] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(control_store, render_store, source_store, **kwargs)
        self._model_host = model_host
        self._on_osc_port_change = on_osc_port_change
        self._last_pkl_path: str | None = None
        # Seeded from the store directly, at construction, rather than
        # lazily adopted on the first tick: nothing can have submitted an
        # event yet (this object's own `submit` does not exist until this
        # constructor returns), so this is guaranteed to be the pristine
        # value nothing has requested changing. A lazy "adopt on first
        # tick" sentinel would silently skip forwarding a device or
        # osc_port change that happened to already be applied to state by
        # the very first tick this loop ever runs.
        initial = control_store.snapshot()
        self._last_device: str = initial.device
        self._seen_device_status = None
        self._last_osc_port: int = initial.osc_port

    def tick(self):
        result = super().tick()
        if result.pkl_path != self._last_pkl_path:
            self._last_pkl_path = result.pkl_path
            if result.pkl_path:
                self._model_host.request_load(result.pkl_path)
        self._watch_device()
        self._watch_osc_port()
        return result

    def _watch_device(self) -> None:
        """Forward a `device` transition to the host, and revert it if the
        host reports the switch failed.

        Edge triggered exactly like `pkl_path` above: `_last_device` moves to
        the new value the instant the switch is requested, not once it
        completes, so a request in flight is never re-issued every tick.

        A failed switch reverts to `status.active`, the device string
        `ModelHost` read straight off the still-current model at the moment
        it recorded the failure, by definition where rendering is actually
        happening right now. Deliberately not a value this loop tracks
        itself (an earlier version kept "the last device a status
        confirmed", updated only once a tick had actually observed that
        status): that bookkeeping lags one tick behind the loader thread,
        so a second switch fired before this loop ever ticked past the
        first one's success would revert past it, to whatever was
        confirmed before *that* one. `status.active` carries no such lag,
        since `ModelHost` reads it fresh from the model at failure time,
        not from anything this loop reported earlier.
        """
        state = self._control_store.snapshot()
        if state.device != self._last_device:
            self._last_device = state.device
            self._model_host.request_device(state.device)

        status = self._model_host.device_store.snapshot()
        if status is self._seen_device_status:
            return
        self._seen_device_status = status
        if status.requested != self._last_device:
            return
        if status.error:
            reverted = status.active if status.active is not None else "auto"
            self._last_device = reverted
            # source="ui": this correction must always land, whatever the
            # mapping panel has bound to /render/device, and "ui" is the one
            # source `_apply`'s remote gate never blocks (see
            # `ControlLoop._accepts_direct`). It is not a stand-in for a real
            # click; it is the one source that means "trusted, apply it".
            self.submit(ControlEvent("/render/device", reverted, source="ui"))

    def _watch_osc_port(self) -> None:
        if self._on_osc_port_change is None:
            return
        port = self._control_store.snapshot().osc_port
        if port != self._last_osc_port:
            self._last_osc_port = port
            self._on_osc_port_change(port)


def build_runtime(
    model_host: ModelHost | None = None,
    osc_port: int = 1338,
    start_osc: bool = True,
    start_audio: bool = True,
    audio_engine: AudioEngineLike | None = None,
    emit: Callable[[str, int, str, float], None] | None = None,
    osc_factory: Callable[[int], object] | None = None,
) -> Runtime:
    return Runtime(
        model_host=model_host or ModelHost(),
        osc_port=osc_port,
        start_osc=start_osc,
        start_audio=start_audio,
        audio_engine=audio_engine,
        emit=emit,
        osc_factory=osc_factory,
    )
