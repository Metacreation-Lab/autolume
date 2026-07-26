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
from autolume.live.core.sources import SourceTable, as_float, canonical_address
from autolume.live.core.store import LatestValueStore
from autolume.live.io.audio import AudioEngineLike, AudioInput
from autolume.live.io.ndi import NdiSink
from autolume.live.io.osc import OscEmitter, OscInput
from autolume.live.io.recorder import (
    SCREENSHOT_ADDRESS,
    Recorder,
    ScreenshotWorker,
    capture_path,
)

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
        # Output sinks. Both are registered with the fan-out for the whole
        # session and are inert until their parameter turns them on, so the
        # render thread never sees the sink list change under it.
        self.ndi = NdiSink()
        self.recorder = Recorder()
        self.screenshots = ScreenshotWorker()
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
            ndi=self.ndi,
            recorder=self.recorder,
            on_screenshot=self._request_screenshot,
            is_running=lambda: self._started,
        )
        self.render_loop = RenderLoop(
            self.render_store,
            self.model_host,
            [self.preview, self.ndi, self.recorder],
            screenshot=self.screenshots.save_png,
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
        # The control loop before the sinks it drives. Its watchers check
        # `_is_running` and then call `start`, which is two statements with a
        # preemption point between them, so stopping a sink while a tick can
        # still be inside that window leaves behind a thread nothing is left
        # to stop.
        #
        # What this buys is that a tick which finishes normally is finished
        # before any sink is asked to stop. It is not an absolute guarantee:
        # `ControlLoop.stop()` joins with its own two second timeout and
        # gives up quietly, so a tick that overruns that (measured with an
        # artificial 2.6 s tick) can still start a sink behind this. Closing
        # that residual belongs in `core/control.py`, which reports nothing
        # when its join expires.
        self.control_loop.stop()
        # Sinks before the render loop: each one stops accepting frames the
        # moment it is told to, so the frames the loop is still fanning out
        # while it winds down go nowhere, and the sender and the writer are
        # provably released before the process ends. The screenshot worker
        # goes after the loop instead, so a request latched on the last frame
        # is still written rather than dropped on the doorstep.
        self.ndi.stop()
        # `abort_on_timeout`: the encoder is a daemon thread, so a flush that
        # outlives the join is abandoned mid write and the mp4 never gets its
        # header, which loses the whole take rather than its tail. A short
        # recording beats an unopenable one.
        self.recorder.stop(abort_on_timeout=True)
        self.render_loop.stop()
        self.screenshots.stop()
        self.model_host.stop()

    def _request_screenshot(self, path: str) -> None:
        """Latch a screenshot on the render loop, from the control thread.

        A method rather than the render loop's own bound method, because the
        control loop is built before the render loop exists.
        """
        self.render_loop.request_screenshot(path)

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
    """Control loop that turns state changes into the side effects that live
    outside state: model loads, device switches, the OSC transport's port, and
    the output sinks' lifecycles.

    All of them are side effects of a state change, so they trigger on the
    control tick where every state change already flows.
    """

    def __init__(
        self,
        control_store,
        render_store,
        source_store,
        model_host,
        on_osc_port_change: Callable[[int], None] | None = None,
        ndi=None,
        recorder=None,
        on_screenshot: Callable[[str], None] | None = None,
        is_running: Callable[[], bool] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(control_store, render_store, source_store, **kwargs)
        self._model_host = model_host
        self._on_osc_port_change = on_osc_port_change
        self._ndi = ndi
        self._recorder = recorder
        self._on_screenshot = on_screenshot
        # Second layer under `Runtime.stop()`'s ordering, which stops this
        # loop before the sinks it drives. That ordering is what actually
        # closes the window, but its join can expire on a badly overrunning
        # tick, and this keeps such a tick from starting a sink the runtime
        # has already torn down. Stops are always allowed.
        self._is_running = is_running or (lambda: True)
        self._last_pkl_path: str | None = None
        # Seeded from the store directly, at construction, rather than
        # lazily adopted on the first tick: nothing can have submitted an
        # event yet (this object's own `submit` does not exist until this
        # constructor returns), so this is guaranteed to be the pristine
        # value nothing has requested changing. A lazy "adopt on first
        # tick" sentinel would silently skip forwarding a device change
        # that happened to already be applied to state by the very first
        # tick this loop ever runs. `request_device` is safe to call at
        # any time (it only ever touches `ModelHost`'s own lock-guarded
        # state), so there is no cost to seeding it eagerly.
        initial = control_store.snapshot()
        self._last_device: str = initial.device
        self._seen_device_status = None
        # Deliberately still lazy (None, adopted on the first tick,
        # unlike `_last_device` above): eager seeding here can fire
        # `_restart_osc` on the very first tick, while `Runtime.start()`
        # is between `control_loop.start()` and `self.osc.start()` and
        # `_started` is already True. That races a second real bind and
        # `serve_forever` thread against the one `start()` is about to
        # create. Nothing needs osc_port forwarded before the transport
        # this loop would be restarting has even started once.
        self._last_osc_port: int | None = None
        # Seeded eagerly, like `_last_device` and for the same reason: both
        # sinks are safe to start or stop at any moment, so there is no
        # first-tick hazard, and a lazy sentinel could swallow the very
        # transition it exists to notice.
        self._last_recording: bool = initial.recording
        self._last_ndi_enabled: bool = initial.ndi_enabled
        self._last_ndi_name: str = initial.ndi_name

    def tick(self):
        result = super().tick()
        if result.pkl_path != self._last_pkl_path:
            self._last_pkl_path = result.pkl_path
            if result.pkl_path:
                self._model_host.request_load(result.pkl_path)
        self._watch_device()
        self._watch_osc_port()
        self._watch_ndi()
        self._watch_recording()
        return result

    def _apply(self, state, sources, event, now):
        """Intercept the one structured address that is an action, not a value.

        `/capture/screenshot` asks for a file and changes nothing about the
        show, so it never reaches the mapping. It is also deliberately
        reachable from raw OSC, which is why the message's value never names
        the file: the name is derived here, under the captures folder, and a
        path arriving from the network would be a write-anywhere primitive
        rather than a screenshot button.

        The value still decides *whether* to fire. A momentary button sends 1
        on the press and 0 on the release, and firing on the address alone
        made every press take two pictures.
        """
        if canonical_address(event.address) == SCREENSHOT_ADDRESS:
            if _is_trigger(event.value):
                self._capture_screenshot(state)
            return state, sources
        return super()._apply(state, sources, event, now)

    def _capture_screenshot(self, state: ControlState) -> None:
        if self._on_screenshot is None:
            return
        path = self._capture_path(state, ".png")
        if path is not None:
            self._on_screenshot(path)

    def _watch_ndi(self) -> None:
        """Drive the NDI sink from `ndi_enabled` and `ndi_name`.

        Edge triggered like every other watcher here. `stop` is called with
        no wait at all: a send that has gone slow must never hold the control
        thread, and the sink destroys its own sender either way.

        The last branch is the sink reporting back. A session that failed (no
        library, a sender that would not open, a send that raised) is over
        whatever the parameter says, so the parameter goes back rather than
        leaving a checkbox on with nothing behind it. The reason stays on the
        sink's status for the panel to show.
        """
        if self._ndi is None:
            return
        state = self._control_store.snapshot()
        if state.ndi_enabled != self._last_ndi_enabled:
            self._last_ndi_enabled = state.ndi_enabled
            self._last_ndi_name = state.ndi_name
            if state.ndi_enabled:
                if self._is_running():
                    self._ndi.start(state.ndi_name)
            else:
                self._ndi.stop(timeout=0.0)
            return
        if state.ndi_name != self._last_ndi_name:
            self._last_ndi_name = state.ndi_name
            if state.ndi_enabled:
                self._ndi.set_name(state.ndi_name)
            return
        # `_is_running` again: a tick that outlives shutdown's join sees
        # "enabled but not sending" for every sink already torn down, and
        # there is no panel left to correct by then anyway.
        if state.ndi_enabled and not self._ndi.status().sending and self._is_running():
            self._last_ndi_enabled = False
            self._ndi.stop(timeout=0.0)
            self.submit(ControlEvent("/ndi/enabled", False, source="ui"))

    def _watch_recording(self) -> None:
        """Drive the recorder from `recording`.

        The take's frame rate is the fps cap as it stands when Record is
        pressed, because a `VideoWriter` names its rate once. `stop` waits
        for nothing: the tail of a take is the encoder thread's work, not the
        control thread's, which is the whole of legacy bug 3.

        The last branch mirrors `_watch_ndi`'s: a take that ended itself (the
        frame size changed, the file would not open) puts the parameter back,
        so Record does not stay lit over a recording that is not happening.
        """
        if self._recorder is None:
            return
        state = self._control_store.snapshot()
        if state.recording != self._last_recording:
            self._last_recording = state.recording
            if state.recording:
                path = self._capture_path(state, ".mp4")
                if path is None:
                    self._last_recording = False
                    self.submit(ControlEvent("/record", False, source="ui"))
                    return
                if self._is_running():
                    self._recorder.start(path, state.fps_cap)
            else:
                self._recorder.stop(timeout=0.0)
            return
        if (
            state.recording
            and not self._recorder.status().recording
            and self._is_running()
        ):
            self._last_recording = False
            self._recorder.stop(timeout=0.0)
            self.submit(ControlEvent("/record", False, source="ui"))

    def _capture_path(self, state: ControlState, extension: str) -> str | None:
        """Where this capture goes, or None if that cannot be worked out.

        Resolving the data root reads the preferences file, so it is wrapped:
        the control thread does not raise, and a capture nobody can name is
        not worth stopping a show over.
        """
        try:
            return capture_path(state.pkl_path, extension)
        except Exception:
            logger.exception("Could not work out where to save a capture")
            return None

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
        if self._last_osc_port is None:
            self._last_osc_port = port
            return
        if port != self._last_osc_port:
            self._last_osc_port = port
            self._on_osc_port_change(port)


def _is_trigger(value: object) -> bool:
    """Whether a bare action message means "now" rather than "not any more".

    A number is a switch: zero is a release, anything else is a press. A
    bang carries no number at all (`OscInput` gives it 1.0, a UI click sends
    the same), and neither does a string, so anything unnumbered fires.
    """
    number = as_float(value)
    return number is None or number != 0.0


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
