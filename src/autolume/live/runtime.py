"""Assembles and owns the live runtime's threads.

Thread inventory: control (125 Hz), render (gpu paced), audio (60 Hz),
model loader, osc server. The UI is not part of the runtime; it is one
more producer of control events and one consumer of the preview mailbox.
"""

from typing import Callable

from autolume.live.core.control import ControlLoop
from autolume.live.core.generator import ModelHost
from autolume.live.core.engine import RenderLoop
from autolume.live.core.params import ControlState, to_render_params
from autolume.live.core.sinks import PreviewMailbox
from autolume.live.core.sources import SourceTable
from autolume.live.core.store import LatestValueStore
from autolume.live.io.audio import AudioEngineLike, AudioInput
from autolume.live.io.osc import OscEmitter, OscInput


class Runtime:
    def __init__(
        self,
        model_host: ModelHost,
        osc_port: int,
        start_osc: bool,
        start_audio: bool = True,
        audio_engine: AudioEngineLike | None = None,
        emit: Callable[[str, int, str, float], None] | None = None,
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
        )
        self.render_loop = RenderLoop(
            self.render_store, self.model_host, [self.preview]
        )
        self.submit = self.control_loop.submit
        self._start_osc = start_osc
        self._start_audio = start_audio
        # `audio_engine` stays None in the app so the audio thread builds and
        # owns its engine. Only a test passes one, and only a test may hold a
        # reference to an engine another thread touches.
        self.audio = AudioInput(self.control_loop.submit, engine=audio_engine)
        self.osc = OscInput(self.control_loop.submit, port=osc_port)
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
                self.osc.start()
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


class _ModelWatchingControlLoop(ControlLoop):
    """Control loop that forwards pkl_path changes to the model host.

    Model loading is a side effect of a state change, so it triggers on
    the control tick where every state change already flows.
    """

    def __init__(
        self, control_store, render_store, source_store, model_host, **kwargs
    ) -> None:
        super().__init__(control_store, render_store, source_store, **kwargs)
        self._model_host = model_host
        self._last_pkl_path: str | None = None

    def tick(self):
        result = super().tick()
        if result.pkl_path != self._last_pkl_path:
            self._last_pkl_path = result.pkl_path
            if result.pkl_path:
                self._model_host.request_load(result.pkl_path)
        return result


def build_runtime(
    model_host: ModelHost | None = None,
    osc_port: int = 1338,
    start_osc: bool = True,
    start_audio: bool = True,
    audio_engine: AudioEngineLike | None = None,
    emit: Callable[[str, int, str, float], None] | None = None,
) -> Runtime:
    return Runtime(
        model_host=model_host or ModelHost(),
        osc_port=osc_port,
        start_osc=start_osc,
        start_audio=start_audio,
        audio_engine=audio_engine,
        emit=emit,
    )
