"""Assembles and owns the live runtime's threads.

Thread inventory: control (125 Hz), render (gpu paced), model loader,
osc server. The UI is not part of the runtime; it is one more producer
of control events and one consumer of the preview mailbox.
"""

from autolume.live.core.control import ControlLoop
from autolume.live.core.generator import ModelHost
from autolume.live.core.engine import RenderLoop
from autolume.live.core.params import ControlState, to_render_params
from autolume.live.core.sinks import PreviewMailbox
from autolume.live.core.store import LatestValueStore
from autolume.live.io.osc import OscInput


class Runtime:
    def __init__(self, model_host: ModelHost, osc_port: int, start_osc: bool) -> None:
        self.control_store = LatestValueStore(ControlState())
        self.render_store = LatestValueStore(to_render_params(ControlState()))
        self.model_host = model_host
        self.preview = PreviewMailbox()
        self.control_loop = _ModelWatchingControlLoop(
            self.control_store, self.render_store, model_host
        )
        self.render_loop = RenderLoop(
            self.render_store, self.model_host, [self.preview]
        )
        self.submit = self.control_loop.submit
        self._start_osc = start_osc
        self.osc = OscInput(self.control_loop.submit, port=osc_port)
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self.control_loop.start()
        self.render_loop.start()
        if self._start_osc:
            self.osc.start()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        self.osc.stop()
        self.render_loop.stop()
        self.control_loop.stop()
        self.model_host.stop()


class _ModelWatchingControlLoop(ControlLoop):
    """Control loop that forwards pkl_path changes to the model host.

    Model loading is a side effect of a state change, so it triggers on
    the control tick where every state change already flows.
    """

    def __init__(self, control_store, render_store, model_host, **kwargs) -> None:
        super().__init__(control_store, render_store, **kwargs)
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
) -> Runtime:
    return Runtime(
        model_host=model_host or ModelHost(),
        osc_port=osc_port,
        start_osc=start_osc,
    )
