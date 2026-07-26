"""Performance panel: the machine, not the show.

Everything here is about the box Autolume is running on rather than the
picture it makes. The frame limit, which device renders, where the output
goes, what port OSC listens on. None of it is written to a preset, and that
is the line deciding what belongs here instead of in Perform: a preset is a
look, and a look does not carry somebody else's graphics card. The frame
limit moved out of Perform for exactly that reason.

Every row driving something outside state shows two facts. The parameter is
what was asked for, and the status the subsystem published is what is
actually happening. They agree except while a device switch, an OSC rebind,
an NDI session or a recording is failing, and then the status is the one
telling the truth. A failed switch, session or take puts its own parameter
back on its own, so from here it simply pops back a moment later, and the
status line beside it is the only place the reason exists.
"""

from imgui_bundle import imgui

from autolume.live.core.events import ControlEvent
from autolume.live.io import ndi
from autolume.live.io.recorder import SCREENSHOT_ADDRESS
from autolume.live.ui.controls import ControlBinder
from autolume.live.ui.panels.perform import combo_index, string_combo
from autolume.live.ui.theme import ERROR_COLOR

# The four the device parameter accepts (plan-4 decisions). "auto" first
# because it is the default and the answer for almost every performer.
DEVICE_VALUES = ("auto", "cuda", "mps", "cpu")
DEVICE_LABELS = ("Automatic", "CUDA", "MPS", "CPU")

_NDI_UNAVAILABLE = "NDI is not installed on this machine."
_SUPERRES_LIMIT = (
    "Super-res only runs on frames up to 1024 pixels on the short side. "
    "A larger frame is passed through unchanged."
)
_CAPTURES_NOTE = "Screenshots and recordings go to the captures folder."


def device_note(requested: str, active: str | None, error: str | None) -> str | None:
    """What to say under the device combo, or None when there is nothing to say.

    Three separate facts, in the order they matter. A failed switch is the
    only one that is news, so it comes first and names the device that was
    refused. A device string the combo has no entry for is the second: it can
    only arrive from OSC or a hand edited preset, and a blank combo on its own
    would look broken rather than informative. Otherwise the running device is
    worth stating whenever it is not literally the word that was asked for,
    which is every "auto" session.
    """
    if error:
        return f"Could not switch to {requested}. {error}"
    if combo_index(requested, DEVICE_VALUES) < 0:
        return f"{requested} is not a device this build offers."
    if active and active != requested:
        return f"Rendering on {active}."
    return None


def osc_note(requested: int, bound_port: int | None, error: str | None) -> str | None:
    """What to say under the OSC port field.

    A failed rebind keeps the previous transport serving, so the sentence has
    to name both the port that was refused and the one still working, or a
    performer reads it as "OSC is down" when it is not. A bound port that
    differs from the requested one without an error is the scan upward
    behaviour finding the next free port, which is a success worth stating
    because nothing else on screen would explain the number.
    """
    if error:
        if bound_port is None:
            return f"Could not listen on port {requested}. {error}"
        return (
            f"Could not listen on port {requested}. {error} "
            f"Still listening on port {bound_port}."
        )
    if bound_port is None:
        return "OSC input is off."
    if bound_port != requested:
        return f"Port {requested} was taken. Listening on port {bound_port}."
    return f"Listening on port {bound_port}."


def superres_note(disabled_reason: str | None, last_error: str | None) -> str | None:
    """What to say under the super-res checkbox, or None while it is healthy.

    `last_error` is deliberately not described as being about the current
    frame. It is cleared only by a forward pass that succeeds, never by the
    calls that short circuit before one (the weights are missing, or the frame
    is over the size guard), so it can still read as set while nothing is
    actually failing right now (`core/superres.py`, recorded trade-off). The
    wording says a problem was reported rather than claiming this frame was
    the one that hit it.
    """
    if disabled_reason:
        return f"Super-res is off. {disabled_reason}"
    if last_error:
        return f"Super-res reported a problem. {last_error}"
    return None


def superres_state(model: object) -> tuple[str | None, str | None]:
    """`(disabled_reason, last_error)` read off whatever model is rendering.

    The super-res stage is owned by the render side and has no accessor of its
    own up to this thread, so this reaches for the attribute and settles for
    nothing when it is not there: a loader's test double is not a
    `LoadedModel`, and a `LoadedModel` from a build before this stage existed
    would not carry one either. Both fields are written only on the render
    thread and neither can tear, so a racy read here is at worst one frame
    stale.
    """
    stage = getattr(model, "_superres", None)
    if stage is None:
        return None, None
    reason = getattr(stage, "disabled_reason", None)
    return reason, getattr(stage, "last_error", None)


def elapsed_text(seconds: float) -> str:
    """A take's running time as `m:ss`, which is how a take is talked about."""
    whole = max(0, int(seconds))
    return f"{whole // 60}:{whole % 60:02d}"


def recording_note(status, elapsed: float | None) -> str | None:
    """What to say under the Record button.

    A take in progress reports its length and, only once it is behind, how
    many frames it has lost. A drop count of zero is not worth a word. After a
    take the file is named, because where it went is the one thing a performer
    needs and the only place it is said.
    """
    if status.recording:
        running = "Recording" if elapsed is None else f"Recording {elapsed_text(elapsed)}"
        if status.frames_dropped:
            return (
                f"{running}. The encoder is behind by "
                f"{status.frames_dropped} frames."
            )
        return f"{running}."
    if status.error:
        return status.error
    if status.path:
        return f"Saved to {status.path}."
    return None


def ndi_note(status, available: bool) -> str | None:
    """What to say under the NDI rows."""
    if not available:
        return _NDI_UNAVAILABLE
    if status.error:
        return status.error
    if status.sending:
        return f"Sending as {status.name}."
    return None


def stats_text(fps: float) -> str:
    """The render rate, and the average interval that rate works out to.

    Called an average interval rather than a frame time on purpose: the rate
    behind it is frames over wall clock seconds, so under a frame limit the
    number includes the loop's own sleep and is not how long a frame took to
    draw. Naming it for what it measures keeps it from being read as a
    rendering cost it is not.
    """
    if fps <= 0.0:
        return "Not rendering yet."
    return f"{fps:.1f} fps. Average interval {1000.0 / fps:.1f} ms."


class PerformancePanel:
    def __init__(self, runtime, mapping_popup=None) -> None:
        self._runtime = runtime
        self._binder = ControlBinder(runtime, mapping_popup)
        # When the take on screen started, by the UI clock. The recorder
        # publishes counters rather than a start time, and deriving one from
        # the frame count would need the writer's fps, which is the cap as it
        # stood when Record was pressed and is not published either.
        self._record_started: float | None = None

    def gui(self) -> None:
        self._render_rows()
        self._stats_rows()
        self._osc_rows()
        self._ndi_rows()
        self._output_rows()

    def _emit(self, address: str, value) -> None:
        self._runtime.submit(ControlEvent(address, value, source="ui"))

    def _render_rows(self) -> None:
        """The frame limit, the device, fp32 and super-res.

        The frame limit is a typed field rather than a drag, the same
        reasoning it carried in Perform: it is a setting picked once, not a
        value worth sweeping through and watching change.
        """
        imgui.separator_text("Render")
        self._binder.input_int("fps_cap", "Frame limit")
        self._device_row()
        self._binder.checkbox("force_fp32", "Force fp32")
        self._superres_row()

    def _device_row(self) -> None:
        state = self._binder.state()
        picked = string_combo(
            "Device", state.device, DEVICE_VALUES, DEVICE_LABELS
        )
        if picked is not None:
            self._emit("/render/device", picked)
        status = self._runtime.model_host.device_store.snapshot()
        self._note(device_note(state.device, status.active, status.error))

    def _superres_row(self) -> None:
        self._binder.checkbox("use_superres", "Super-res")
        if not bool(self._binder.value("use_superres")):
            return
        reason, last_error = superres_state(self._runtime.model_host.current())
        note = superres_note(reason, last_error)
        if note is None:
            self._note(_SUPERRES_LIMIT)
        else:
            self._error(note)

    def _stats_rows(self) -> None:
        imgui.separator_text("Stats")
        # Wrapped, not `imgui.text`: a plain text item is as wide as its
        # string and this one is two sentences, so at a scaled up font in a
        # narrow dock it ran past the panel edge and took the separators with
        # it (`test_panel_drawing.py` catches exactly this).
        imgui.text_wrapped(stats_text(self._runtime.render_loop.fps()))

    def _osc_rows(self) -> None:
        imgui.separator_text("OSC")
        # A port has no meaningful neighbour to scrub through on the way to
        # the one a performer wants, and every intermediate value would
        # rebind the transport. Five digits at most, so it is capped narrow.
        self._binder.input_int("osc_port", "Port", natural_ems=7.0)
        status = self._runtime.osc_status_store.snapshot()
        requested = int(self._binder.value("osc_port") or 0)
        self._note(osc_note(requested, status.bound_port, status.error))

    def _ndi_rows(self) -> None:
        """The NDI toggle and the name it advertises.

        Both rows are drawn whether or not the library is installed, greyed
        rather than gone: a missing NDI runtime is a fact about the machine
        worth seeing, and a section that appears and disappears with an
        optional dependency moves everything under it.
        """
        imgui.separator_text("NDI")
        available = ndi.available()
        self._binder.checkbox("ndi_enabled", "Send over NDI", enabled=available)
        self._binder.input_text(
            "ndi_name", "Name", hint="Autolume Live", enabled=available, natural_ems=20.0
        )
        status = self._runtime.ndi.status()
        note = ndi_note(status, available)
        # `available` as well as `error`, because the note a missing runtime
        # produces is not the error: a machine without NDI installed is a fact
        # about the machine, and drawing it red says something went wrong.
        if available and status.error:
            self._error(note)
        else:
            self._note(note)

    def _output_rows(self) -> None:
        imgui.separator_text("Output")
        if imgui.button("Screenshot"):
            self._emit(SCREENSHOT_ADDRESS, 1.0)
        self._note(_CAPTURES_NOTE)
        self._record_row()
        # "Fullscreen", not "Fullscreen output": a checkbox's label is drawn
        # outside its item and nothing can narrow it, so a long one runs past
        # the panel edge in a narrow dock at a scaled up font. The section it
        # sits under already says this is about output.
        self._binder.checkbox("fullscreen", "Fullscreen")

    def _record_row(self) -> None:
        """Record, and what the take is doing.

        A checkbox rather than a button, because `recording` is a state a
        performer can also be put into from OSC and a button would not show
        that. What it shows is the parameter, like every other bound control;
        what the recorder published is on the line under it, and the two
        disagree only while a take is failing.
        """
        self._binder.checkbox("recording", "Record")
        status = self._runtime.recorder.status()
        elapsed = self._take_elapsed(status.recording)
        note = recording_note(status, elapsed)
        if status.error:
            self._error(note)
        else:
            self._note(note)

    def _take_elapsed(self, recording: bool) -> float | None:
        if not recording:
            self._record_started = None
            return None
        now = imgui.get_time()
        if self._record_started is None:
            self._record_started = now
        return now - self._record_started

    def _note(self, text: str | None) -> None:
        if not text:
            return
        imgui.push_style_color(
            imgui.Col_.text, imgui.get_style_color_vec4(imgui.Col_.text_disabled)
        )
        imgui.text_wrapped(text)
        imgui.pop_style_color()

    def _error(self, text: str | None) -> None:
        if not text:
            return
        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*ERROR_COLOR))
        imgui.text_wrapped(text)
        imgui.pop_style_color()
