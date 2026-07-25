"""Preview panel: the newest rendered frame, and what the model is doing.

The model's state is reported here rather than beside the field that sets it,
because this is the surface the performer is already looking at and it is the
one that changes when a load lands. A line under a control in a side panel is
missable in a way this is not.

It also fixes a preview that lied. "Waiting for frames" was drawn whenever
there was no frame, which is true at startup and false in the two states that
matter: a model that is loading says the same thing as no model at all, and a
model that failed to load says something is still coming when nothing is. A
failed load is the case worth getting right, because it is the one where the
performer is waiting for a screen that will never change.

The line is drawn on every frame, empty when there is nothing to say, so
nothing on the panel moves when a load starts or ends. It says one thing at a
time and never wraps, and what it names is the model's filename rather than its
path, so the length of a path cannot push the frame around either.
"""

import os
from dataclasses import dataclass

from imgui_bundle import imgui, immvision

from autolume.live.ui.controls import ERROR_COLOR

_NO_MODEL = "No model loaded. Click Browse in the Controls panel to open one."
_NO_FRAMES = "Waiting for frames."


@dataclass(frozen=True)
class PreviewStatus:
    """What the line above the preview says, and whether it is a failure."""

    text: str
    error: bool = False


def model_name(path: str) -> str:
    """The part of a model path worth putting in a status line.

    The filename, because that is what a performer calls the model, and because
    a status line that grew with the path would push the frame under it around
    every time a model in a deeper folder was loaded.
    """
    return os.path.basename(path) or path


def preview_status(
    pending: str | None, error: str | None, loaded: bool, has_frame: bool
) -> PreviewStatus:
    """What the preview says about the model, in the order it says it.

    Loading comes first and names what it is loading, because a load takes
    seconds and the only alternative is a still frame that looks like a hang.
    It also outranks an error, which by then belongs to the model before this
    one and would be read as this one failing.

    A failure comes next, ahead of everything about frames, and it shows even
    while frames are arriving. That is the whole point: a load that fails
    partway through a set leaves the previous model rendering happily, and
    without this the performer sees a preview that simply ignored them.

    Frames arriving means there is nothing to say. The two remaining silences
    are told apart because they call for different things: no model loaded is
    an invitation, and a model loaded with nothing on screen yet is the app
    working through its first frame.
    """
    if pending is not None:
        return PreviewStatus(f"Loading {model_name(pending)}.")
    if error:
        return PreviewStatus(error, True)
    if has_frame:
        return PreviewStatus("")
    return PreviewStatus(_NO_FRAMES if loaded else _NO_MODEL)


class PreviewPanel:
    def __init__(self, runtime) -> None:
        self._runtime = runtime
        self._last_seq = -1

    def gui(self) -> None:
        seq, frame = self._runtime.preview.latest()
        host = self._runtime.model_host
        # `pending` carries both whether a load is running and which one, so it
        # is read once. Two reads could straddle a load finishing and report a
        # state the host was never in.
        self._status(
            preview_status(
                host.pending(),
                host.error(),
                host.current() is not None,
                frame is not None,
            )
        )
        if frame is None:
            return
        refresh = seq != self._last_seq
        self._last_seq = seq
        immvision.image_display_resizable(
            "##preview", frame, refresh_image=refresh
        )

    def _status(self, status: PreviewStatus) -> None:
        """Draw the line, always, whether or not it says anything.

        Unwrapped, so a long failure message is one line however long it is and
        the frame below never moves. The whole of it is on the tooltip, which
        is where a message too long for the panel stays readable.
        """
        color = (
            imgui.ImVec4(*ERROR_COLOR)
            if status.error
            else imgui.get_style_color_vec4(imgui.Col_.text_disabled)
        )
        imgui.push_style_color(imgui.Col_.text, color)
        imgui.text(status.text)
        imgui.pop_style_color()
        if status.text and imgui.is_item_hovered(imgui.HoveredFlags_.delay_normal):
            imgui.set_tooltip(status.text)
