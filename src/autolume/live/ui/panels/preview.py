"""Preview panel: the newest rendered frame, centred, and what the model is doing.

The frame is fitted to the panel rather than drawn at its own resolution. A
model's output size is a property of the model, not a choice the performer made
about how big they want the picture, so a 1024 model overflowing a docked panel
and putting scrollbars on it was the panel following the wrong one of the two.

Two modes, and the names are the point.

    Fit         scaled to fit inside the panel, aspect kept, letterboxed
    Stretch     filled to the panel, aspect ignored

The old app called these "Raw" and "Fit" and had both backwards: its "Raw" kept
the aspect ratio, which is what everyone means by fit, and its "Fit" filled the
area while distorting the image, which is stretching. The names here say what
happens. The mode is ordinary panel state and deliberately not a registry
parameter: it describes this window, not the performance, so nothing carries it
into a preset or out over OSC. It lives in the right click menu until the
display options panel that owns it exists.

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

The status is drawn over the frame, centred, and nothing at all when there is
nothing to say. Over arbitrary imagery no text colour is safe, so the whole
viewport is dimmed behind it. The dim is the backdrop, it covers everything
equally, and it doubles as the signal that the panel is busy.

**The model being replaced keeps rendering while the new one loads.** Blanking
the output would have made the status legible for free, and it is the wrong
trade: the loader runs on its own thread precisely so the picture never stops,
and unloading would black the output for the whole of every switch, including
the ones a controller makes mid set, and later the NDI and fullscreen outputs
with it.

**The dim is presentation and stops here.** It is a rectangle painted on this
panel's draw list after the image, so it exists only in this window: the frame
array is never read for it and never written. That has to stay true. The render
loop fans the same frames out to every sink, and the parity plan adds NDI, a
recorder and a fullscreen output, so a dim that reached the frame would dim the
show and the recording every time a model was switched. The consequence,
accepted deliberately, is that this panel will not look like the show output
once separate outputs exist. It is an affordance for the person driving, not a
property of the performance.

The render loop marks each frame read-only before handing it out, which is what
makes that guarantee structural rather than a promise this file keeps. The one
copy this panel does make is for immvision, which will not accept a read-only
array at all. See `_displayable`.
"""

import os
from dataclasses import dataclass
from enum import Enum

import numpy as np
from imgui_bundle import imgui, immvision

from autolume.live.ui.controls import ERROR_COLOR

_NO_MODEL = "No model loaded. Click Browse in the Controls panel to open one."
_NO_FRAMES = "Waiting for frames."

# How much of the frame the dim keeps out from behind the words. Set from the
# worst case rather than from a look, and the worst case is a model producing
# white: over white this leaves a 0.2 grey, which carries the error red at
# about four to one and the ordinary text colour at about thirteen. The red is
# what binds. It is a mid luminance colour, so it has the least room of
# anything drawn here, and it carries the message that most needs reading.
#
# The other side of the trade is that the frame underneath has to stay visible,
# because the whole point of not unloading is that the performer keeps seeing
# what is still rendering. A fifth of the way to black is plainly an image and
# plainly interrupted. Anything lighter loses the red.
_DIM_ALPHA = 0.8
_VIEWPORT_FLAGS = (
    imgui.WindowFlags_.no_scrollbar | imgui.WindowFlags_.no_scroll_with_mouse
)


class DisplayMode(Enum):
    """How the frame is sized to the panel. The value is what the menu says."""

    FIT = "Fit"
    STRETCH = "Stretch"


@dataclass(frozen=True)
class PreviewStatus:
    """What the line over the preview says, and whether it is a failure."""

    text: str
    error: bool = False


def model_name(path: str) -> str:
    """The part of a model path worth putting in a status line.

    The filename, because that is what a performer calls the model, and because
    a status line that grew with the path would be a wider plate over the image
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


def dims_the_frame(status: PreviewStatus, has_frame: bool) -> bool:
    """Whether the viewport is dimmed behind the status this frame.

    Only over a frame. With nothing rendered there is nothing to dim and
    nothing to read the words against but the empty panel, which they already
    stand out on, so a dim there would be a grey rectangle explaining itself.

    And only with something to say. A dim over a preview that is running fine
    is the panel claiming to be busy when it is not.
    """
    return bool(status.text) and has_frame


def displayed_size(
    frame: tuple[int, int], area: tuple[float, float], mode: DisplayMode
) -> tuple[int, int]:
    """How big the frame is drawn in `area`, in whole pixels.

    Fit scales by whichever axis runs out first, so the whole image is visible
    and the rest of the panel is letterboxing. It scales **up** as well as
    down. The preview is how a performer judges what they are making, and a 256
    model shown at 256 in the middle of a large panel is a smaller picture for
    no reason the performer chose. Refusing to scale up would also make Fit and
    Stretch differ in two ways at once for a small model, which is one more
    thing to explain than the pair is worth.

    Stretch takes the area whatever shape it is.

    Never rounds up, so the result cannot exceed the area it was measured
    against by a pixel and the panel can never be made to scroll by the
    rounding. A panel or a frame with no area yields nothing to draw, which is
    a real state on both counts: a dock split dragged shut has no width, and
    there is no frame at all until the first one arrives.
    """
    width, height = area
    frame_width, frame_height = frame
    if width <= 0.0 or height <= 0.0 or frame_width <= 0 or frame_height <= 0:
        return (0, 0)
    if mode is DisplayMode.STRETCH:
        return (int(width), int(height))
    scale = min(width / frame_width, height / frame_height)
    return (int(frame_width * scale), int(frame_height * scale))


def centred_offset(
    size: tuple[float, float], area: tuple[float, float]
) -> tuple[float, float]:
    """Where to put something of `size` to centre it in `area`.

    Never negative, so something wider than the area it is centred in starts at
    the edge and runs off the far side rather than off both. That is the case
    of a status line longer than a narrow panel, and losing the end of it beats
    losing the beginning as well.
    """
    return (
        max((area[0] - size[0]) * 0.5, 0.0),
        max((area[1] - size[1]) * 0.5, 0.0),
    )


def needs_refresh(
    seq: int, last_seq: int, size: tuple[int, int], last_size: tuple[int, int]
) -> bool:
    """Whether the image texture has to be uploaded again this frame.

    A new frame from the render loop, which is the one that matters and the one
    the sequence number exists for. An unchanged frame at an unchanged size
    never re-uploads, which is the whole reason the mailbox carries a sequence
    rather than the panel comparing arrays.

    Also a change of displayed size, because immvision draws through a texture
    it builds at the size it was asked for, so the size is part of what is
    cached and not only of where it is put. That is a dock split being dragged
    or a mode being switched, both of which are gestures rather than anything
    that happens during a performance.
    """
    return seq != last_seq or size != last_size


class PreviewPanel:
    def __init__(self, runtime) -> None:
        self._runtime = runtime
        self._last_seq = -1
        self._last_size = (0, 0)
        self._mode = DisplayMode.FIT
        self._display: np.ndarray | None = None

    def gui(self) -> None:
        seq, frame = self._runtime.preview.latest()
        host = self._runtime.model_host
        # `pending` carries both whether a load is running and which one, so it
        # is read once. Two reads could straddle a load finishing and report a
        # state the host was never in.
        status = preview_status(
            host.pending(),
            host.error(),
            host.current() is not None,
            frame is not None,
        )
        area = imgui.get_content_region_avail()
        # A child of exactly the space available, with no padding of its own
        # and no scrollbars. The frame is fitted to it, so it should never have
        # anything to scroll, and the flags are what make that a property of
        # the panel rather than of the arithmetic being right every time.
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(0.0, 0.0))
        opened = imgui.begin_child("##viewport", area, 0, _VIEWPORT_FLAGS)
        imgui.pop_style_var()
        if opened:
            origin = imgui.get_cursor_screen_pos()
            if frame is not None:
                self._draw_frame(seq, frame, (area.x, area.y))
            self._mode_menu()
            self._overlay(status, origin, (area.x, area.y), frame is not None)
        imgui.end_child()

    def _draw_frame(self, seq: int, frame: np.ndarray, area: tuple[float, float]):
        size = displayed_size((frame.shape[1], frame.shape[0]), area, self._mode)
        if size[0] <= 0 or size[1] <= 0:
            return
        offset = centred_offset(size, area)
        start = imgui.get_cursor_pos()
        imgui.set_cursor_pos(
            imgui.ImVec2(start.x + offset[0], start.y + offset[1])
        )
        refresh = needs_refresh(seq, self._last_seq, size, self._last_size)
        self._last_seq = seq
        self._last_size = size
        immvision.image_display(
            "##preview", self._displayable(frame, refresh), size, refresh_image=refresh
        )

    def _displayable(self, frame: np.ndarray, refresh: bool) -> np.ndarray:
        """A writeable copy of the frame, because immvision will not take one.

        Frames come off the render loop read-only, so that no sink can corrupt
        what the others are handed. immvision converts a numpy array through a
        mutable cv::Mat and rejects a read-only buffer outright, which makes the
        copy unavoidable rather than a choice, and this panel is where it
        belongs: it is the one consumer that needs it.

        Into a buffer it keeps, and only when the texture is being refreshed.
        immvision does not look at the pixels otherwise, so copying a megabyte
        image on every UI frame would be paying for uploads that are not
        happening, several times per rendered frame.
        """
        stale = (
            self._display is None
            or self._display.shape != frame.shape
            or self._display.dtype != frame.dtype
        )
        if stale:
            self._display = np.empty(frame.shape, dtype=frame.dtype)
        if refresh or stale:
            np.copyto(self._display, frame)
        return self._display

    def _mode_menu(self) -> None:
        """The mode picker, until the display options panel that owns it exists.

        On the window rather than on the image, so it opens over the
        letterboxing and before the first frame as well.
        """
        if not imgui.begin_popup_context_window("##display"):
            return
        for mode in DisplayMode:
            if imgui.menu_item(mode.value, "", self._mode is mode)[0]:
                self._mode = mode
        imgui.end_popup()

    def _overlay(
        self,
        status: PreviewStatus,
        origin,
        area: tuple[float, float],
        has_frame: bool,
    ) -> None:
        """Dim the viewport and draw the status, or draw nothing whatsoever.

        Nothing means nothing: no dim, no glyph, and no item, so a running
        preview is the frame and the frame only. It takes no layout in either
        state, because it is painted at a position rather than placed, which is
        what lets it sit on the image without having pushed it down first.

        The dim is a rectangle over this window, after the image and before the
        words. The frame array is not touched by any of it, which is what keeps
        the effect off every other sink the render loop feeds.
        """
        if not status.text:
            return
        draw_list = imgui.get_window_draw_list()
        far = imgui.ImVec2(origin.x + area[0], origin.y + area[1])
        if dims_the_frame(status, has_frame):
            draw_list.add_rect_filled(
                origin,
                far,
                imgui.get_color_u32(imgui.ImVec4(0.0, 0.0, 0.0, _DIM_ALPHA)),
            )
        text_size = imgui.calc_text_size(status.text)
        offset = centred_offset((text_size.x, text_size.y), area)
        color = (
            imgui.ImVec4(*ERROR_COLOR)
            if status.error
            else imgui.get_style_color_vec4(imgui.Col_.text)
        )
        draw_list.add_text(
            imgui.ImVec2(origin.x + offset[0], origin.y + offset[1]),
            imgui.get_color_u32(color),
            status.text,
        )
