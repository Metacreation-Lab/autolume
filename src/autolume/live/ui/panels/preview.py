"""Preview panel: the newest rendered frame, centred, and what the model is doing.

The frame is centred and never allowed to overflow, where before it was drawn
at its own resolution in the top left and a 1024 model put scrollbars on a
docked panel.

Two modes, and the names are the point.

    Fit         native size, shrunk only when it does not fit
    Stretch     grown to meet the panel on its tighter axis

Both keep the aspect ratio, both centre, and neither crops. The whole frame is
always visible and the rest of the panel is letterboxing. The modes differ in
exactly one thing: whether a frame smaller than the panel is magnified.

**Fit never magnifies.** A model smaller than the panel is meant to look small.
Magnifying is not fitting, so a Fit that enlarged would be misnamed, and the
size a model renders at is worth being able to see for what it is. This is the
old app's "Raw" behaviour under a name that describes it.

The old app called these "Raw" and "Fit" and had both backwards: its "Raw" kept
the aspect ratio and its native size, which is what everyone means by fit, and
its "Fit" scaled the image up to the preview area, which is what Stretch does
here. The names say what happens. The mode is ordinary panel state and deliberately
not a registry parameter: it describes this window, not the performance, so
nothing carries it into a preset or out over OSC. It lives in the right click
menu until the display options panel that owns it exists.

The frame is drawn as a textured quad rather than through
`immvision.image_display`, and that is not a preference. `image_display` fits
whatever it is given into the size it is asked for **with the aspect ratio
kept**, and pins the result to the top left of that box. Asking it for a box of
the panel's shape therefore produced neither of our modes: it letterboxed like
Fit and aligned like nothing at all. Uploading the frame once and sizing the
quad ourselves gives both modes exactly, puts the scaling on the GPU, and makes
the upload independent of the panel's size. See `_texture_id`.

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

    Both modes scale by whichever axis runs out first, so the aspect ratio is
    kept, the whole image stays visible, and what is left of the panel is
    letterboxing. Neither crops and neither distorts.

    They differ in one thing: Fit never magnifies, Stretch does. Fit is native
    size where the frame fits and a shrink where it does not, because
    enlarging is not fitting and a model that renders smaller than the panel is
    worth seeing at the size it renders. Stretch grows it to meet the panel on
    its tighter axis.

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
    scale = min(width / frame_width, height / frame_height)
    if mode is DisplayMode.FIT:
        scale = min(scale, 1.0)
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


def needs_refresh(seq: int, last_seq: int) -> bool:
    """Whether the frame has to be uploaded to the GPU again.

    A new frame from the render loop and nothing else, which is the whole
    reason the mailbox carries a sequence number rather than the panel
    comparing arrays. An unchanged frame never re-uploads, however many UI
    frames are drawn from it.

    The displayed size is deliberately not part of this. The texture holds the
    frame at its own resolution and the quad drawn from it carries the scaling,
    so dragging a dock split or switching mode changes the quad and never the
    texture. An earlier pass did include the size, because
    `immvision.image_display` builds its texture at the size it is asked for,
    and drawing the quad here is what made that unnecessary.
    """
    return seq != last_seq


class PreviewPanel:
    def __init__(self, runtime) -> None:
        self._runtime = runtime
        self._last_seq = -1
        self._mode = DisplayMode.FIT
        self._display: np.ndarray | None = None
        self._texture: immvision.GlTexture | None = None

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
        imgui.set_cursor_pos(imgui.ImVec2(start.x + offset[0], start.y + offset[1]))
        texture = self._texture_id(frame, needs_refresh(seq, self._last_seq))
        self._last_seq = seq
        imgui.image(
            imgui.ImTextureRef(texture),
            imgui.ImVec2(float(size[0]), float(size[1])),
        )

    def _texture_id(self, frame: np.ndarray, refresh: bool) -> int:
        """The texture holding the newest frame, uploaded only when it is new.

        Always at the frame's own resolution. The quad drawn from it carries
        all the scaling, which is what makes both modes cost nothing at draw
        time and keeps the upload independent of the panel's size: dragging a
        dock split or switching mode changes the quad and never the texture.

        Built on first use rather than in the constructor, because it takes a
        GL context and a panel is built before there is one.
        """
        if self._texture is None:
            self._texture = immvision.GlTexture()
        if refresh:
            # Explicit rather than left to immvision's global colour order,
            # because the frames are RGB and a wrong guess here is a silent
            # swap of red and blue in the only place anyone would see it.
            self._texture.update_from_image(
                self._displayable(frame), is_color_order_bgr=False
            )
        return self._texture.texture_id

    def _displayable(self, frame: np.ndarray) -> np.ndarray:
        """A writeable copy of the frame, because the uploader will not take one.

        Frames come off the render loop read-only, so that no sink can corrupt
        what the others are handed. The upload converts a numpy array through a
        mutable cv::Mat and rejects a read-only buffer outright, which makes the
        copy unavoidable rather than a choice, and this panel is where it
        belongs: it is the one consumer that needs it.

        Into a buffer it keeps, and only ever called on an upload, so a
        megabyte is copied once per rendered frame rather than once per UI
        frame drawn from it.
        """
        if (
            self._display is None
            or self._display.shape != frame.shape
            or self._display.dtype != frame.dtype
        ):
            self._display = np.empty(frame.shape, dtype=frame.dtype)
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
