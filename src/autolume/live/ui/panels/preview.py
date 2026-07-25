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

**The geometry is in device pixels, not in imgui's points.** Everything imgui
is told is in points, and on a display that scales them the two are not the
same number: a frame drawn at "its own size" in points covers twice as many
pixels on a 2x screen, which is a 2x magnification wearing the name native. So
the fitting is done in pixels, against a panel measured in pixels, and only
converted to points at the two calls that need points. That is what makes Fit
mean what it says on every display, and it is what lets a frame that fits land
on the pixel grid exactly rather than half a pixel off it. See `display_scale`.

**Magnification is done to the frame, not to the quad.** A texture magnified by
the GPU is interpolated, and there is no per-texture escape from it: imgui's
OpenGL renderer binds a sampler object for every draw it makes, and a sampler
overrides whatever filtering the texture itself was given, so
`glTexParameteri(GL_TEXTURE_MAG_FILTER, GL_NEAREST)` on our texture changes
nothing at all. The pixel grid is part of what a generative model produces, and
a performer scaling a small model up wants to see it rather than a blurred
interpolation of it, so a magnified frame is enlarged into the texture before
it is uploaded and the quad then draws it one for one. See `magnified`.

Minification is left to the GPU, where the interpolation is the right
behaviour: a 1024 frame shrunk into a docked panel would alias and crawl if it
were sampled every nth pixel, and the blend is what keeps it readable.

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
class Placement:
    """Where the frame goes in the panel, how big, and what it is made of.

    `offset` and `size` are in imgui's points, because points are the only
    unit imgui accepts. `pixels` is that same size in device pixels, which is
    what the texture has to hold: the picture is made of pixels, and the size
    in points is only a statement of how big to draw them.
    """

    offset: tuple[float, float]
    size: tuple[float, float]
    pixels: tuple[int, int]


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


def display_scale() -> float:
    """Device pixels per imgui point, on the display the window is on.

    One on a display that does not scale, two on a Retina Mac, and whatever
    the window manager says elsewhere. Everything imgui is told is in points,
    and this is the only number that turns them into what the screen actually
    draws.

    Falls back to one rather than to nothing, because a scale of zero would
    divide the geometry by zero and every backend that reports honestly
    reports at least one.
    """
    scale = imgui.get_io().display_framebuffer_scale.x
    return float(scale) if scale > 0.0 else 1.0


def magnified(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """`frame` enlarged to `size` by repeating pixels, never blending them.

    Each output pixel is one input pixel, chosen by where it falls in the
    frame, so the result is the frame's own pixels drawn bigger and its edges
    stay where they were. That is the whole point: the alternative is the
    GPU's interpolation, which is the soft result this exists to avoid.

    The scale does not have to be a whole number and the blocks are allowed to
    come out uneven, because Stretch fills the panel and a panel is not an
    exact multiple of a model's resolution except by accident. Uneven and
    crisp reads as pixels. Even and blurred does not.

    Returns a new array every time. That is what the uploader needs anyway,
    since it will not take the read-only frames the render loop hands out.
    """
    width, height = size
    columns = (np.arange(width) * frame.shape[1]) // width
    rows = (np.arange(height) * frame.shape[0]) // height
    return frame[rows[:, None], columns]


def upload_size(frame: tuple[int, int], size: tuple[int, int]) -> tuple[int, int]:
    """The resolution to put on the GPU to draw the frame at `size`.

    Its own, unless it is being magnified, in which case the enlarging is done
    to the pixels before they are uploaded and the texture is the size of the
    quad. Shrinking is left at native resolution and handed to the GPU, which
    interpolates, which is what a frame being made smaller wants.
    """
    if size[0] > frame[0] or size[1] > frame[1]:
        return size
    return frame


def displayed_size(
    frame: tuple[int, int], area: tuple[float, float], mode: DisplayMode
) -> tuple[int, int]:
    """How big the frame is drawn in `area`, in whole device pixels.

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

    Whole pixels, never half of one. An image landing on a half pixel is
    resampled across the whole grid however exactly it was sized, so the
    centring would undo the fitting. Down rather than to the nearest, so a
    frame measured to fit its area cannot be pushed a pixel past it.
    """
    return (
        float(int(max((area[0] - size[0]) * 0.5, 0.0))),
        float(int(max((area[1] - size[1]) * 0.5, 0.0))),
    )


def frame_placement(
    frame: tuple[int, int],
    area: tuple[float, float],
    mode: DisplayMode,
    scale: float,
) -> Placement:
    """Where to draw the frame in a panel `area` points across.

    The fitting and the centring both happen in device pixels, and the result
    is converted back to points at the end, because a panel measured in points
    is a different number of pixels on every display and it is the pixels the
    picture is made of. Sizing the quad in points instead would draw a frame at
    "its own size" over twice as many pixels as it has wherever points are
    scaled, and every mode would magnify without saying so.
    """
    pixels = (area[0] * scale, area[1] * scale)
    size = displayed_size(frame, pixels, mode)
    offset = centred_offset(size, pixels)
    return Placement(
        (offset[0] / scale, offset[1] / scale),
        (size[0] / scale, size[1] / scale),
        size,
    )


def needs_refresh(
    seq: int, last_seq: int, size: tuple[int, int], last_size: tuple[int, int]
) -> bool:
    """Whether the frame has to be uploaded to the GPU again.

    A new frame from the render loop, which is the whole reason the mailbox
    carries a sequence number rather than the panel comparing arrays. An
    unchanged frame never re-uploads, however many UI frames are drawn from it.

    And a change in the resolution it is uploaded at, which is the frame's own
    until the frame is magnified and the size of the quad after that. So in
    Fit, and in Stretch below native size, dragging a dock split changes the
    quad and never the texture. Above native size it changes both, because that
    is where the enlarging lives, and an upload that ignored the size would
    leave the picture at whatever width the panel happened to be when the frame
    arrived.
    """
    return seq != last_seq or size != last_size


class PreviewPanel:
    def __init__(self, runtime) -> None:
        self._runtime = runtime
        self._last_seq = -1
        self._last_upload = (0, 0)
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
        place = frame_placement(
            (frame.shape[1], frame.shape[0]), area, self._mode, display_scale()
        )
        if place.pixels[0] <= 0 or place.pixels[1] <= 0:
            return
        start = imgui.get_cursor_pos()
        imgui.set_cursor_pos(
            imgui.ImVec2(start.x + place.offset[0], start.y + place.offset[1])
        )
        imgui.image(
            imgui.ImTextureRef(self._texture_id(seq, frame, place.pixels)),
            imgui.ImVec2(*place.size),
        )

    def _texture_id(self, seq: int, frame: np.ndarray, size: tuple[int, int]) -> int:
        """The texture the quad is drawn from, uploaded only when it changes.

        At the frame's own resolution while the frame is being shrunk, so that
        dragging a dock split changes the quad and never the texture and the
        GPU does the interpolating that a shrink wants. At the size of the quad
        once the frame is being magnified, because that is the case the GPU
        cannot be left to do: see the module docstring.

        Built on first use rather than in the constructor, because it takes a
        GL context and a panel is built before there is one.
        """
        if self._texture is None:
            self._texture = immvision.GlTexture()
        upload = upload_size((frame.shape[1], frame.shape[0]), size)
        if needs_refresh(seq, self._last_seq, upload, self._last_upload):
            # Explicit rather than left to immvision's global colour order,
            # because the frames are RGB and a wrong guess here is a silent
            # swap of red and blue in the only place anyone would see it.
            self._texture.update_from_image(
                self._displayable(frame, upload), is_color_order_bgr=False
            )
            self._last_seq = seq
            self._last_upload = upload
        return self._texture.texture_id

    def _displayable(self, frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        """The frame at `size`, writeable, because the uploader will not take one.

        Frames come off the render loop read-only, so that no sink can corrupt
        what the others are handed. The upload converts a numpy array through a
        mutable cv::Mat and rejects a read-only buffer outright, which makes the
        copy unavoidable rather than a choice, and this panel is where it
        belongs: it is the one consumer that needs it.

        Into a buffer it keeps, and only ever called on an upload, so a
        megabyte is copied once per rendered frame rather than once per UI
        frame drawn from it. A magnified frame is a new array each time
        instead: it is already a copy, and its size follows the panel rather
        than the model, so there is no shape worth holding on to.
        """
        if (size[1], size[0]) != frame.shape[:2]:
            return magnified(frame, size)
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
