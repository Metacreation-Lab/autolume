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

**Fit never magnifies past native size.** A model smaller than the panel is
meant to look small. Magnifying is not fitting, so a Fit that enlarged would be
misnamed, and the size a model renders at is worth being able to see for what
it is. This is the old app's "Raw" behaviour under a name that describes it.

**Native size is one frame pixel per point**, the same DPI-independent unit the
rest of the interface is laid out in, so a model is the same physical size on
every display. One frame pixel per *device* pixel would be a different rule and
a worse one: a 64 model would come out a centimetre across on a 2x screen and
twice that on a 1x one, and the app's own convention is that nothing changes
physical size with the display. The old app arrives at the same place by a
different route, drawing through a projection set up in points over a
framebuffer in pixels, so its "Raw" and this Fit agree exactly.

The old app called these "Raw" and "Fit" and had both backwards: its "Raw" kept
the aspect ratio and its native size, which is what everyone means by fit, and
its "Fit" scaled the image up to the preview area, which is what Stretch does
here. The names say what happens. The mode is ordinary panel state and deliberately
not a registry parameter: it describes this window, not the performance, so
nothing carries it into a preset or out over OSC. It lives in the right click
menu until the display options panel that owns it exists.

**The geometry is measured in device pixels even though the sizes are named in
points.** Everything imgui is told is in points, and on a display that scales
them the two are not the same number. Measuring in pixels is what lets the drawn
frame land on the pixel grid exactly rather than half a pixel off it, and it is
what tells the uploader how many pixels the picture actually has to fill. The
conversion back to points happens at the two calls that need points, and the
display scale appears nowhere else. See `display_scale`.

**Magnification happens before imgui sees the texture.** Anything imgui draws
magnified is interpolated, and there is no per-texture escape from it: its
OpenGL renderer binds a sampler object for every draw it makes, and a sampler
overrides whatever filtering the texture itself was given, so
`glTexParameteri(GL_TEXTURE_MAG_FILTER, GL_NEAREST)` on our texture changes
nothing at all. The pixel grid is part of what a generative model produces, and
a performer scaling a small model up wants to see it rather than a blurred
interpolation of it, so a magnified frame is resampled into a texture of the
size it will be drawn at and the quad then draws that one for one. The resample
is a `GL_NEAREST` framebuffer blit, which is the GPU doing in a tenth of a
millisecond what the same arithmetic in numpy costs tens of. See `Enlarger`.

Minification is left to imgui's sampler, where the interpolation is the right
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
from OpenGL import GL

from autolume.live.ui import theme
from autolume.live.ui.theme import ERROR_COLOR

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


def magnifies(frame: tuple[int, int], size: tuple[int, int]) -> bool:
    """Whether drawing the frame at `size` enlarges it.

    Only then does the enlargement have to be done for the GPU rather than by
    it. Shrinking is handed straight to imgui's sampler, which interpolates,
    which is what a frame being made smaller wants.
    """
    return size[0] > frame[0] or size[1] > frame[1]


class Enlarger:
    """Nearest-neighbour magnification, done on the GPU by a framebuffer blit.

    `glBlitFramebuffer` with `GL_NEAREST` is a nearest-neighbour resample, at
    any scale, whole-numbered or not, and it is the one way left to get one
    once imgui's renderer stopped honouring per-texture filtering. It costs
    around a tenth of a millisecond for a 1024 frame, against 17 to 41 doing
    the same arithmetic in numpy, which at 1024 on CUDA would have been more
    than a whole frame's budget spent enlarging a picture.

    The blit is not the render path. It runs on the UI thread, once per frame
    the panel actually draws, and it touches the render loop's frames only by
    reading the texture they were uploaded to.

    Nothing here is created until it is used, because a panel is built before
    there is a GL context to build it in. There is no fallback if the blit
    fails: this is not an optional hint, it is how the picture gets its pixels,
    and the same context draws the rest of the interface, so a failure here is
    a failure everywhere and should look like one rather than quietly halving
    the frame rate on a path nobody tests.
    """

    def __init__(self) -> None:
        self._read = 0
        self._draw = 0
        self._texture = 0
        self._size = (0, 0)

    def enlarge(
        self, source: int, frame: tuple[int, int], size: tuple[int, int]
    ) -> int:
        """`source` resampled up to `size`, as a texture id to draw from."""
        if not self._read:
            self._read = int(GL.glGenFramebuffers(1))
            self._draw = int(GL.glGenFramebuffers(1))
        if size != self._size:
            self._resize(size)
        read_was = int(GL.glGetIntegerv(GL.GL_READ_FRAMEBUFFER_BINDING))
        draw_was = int(GL.glGetIntegerv(GL.GL_DRAW_FRAMEBUFFER_BINDING))
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self._read)
        GL.glFramebufferTexture2D(
            GL.GL_READ_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, source, 0
        )
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self._draw)
        GL.glBlitFramebuffer(
            0, 0, frame[0], frame[1],
            0, 0, size[0], size[1],
            GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST,
        )  # fmt: skip
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, read_was)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, draw_was)
        return self._texture

    def _resize(self, size: tuple[int, int]) -> None:
        """Point the destination at a texture of exactly `size`.

        Reallocated rather than grown, because the size follows the panel and a
        texture kept at the largest size ever seen would hold onto the memory
        of a window that was briefly dragged full screen.
        """
        if not self._texture:
            self._texture = int(GL.glGenTextures(1))
        was = int(GL.glGetIntegerv(GL.GL_TEXTURE_BINDING_2D))
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D, 0, GL.GL_RGB8, size[0], size[1],
            0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, None,
        )  # fmt: skip
        GL.glBindTexture(GL.GL_TEXTURE_2D, was)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self._draw)
        GL.glFramebufferTexture2D(
            GL.GL_DRAW_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT0,
            GL.GL_TEXTURE_2D,
            self._texture,
            0,
        )
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, 0)
        self._size = size


def displayed_size(
    frame: tuple[int, int],
    area: tuple[float, float],
    mode: DisplayMode,
    native: float,
) -> tuple[int, int]:
    """How big the frame is drawn in `area`, in whole device pixels.

    `area` is in device pixels and `native` is how many device pixels one frame
    pixel covers at native size, which is the display scale, because native
    means one frame pixel per point. It is a parameter rather than a constant
    so that the one place that knows the display scale is the panel, and so
    that this stays a function of its arguments.

    Both modes scale by whichever axis runs out first, so the aspect ratio is
    kept, the whole image stays visible, and what is left of the panel is
    letterboxing. Neither crops and neither distorts.

    They differ in one thing: Fit never magnifies past native size, Stretch
    does. Fit is native size where the frame fits and a shrink where it does
    not, because enlarging is not fitting and a model that renders smaller than
    the panel is worth seeing at the size it renders. Stretch grows it to meet
    the panel on its tighter axis.

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
        scale = min(scale, native)
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

    The fitting and the centring both happen in device pixels and the result is
    converted back to points at the end, because the picture is made of pixels
    and the quad has to land on whole ones. Native size is one frame pixel per
    point, so on a scaled display it is `scale` pixels per frame pixel, and the
    enlarging that keeps it crisp is done to the frame rather than by the GPU.
    """
    pixels = (area[0] * scale, area[1] * scale)
    size = displayed_size(frame, pixels, mode, scale)
    offset = centred_offset(size, pixels)
    return Placement(
        (offset[0] / scale, offset[1] / scale),
        (size[0] / scale, size[1] / scale),
        size,
    )


def needs_refresh(
    seq: int, last_seq: int, size: tuple[int, int], last_size: tuple[int, int]
) -> bool:
    """Whether what the quad is drawn from has to be rebuilt.

    A new frame from the render loop, which is the whole reason the mailbox
    carries a sequence number rather than the panel comparing arrays. An
    unchanged frame is never uploaded again, however many UI frames are drawn
    from it.

    And a change in the size it is drawn at, because a magnified frame is
    resampled to that size before the quad is drawn from it. The upload itself
    does not depend on the size, only the resample does, so dragging a dock
    split repeats a blit and never an upload.
    """
    return seq != last_seq or size != last_size


class PreviewPanel:
    def __init__(self, runtime) -> None:
        self._runtime = runtime
        self._last_seq = -1
        self._last_size = (0, 0)
        self._mode = DisplayMode.FIT
        self._display: np.ndarray | None = None
        self._texture: immvision.GlTexture | None = None
        self._enlarger: Enlarger | None = None
        self._enlarged = 0

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
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(*theme.NO_PADDING))
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
        """The texture the quad is drawn from, rebuilt only when it changes.

        The frame is always uploaded at its own resolution, so an upload costs
        the same whatever the panel is doing. What follows depends on which way
        the frame is being scaled: shrinking is handed to imgui's sampler,
        which interpolates, which is what a shrink wants. Magnifying goes
        through a blit first, because imgui's sampler would interpolate that
        too and a magnified frame wants its pixels: see `Enlarger`.

        Built on first use rather than in the constructor, because it takes a
        GL context and a panel is built before there is one.
        """
        if self._texture is None:
            self._texture = immvision.GlTexture()
            self._enlarger = Enlarger()
        native = (frame.shape[1], frame.shape[0])
        if needs_refresh(seq, self._last_seq, size, self._last_size):
            if seq != self._last_seq:
                # The colour order is explicit rather than left to immvision's
                # global setting, because the frames are RGB and a wrong guess
                # here is a silent swap of red and blue in the only place
                # anyone would see it.
                self._texture.update_from_image(
                    self._displayable(frame), is_color_order_bgr=False
                )
            self._enlarged = (
                self._enlarger.enlarge(self._texture.texture_id, native, size)
                if magnifies(native, size)
                else 0
            )
            self._last_seq = seq
            self._last_size = size
        return self._enlarged or self._texture.texture_id

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
