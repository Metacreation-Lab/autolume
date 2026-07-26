"""Fullscreen output window: an undecorated GLFW window with its own GL
context, driven from the UI thread's per-frame callback.

Legacy fullscreen shared the app's GL context, which macOS refused to open a
second surface into, so it never worked there. This window owns its own
context and uploads the frame itself instead, which is what makes it work on
every platform. It is driven from `window.py`'s per-frame callback rather
than a thread of its own because GLFW window creation has to happen on the
main thread, and this app only has the one.

`decide_action`, `letterbox_rect` and `suppressed_fullscreen` are the pure
logic and are unit tested. Everything else here is GL and GLFW calls, which
cannot be driven headless and stay manual-only (see
`tests/live/test_output_window.py`).
"""

import ctypes
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np
from imgui_bundle import glfw_utils, imgui

# `glfw_utils.glfw` rather than a bare `import glfw`: `imgui_bundle` points
# pyglfw's dynamic library search at its own bundled libglfw as a side effect
# of being imported, and `glfw_utils` is the module that does that import
# itself, so this is guaranteed to be the same native library
# `glfw_window_hello_imgui` uses. A plain `import glfw` here would work today
# only because `imgui_bundle` happens to be imported first in this file, an
# order an import sorter would happily reverse, which would silently load a
# second, independent GLFW instance whose window handles are not valid for
# the other's calls, including the `make_context_current` this module relies
# on every frame to hand control back to hello_imgui.
glfw = glfw_utils.glfw

from OpenGL import GL

from autolume.live.core.events import ControlEvent
from autolume.live.ui.theme import ERROR_COLOR

logger = logging.getLogger(__name__)

_ADDRESS = "/output/fullscreen"
_TITLE = "Autolume Output"
_UNAVAILABLE_STATUS = "Fullscreen output is unavailable. Check the log for details."
# Comfortably longer than one control loop tick (its default is 125 Hz, an
# 8 ms period) so a stale `True` this window's own submit has not caught up
# with yet reliably clears, short enough that a performer re-enabling
# fullscreen right after closing it never feels delayed.
_SUPPRESS_SECONDS = 0.25


class Action(Enum):
    """What the output window should do this frame."""

    NONE = "none"
    CREATE = "create"
    DESTROY = "destroy"
    UPLOAD = "upload"


def decide_action(
    fullscreen: bool,
    exists: bool,
    close_requested: bool,
    latest_seq: int,
    last_seq: int,
) -> Action:
    """What the output window does this frame, from state alone.

    Destroying wins over everything else. A window that reports it should
    close, or a param that has gone false, both end the window whatever else
    is true, including a frame still waiting to be uploaded, which is what
    keeps ESC and the param from disagreeing about whether a window exists.

    Creating only happens from nothing. Uploading only happens once there is
    something new to show: an unchanged sequence number, the ordinary case of
    a UI frame drawn between two render frames, does nothing.
    """
    if exists and (not fullscreen or close_requested):
        return Action.DESTROY
    if not exists:
        return Action.CREATE if fullscreen else Action.NONE
    if latest_seq != last_seq:
        return Action.UPLOAD
    return Action.NONE


def suppressed_fullscreen(
    fullscreen: bool, suppress_until: float | None, now: float
) -> tuple[bool, float | None]:
    """The fullscreen value `decide_action` should see this poll, and the
    suppression deadline to carry into the next one.

    A destroy this window starts on its own initiative (ESC, the OS closing
    it, a failed create, or a drawing failure) submits `fullscreen=False` to
    the control loop, which publishes it asynchronously. The very next poll
    can still read the stale `True` the loop has not caught up with yet,
    which without this would create the window right back, or in the failed
    create case, retry the failing create every poll.

    Bounded by `suppress_until` rather than held until an observed `False`:
    a source that keeps asserting `fullscreen=True` past the control loop's
    own propagation delay, such as a binding or an OSC surface resending
    stale state, must not be locked out of fullscreen for the rest of the
    session. Once `now` reaches the deadline, whatever `fullscreen` actually
    says wins again, stale or genuine, which is what lets a real re-enable
    come back rather than staying masked forever.
    """
    if suppress_until is None or now >= suppress_until:
        return fullscreen, None
    return False, suppress_until


@dataclass(frozen=True)
class Rect:
    """A destination rectangle, in whole pixels."""

    x: int
    y: int
    width: int
    height: int


def letterbox_rect(frame: tuple[int, int], monitor: tuple[int, int]) -> Rect:
    """Where to draw `frame` inside `monitor`, aspect kept, centred.

    Scaled by whichever axis is tighter, so the whole frame stays visible and
    whatever is left over on the other axis is the letterbox. Unlike the
    preview panel's Fit mode, this magnifies past native size on purpose: a
    projector is meant to be filled, and a model rendering smaller than it is
    the ordinary case, not the exception.

    A degenerate size on either side draws nothing, which is a real state:
    there is no frame at all before the first one arrives.
    """
    frame_w, frame_h = frame
    mon_w, mon_h = monitor
    if frame_w <= 0 or frame_h <= 0 or mon_w <= 0 or mon_h <= 0:
        return Rect(0, 0, 0, 0)
    scale = min(mon_w / frame_w, mon_h / frame_h)
    width = int(round(frame_w * scale))
    height = int(round(frame_h * scale))
    x = (mon_w - width) // 2
    y = (mon_h - height) // 2
    return Rect(x, y, width, height)


_VERTEX_SRC = """#version 330 core
layout (location = 0) in vec2 in_pos;
layout (location = 1) in vec2 in_uv;
out vec2 uv;
void main() {
    uv = in_uv;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

_FRAGMENT_SRC = """#version 330 core
in vec2 uv;
out vec4 out_color;
uniform sampler2D frame_texture;
void main() {
    out_color = texture(frame_texture, uv);
}
"""

# A full screen quad in clip space. Aspect fitting is done by shrinking the
# viewport to the letterbox rect before this is drawn, not by moving these
# vertices, so the quad always simply fills whatever viewport is bound.
#
# The uv is flipped relative to the position on purpose: glTexImage2D stores
# row 0 of the array first, which OpenGL then treats as the bottom of the
# texture, so the row that is visually the top of the frame has to be sampled
# at uv.y == 0 to land at the top of the screen (position y == 1).
_QUAD_VERTICES = np.array(
    [
        -1.0, -1.0, 0.0, 1.0,
         1.0, -1.0, 1.0, 1.0,
         1.0,  1.0, 1.0, 0.0,
        -1.0,  1.0, 0.0, 0.0,
    ],
    dtype=np.float32,
)  # fmt: skip
_QUAD_INDICES = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)


def _compile_shader(source: str, kind: int) -> int:
    shader = GL.glCreateShader(kind)
    GL.glShaderSource(shader, source)
    GL.glCompileShader(shader)
    if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
        log = GL.glGetShaderInfoLog(shader)
        GL.glDeleteShader(shader)
        raise RuntimeError(f"Output window shader failed to compile: {log}")
    return shader


def _link_program() -> int:
    vertex = _compile_shader(_VERTEX_SRC, GL.GL_VERTEX_SHADER)
    fragment = _compile_shader(_FRAGMENT_SRC, GL.GL_FRAGMENT_SHADER)
    program = GL.glCreateProgram()
    GL.glAttachShader(program, vertex)
    GL.glAttachShader(program, fragment)
    GL.glLinkProgram(program)
    GL.glDeleteShader(vertex)
    GL.glDeleteShader(fragment)
    if not GL.glGetProgramiv(program, GL.GL_LINK_STATUS):
        log = GL.glGetProgramInfoLog(program)
        GL.glDeleteProgram(program)
        raise RuntimeError(f"Output window shader failed to link: {log}")
    GL.glUseProgram(program)
    GL.glUniform1i(GL.glGetUniformLocation(program, "frame_texture"), 0)
    GL.glUseProgram(0)
    return program


def _build_quad() -> tuple[int, int, int]:
    vao = int(GL.glGenVertexArrays(1))
    vbo = int(GL.glGenBuffers(1))
    ebo = int(GL.glGenBuffers(1))
    GL.glBindVertexArray(vao)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
    GL.glBufferData(
        GL.GL_ARRAY_BUFFER, _QUAD_VERTICES.nbytes, _QUAD_VERTICES, GL.GL_STATIC_DRAW
    )
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, ebo)
    GL.glBufferData(
        GL.GL_ELEMENT_ARRAY_BUFFER,
        _QUAD_INDICES.nbytes,
        _QUAD_INDICES,
        GL.GL_STATIC_DRAW,
    )
    stride = 4 * _QUAD_VERTICES.itemsize
    GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(0))
    GL.glEnableVertexAttribArray(0)
    GL.glVertexAttribPointer(
        1,
        2,
        GL.GL_FLOAT,
        GL.GL_FALSE,
        stride,
        ctypes.c_void_p(2 * _QUAD_VERTICES.itemsize),
    )
    GL.glEnableVertexAttribArray(1)
    GL.glBindVertexArray(0)
    return vao, vbo, ebo


def _create_texture() -> int:
    texture = int(GL.glGenTextures(1))
    GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
    # Default unpack alignment is 4 bytes, which skews any upload whose row
    # width in bytes is not a multiple of that. Today's frames are
    # power-of-two and would never show it, but a future non-power-of-two
    # model would upload diagonally sheared, and that is not a bug anyone
    # would think to look for from the projector.
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    return texture


class OutputWindow:
    """Owns the fullscreen output's GLFW window, GL context and texture.

    `poll` is the only entry point, called once per UI frame from
    `window.py`'s per-frame callback, and it never blocks: `swap_interval(0)`
    is set on this window's own context precisely so its swap cannot stall
    the UI thread waiting on a projector's vblank.
    """

    def __init__(
        self,
        submit: Callable[[ControlEvent], None],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._submit = submit
        self._clock = clock
        self._window = None
        self._program = 0
        self._vao = 0
        self._vbo = 0
        self._ebo = 0
        self._texture = 0
        self._frame_size = (0, 0)
        self._last_seq = -1
        self._status: str | None = None
        # See `suppressed_fullscreen`: set whenever this window submits
        # `fullscreen=False` on its own initiative, cleared once that
        # suppression window elapses or the param is genuinely false again.
        self._suppress_until: float | None = None

    def poll(self, fullscreen: bool, preview) -> None:
        """Run one frame of the output window's lifecycle.

        Always restores hello_imgui's own context before returning, even if
        this window had nothing to do or just failed outright: leaving our
        context current is the one mistake that stops the whole app
        rendering, not just this window.

        The outer `except` is a last resort beyond the create and render
        paths, which already degrade on their own: a projector hiccup must
        not kill a live set, and that has to hold for whatever GLFW or imgui
        call turns out to be the one that fails, not only the ones
        anticipated here today, which is also why drawing the status is
        inside this same guard rather than after it.
        """
        try:
            self._poll(fullscreen, preview)
            self._draw_status()
        except Exception:
            logger.exception("Fullscreen output failed, closing it")
            self._force_close()
        finally:
            self._restore_main_context()

    def _poll(self, fullscreen: bool, preview) -> None:
        if not fullscreen:
            # Off is never itself a failure, whether it was always off or
            # just settled there after one: there is nothing left to warn
            # about, which is what keeps a status from an old failure stuck
            # on screen for the rest of the session.
            self._status = None
        now = self._clock()
        effective, self._suppress_until = suppressed_fullscreen(
            fullscreen, self._suppress_until, now
        )
        seq, frame = preview.latest()
        close_requested = self._window is not None and glfw.window_should_close(
            self._window
        )
        action = decide_action(
            effective, self._window is not None, close_requested, seq, self._last_seq
        )
        if action is Action.CREATE:
            self._try_create()
            return
        if action is Action.DESTROY:
            # Nothing to suppress if the destroy is only catching up to a
            # param that already reads false: the two are already in
            # agreement, so there is no race for the guard to protect.
            self._close(suppress=fullscreen)
            return
        if self._window is None:
            return
        try:
            self._render(frame, seq, action is Action.UPLOAD)
        except Exception:
            logger.exception("Fullscreen output failed while drawing, closing it")
            self._close(suppress=True)
            self._status = _UNAVAILABLE_STATUS

    def _try_create(self) -> None:
        try:
            self._create()
        except Exception:
            logger.exception("Could not open the fullscreen output window")
            self._close(suppress=True)
            self._status = _UNAVAILABLE_STATUS
        else:
            self._status = None

    def _create(self) -> None:
        glfw.default_window_hints()
        glfw.window_hint(glfw.DECORATED, False)
        glfw.window_hint(glfw.RESIZABLE, False)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        monitor = glfw.get_primary_monitor()
        if monitor is None:
            raise RuntimeError("No primary monitor reported by GLFW")
        mode = glfw.get_video_mode(monitor)
        if mode is None:
            raise RuntimeError("No video mode reported for the primary monitor")
        window = glfw.create_window(mode.size.width, mode.size.height, _TITLE, None, None)
        # Hints are process-global and only consumed at creation, so they are
        # put back to GLFW's own defaults immediately after, rather than left
        # sitting there for whatever creates the next GLFW window, which
        # would otherwise inherit undecorated, non-resizable and a pinned GL
        # version meant only for this one.
        glfw.default_window_hints()
        if window is None:
            raise RuntimeError("glfw.create_window returned no window")
        self._window = window
        mon_x, mon_y = glfw.get_monitor_pos(monitor)
        glfw.set_window_pos(window, mon_x, mon_y)
        glfw.set_key_callback(window, self._on_key)
        glfw.make_context_current(window)
        glfw.swap_interval(0)
        self._program = _link_program()
        self._vao, self._vbo, self._ebo = _build_quad()
        self._texture = _create_texture()
        self._frame_size = (0, 0)
        self._last_seq = -1

    def _on_key(self, window, key, scancode, action, mods) -> None:
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)

    def _render(self, frame: np.ndarray | None, seq: int, do_upload: bool) -> None:
        glfw.make_context_current(self._window)
        fb_width, fb_height = glfw.get_framebuffer_size(self._window)
        GL.glViewport(0, 0, fb_width, fb_height)
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        if do_upload and frame is not None:
            self._upload(frame)
            self._last_seq = seq
        if self._frame_size != (0, 0) and fb_width > 0 and fb_height > 0:
            rect = letterbox_rect(self._frame_size, (fb_width, fb_height))
            if rect.width > 0 and rect.height > 0:
                GL.glViewport(rect.x, rect.y, rect.width, rect.height)
                GL.glUseProgram(self._program)
                GL.glBindVertexArray(self._vao)
                GL.glActiveTexture(GL.GL_TEXTURE0)
                GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
                GL.glDrawElements(GL.GL_TRIANGLES, 6, GL.GL_UNSIGNED_INT, ctypes.c_void_p(0))
                GL.glBindVertexArray(0)
        glfw.swap_buffers(self._window)

    def _upload(self, frame: np.ndarray) -> None:
        height, width = frame.shape[0], frame.shape[1]
        data = np.ascontiguousarray(frame)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
        if (width, height) != self._frame_size:
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D,
                0,
                GL.GL_RGB8,
                width,
                height,
                0,
                GL.GL_RGB,
                GL.GL_UNSIGNED_BYTE,
                data,
            )
            self._frame_size = (width, height)
        else:
            GL.glTexSubImage2D(
                GL.GL_TEXTURE_2D,
                0,
                0,
                0,
                width,
                height,
                GL.GL_RGB,
                GL.GL_UNSIGNED_BYTE,
                data,
            )

    def _close(self, *, suppress: bool) -> None:
        """Tear the window down and always report it gone.

        `suppress` is whatever `fullscreen` still read at the moment this
        window decided to close on its own: see `suppressed_fullscreen` for
        why that matters and `_submit_off` for what it does with it.

        `_submit_off` runs in a `finally` around `_destroy_gl` so that even an
        unexpected failure while releasing GL resources still ends with the
        param telling the truth. `_destroy_gl` itself is written the same
        way: it forgets the window first and only then tries the native
        calls, so the bookkeeping cannot be left claiming a window that a
        failed teardown left half-destroyed.
        """
        try:
            self._destroy_gl()
        finally:
            self._submit_off(suppress=suppress)

    def _force_close(self) -> None:
        """Best-effort teardown after a failure `_close` itself did not expect."""
        try:
            self._close(suppress=True)
        except Exception:
            logger.exception("Failed to close the fullscreen output cleanly")
            self._window = None
        self._status = _UNAVAILABLE_STATUS

    def _destroy_gl(self) -> None:
        window = self._window
        if window is None:
            return
        texture, vbo, ebo, vao, program = (
            self._texture,
            self._vbo,
            self._ebo,
            self._vao,
            self._program,
        )
        # State is forgotten before any native call is attempted, so a
        # failure below leaves the bookkeeping truthful even though the
        # underlying resources may or may not actually be gone.
        self._window = None
        self._program = 0
        self._vao = 0
        self._vbo = 0
        self._ebo = 0
        self._texture = 0
        self._frame_size = (0, 0)
        self._last_seq = -1
        try:
            glfw.make_context_current(window)
            if texture:
                GL.glDeleteTextures([texture])
            if vbo:
                GL.glDeleteBuffers(1, [vbo])
            if ebo:
                GL.glDeleteBuffers(1, [ebo])
            if vao:
                GL.glDeleteVertexArrays(1, [vao])
            if program:
                GL.glDeleteProgram(program)
        except Exception:
            logger.exception("Error releasing fullscreen output GL resources")
        try:
            glfw.destroy_window(window)
        except Exception:
            logger.exception("Error destroying the fullscreen output window")

    def _submit_off(self, *, suppress: bool) -> None:
        if suppress:
            self._suppress_until = self._clock() + _SUPPRESS_SECONDS
        self._submit(ControlEvent(_ADDRESS, False, source="ui"))

    def _restore_main_context(self) -> None:
        try:
            glfw.make_context_current(glfw_utils.glfw_window_hello_imgui())
        except Exception:
            logger.exception("Could not restore the main GL context")

    def _draw_status(self) -> None:
        if not self._status:
            return
        io = imgui.get_io()
        text = self._status
        size = imgui.calc_text_size(text)
        pad = 8.0
        x = (io.display_size.x - size.x) * 0.5
        y = io.display_size.y - size.y - pad * 3.0
        draw_list = imgui.get_foreground_draw_list()
        draw_list.add_rect_filled(
            imgui.ImVec2(x - pad, y - pad),
            imgui.ImVec2(x + size.x + pad, y + size.y + pad),
            imgui.get_color_u32(imgui.ImVec4(0.0, 0.0, 0.0, 0.8)),
        )
        draw_list.add_text(
            imgui.ImVec2(x, y), imgui.get_color_u32(imgui.ImVec4(*ERROR_COLOR)), text
        )
