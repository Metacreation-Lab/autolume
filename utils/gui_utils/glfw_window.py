# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import logging
import os
import sys
import time
import glfw
import OpenGL.GL as gl
import PIL.Image
from . import gl_utils
from . import dpi
from utils.resource_paths import resource_path


logger = logging.getLogger(__name__)

def _wayland_session():
    return sys.platform == 'linux' and bool(os.environ.get('WAYLAND_DISPLAY'))

#----------------------------------------------------------------------------

class GlfwWindow: # pylint: disable=too-many-public-methods
    def __init__(self, *, title='GlfwWindow', window_width=1920, window_height=1080, deferred_show=True, close_on_esc=True):
        self.stopped = False
        self._glfw_window           = None
        self._drawing_frame         = False
        self._frame_start_time      = None
        self._frame_delta           = 0
        self._fps_limit             = None
        self._vsync                 = None
        self._skip_frames           = 0
        self._deferred_show         = deferred_show
        self._close_on_esc          = close_on_esc
        self._esc_pressed           = False
        self._drag_and_drop_paths   = None
        self._capture_next_frame    = False
        self._captured_frame        = None

        # Create window.
        glfw.init()
        glfw.window_hint(glfw.VISIBLE, False)
        # Windows measures windows in physical pixels and keeps their pixel size
        # when dragged across monitors, while the UI font follows the monitor's
        # DPI scale — the layout would reflow. Let GLFW resize the window by the
        # DPI ratio on monitor change so the font-to-window ratio stays constant,
        # matching the point-based (reflow-free) behavior macOS gives for free.
        if sys.platform == 'win32':
            glfw.window_hint(glfw.SCALE_TO_MONITOR, glfw.TRUE)
        # XWayland: force an EGL context so PyOpenGL's EGL platform (not the
        # buggy GLX one) picks it up.
        if _wayland_session():
            glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)
        self._glfw_window = glfw.create_window(width=window_width, height=window_height, title=title, monitor=None, share=None)
        self._set_window_icon()
        self._attach_glfw_callbacks()
        self.make_context_current()

        # Adjust window.
        self.set_vsync(False)
        self.set_window_size(window_width, window_height)
        if not self._deferred_show:
            glfw.show_window(self._glfw_window)



    def _set_window_icon(self):
        # macOS and Wayland don't support per-window icons.
        if sys.platform == 'darwin' or glfw.get_platform() == glfw.PLATFORM_WAYLAND:
            return
        # GLFW needs raw RGBA pixels, so load the PNG rather than the .exe .ico.
        try:
            image = PIL.Image.open(resource_path('assets', 'metacreation-logo.png')).convert('RGBA')
            glfw.set_window_icon(self._glfw_window, 1, [image])
        except Exception as err: # pylint: disable=broad-except
            logger.warning('Failed to set window icon: %s', err)

    def native_handle(self):
        # Win32 HWND, for the few things GLFW does not expose. None elsewhere,
        # and on a GLFW built without the Win32 native access header.
        if sys.platform != 'win32' or not hasattr(glfw, 'get_win32_window'):
            return None
        if self._glfw_window is None:
            return None
        try:
            return glfw.get_win32_window(self._glfw_window)
        except Exception as err: # pylint: disable=broad-except
            logger.warning('Failed to get the native window handle: %s', err)
            return None

    def close(self):
        if self._drawing_frame:
            self.end_frame()
        if self._glfw_window is not None:
            glfw.destroy_window(self._glfw_window)
            self._glfw_window = None
        #glfw.terminate() # Commented out to play it nice with other glfw clients.

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def _content_scale(self):
        return dpi.window_unit_scale(self._glfw_window)

    @property
    def window_width(self):
        return self.content_width

    @property
    def window_height(self):
        return self.content_height + self.title_bar_height

    @property
    def content_width(self):
        width, _ = glfw.get_window_size(self._glfw_window)
        return round(width / self._content_scale())

    @property
    def content_height(self):
        _, height = glfw.get_window_size(self._glfw_window)
        return round(height / self._content_scale())

    @property
    def title_bar_height(self):
        _left, top, _right, _bottom = glfw.get_window_frame_size(self._glfw_window)
        return round(top / self._content_scale())

    def _get_work_area(self):
        monitor = glfw.get_primary_monitor()
        # get_monitor_workarea can crash on macOS; use it elsewhere only.
        # A zero-sized result falls through to the always-valid video mode.
        if sys.platform != 'darwin':
            area_x, area_y, area_width, area_height = glfw.get_monitor_workarea(monitor)
            if area_width > 0 and area_height > 0:
                return area_x, area_y, area_width, area_height
        area_x, area_y = glfw.get_monitor_pos(monitor)
        mode = glfw.get_video_mode(monitor)
        return area_x, area_y, mode.size.width, mode.size.height

    @property
    def monitor_width(self):
        _, _, width, _ = self._get_work_area()
        return round(width / self._content_scale())

    @property
    def monitor_height(self):
        _, _, _, height = self._get_work_area()
        return round(height / self._content_scale())

    @property
    def frame_delta(self):
        return self._frame_delta

    def set_title(self, title):
        glfw.set_window_title(self._glfw_window, title)

    def set_window_size(self, width, height):
        width = min(width, self.monitor_width)
        height = min(height, self.monitor_height)
        scale = self._content_scale()
        tbh = round(self.title_bar_height * scale)  # physical title-bar height
        glfw.restore_window(self._glfw_window)
        glfw.set_window_size(self._glfw_window,
                             round(width * scale),
                             max(round(height * scale) - tbh, 0))
        if width == self.monitor_width and height == self.monitor_height:
            self.maximize()

    def set_content_size(self, width, height):
        self.set_window_size(width, height + self.title_bar_height)

    def maximize(self):
        # Fill the work area explicitly rather than entering the OS "maximized"
        # state: a maximized window snaps back to its smaller restored size the
        # moment the user drags it (Windows), and maximize_window is also
        # unreliable on XWayland and animates badly on undecorated macOS windows.
        area_x, area_y, area_width, area_height = self._get_work_area()
        tbh = round(self.title_bar_height * self._content_scale())  # logical → OS units
        if sys.platform != 'darwin':
            # Clear any stale maximized state. Skipped on macOS, which never has
            # one and where restoring a screen-filling window plays the NSWindow
            # zoom animation.
            glfw.restore_window(self._glfw_window)
        if glfw.get_platform() != glfw.PLATFORM_WAYLAND:
            # Wayland: compositor owns window position.
            glfw.set_window_pos(self._glfw_window, area_x, area_y + tbh)
        glfw.set_window_size(self._glfw_window, area_width, max(area_height - tbh, 1))

    def set_position(self, x, y):
        # Wayland: compositor owns window position.
        if glfw.get_platform() == glfw.PLATFORM_WAYLAND:
            return
        # Offset by the work-area origin (e.g. macOS menu bar).
        area_x, area_y, _, _ = self._get_work_area()
        scale = self._content_scale()
        glfw.set_window_pos(self._glfw_window,
                            area_x + round(x * scale),
                            area_y + round((y + self.title_bar_height) * scale))

    def center(self):
        self.set_position((self.monitor_width - self.window_width) // 2, (self.monitor_height - self.window_height) // 2)

    def hide_title_bar(self):
        glfw.set_window_attrib(self._glfw_window, glfw.DECORATED, False)

    def show_title_bar(self):
        glfw.set_window_attrib(self._glfw_window, glfw.DECORATED, True)

    def set_vsync(self, vsync):
        vsync = bool(vsync)
        if vsync != self._vsync:
            glfw.swap_interval(1 if vsync else 0)
            self._vsync = vsync

    def set_fps_limit(self, fps_limit):
        self._fps_limit = int(fps_limit)

    def should_close(self):
        return glfw.window_should_close(self._glfw_window) # or (self._close_on_esc and self._esc_pressed)

    def stop(self):
        self.stopped = True

    def skip_frame(self):
        self.skip_frames(1)

    def skip_frames(self, num): # Do not update window for the next N frames.
        self._skip_frames = max(self._skip_frames, int(num))

    def is_skipping_frames(self):
        return self._skip_frames > 0

    def capture_next_frame(self):
        self._capture_next_frame = True

    def pop_captured_frame(self):
        frame = self._captured_frame
        self._captured_frame = None
        return frame

    def pop_drag_and_drop_paths(self):
        paths = self._drag_and_drop_paths
        self._drag_and_drop_paths = None
        return paths

    def draw_frame(self): # To be overridden by subclass.
        self.begin_frame()
        # Rendering code goes here.
        self.end_frame()

    def make_context_current(self):
        if self._glfw_window is not None:
            glfw.make_context_current(self._glfw_window)

    def begin_frame(self):
        # End previous frame.
        if self._drawing_frame:
            self.end_frame()

        # Apply FPS limit.
        if self._frame_start_time is not None and self._fps_limit is not None:
            delay = self._frame_start_time - time.perf_counter() + 1 / self._fps_limit
            if delay > 0:
                time.sleep(delay)
        cur_time = time.perf_counter()
        if self._frame_start_time is not None:
            self._frame_delta = cur_time - self._frame_start_time
        self._frame_start_time = cur_time

        # Process events.
        glfw.poll_events()

        # Begin frame.
        self._drawing_frame = True
        self.make_context_current()

        # Initialize GL state. The viewport covers the framebuffer, which is larger
        # than the window on scaled (retina) displays; the projection below keeps
        # all drawing in window coordinates.
        fb_width, fb_height = glfw.get_framebuffer_size(self._glfw_window)
        gl.glViewport(0, 0, fb_width, fb_height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glTranslate(-1, 1, 0)
        gl.glScale(2 / max(self.content_width, 1), -2 / max(self.content_height, 1), 1)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE_MINUS_SRC_ALPHA) # Pre-multiplied alpha.

        # Clear.
        gl.glClearColor(0, 0, 0, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    def end_frame(self):
        assert self._drawing_frame
        self._drawing_frame = False

        # Skip frames if requested.
        if self._skip_frames > 0:
            self._skip_frames -= 1
            return

        # Capture frame if requested.
        if self._capture_next_frame:
            self._captured_frame = gl_utils.read_pixels(*glfw.get_framebuffer_size(self._glfw_window))
            self._capture_next_frame = False

        # Update window.
        if self._deferred_show:
            glfw.show_window(self._glfw_window)
            self._deferred_show = False
        glfw.swap_buffers(self._glfw_window)

    def _attach_glfw_callbacks(self):
        glfw.set_key_callback(self._glfw_window, self._glfw_key_callback)
        glfw.set_drop_callback(self._glfw_window, self._glfw_drop_callback)

    def _glfw_key_callback(self, _window, key, _scancode, action, _mods):
        if action == glfw.PRESS and key == glfw.KEY_ESCAPE:
            self._esc_pressed = True

    def _glfw_drop_callback(self, _window, paths):
        self._drag_and_drop_paths = paths

#----------------------------------------------------------------------------
