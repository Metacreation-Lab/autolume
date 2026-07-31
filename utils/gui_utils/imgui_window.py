# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
import glfw
import imgui
import imgui.integrations.glfw
import numpy as np
import OpenGL.GL as gl

from . import glfw_window
from . import imgui_utils
from . import text_utils
from . import dpi

#----------------------------------------------------------------------------

class ImguiWindow(glfw_window.GlfwWindow):
    def __init__(self, *, title='ImguiWindow', font=None, font_sizes=range(14,24), **glfw_kwargs):
        if font is None:
            font = text_utils.get_default_font()
        font_sizes = {int(size) for size in font_sizes}
        super().__init__(title=title, **glfw_kwargs)

        # Init fields.
        self._imgui_context  = None
        self._imgui_renderer = None
        self._imgui_fonts    = None
        self._cur_font_size  = max(font_sizes)

        # Delete leftover imgui.ini to avoid unexpected behavior.
        if os.path.isfile('imgui.ini'):
            os.remove('imgui.ini')

        # Init ImGui.
        self._imgui_context = imgui.create_context()
        self._imgui_renderer = _GlfwRenderer(self._glfw_window)
        self._attach_glfw_callbacks()
        imgui.get_io().ini_saving_rate = 0 # Disable creating imgui.ini at runtime.
        imgui.get_io().mouse_drag_threshold = 0 # Improve behavior with imgui_utils.drag_custom().
        self._font_path  = font
        self._font_sizes = font_sizes
        self._pending_font_sizes = None
        self._font_atlas_key = self._current_font_atlas_key()
        self._rebuild_font_atlas()

    def close(self):
        self.make_context_current()
        self._imgui_fonts = None
        if self._imgui_renderer is not None:
            self._imgui_renderer.shutdown()
            self._imgui_renderer = None
        if self._imgui_context is not None:
            #imgui.destroy_context(self._imgui_context) # Commented out to avoid creating imgui.ini at the end.
            self._imgui_context = None
        super().close()

    def _glfw_key_callback(self, *args):
        super()._glfw_key_callback(*args)
        self._imgui_renderer.keyboard_callback(*args)

    @property
    def font_size(self):
        return self._cur_font_size

    @property
    def spacing(self):
        return round(self._cur_font_size * 0.4)

    def set_font_size(self, target): # Applied on next frame.
        self._cur_font_size = min((abs(key - target), key) for key in self._imgui_fonts.keys())[1]

    def set_font_sizes(self, font_sizes): # Applied on next frame.
        # Deferred like set_font_size: the atlas cannot be rebuilt mid-frame
        # because the frame being drawn still references the old font objects.
        self._pending_font_sizes = {int(size) for size in font_sizes}

    def _apply_pending_font_sizes(self):
        if self._pending_font_sizes is None:
            return
        sizes, self._pending_font_sizes = self._pending_font_sizes, None
        if sizes == self._font_sizes:
            return
        self._font_sizes = sizes
        self._rebuild_font_atlas()
        self.set_font_size(self._cur_font_size) # Re-snap onto an existing size.
        self.skip_frame()

    def scale_ui_size(self, size):
        # Convert a DPI-independent UI size to logical units for the current monitor.
        return dpi.scale_ui_size(size, self._glfw_window)

    def _current_font_atlas_key(self):
        # The atlas must be rebuilt whenever either of these changes: the raster
        # scale (physical px per logical unit) or the 1x-sharpening state. On
        # Windows the raster scale is always 1, so the 1x flag is the only signal
        # that the window crossed onto a different-DPI monitor.
        w = self._glfw_window
        return (dpi.pixels_per_logical(w), dpi.is_native_1x(w))

    def _rebuild_font_atlas(self):
        # Rasterize each UI size at the monitor's physical resolution so text
        # stays sharp; the sizes the app sees stay in logical units via
        # font_global_scale.
        raster_scale = self._font_atlas_key[0]
        io = imgui.get_io()
        io.font_global_scale = 1 / raster_scale
        io.fonts.clear()
        self._imgui_fonts = {size: io.fonts.add_font_from_file_ttf(
            self._font_path, size * raster_scale) for size in self._font_sizes}
        self._imgui_renderer.refresh_font_texture()

    def _update_font_scale(self):
        # Rebuild the atlas when the window moves to a monitor that changes the
        # atlas key (raster scale or 1x-sharpening state).
        key = self._current_font_atlas_key()
        if key == self._font_atlas_key:
            return
        self._font_atlas_key = key
        self._rebuild_font_atlas()
        self.skip_frame()

    def begin_frame(self):
        # Begin glfw frame.
        super().begin_frame()

        # Process imgui events.
        if self.content_width > 0 and self.content_height > 0:
            self._imgui_renderer.process_inputs()
        self._apply_pending_font_sizes()
        self._update_font_scale()

        # Begin imgui frame.
        imgui.new_frame()
        imgui.push_font(self._imgui_fonts[self._cur_font_size])
        imgui_utils.set_default_style(spacing=self.spacing, indent=self.font_size, scrollbar=self.font_size+4)

    def end_frame(self):
        imgui.pop_font()
        imgui.render()
        imgui.end_frame()
        self._imgui_renderer.render(imgui.get_draw_data())
        super().end_frame()

#----------------------------------------------------------------------------

def _refresh_font_texture(renderer, alpha8):
    # Upload the imgui font atlas, sharpened on 1x displays. There the font's
    # fractional glyph advances put glyphs on fractional pixels, so bilinear
    # sampling smears them: nearest-neighbor keeps glyphs bit-exact and the
    # contrast curve restores stem solidity. On HiDPI the smear is
    # sub-physical-pixel, so the stock bilinear path is kept. Shared by both
    # backends; alpha8 for the fixed pipeline (macOS), rgba32 otherwise. The base
    # renderer calls this once from __init__ before self.window is set (window is
    # None -> not sharpened); ImguiWindow refreshes again after adding its fonts.
    io = renderer.io
    native_1x = dpi.is_native_1x(getattr(renderer, 'window', None))
    last_texture = gl.glGetIntegerv(gl.GL_TEXTURE_BINDING_2D)
    gl_format = gl.GL_ALPHA if alpha8 else gl.GL_RGBA
    if alpha8:
        width, height, pixels = io.fonts.get_tex_data_as_alpha8()
        if native_1x:
            pixels = dpi.sharpen_font_alpha(np.frombuffer(pixels, dtype=np.uint8)).astype(np.uint8).tobytes()
    else:
        width, height, pixels = io.fonts.get_tex_data_as_rgba32()
        if native_1x:
            rgba = np.frombuffer(pixels, dtype=np.uint8).reshape(-1, 4).copy()
            rgba[:, 3] = dpi.sharpen_font_alpha(rgba[:, 3]).astype(np.uint8)
            pixels = rgba.tobytes()
    if renderer._font_texture is not None:
        gl.glDeleteTextures([renderer._font_texture])
    renderer._font_texture = gl.glGenTextures(1)
    tex_filter = gl.GL_NEAREST if native_1x else gl.GL_LINEAR
    gl.glBindTexture(gl.GL_TEXTURE_2D, renderer._font_texture)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, tex_filter)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, tex_filter)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl_format, width, height, 0,
                    gl_format, gl.GL_UNSIGNED_BYTE, pixels)
    io.fonts.texture_id = renderer._font_texture
    gl.glBindTexture(gl.GL_TEXTURE_2D, last_texture)
    io.fonts.clear_tex_data()

#----------------------------------------------------------------------------
# Wrapper class for GlfwRenderer to fix a mouse wheel bug on Linux.

# GLFW does not normalize scroll deltas: Windows and X11 report one unit per
# wheel notch, macOS reports the OS line rate (3 per notch) or, on precise
# devices, points/10. imgui then scrolls by mouse_wheel * 5 * font_size, which
# is already constant in DPI-independent units -- deriving the scale from the
# font size would apply the display scale a second time. Both values put a wheel
# notch at ~60 units, matching native scrolling.
_MOUSE_WHEEL_SCALE = 0.25 if sys.platform == 'darwin' else 0.75

class _GlfwRenderer(imgui.integrations.glfw.GlfwRenderer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The base class only polls button state once per frame, so a click whose
        # press and release both happen between two frames (e.g. a trackpad tap,
        # or any click while the frame rate is low) is lost. Latch presses from
        # the event callback and hold them for one frame, like the official
        # Dear ImGui GLFW backend does.
        self._mouse_just_pressed = [False, False, False]
        glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)

    def _mouse_button_callback(self, _window, button, action, _mods):
        if action == glfw.PRESS and 0 <= button < len(self._mouse_just_pressed):
            self._mouse_just_pressed[button] = True

    def process_inputs(self):
        super().process_inputs()
        for i in range(len(self._mouse_just_pressed)):
            if self._mouse_just_pressed[i]:
                self.io.mouse_down[i] = True
                self._mouse_just_pressed[i] = False
        # XWayland HiDPI: correct display_size, display_fb_scale, and mouse_pos
        # to logical units (GLFW returns physical pixels on XWayland).
        if sys.platform == 'linux':
            fb_w, fb_h = glfw.get_framebuffer_size(self.window)
            win_w, win_h = glfw.get_window_size(self.window)
            if fb_w == win_w:
                xscale, yscale = glfw.get_window_content_scale(self.window)
                if xscale > 1.0:
                    self.io.display_size = win_w / xscale, win_h / yscale
                    self.io.display_fb_scale = xscale, yscale
                    cx, cy = glfw.get_cursor_pos(self.window)
                    self.io.mouse_pos = cx / xscale, cy / yscale

    def scroll_callback(self, window, x_offset, y_offset):
        # Accumulate rather than assign: several scroll events can arrive between
        # two frames, and imgui clears mouse_wheel at the end of each one.
        # x_offset is dropped on purpose -- trackpads leak sideways deltas into
        # vertical swipes, and nothing in the UI scrolls horizontally.
        self.io.mouse_wheel += y_offset * _MOUSE_WHEEL_SCALE

    def refresh_font_texture(self):
        _refresh_font_texture(self, alpha8=False)

if sys.platform == 'darwin':
    # macOS offers either a GL 2.1 context or a 3.2+ core profile, never both.
    # The app draws with fixed-function GL, so it runs on the 2.1 context, where
    # the default imgui backend cannot compile its GLSL 330 shaders. Swap the
    # rendering methods for pyimgui's fixed-function backend instead.
    from imgui.integrations.opengl import FixedPipelineRenderer

    class _GlfwRenderer(_GlfwRenderer):
        render = FixedPipelineRenderer.render
        _create_device_objects = FixedPipelineRenderer._create_device_objects
        _invalidate_device_objects = FixedPipelineRenderer._invalidate_device_objects

        def refresh_font_texture(self):
            # The fixed pipeline needs the alpha8 atlas format.
            _refresh_font_texture(self, alpha8=True)

#----------------------------------------------------------------------------
