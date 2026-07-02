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

from . import glfw_window
from . import imgui_utils
from . import text_utils

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
        # Rasterize fonts at the framebuffer scale (2x on HiDPI) so text stays
        # sharp; sizes exposed to the UI remain in logical (content) coordinates.
        self._font_path  = font
        self._font_sizes = font_sizes
        fb_width = glfw.get_framebuffer_size(self._glfw_window)[0]
        font_scale = max(1, round(fb_width / max(self.content_width, 1)))
        self._font_dpi_scale = font_scale
        imgui.get_io().font_global_scale = 1 / font_scale
        self._imgui_fonts = {size: imgui.get_io().fonts.add_font_from_file_ttf(font, size * font_scale) for size in font_sizes}
        self._imgui_renderer.refresh_font_texture()

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

    def _update_font_scale(self):
        # Rebuild the font atlas when the window moves to a monitor with a different DPI scale.
        fb_width = glfw.get_framebuffer_size(self._glfw_window)[0]
        new_scale = max(1, round(fb_width / max(self.content_width, 1)))
        if new_scale == self._font_dpi_scale:
            return
        self._font_dpi_scale = new_scale
        imgui.get_io().font_global_scale = 1 / new_scale
        imgui.get_io().fonts.clear()
        self._imgui_fonts = {size: imgui.get_io().fonts.add_font_from_file_ttf(
            self._font_path, size * new_scale) for size in self._font_sizes}
        self._imgui_renderer.refresh_font_texture()
        self.skip_frame()

    def begin_frame(self):
        # Begin glfw frame.
        super().begin_frame()

        # Process imgui events.
        self._imgui_renderer.mouse_wheel_multiplier = self._cur_font_size / 10
        if self.content_width > 0 and self.content_height > 0:
            self._imgui_renderer.process_inputs()
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
# Wrapper class for GlfwRenderer to fix a mouse wheel bug on Linux.

class _GlfwRenderer(imgui.integrations.glfw.GlfwRenderer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mouse_wheel_multiplier = 1
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
        self.io.mouse_wheel += y_offset * self.mouse_wheel_multiplier

if sys.platform == 'darwin':
    # macOS offers either a GL 2.1 context or a 3.2+ core profile, never both.
    # The app draws with fixed-function GL, so it runs on the 2.1 context, where
    # the default imgui backend cannot compile its GLSL 330 shaders. Swap the
    # rendering methods for pyimgui's fixed-function backend instead.
    from imgui.integrations.opengl import FixedPipelineRenderer

    class _GlfwRenderer(_GlfwRenderer):
        refresh_font_texture = FixedPipelineRenderer.refresh_font_texture
        render = FixedPipelineRenderer.render
        _create_device_objects = FixedPipelineRenderer._create_device_objects
        _invalidate_device_objects = FixedPipelineRenderer._invalidate_device_objects

#----------------------------------------------------------------------------
