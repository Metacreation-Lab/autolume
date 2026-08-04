# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import logging
import threading
import time
import numpy as np
import queue
import imgui
import cv2

import dnnlib
from utils import device_utils
from utils.gui_utils import imgui_utils
from utils.gui_utils import gl_utils
from utils.gui_utils import text_utils
from widgets import pickle_widget
from widgets import latent_widget
from widgets import display_capture_widget
from widgets import trunc_noise_widget
from widgets import performance_widget
from widgets import layer_widget
from widgets import adjuster_widget
from widgets import looping_widget
from widgets import preset_widget
from widgets import audio_widget
from widgets import mixing_widget
from widgets import collapsable_layer
from widgets import diffusion_widget
from widgets.help_icon_widget import HelpIconWidget

from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.dispatcher import Dispatcher
from pythonosc.udp_client import SimpleUDPClient
try:
    import NDIlib as ndi
except ImportError:
    ndi = None  # NDIlib is optional; NDI streaming is disabled when it isn't installed.

import glfw
from OpenGL import GL as gl
import ctypes
import os

logger = logging.getLogger(__name__)

#----------------------------------------------------------------------------
class RecordingDispatcher(Dispatcher):
    """Dispatcher that records when each incoming OSC address was last seen."""

    def __init__(self):
        super().__init__()
        self._last_seen = {}

    def handlers_for_address(self, address_pattern):
        self._last_seen[address_pattern] = time.time()
        return super().handlers_for_address(address_pattern)

    def streaming_addresses(self, window=60.0):
        # Segments starting with "_" are sender metadata (e.g. TouchDesigner's /_samplerate).
        cutoff = time.time() - window
        return sorted(addr for addr, seen in list(self._last_seen.items())
                      if seen >= cutoff and not addr.rsplit("/", 1)[-1].startswith("_"))

#----------------------------------------------------------------------------
class Visualizer:
    def __init__(self, app, renderer):
        self.app = app
        self.help_icon = HelpIconWidget()
        self.help_texts, self.help_urls = self.help_icon.load_help_texts("visualizer")

        #COMMUNICATIONS
        self.in_ip = "0.0.0.0"
        self.in_port = 1338
        self.out_ip = "127.0.0.1"
        self.out_port = 1337
        self.osc_dispatcher = RecordingDispatcher()
        self.osc_client = SimpleUDPClient(self.out_ip, self.out_port)
        self.server = None
        self.server_thread = None
        self.start_osc_server()
        self.osc_dispatcher.map("/*", self.osc_message_handler)



        # NDI parameters
        self.ndi_name = 'Autolume Live'
        self.ndi_send = None
        self.video_frame = None
        self.start_ndi()

        # Internals.

        self.pane_w = 0
        self._last_error_print  = None
        self._async_renderer    = renderer
        self._defer_rendering   = 0
        self._tex_img           = None
        self._tex_obj           = None

        # Widget interface.
        self.args               = dnnlib.EasyDict()
        self.result             = dnnlib.EasyDict()

        # Widgets.
        self.display_capture_widget = display_capture_widget.DisplayCaptureWidget(self)
        self.pickle_widget      = pickle_widget.PickleWidget(self)
        self.latent_widget      = latent_widget.LatentWidget(self)
        self.trunc_noise_widget = trunc_noise_widget.TruncationNoiseWidget(self)
        self.perf_widget        = performance_widget.PerformanceWidget(self)
        self.layer_widget       = layer_widget.LayerWidget(self)
        self.adjuster_widget = adjuster_widget.AdjusterWidget(self)
        self.looping_widget = looping_widget.LoopingWidget(self)
        self.preset_widget = preset_widget.PresetWidget(self)
        self.mixing_widget = mixing_widget.MixingWidget(self)
        self.collapsed_widget = collapsable_layer.LayerWidget(self)
        self.audio_widget = audio_widget.AudioWidget(self)
        self.diffusion_widget = diffusion_widget.DiffusionWidget(self)

    #Screen capture and screen recording
        self.is_recording = False
        self.frame_queue = queue.Queue()
        self.recording_thread = None
        self.recording_file_path = None

    #Fullscreen
        self.is_fullscreen_display = False
        self.fullscreen_window = None
        self.main_window_context = None
        self.fullscreen_shader = None
        self.fullscreen_vao = None
        self.fullscreen_vbo = None
        self.window_created = False

        self.fit_screen = False

        # reset scroll position on initialization
        self.reset_scroll = True

    def create_shader_program(self):
        try:
            vertex_shader = """
            #version 330 core
            layout (location = 0) in vec2 aPos;
            layout (location = 1) in vec2 aTexCoord;
            uniform vec2 uScale;
            uniform vec2 uOffset;
            out vec2 TexCoord;
            void main() {
                vec2 pos = aPos * uScale + uOffset;
                gl_Position = vec4(pos, 0.0, 1.0);
                TexCoord = aTexCoord;
            }
            """

            fragment_shader = """
            #version 330 core
            in vec2 TexCoord;
            out vec4 FragColor;
            uniform sampler2D ourTexture;
            void main() {
                FragColor = texture(ourTexture, TexCoord);
            }
            """

            vs = gl.glCreateShader(gl.GL_VERTEX_SHADER)
            gl.glShaderSource(vs, vertex_shader)
            gl.glCompileShader(vs)
            
            if not gl.glGetShaderiv(vs, gl.GL_COMPILE_STATUS):
                return None

            fs = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
            gl.glShaderSource(fs, fragment_shader)
            gl.glCompileShader(fs)
            
            if not gl.glGetShaderiv(fs, gl.GL_COMPILE_STATUS):
                return None

            program = gl.glCreateProgram()
            gl.glAttachShader(program, vs)
            gl.glAttachShader(program, fs)
            gl.glLinkProgram(program)
            
            if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
                return None

            gl.glDeleteShader(vs)
            gl.glDeleteShader(fs)
            
            return program
            
        except Exception:
            logger.exception("Failed to compile shader program")
            return None

    def create_fullscreen_window(self):
        try:
            self.main_window_context = glfw.get_current_context()
            
            monitor = glfw.get_primary_monitor()
            mode = glfw.get_video_mode(monitor)
            
            glfw.window_hint(glfw.DECORATED, True)           
            glfw.window_hint(glfw.FLOATING, False)           
            glfw.window_hint(glfw.MAXIMIZED, False)          
            glfw.window_hint(glfw.FOCUSED, True)             
            glfw.window_hint(glfw.AUTO_ICONIFY, False)       
            glfw.window_hint(glfw.RESIZABLE, True)           
            glfw.window_hint(glfw.VISIBLE, True)             
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            
            window_width = int(mode.size.width * 1.0)
            window_height = int(mode.size.height * 0.95)
            
            window = glfw.create_window(window_width, window_height, 
                                      "Preview Window (Press ESC to exit)", None, self.main_window_context)
            
            if window:
                x_pos = (mode.size.width - window_width) // 2
                y_pos = (mode.size.height - window_height) // 2
                glfw.set_window_pos(window, x_pos, y_pos)
                
                self.resize_edge = None
                self.resize_start_pos = None
                self.resize_start_size = None
                self.resize_start_window_pos = None
                
                def mouse_button_callback(window, button, action, mods):
                    if button == glfw.MOUSE_BUTTON_LEFT:
                        if action == glfw.PRESS:
                            x, y = glfw.get_cursor_pos(window)
                            win_w, win_h = glfw.get_window_size(window)
                            win_x, win_y = glfw.get_window_pos(window)
                            border = 8
                            
                            if x < border:
                                self.resize_edge = 'left'
                            elif x > win_w - border:
                                self.resize_edge = 'right'
                            elif y < border:
                                self.resize_edge = 'top'
                            elif y > win_h - border:
                                self.resize_edge = 'bottom'
                            else:
                                self.resize_edge = None
                                
                            if self.resize_edge:
                                self.resize_start_pos = (x, y)
                                self.resize_start_size = (win_w, win_h)
                                self.resize_start_window_pos = (win_x, win_y)
                        else:
                            self.resize_edge = None
                
                def cursor_pos_callback(window, x, y):
                    if self.resize_edge:
                        dx = x - self.resize_start_pos[0]
                        dy = y - self.resize_start_pos[1]
                        win_w, win_h = self.resize_start_size
                        win_x, win_y = self.resize_start_window_pos
                        
                        new_w = win_w
                        new_h = win_h
                        new_x = win_x
                        new_y = win_y
                        
                        if self.resize_edge == 'left':
                            new_w = win_w - dx
                            new_x = win_x + dx
                        elif self.resize_edge == 'right':
                            new_w = win_w + dx
                        elif self.resize_edge == 'top':
                            new_h = win_h - dy
                            new_y = win_y + dy
                        elif self.resize_edge == 'bottom':
                            new_h = win_h + dy
                        elif self.resize_edge == 'topleft':
                            new_w = win_w - dx
                            new_h = win_h - dy
                            new_x = win_x + dx
                            new_y = win_y + dy
                        elif self.resize_edge == 'topright':
                            new_w = win_w + dx
                            new_h = win_h - dy
                            new_y = win_y + dy
                        elif self.resize_edge == 'bottomleft':
                            new_w = win_w - dx
                            new_h = win_h + dy
                            new_x = win_x + dx
                        elif self.resize_edge == 'bottomright':
                            new_w = win_w + dx
                            new_h = win_h + dy
                        
                        min_size = 200
                        if new_w >= min_size and new_h >= min_size:
                            glfw.set_window_pos(window, int(new_x), int(new_y))
                            glfw.set_window_size(window, int(new_w), int(new_h))
                    else:
                        win_w, win_h = glfw.get_window_size(window)
                        border = 8
                        
                        if x < border and y < border:
                            glfw.set_cursor(window, glfw.create_standard_cursor(glfw.ARROW_CURSOR))
                        elif x > win_w - border and y < border:
                            glfw.set_cursor(window, glfw.create_standard_cursor(glfw.ARROW_CURSOR))
                        elif x < border and y > win_h - border:
                            glfw.set_cursor(window, glfw.create_standard_cursor(glfw.ARROW_CURSOR))
                        elif x > win_w - border and y > win_h - border:
                            glfw.set_cursor(window, glfw.create_standard_cursor(glfw.ARROW_CURSOR))
                        elif x < border or x > win_w - border:
                            glfw.set_cursor(window, glfw.create_standard_cursor(glfw.HRESIZE_CURSOR))
                        elif y < border or y > win_h - border:
                            glfw.set_cursor(window, glfw.create_standard_cursor(glfw.VRESIZE_CURSOR))
                        else:
                            glfw.set_cursor(window, glfw.create_standard_cursor(glfw.ARROW_CURSOR))
                
                def key_callback(window, key, scancode, action, mods):
                    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
                        self.is_fullscreen_display = False
                        self.window_created = False
                
                glfw.set_key_callback(window, key_callback)
                glfw.set_mouse_button_callback(window, mouse_button_callback)
                glfw.set_cursor_pos_callback(window, cursor_pos_callback)
                
                glfw.make_context_current(window)
                self.init_gl_resources()
                glfw.make_context_current(self.main_window_context)
                
                return window
                
            return None
            
        except Exception as e:
            logger.error("Error creating fullscreen window: %s", e)
            if 'window' in locals() and window:
                glfw.destroy_window(window)
            return None

    def init_gl_resources(self):
        self.fullscreen_shader = self.create_shader_program()
        
        vertices = np.array([
            # 位置          # ��理坐标
             0.0,  0.0,    0.0, 1.0,
             2.0,  0.0,    1.0, 1.0,
             2.0,  2.0,    1.0, 0.0,
             0.0,  2.0,    0.0, 0.0
        ], dtype=np.float32)
        
        self.fullscreen_vao = gl.glGenVertexArrays(1)
        self.fullscreen_vbo = gl.glGenBuffers(1)
        
        gl.glBindVertexArray(self.fullscreen_vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.fullscreen_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)
        
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 16, None)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, 16, ctypes.c_void_p(8))
        gl.glEnableVertexAttribArray(1)

    def render_fullscreen(self):
        if not self.fullscreen_window or not self._tex_obj:
            return

        try:
            glfw.make_context_current(self.fullscreen_window)
            
            window_w, window_h = glfw.get_window_size(self.fullscreen_window)
            gl.glViewport(0, 0, window_w, window_h)
            
            gl.glClearColor(0, 0, 0, 1)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            
            tex_aspect = self._tex_obj.width / self._tex_obj.height
            window_aspect = window_w / window_h
            
            if window_aspect > tex_aspect:
                scale_h = 1.0
                scale_w = (tex_aspect / window_aspect)
                offset_x = (1.0 - scale_w) / 2.0
                offset_y = 0.0
            else:
                scale_w = 1.0
                scale_h = (window_aspect / tex_aspect)
                offset_x = 0.0
                offset_y = (1.0 - scale_h) / 2.0
            
            gl.glUseProgram(self.fullscreen_shader)
            
            gl.glUniform2f(gl.glGetUniformLocation(self.fullscreen_shader, "uScale"), scale_w, scale_h)
            gl.glUniform2f(gl.glGetUniformLocation(self.fullscreen_shader, "uOffset"), offset_x * 2.0 - 1.0, offset_y * 2.0 - 1.0)
            
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self._tex_obj.gl_id)
            gl.glUniform1i(gl.glGetUniformLocation(self.fullscreen_shader, "ourTexture"), 0)
            
            gl.glBindVertexArray(self.fullscreen_vao)
            gl.glDrawArrays(gl.GL_TRIANGLE_FAN, 0, 4)
            
            glfw.swap_buffers(self.fullscreen_window)
            glfw.make_context_current(self.main_window_context)
            
        except Exception as e:
            logger.error("Fullscreen render failed: %s", e)

    def start_recording(self, file_path):
        self.is_recording = True
        self.recording_file_path = file_path

    def stop_recording(self):
        self.recording_thread = threading.Thread(target=self._record_frames, daemon=True)
        self.recording_thread.start()
        
        self.is_recording = False
        if self.recording_thread is not None:
            self.recording_thread.join()
            self.recording_thread = None

    def _record_frames(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # H.264 codec
        out = None
        while self.is_recording or not self.frame_queue.empty():
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                if out is None:
                    height, width, channels = frame.shape
                    os.makedirs(os.path.dirname(self.recording_file_path), exist_ok=True)
                    out = cv2.VideoWriter(self.recording_file_path, fourcc, 30.0, (width, height))
                out.write(frame)
        if out is not None:
            out.release()
            logger.info("Recording saved to %s", self.recording_file_path)

    def capture_screenshot(self, file_path):
        if 'image' in self.result:
            image_data = self.result.image  # Convert tensor to numpy array if needed

            # Convert from RGB to BGR
            image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGRA)

            # Save the image using OpenCV
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            if cv2.imwrite(file_path, image_data):
                logger.info("Screenshot saved to %s", file_path)
            else:
                logger.error("Failed to save screenshot to %s", file_path)
        else:
            logger.warning("No render result available to capture")

    def osc_message_handler(self, address, *args):
        logger.debug("OSC message received at %s with arguments: %s", address, args)

    def stop_osc_server(self):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
        if self.server_thread is not None:
            self.server_thread.join()
            self.server_thread = None

    def start_osc_server(self, max_attempts=20):
        self.stop_osc_server()
        requested_port = self.in_port
        for attempt in range(max_attempts):
            port = requested_port + attempt
            try:
                self.server = BlockingOSCUDPServer((self.in_ip, port), self.osc_dispatcher)
            except OSError as exc:
                logger.debug("OSC port %s unavailable: %s", port, exc)
                continue
            self.in_port = port
            self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.server_thread.start()
            if port != requested_port:
                logger.info("OSC port %s in use, server started on fallback %s:%s", requested_port, self.in_ip, port)
            else:
                logger.info("OSC server started on %s:%s", self.in_ip, port)
            return
        logger.error("Failed to start OSC server on %s: ports %s-%s unavailable",
                     self.in_ip, requested_port, requested_port + max_attempts - 1)



    def start_ndi(self):
        if ndi is None:
            return
        self.stop_ndi()
        send_settings = ndi.SendCreate()
        send_settings.ndi_name = self.ndi_name
        self.ndi_send = ndi.send_create(send_settings)
        self.video_frame = ndi.VideoFrameV2()

    def stop_ndi(self):
        if ndi is not None and self.ndi_send is not None:
            ndi.send_destroy(self.ndi_send)
            self.ndi_send = None

    def close(self):
        self.audio_widget.close()
        if self._async_renderer is not None:
            self._async_renderer.close()
            self._async_renderer = None

        self.stop_osc_server()

        self.stop_ndi()


    def add_recent_pickle(self, pkl, ignore_errors=False):
        self.pickle_widget.add_recent(pkl, ignore_errors=ignore_errors)

    def load_pickle(self, pkl, ignore_errors=False):
        self.pickle_widget.load(pkl, ignore_errors=ignore_errors)

    def print_error(self, error):
        error = str(error)
        if error != self._last_error_print:
            logger.error("%s", error)
            self._last_error_print = error

    def defer_rendering(self, num_frames=1):
        self._defer_rendering = max(self._defer_rendering, num_frames)

    def clear_result(self):
        self._async_renderer.clear_result()

    def _header_help_icon(self, help_key, hyperlink_text="Read More"):
        """Position and render the help icon to the right of a collapsing header."""
        self.help_icon.render(self.help_texts.get(help_key),
                              url=self.help_urls.get(help_key),
                              hyperlink_text=hyperlink_text, align_right=True)

    @imgui_utils.scoped_by_object_id
    def __call__(self):
        self.pane_w = self.app.font_size * 45
        self.args = dnnlib.EasyDict()

        navbar_h = self.app.navbar_height

        # Detect mouse dragging in the result area.
        dragging, dx, dy = imgui_utils.drag_hidden_window('##result_area', x=self.pane_w, y=navbar_h, width=self.app.content_width-self.pane_w, height=self.app.content_height - navbar_h)
        if dragging:
            self.latent_widget.drag(dx, dy)

        imgui.set_next_window_position(0, navbar_h)
        imgui.set_next_window_size(self.pane_w, self.app.content_height - navbar_h)
        imgui.begin('##control_pane', closable=False, flags=(imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE))
        if self.reset_scroll:
            imgui.set_scroll_y(0)
            self.reset_scroll = False

        # Display & Capture
        header_opened = imgui_utils.collapsing_header('Display & Capture', default=True)[0]
        self._header_help_icon('display_capture')
        self.display_capture_widget(header_opened)

        # Network & Latent
        header_opened = imgui_utils.collapsing_header('Network & Latent', default=True)[0]
        self._header_help_icon('network_latent')

        self.pickle_widget(header_opened)
        self.latent_widget(header_opened)

        # Diversity & Noise
        header_opened = imgui_utils.collapsing_header('Diversity & Noise', default=True)[0]
        self._header_help_icon('diversity_noise')
        self.trunc_noise_widget(header_opened)

        # Looping
        header_opened = imgui_utils.collapsing_header('Looping', default=True)[0]
        self._header_help_icon('looping')
        self.looping_widget(header_opened)

        # Performance & OSC
        header_opened = imgui_utils.collapsing_header('Performance & OSC', default=True)[0]
        self._header_help_icon('performance_osc')
        self.perf_widget(header_opened)

        # Adjust Input
        header_opened = imgui_utils.collapsing_header('Adjust Input', default=True)[0]
        self._header_help_icon('adjust_input')
        self.adjuster_widget(header_opened)

        # Layer Transformations
        header_opened = imgui_utils.collapsing_header('Layer Transformations', default=True)[0]
        self._header_help_icon('layer_transform')
        self.collapsed_widget(header_opened)

        # Model Mixing
        header_opened = imgui_utils.collapsing_header('Model Mixing', default=True)[0]
        self._header_help_icon('model_mixing')
        self.mixing_widget(header_opened)

        # Diffusion. streamdiffusion has no macOS build and is not installed
        # there, so the section is absent rather than permanently disabled.
        # The widget is still constructed, because presets save and load it.
        if not device_utils.is_macos():
            header_opened = imgui_utils.collapsing_header('Diffusion', default=True)[0]
            self._header_help_icon('diffusion')
            self.diffusion_widget(header_opened)

        # Presets
        header_opened = imgui_utils.collapsing_header('Presets', default=True)[0]
        self._header_help_icon('presets')
        self.preset_widget(header_opened)

        # Audio Module
        header_opened = imgui_utils.collapsing_header('Audio Module', default=True)[0]
        self._header_help_icon('audio')
        self.audio_widget(header_opened)

        # Render.
        if self.app.is_skipping_frames():
            pass
        elif self._defer_rendering > 0:
            self._defer_rendering -= 1
        elif self.args.pkl is not None:
            self._async_renderer.set_args(**self.args)
            result = self._async_renderer.get_result()
            if result is not None:
                self.result = result

        # Display.
        max_w = self.app.content_width - self.pane_w
        max_h = self.app.content_height - navbar_h
        pos = np.array([self.pane_w + max_w / 2, navbar_h + max_h / 2])
        if 'image' in self.result:
            if self._tex_img is not self.result.image:
                self._tex_img = self.result.image
                send_ndi = ndi is not None and self.ndi_send is not None
                # The BGRA copy is only needed for NDI and recording; skip the
                # per-frame conversion otherwise.
                img = cv2.cvtColor(self._tex_img, cv2.COLOR_RGB2BGRA) if (self.is_recording or send_ndi) else None
                
                # Recording 逻辑保持不变
                if self.is_recording:
                    try:
                        frame_to_record = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                        self.frame_queue.put(frame_to_record)
                        self.frames_captured += 1
                        if self.frames_captured % 30 == 0:  
                            logger.debug("Captured %d frames", self.frames_captured)
                    except Exception:
                        pass
                
                if send_ndi:
                    self.video_frame.data = img
                    self.video_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRX
                    ndi.send_send_video_v2(self.ndi_send, self.video_frame)
                if self._tex_obj is None or not self._tex_obj.is_compatible(image=self._tex_img):
                    self._tex_obj = gl_utils.Texture(image=self._tex_img, bilinear=False, mipmap=False)
                else:
                    self._tex_obj.update(self._tex_img)
            
            if self.fit_screen:
                zoom_w = max_w / self._tex_obj.width  
                zoom_h = max_h / self._tex_obj.height
                zoom = min(zoom_w, zoom_h)
            else:
                base_zoom = min(max_w / self._tex_obj.width, max_h / self._tex_obj.height)
                
                if self._tex_obj.width >= 1024 or self._tex_obj.height >= 1024:
                    zoom = base_zoom  
                else:
                    zoom = min(1.0, base_zoom)
                
                zoom = np.floor(zoom) if zoom >= 1 else zoom
            
            self._tex_obj.draw(pos=pos, zoom=zoom, align=0.5, rint=True)
        if 'error' in self.result:
            self.print_error(self.result.error)
            if 'message' not in self.result:
                self.result.message = str(self.result.error)
        if 'message' in self.result:
            tex = text_utils.get_texture(self.result.message, size=self.app.font_size, max_width=max_w, max_height=max_h, outline=2)
            tex.draw(pos=pos, align=0.5, rint=True, color=1)

        # End frame.
        imgui.end()


        
        if self.is_fullscreen_display:
            if not self.window_created:
                self.fullscreen_window = self.create_fullscreen_window()
                if self.fullscreen_window:
                    self.window_created = True
                else:
                    self.is_fullscreen_display = False
            
            if self.fullscreen_window and self.window_created:
                if glfw.window_should_close(self.fullscreen_window):
                    self.is_fullscreen_display = False
                    self.window_created = False
                else:
                    self.render_fullscreen()
                    glfw.poll_events()
        else:
            if self.fullscreen_window:
                glfw.destroy_window(self.fullscreen_window)
                self.fullscreen_window = None
                self.window_created = False

    def __del__(self):
        if hasattr(self, 'fullscreen_window') and self.fullscreen_window:
            glfw.make_context_current(self.fullscreen_window)
            if hasattr(self, 'fullscreen_shader'):
                gl.glDeleteProgram(self.fullscreen_shader)
            if hasattr(self, 'fullscreen_vao'):
                gl.glDeleteVertexArrays(1, [self.fullscreen_vao])
            if hasattr(self, 'fullscreen_vbo'):
                gl.glDeleteBuffers(1, [self.fullscreen_vbo])
            glfw.destroy_window(self.fullscreen_window)


#----------------------------------------------------------------------------



