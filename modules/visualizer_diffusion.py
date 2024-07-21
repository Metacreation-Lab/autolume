# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import threading
import queue
import numpy as np
import imgui
import cv2
import time
import dnnlib
from assets import GRAY, DARKGRAY, LIGHTGRAY
from utils.gui_utils import imgui_utils
from utils.gui_utils import gl_utils
from utils.gui_utils import text_utils
from widgets import diffusion_widget
# from widgets import pickle_widget

from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.dispatcher import Dispatcher
from pythonosc.udp_client import SimpleUDPClient
import NDIlib as ndi

from PIL import Image
import datetime


# ----------------------------------------------------------------------------

class VisualizerDiffusion:
    def __init__(self, app, renderer):
        self.app = app

        # Internals.
        self._async_renderer = renderer

        # NDI parameters
        self.ndi_source = None
        self.ndi_recv = None

        self.ndi_name = 'Diffusion Live'
        send_settings = ndi.SendCreate()
        send_settings.ndi_name = self.ndi_name
        self.ndi_send = ndi.send_create(send_settings)
        self.video_frame = ndi.VideoFrameV2()

        # Internals.

        self.pane_w = 0
        self._last_error_print = None
        self._defer_rendering = 0
        self._tex_img = None
        self._tex_obj = None

        # Widget interface.
        self.args = dnnlib.EasyDict()
        self.result = dnnlib.EasyDict()

        # Widgets.
        self.diffusion_widget = diffusion_widget.DiffusionWidget(self)

        self.logo = cv2.imread("assets/Autolume-logo.png", cv2.IMREAD_UNCHANGED)
        self.logo_texture = gl_utils.Texture(image=self.logo, width=self.logo.shape[1], height=self.logo.shape[0],
                                             channels=self.logo.shape[2])

        self.metacreation = cv2.imread("assets/metalogo.png", cv2.IMREAD_UNCHANGED)
        self.metacreation_texture = gl_utils.Texture(image=self.metacreation, width=self.metacreation.shape[1],
                                                     height=self.metacreation.shape[0],
                                                     channels=self.metacreation.shape[2])

        self.is_recording = False
        self.frame_queue = queue.Queue()
        self.recording_thread = None
        self.recording_file_path = None

        self.is_processing = False

        self.last_frame_time = None
        self.fps = 0

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

    def calculate_fps(self):
        current_time = time.perf_counter()
        if self.last_frame_time is not None:
            time_delta = current_time - self.last_frame_time
            self.fps = 1 / time_delta if time_delta > 0 else 0
        self.last_frame_time = current_time
        # Optionally print or display the FPS
        print(f"Current FPS: {self.fps}")

    def _record_frames(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # H.264 codec
        out = None
        while self.is_recording or not self.frame_queue.empty():
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                if out is None:
                    height, width, channels = frame.shape
                    out = cv2.VideoWriter(self.recording_file_path, fourcc, 30.0, (width, height))
                out.write(frame)
        if out is not None:
            out.release()

    def capture_screenshot(self, file_path):
        if 'image' in self.result:
            image_data = self.result.image  # Convert tensor to numpy array if needed

            image_data = (image_data * 255).clip(0, 255).astype(np.uint8)

            # Convert from RGB to BGR
            image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGRA)

            # Save the image using OpenCV
            cv2.imwrite(file_path, image_data)
        else:
            print("No render result available to capture.")

    def close(self):
        if self._async_renderer is not None:
            self._async_renderer.close()
            self._async_renderer = None

        if self.ndi_send is not None:
            ndi.send_destroy(self.ndi_send)
            self.ndi_send = None

    def add_recent_pickle(self, pkl, ignore_errors=False):
        self.pickle_widget.add_recent(pkl, ignore_errors=ignore_errors)

    def load_pickle(self, pkl, ignore_errors=False):
        self.pickle_widget.load(pkl, ignore_errors=ignore_errors)

    def print_error(self, error):
        error = str(error)
        if error != self._last_error_print:
            print('\n' + error + '\n')
            self._last_error_print = error

    def defer_rendering(self, num_frames=1):
        self._defer_rendering = max(self._defer_rendering, num_frames)

    def clear_result(self):
        self._async_renderer.clear_result()

    def set_ndi_source(self, ndi_source):
        self.ndi_source = ndi_source
        if self.ndi_source is not None:
            recv_create_desc = ndi.RecvCreateV3()
            recv_create_desc.color_format = ndi.RECV_COLOR_FORMAT_BGRX_BGRA
            self.ndi_recv = ndi.recv_create_v3(recv_create_desc)
            ndi.recv_connect(self.ndi_recv, self.ndi_source)

    @imgui_utils.scoped_by_object_id
    def __call__(self):
        self.pane_w = self.app.font_size * 45
        self.args = dnnlib.EasyDict()

        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(self.pane_w, self.app.content_height)
        imgui.begin('##control_pane', closable=False,
                    flags=(imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE))
        # set red background
        imgui.get_window_draw_list().add_rect_filled(0, 0, self.pane_w, 42,
                                                     imgui.get_color_u32_rgba(*DARKGRAY))
        # draw gray line
        imgui.get_window_draw_list().add_line(0, 42, self.pane_w, 42, imgui.get_color_u32_rgba(*LIGHTGRAY), 1)

        # calculate logo shape ratio
        logo_ratio = self.logo.shape[1] / self.logo.shape[0]
        # logo with height of 30px centered in y axis
        imgui.set_cursor_pos_y(18 - (18 / 2))
        imgui.set_cursor_pos_x(self.app.spacing * 2)

        imgui.image(self.logo_texture.gl_id, 18 * logo_ratio, 18, tint_color=(1, 1, 1, 0.5))

        # Position the button in the middle
        imgui.same_line(self.app.spacing * 18)

        if imgui.button('Screen Capture'):
            now = datetime.datetime.now()
            current_time_str = now.strftime("%Y-%m-%d %H-%M-%S")
            self.capture_screenshot(f'screenshots/{current_time_str}.png')

        # Capture frame if recording
        if self.is_recording and 'image' in self.result:
            frame = cv2.cvtColor(self.result.image, cv2.COLOR_RGB2BGR)
            self.frame_queue.put(frame)

        imgui.same_line()
        if imgui.button('Start Recording' if not self.is_recording else 'Stop Recording'):
            if not self.is_recording:
                now = datetime.datetime.now()
                current_time_str = now.strftime("%Y-%m-%d %H-%M-%S")
                self.start_recording(f'recordings/{current_time_str}.mp4')
            else:
                self.stop_recording()

        # calculate metacreation shape ratio
        metacreation_ratio = self.metacreation.shape[1] / self.metacreation.shape[0]
        # metacreation with height of 30px centered in y axis
        imgui.same_line(self.pane_w - ((18 * metacreation_ratio) + (self.app.spacing * 6)))
        imgui.set_cursor_pos_y(18 - (18 / 2))
        imgui.image(self.metacreation_texture.gl_id, 18 * metacreation_ratio, 18, tint_color=(1, 1, 1, 0.5))
        imgui.set_cursor_pos_y(36 + self.app.spacing)

        # Widgets.
        expanded, _visible = imgui_utils.collapsing_header('Diffusion Settings', default=True)
        self.diffusion_widget(expanded)

        # go back to menu
        imgui.separator()
        if imgui.button('Back to menu'):
            self.defer_rendering(10)
            self.app.set_visible_menu()

        # Render.
        if self.app.is_skipping_frames():
            pass
        elif self._defer_rendering > 0:
            self._defer_rendering -= 1
        elif self.ndi_recv is not None and self.is_processing:
            self._async_renderer.set_args(**self.args)
            t, v, _, _ = ndi.recv_capture_v2(self.ndi_recv, 5000)
            if t == ndi.FRAME_TYPE_VIDEO:
                frame = np.copy(v.data)
                self._async_renderer.set_frame(frame)
                result = self._async_renderer.get_result()
                ndi.recv_free_video_v2(self.ndi_recv, v)
                if result is not None:
                    self.result = result

        # Display.
        max_w = self.app.content_width - self.pane_w
        max_h = self.app.content_height
        pos = np.array([self.pane_w + max_w / 2, max_h / 2])
        if 'image' in self.result:
            if self._tex_img is not self.result.image:
                self._tex_img = self.result.image
                img = (self._tex_img * 255).clip(0, 255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGRA)
                self.video_frame.data = img
                self.video_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRX
                try:
                    ndi.send_send_video_v2(self.ndi_send, self.video_frame)
                except Exception as e:
                    print(e)
                if self._tex_obj is None or not self._tex_obj.is_compatible(image=img):
                    self._tex_obj = gl_utils.Texture(image=img, bilinear=False, mipmap=False)
                else:
                    self._tex_obj.update(img)
                self.calculate_fps()
            zoom = min(max_w / self._tex_obj.width, max_h / self._tex_obj.height)
            zoom = np.floor(zoom) if zoom >= 1 else zoom
            self._tex_obj.draw(pos=pos, zoom=zoom, align=0.5, rint=True)

        if 'error' in self.result:
            self.print_error(self.result.error)
            if 'message' not in self.result:
                self.result.message = str(self.result.error)
        if 'message' in self.result:
            tex = text_utils.get_texture(self.result.message, size=self.app.font_size, max_width=max_w,
                                         max_height=max_h, outline=2)
            tex.draw(pos=pos, align=0.5, rint=True, color=1)

        # End frame.
        imgui.end()

# ----------------------------------------------------------------------------
