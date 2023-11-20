# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import threading
import numpy as np
import imgui
import cv2
import pyaudio
print("imported pyaudio")

import dnnlib
from assets import GRAY, DARKGRAY, LIGHTGRAY
from utils.gui_utils import imgui_utils
from utils.gui_utils import gl_utils
from utils.gui_utils import text_utils
from widgets import pickle_widget
from widgets import latent_widget
from widgets import trunc_noise_widget
from widgets import performance_widget
from widgets import layer_widget
from widgets import adjuster_widget
from widgets import audio_widget
from widgets import looping_widget
from widgets import preset_widget
from widgets import mixing_widget
from widgets import collapsable_layer

from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.dispatcher import Dispatcher
from pythonosc.udp_client import SimpleUDPClient
import NDIlib as ndi

#----------------------------------------------------------------------------

class Visualizer:
    def __init__(self, app, renderer):
        self.app = app

        #COMMUNICATIONS
        self.has_microphone = False
        # check if microphone is available
        try:
            print("checking for microphone")
            self.pa = pyaudio.PyAudio()
            print(self.pa)
            print(self.pa.get_default_input_device_info())
            self.has_microphone = True
        except:
            print("except no microphone found")
            pass
        self.in_ip = "127.0.0.1"
        self.in_port = 1338
        self.out_ip = "127.0.0.1"
        self.out_port = 1337
        self.osc_dispatcher = Dispatcher()
        self.osc_client = SimpleUDPClient(self.out_ip, self.out_port)
        self.server = BlockingOSCUDPServer((self.in_ip, self.in_port), self.osc_dispatcher)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()

        # NDI parameters
        self.ndi_name = 'Autolume Live'
        send_settings = ndi.SendCreate()
        send_settings.ndi_name = self.ndi_name
        self.ndi_send = ndi.send_create(send_settings)
        self.video_frame = ndi.VideoFrameV2()

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
        if self.has_microphone:
            print("Setting up audio widget")
            self.audio_widget = audio_widget.AudioWidget(self)

        self.logo = cv2.imread("assets/Autolume-logo.png", cv2.IMREAD_UNCHANGED)
        self.logo_texture = gl_utils.Texture(image=self.logo, width=self.logo.shape[1], height=self.logo.shape[0],
                                             channels=self.logo.shape[2])

        self.metacreation = cv2.imread("assets/metalogo.png", cv2.IMREAD_UNCHANGED)
        self.metacreation_texture = gl_utils.Texture(image=self.metacreation, width=self.metacreation.shape[1],
                                                     height=self.metacreation.shape[0],
                                                     channels=self.metacreation.shape[2])

    def close(self):
        if self._async_renderer is not None:
            self._async_renderer.close()
            self._async_renderer = None

        if self.server is not None:
            self.server.shutdown()
            self.server = None

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


    @imgui_utils.scoped_by_object_id
    def __call__(self):
        self.pane_w = self.app.font_size * 45
        self.args = dnnlib.EasyDict()

        # Detect mouse dragging in the result area.
        dragging, dx, dy = imgui_utils.drag_hidden_window('##result_area', x=self.pane_w, y=0, width=self.app.content_width-self.pane_w, height=self.app.content_height)
        if dragging:
            self.latent_widget.drag(dx, dy)

        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(self.pane_w, self.app.content_height)
        imgui.begin('##control_pane', closable=False, flags=(imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE))
        # set red background
        imgui.get_window_draw_list().add_rect_filled(0, 0, self.pane_w, 36,
                                                     imgui.get_color_u32_rgba(*DARKGRAY))
        # draw gray line
        imgui.get_window_draw_list().add_line(0, 36, self.pane_w, 36, imgui.get_color_u32_rgba(*LIGHTGRAY), 1)

        # calculate logo shape ratio
        logo_ratio = self.logo.shape[1] / self.logo.shape[0]
        # logo with height of 30px centered in y axis
        imgui.set_cursor_pos_y(18 - (18 / 2))
        imgui.set_cursor_pos_x(self.app.spacing * 2)
        imgui.image(self.logo_texture.gl_id, 18 * logo_ratio, 18, tint_color=(1, 1, 1, 0.5))

        # calculate metacreation shape ratio
        metacreation_ratio = self.metacreation.shape[1] / self.metacreation.shape[0]
        # metacreation with height of 30px centered in y axis
        imgui.same_line(self.pane_w - ((18 * metacreation_ratio) + (self.app.spacing * 6)))
        imgui.set_cursor_pos_y(18 - (18 / 2))
        imgui.image(self.metacreation_texture.gl_id, 18 * metacreation_ratio, 18, tint_color=(1, 1, 1, 0.5))
        imgui.set_cursor_pos_y(36 + self.app.spacing)
        # Widgets.
        expanded, _visible = imgui_utils.collapsing_header('Network & latent', default=True)
        self.pickle_widget(expanded)
        self.latent_widget(expanded)
        expanded, _visible = imgui_utils.collapsing_header('Diversity & Noise', default=True)
        self.trunc_noise_widget(expanded)
        expanded, _visible = imgui_utils.collapsing_header('Looping', default=True)
        self.looping_widget(expanded)
        expanded, _visible = imgui_utils.collapsing_header('Performance & OSC', default=True)
        self.perf_widget(expanded)
        expanded, _visible = imgui_utils.collapsing_header('Adjust Input', default=True)
        self.adjuster_widget(expanded)
        expanded, _visible = imgui_utils.collapsing_header('Layer Transformations', default=True)
        self.collapsed_widget(expanded)
        expanded, _visible = imgui_utils.collapsing_header('Model Mixing', default=True)
        self.mixing_widget(expanded)
        expanded, _visible = imgui_utils.collapsing_header('Presets', default=True)
        self.preset_widget(expanded)

        expanded, _visible = imgui_utils.collapsing_header('Audio Module', default=True)
        if self.has_microphone:
            self.audio_widget(expanded)
        else:
            if expanded:
                imgui.text('No microphone detected')


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
        elif self.args.pkl is not None:
            self._async_renderer.set_args(**self.args)
            result = self._async_renderer.get_result()
            if result is not None:
                self.result = result

        # Display.
        max_w = self.app.content_width - self.pane_w
        max_h = self.app.content_height
        pos = np.array([self.pane_w + max_w / 2, max_h / 2])
        if 'image' in self.result:
            if self._tex_img is not self.result.image:
                self._tex_img = self.result.image
                img = cv2.cvtColor(self._tex_img, cv2.COLOR_RGB2BGRA)
                self.video_frame.data = img
                self.video_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRX
                ndi.send_send_video_v2(self.ndi_send, self.video_frame)
                if self._tex_obj is None or not self._tex_obj.is_compatible(image=self._tex_img):
                    self._tex_obj = gl_utils.Texture(image=self._tex_img, bilinear=False, mipmap=False)
                else:
                    self._tex_obj.update(self._tex_img)
            zoom = min(max_w / self._tex_obj.width, max_h / self._tex_obj.height)
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

#----------------------------------------------------------------------------



