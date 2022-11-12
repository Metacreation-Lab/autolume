# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import threading

import click
import os
import gc

import multiprocessing

import glfw
import numpy as np
import imgui
# import pyaudio
import torch.cuda
import ray

import dnnlib
from utils.gui_utils import imgui_utils
from utils.gui_utils import gl_utils
from utils.gui_utils import text_utils
from widgets import pickle_widget
from widgets import latent_widget
from widgets import trunc_noise_widget
from widgets import performance_widget
from widgets import layer_widget
from widgets import adjuster_widget
# from widgets import audio_widget
from widgets import looping_widget
from widgets import preset_widget

from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.dispatcher import Dispatcher
from pythonosc.udp_client import SimpleUDPClient
import NDIlib as ndi

#----------------------------------------------------------------------------

class Visualizer:
    def __init__(self, app, renderer):
        self.app = app

        #COMMUNICATIONS
        # self.pa = pyaudio.PyAudio()
        self.in_ip = "127.0.0.1"
        self.in_port = 1337
        self.out_ip = "127.0.0.1"
        self.out_port = 1337
        self.osc_dispatcher = Dispatcher()
        self.osc_client = SimpleUDPClient(self.out_ip, self.out_port)
        self.server = BlockingOSCUDPServer((self.in_ip, self.in_port), self.osc_dispatcher)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()

        # NDI parameters
        send_settings = ndi.SendCreate()
        send_settings.ndi_name = 'Autolume Live'
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
        # self.audio_widget = audio_widget.AudioWidget(self)


    def close(self):
        if self._async_renderer is not None:
            self._async_renderer.close()
            self._async_renderer = None

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

        # Begin control pane.
        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(self.pane_w, self.app.content_height)
        imgui.begin('##control_pane', closable=False, flags=(imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE))

        # Widgets.
        expanded, _visible = imgui_utils.collapsing_header('Network & latent', default=True)
        self.pickle_widget(expanded)
        self.latent_widget(expanded)
        self.trunc_noise_widget(expanded)
        self.looping_widget(expanded)
        expanded, _visible = imgui_utils.collapsing_header('Performance & capture', default=True)
        self.perf_widget(expanded)
        expanded, _visible = imgui_utils.collapsing_header('Adjust Input', default=True)
        self.adjuster_widget(expanded)
        expanded, _visible = imgui_utils.collapsing_header('Layers & channels', default=True)
        self.layer_widget(expanded)
        expanded, _visible = imgui_utils.collapsing_header('Preset Module', default=True)
        self.preset_widget(expanded)
        # expanded, _visible = imgui_utils.collapsing_header('Audio Module', default=True)
        # self.audio_widget(expanded)

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



