# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import array
import threading

import numpy as np
import imgui
import torch.cuda

from utils.gui_utils import imgui_utils
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient
import NDIlib as ndi

# ----------------------------------------------------------------------------
class PerformanceWidget:
    def __init__(self, viz):
        self.viz = viz
        self.gui_times = [float('nan')] * 60
        self.render_times = [float('nan')] * 30
        self.fps_limit = 60
        self.use_vsync = False
        self.force_fp32 = False
        self.use_superres = False
        self.scale_factor = 0
        self.device = "cuda" if torch.cuda.is_available() else 'cpu'
        self.custom_kernel_available = False


    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if "has_custom" in viz.result:
            self.custom_kernel_available = viz.result.has_custom
            del viz.result.has_custom

        self.gui_times = self.gui_times[1:] + [viz.app.frame_delta]
        if 'render_time' in viz.result:
            self.render_times = self.render_times[1:] + [viz.result.render_time]
            del viz.result.render_time

        if show:
            imgui.text('GUI')
            imgui.same_line(viz.app.label_w)
            with imgui_utils.item_width(viz.app.font_size * 8):
                imgui.plot_lines('##gui_times', array.array('f', self.gui_times), scale_min=0)
            imgui.same_line(viz.app.label_w + viz.app.font_size * 9)
            t = [x for x in self.gui_times if x > 0]
            t = np.mean(t) if len(t) > 0 else 0
            imgui.text(f'{t * 1e3:.1f} ms' if t > 0 else 'N/A')
            imgui.same_line(viz.app.label_w + viz.app.font_size * 14)
            imgui.text(f'{1 / t:.1f} FPS' if t > 0 else 'N/A')
            imgui.same_line(viz.app.label_w + viz.app.font_size * 18 + viz.app.spacing * 3)
            with imgui_utils.item_width(viz.app.font_size * 6):
                _changed, self.fps_limit = imgui.input_int('FPS limit', self.fps_limit,
                                                           flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
                self.fps_limit = min(max(self.fps_limit, 5), 1000)
            pos_x = imgui.get_item_rect_max()[0] + (viz.app.spacing*3)
            imgui.same_line(pos_x)
            _clicked, self.use_vsync = imgui.checkbox('Vertical sync', self.use_vsync)

        if show:
            imgui.text('Render')
            imgui.same_line(viz.app.label_w)
            with imgui_utils.item_width(viz.app.font_size * 8):
                imgui.plot_lines('##render_times', array.array('f', self.render_times), scale_min=0)
            imgui.same_line(viz.app.label_w + viz.app.font_size * 9)
            t = [x for x in self.render_times if x > 0]
            t = np.mean(t) if len(t) > 0 else 0
            imgui.text(f'{t * 1e3:.1f} ms' if t > 0 else 'N/A')
            imgui.same_line(viz.app.label_w + viz.app.font_size * 14)
            imgui.text(f'{1 / t:.1f} FPS' if t > 0 else 'N/A')
            imgui.same_line(pos_x)
            _clicked, self.force_fp32 = imgui.checkbox('Force FP32', self.force_fp32)

            imgui.text('Server')
            imgui.same_line()
            with imgui_utils.item_width(viz.app.font_size * 6):
                changed_ip, self.viz.in_ip = imgui.input_text(f"OSC IP Addresse", self.viz.in_ip, 256,
                                                              imgui.INPUT_TEXT_CHARS_NO_BLANK | imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
                imgui.same_line()
                changed_port, self.viz.in_port = imgui.input_int(f"OSC port", self.viz.in_port,
                                                                 flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)

                imgui.same_line()
                # NDI parameters
                changed_ndi, self.viz.ndi_name = imgui.input_text(f"NDI Name", self.viz.ndi_name,
                                                                  256, imgui.INPUT_TEXT_CHARS_NO_BLANK | imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)

            if changed_port or changed_ip:
                self.viz.server.shutdown()
                self.viz.server.server_close()
                self.viz.server_thread.join()
                self.viz.server = BlockingOSCUDPServer((self.viz.in_ip, self.viz.in_port), self.viz.osc_dispatcher)
                print("new server", self.viz.in_ip, self.viz.in_port)
                self.viz.server_thread = threading.Thread(target=self.viz.server.serve_forever, daemon=True)
                self.viz.server_thread.start()
                self.viz.osc_client = SimpleUDPClient(self.viz.in_ip, self.viz.in_port)

            if changed_ndi:
                        send_settings = ndi.SendCreate()
                        send_settings.ndi_name = self.viz.ndi_name
                        ndi.send_destroy(self.viz.ndi_send)
                        self.viz.ndi_send = ndi.send_create(send_settings)

            if imgui.checkbox("CPU", self.device=="cpu")[0]:
                self.device = "cpu"

            imgui.same_line()
            with imgui_utils.grayed_out(not torch.cuda.is_available()):
                if imgui.checkbox("GPU", self.device == "cuda")[0]:
                    if torch.cuda.is_available():
                        self.device = "cuda"

            imgui.same_line()

            with imgui_utils.grayed_out(not self.custom_kernel_available):
                if imgui.checkbox("Custom Kernel", self.device == "custom")[0]:
                    if self.custom_kernel_available:
                        self.device = "custom"


            imgui.same_line(spacing=viz.app.spacing*3)
            _, self.use_superres = imgui.checkbox('Super Resolution', self.use_superres)

        viz.app.set_fps_limit(self.fps_limit)
        viz.app.set_vsync(self.use_vsync)
        viz.args.force_fp32 = self.force_fp32
        viz.args.use_superres = self.use_superres
        viz.args.device = self.device

# ----------------------------------------------------------------------------
