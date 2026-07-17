# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import array

import numpy as np
import imgui

from utils import device_utils
from utils import network_utils
from utils.gui_utils import imgui_utils

try:
    import NDIlib as ndi
except ImportError:
    ndi = None  # NDIlib is optional; NDI streaming is disabled when it isn't installed.

# ----------------------------------------------------------------------------
class PerformanceWidget:
    def __init__(self, viz):
        self.viz = viz
        self.gui_times = [float('nan')] * 60
        self.render_times = [float('nan')] * 30
        self.fps_limit = 60
        self.use_vsync = False
        self.force_fp32 = False
        self.scale_factor = 0
        self.device = device_utils.get_device().type
        self.bind_addresses = network_utils.list_bind_addresses()


    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        self.gui_times = self.gui_times[1:] + [viz.app.frame_delta]
        if 'render_time' in viz.result:
            self.render_times = self.render_times[1:] + [viz.result.render_time]
            del viz.result.render_time

        if show:
            fs = viz.app.font_size
            # Wider than viz.app.label_w so "OSC input"/"NDI output" fit.
            label_w = max(viz.app.label_w, round(fs * 5.5))
            stats_x = label_w + fs * 6.5
            mid_col_x = label_w + fs * 15.5
            right_col_x = mid_col_x + fs * 10.5 + viz.app.spacing

            # Row 1: GUI stats | FPS limit | Vertical sync.
            imgui.text('GUI')
            imgui.same_line(label_w)
            with imgui_utils.item_width(fs * 6):
                imgui.plot_lines('##gui_times', array.array('f', self.gui_times), scale_min=0)
            imgui.same_line(stats_x)
            t = [x for x in self.gui_times if x > 0]
            t = np.mean(t) if len(t) > 0 else 0
            imgui.text(f'{t * 1e3:.1f} ms · {1 / t:.1f} FPS' if t > 0 else 'N/A')
            imgui.same_line(mid_col_x)
            with imgui_utils.item_width(fs * 6):
                _changed, self.fps_limit = imgui.input_int('FPS limit', self.fps_limit,
                                                           flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
                self.fps_limit = min(max(self.fps_limit, 5), 1000)
            imgui.same_line(right_col_x)
            _clicked, self.use_vsync = imgui.checkbox('Vertical sync', self.use_vsync)

            # Row 2: Render stats | Device | Force FP32.
            imgui.text('Render')
            imgui.same_line(label_w)
            with imgui_utils.item_width(fs * 6):
                imgui.plot_lines('##render_times', array.array('f', self.render_times), scale_min=0)
            imgui.same_line(stats_x)
            t = [x for x in self.render_times if x > 0]
            t = np.mean(t) if len(t) > 0 else 0
            imgui.text(f'{t * 1e3:.1f} ms · {1 / t:.1f} FPS' if t > 0 else 'N/A')
            imgui.same_line(mid_col_x)
            accel_type = device_utils.get_device().type
            with imgui_utils.grayed_out(accel_type == 'cpu'):
                if imgui.radio_button('GPU', self.device in ('cuda', 'mps')) and accel_type != 'cpu':
                    self.device = accel_type
            imgui.same_line()
            if imgui.radio_button('CPU', self.device == 'cpu'):
                self.device = 'cpu'
            imgui.same_line(right_col_x)
            _clicked, self.force_fp32 = imgui.checkbox('Force FP32', self.force_fp32)

            # Row 3: OSC input server.
            imgui.text('OSC input')
            imgui.same_line(label_w)
            imgui.text('Source')
            imgui.same_line()
            addresses = self.bind_addresses
            if self.viz.in_ip not in [ip for _, ip in addresses]:
                addresses = addresses + [(self.viz.in_ip, self.viz.in_ip)]
            current = [ip for _, ip in addresses].index(self.viz.in_ip)
            with imgui_utils.item_width(fs * 9):
                changed_source, selected = imgui.combo('##osc_source', current, [label for label, _ in addresses])
            if imgui.is_item_clicked():
                self.bind_addresses = network_utils.list_bind_addresses()
            if changed_source:
                self.viz.in_ip = addresses[selected][1]
            imgui.same_line()
            imgui.text('Port')
            imgui.same_line()
            with imgui_utils.item_width(fs * 6):
                changed_port, self.viz.in_port = imgui.input_int('##osc_port', self.viz.in_port,
                                                                 flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
            imgui.same_line()
            if imgui.button('Restart') or changed_port or changed_source:
                self.viz.start_osc_server()

            # Row 4: NDI video output.
            imgui.text('NDI output')
            imgui.same_line(label_w)
            with imgui_utils.grayed_out(ndi is None):
                clicked_ndi, ndi_enabled = imgui.checkbox('Enabled##ndi', self.viz.ndi_send is not None)
                imgui.same_line()
                imgui.text('Name')
                imgui.same_line()
                with imgui_utils.item_width(fs * 8):
                    changed_ndi, self.viz.ndi_name = imgui.input_text('##ndi_name', self.viz.ndi_name, 256,
                                                                      imgui.INPUT_TEXT_CHARS_NO_BLANK | imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)

            if ndi is not None:
                if clicked_ndi:
                    if ndi_enabled:
                        self.viz.start_ndi()
                    else:
                        self.viz.stop_ndi()
                elif changed_ndi and self.viz.ndi_send is not None:
                    self.viz.start_ndi()

        viz.app.set_fps_limit(self.fps_limit)
        viz.app.set_vsync(self.use_vsync)
        viz.args.force_fp32 = self.force_fp32
        viz.args.device = self.device

# ----------------------------------------------------------------------------
