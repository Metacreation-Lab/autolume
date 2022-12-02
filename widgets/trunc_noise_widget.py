# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import imgui

import dnnlib
from utils.gui_utils import imgui_utils
from widgets import osc_menu

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

#----------------------------------------------------------------------------

class TruncationNoiseWidget:
    def __init__(self, viz):
        self.viz            = viz
        self.params = dnnlib.EasyDict(trunc_psi=0.8, trunc_cutoff=0, noise_enable=True, noise_seed=0, noise_anim=False)
        self.prev_num_ws    = 0

        funcs = dict(zip(["Truncation", "Cut Off", "Noise ON/Off", "Noise Seed", "Noise Anim"], [self.osc_handler(param) for param in
                                                     self.params]))

        self.osc_menu = osc_menu.OscMenu(self.viz, funcs,
                                         label="##TruncationOSC")

    def osc_handler(self, param):
        def func(address, *args):
            try:
                assert (type(args[-1]) is type(self.params[
                                                   param])), f"OSC Message and Parameter type must align [OSC] {type(args[-1])} != [Param] {type(self.params[param])}"

                self.params[param] = args[-1]
            except Exception as e:
                print(e)
        return func

    def get_params(self):
        return self.params

    def set_params(self, params):
        self.params = params

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_params(), f)

    def load(self, path):
        with open(path, "rb") as f:
            self.set_params(pickle.load(f))

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        num_ws = viz.result.get('num_ws', 0)
        has_noise = viz.result.get('has_noise', False)
        if num_ws > 0 and num_ws != self.prev_num_ws:
            if self.params.trunc_cutoff > num_ws or self.params.trunc_cutoff == self.prev_num_ws:
                self.params.trunc_cutoff = num_ws
            self.prev_num_ws = num_ws

        if show:
            imgui.text('Truncate')
            imgui.same_line(viz.app.label_w)
            with imgui_utils.item_width(viz.app.font_size * 10), imgui_utils.grayed_out(num_ws == 0):
                _changed, self.params.trunc_psi = imgui.slider_float('##psi', self.params.trunc_psi, -1, 2, format='Psi %.2f')
            imgui.same_line()
            if num_ws == 0:
                imgui_utils.button('Cutoff 0', width=(viz.app.font_size * 8 + viz.app.spacing), enabled=False)
            else:
                with imgui_utils.item_width(viz.app.font_size * 8 + viz.app.spacing):
                    changed, new_cutoff = imgui.slider_int('##cutoff', self.params.trunc_cutoff, 0, num_ws, format='Cutoff %d')
                    if changed:
                        self.params.trunc_cutoff = min(max(new_cutoff, 0), num_ws)

            with imgui_utils.grayed_out(not has_noise):
                imgui.same_line()
                _clicked, self.params.noise_enable = imgui.checkbox('Noise##enable', self.params.noise_enable)
                imgui.same_line(round(viz.app.font_size * 27.7))
                with imgui_utils.grayed_out(not self.params.noise_enable):
                    with imgui_utils.item_width(-1 - viz.app.button_w - viz.app.spacing - viz.app.font_size * 4):
                        _changed, self.params.noise_seed = imgui.input_int('##seed', self.params.noise_seed)
                    imgui.same_line(spacing=0)
                    _clicked, self.params.noise_anim = imgui.checkbox('Anim##noise', self.params.noise_anim)

            is_def_trunc = (self.params.trunc_psi == 1 and self.params.trunc_cutoff == num_ws)
            is_def_noise = (self.params.noise_enable and self.params.noise_seed == 0 and not self.params.noise_anim)
            with imgui_utils.grayed_out(is_def_trunc and not has_noise):
                imgui.same_line(imgui.get_content_region_max()[0] - 1 - viz.app.button_w)
                if imgui_utils.button('Reset', width=-1, enabled=(not is_def_trunc or not is_def_noise)):
                    self.prev_num_ws = num_ws
                    self.params.trunc_psi = 1
                    self.params.trunc_cutoff = num_ws
                    self.params.noise_enable = True
                    self.params.noise_seed = 0
                    self.params.noise_anim = False

        self.osc_menu()
        if self.params.noise_anim:
            self.params.noise_seed += 1
        viz.args.update(trunc_psi=self.params.trunc_psi, trunc_cutoff=self.params.trunc_cutoff, random_seed=self.params.noise_seed)
        viz.args.noise_mode = ('none' if not self.params.noise_enable else 'const' if self.params.noise_seed == 0 else 'random')

#----------------------------------------------------------------------------
