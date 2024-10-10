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
        self.params = dnnlib.EasyDict(trunc_psi=0.8, global_noise=1, noise_enable=True, noise_seed=0, noise_anim=False, reset=False)
        self.prev_num_ws    = 0

        funcs = dict(zip(["Diversity", "Global Noise", "Noise On/Off", "Noise Seed", "Animate Noise", "Reset"], [self.osc_handler(param) for param in
                                                     self.params]))

        self.osc_menu = osc_menu.OscMenu(self.viz, funcs,
                                         label="##TruncationOSC")

    def osc_handler(self, param):
        def func(address, *args):
            try:
                nec_type = type(self.params[param])
                print(self.params)
                self.params[param] = nec_type(args[-1])
            except Exception as e:
                print(e)
        return func

    def get_params(self):
        return self.params, self.osc_menu.get_params()

    def set_params(self, params):
        self.params, self.osc_params = params
        self.osc_menu.set_params(self.osc_params)

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

        if show:
            with imgui_utils.item_width(viz.app.button_w * 1.5), imgui_utils.grayed_out(num_ws == 0):
                _changed, self.params.trunc_psi = imgui.slider_float('##psi', self.params.trunc_psi, -1, 2, format='Diversity %.2f')
            imgui.same_line(spacing=self.viz.app.spacing*2)
            imgui.text("|")
            imgui.same_line(spacing=self.viz.app.spacing*2)
            with imgui_utils.grayed_out(not has_noise):
                _clicked, self.params.noise_enable = imgui.checkbox('Noise##enable', self.params.noise_enable)
                imgui.same_line()


                with imgui_utils.grayed_out(not self.params.noise_enable):
                    with imgui_utils.item_width(viz.app.button_w * 1.5):
                        _changed, g_noise= imgui_utils.drag_float_slider('##global_noise', self.params.global_noise, 0, 2, format='Global Noise %.2f')
                        if _changed and has_noise:
                            self.params.global_noise = g_noise
                    imgui.same_line()
                    with imgui_utils.item_width(-1 - viz.app.button_w - viz.app.spacing - viz.app.font_size * 4):
                        _changed, self.params.noise_seed = imgui.input_int('##seed', self.params.noise_seed)
                    imgui.same_line(spacing=0)
                    _clicked, self.params.noise_anim = imgui.checkbox('Anim##noise', self.params.noise_anim)

            is_def_trunc = self.params.trunc_psi == 1
            is_def_noise = (self.params.noise_enable and self.params.noise_seed == 0 and not self.params.noise_anim)
            with imgui_utils.grayed_out(is_def_trunc and not has_noise):
                imgui.same_line(imgui.get_content_region_max()[0] - 1 - viz.app.button_w)
                if imgui_utils.button('Reset', width=-1, enabled=(not is_def_trunc or not is_def_noise)) or self.params.reset:
                    self.params.trunc_psi = 0.8
                    self.params.noise_enable = True
                    self.params.noise_seed = 0
                    self.params.noise_anim = False
                    self.params.global_noise = 1
                    self.params.reset = False

            self.osc_menu()
        if self.params.noise_anim:
            self.params.noise_seed += 1
        viz.args.update(trunc_psi=self.params.trunc_psi, trunc_cutoff= num_ws,random_seed=self.params.noise_seed, global_noise=self.params.global_noise)
        viz.args.noise_mode = ('none' if not self.params.noise_enable else 'const' if self.params.noise_seed == 0 else 'random')

#----------------------------------------------------------------------------
