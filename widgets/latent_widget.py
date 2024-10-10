# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import copy
import os

import numpy as np
import torch
import torch.nn.functional as F

from widgets.browse_widget import BrowseWidget

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

import dnnlib
from utils.gui_utils import imgui_utils
from widgets import osc_menu, save_widget
import imgui


#----------------------------------------------------------------------------


modes = ["Stop", "Anim"]
class LatentWidget:
    def __init__(self, viz):
        self.viz        = viz
        self.latent = dnnlib.EasyDict(vec=torch.randn(1, 512), next=torch.randn(1, 512), x=0, y=0, frac_x=0., frac_y=0.,
                                      update_mode=0, speed=0.25, mode=True, project=True)
        self.step_y = 100
        funcs = dict(zip(["project", "seed",  "anim", "speed"],
                         [self.osc_handler(param) for param in
                          ["project", "x", "update_mode", "speed"]]))
        funcs["speed"] = self.speed_handler()
        funcs["model"] = self.model_handler()
        self.seed_osc_menu = osc_menu.OscMenu(self.viz, copy.deepcopy(funcs),
                                         label="##SeedOSC")
        del funcs["seed"]
        funcs["vector"] = self.list_handler("vec")
        funcs["randomize"] = self.randomize_handler()
        self.vec_osc_menu = osc_menu.OscMenu(self.viz, copy.deepcopy(funcs),
                                            label="##VecOSC")
        self.update = False
        self.latent_def = dnnlib.EasyDict(self.latent)
        self.vec_path   = ""
        self.vec_save_path = ""
        self.file_dialog = BrowseWidget(viz, "Browse", os.path.relpath(os.getcwd()),
                                      ["*", ".pth", ".pt", ".npy", ".npz", ],
                                      width=self.viz.app.button_w, multiple=False, traverse_folders=False)
        self.pt_save_dialog = save_widget.SaveWidget(viz, "Save Vector", os.path.abspath(os.getcwd()), ".pt")


    def save(self, path):
        with open(path, "wb") as f:
            print(self.get_params())
            pickle.dump(self.get_params(), f)

    def load(self, path):
        with open(path, "rb") as f:
            self.set_params(pickle.load(f))

    def get_params(self):
        return self.latent, self.seed_osc_menu.get_params(), self.vec_osc_menu.get_params()

    def set_params(self, params):
        self.latent, osc_params, vec_osc = params
        self.seed_osc_menu.set_params(osc_params)
        self.vec_osc_menu.set_params(vec_osc)

        self.viz.args.mode = self.latent.mode
        self.viz.args.project = self.latent.project
        self.viz.args.seed = [self.latent.x, self.latent.y]  # [[seed, weight], ...]
        self.viz.args.vec = self.latent.vec.pin_memory() if torch.cuda.is_available else self.latent.vec

    def drag(self, dx, dy):
        viz = self.viz
        self.latent.x += dx / viz.app.font_size * 4e-2
        self.latent.y += dy / viz.app.font_size * 4e-2

    def osc_handler(self, param):
        def func(address, *args):
            try:
                nec_type = type(self.latent[param])
                self.latent[param] = nec_type(args[-1])
            except Exception as e:
                self.viz.print_error(e)
        return func

    def speed_handler(self):
        def func(address, *args):
            try:
                nec_type = type(self.latent.speed)
                self.latent.speed = nec_type(args[-1])
                self.update = True
            except Exception as e:
                self.viz.print_error(e)
        return func

    def list_handler(self, param):
        def func(address, *args):
            try:
                value = torch.as_tensor(args)[None]
                assert value.shape == self.latent[
                    param].shape, f"Shapes of osc message and parameter must align, [OSC] {value.shape} != [Param] {self.latent[param].shape}"
                self.latent[param] = value
            except Exception as e:
                self.viz.print_error(e)

        return func

    @imgui_utils.scoped_by_object_id
    def seed_viz(self):
        viz = self.viz
        seed = round(self.latent.x) + round(self.latent.y) * self.step_y
        with imgui_utils.item_width(viz.app.font_size * 8):
            _changed, seed = imgui.input_int("##seed", seed)
        if _changed:
            self.latent.x = seed
            self.latent.y = 0
        imgui.same_line()
        frac_x = self.latent.x - round(self.latent.x)
        frac_y = self.latent.y - round(self.latent.y)
        with imgui_utils.item_width(viz.app.font_size * 5):
            _changed, (new_frac_x, new_frac_y) = imgui.input_float2('##frac', frac_x, frac_y, format='%+.2f',
                                                                    flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
        if _changed:
            self.latent.x += new_frac_x - frac_x
            self.latent.y += new_frac_y - frac_y
        imgui.same_line(spacing=0)
        _clicked, dragging, dx, dy = imgui_utils.drag_button('Drag', width=viz.app.button_w)
        if dragging:
            self.drag(dx, dy)
        imgui.same_line(imgui.get_item_rect_max()[0] + (viz.app.spacing * 2))
        if imgui_utils.button(f"{modes[(int(self.latent.update_mode) + 1) % len(modes)]}##latent"):
            self.latent.update_mode = (int(self.latent.update_mode) + 1) % len(modes)
        imgui.same_line()
        with imgui_utils.item_width(viz.app.button_w * 2 - viz.app.spacing * 2):
            changed, speed = imgui.slider_float('##speed', self.latent.speed, -5, 5,
                                                format='Speed %.3f',
                                                power=3)
            if changed:
                self.latent.speed = speed
                self.update = True

    def update_vec(self):
        viz = self.viz
        if self.latent.mode:
            self.latent.x += abs(viz.app.frame_delta) * self.latent.speed
        else:
            diff = self.latent.next - self.latent.vec
            if torch.linalg.vector_norm(diff) < 1:
                self.latent.next = torch.randn(1, 512)
            direc = F.normalize(diff)
            self.latent.vec += abs(viz.app.frame_delta) * self.latent.speed * 10 * direc

    @imgui_utils.scoped_by_object_id
    def vec_viz(self):
        viz = self.viz
        if imgui_utils.button("Randomize##vecmode", width=viz.app.button_w):
            self.latent.vec = torch.randn(self.latent.vec.shape)
            self.latent.next = torch.randn(self.latent.next.shape)

        imgui.same_line()
        if imgui_utils.button(f"{modes[self.latent.update_mode]}##latent"):
            self.latent.update_mode = (int(self.latent.update_mode) + 1) % len(modes)
        imgui.same_line()
        with imgui_utils.item_width(viz.app.button_w * 2 - viz.app.spacing * 2):
            changed, speed = imgui.slider_float('##speed', self.latent.speed, -5, 5,
                                                format='Speed %.3f',
                                                power=3)
            if changed:
                self.latent.speed = speed
                self.update = True

        _changed, vec_path = imgui_utils.input_text("##vecpath", self.vec_path, 1024, width=viz.app.button_w, flags=imgui.INPUT_TEXT_AUTO_SELECT_ALL, help_text="Load vector from file")
        imgui.same_line()
        if _changed:
            # if the file exists and is a valid vector, load it
            if os.path.exists(vec_path):
                self.vec_path = vec_path
        _clicked, paths = self.file_dialog()
        if _clicked and len(paths) > 0:
            self.vec_path = paths[0]
            print("Selected vector at", self.vec_path)

        imgui.same_line()
        if imgui_utils.button("Load##vecmode", width=viz.app.button_w, enabled=self.vec_path is not None and self.vec_path != ''):
            print("Loading vector from", self.vec_path)
            if self.vec_path:
                if self.vec_path.endswith('.npy'):
                    self.latent.vec = torch.from_numpy(np.load(self.vec_path))
                    if len(self.latent.vec.shape) == 1:
                        self.latent.vec = self.latent.vec.unsqueeze(0)
                elif self.vec_path.endswith('.pt'):
                    self.latent.vec = torch.load(self.vec_path)
                    if len(self.latent.vec.shape) == 1:
                        self.latent.vec = self.latent.vec.unsqueeze(0)
                else:
                    print("Unsupported file format")
                print("Loaded vector of shape", self.latent.vec.shape)
                self.latent.next = torch.randn(self.latent.next.shape)


        imgui.same_line()
        _changed, fname = self.pt_save_dialog()
        if _changed:
            self.vec_save_path = fname
            torch.save(self.latent.vec, self.vec_save_path)

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            with torch.inference_mode(True):
                try:
                    imgui.text('Latent')
                    imgui.same_line(viz.app.label_w)
                    self.latent.mode = imgui.checkbox('Seed', self.latent.mode)[1]
                    imgui.same_line()
                    self.latent.mode = not (imgui.checkbox('Vector', not self.latent.mode)[1])
                    viz.args.mode = "seed" if self.latent.mode else "vec"
                    _clicked, self.latent.project = imgui.checkbox('Project', self.latent.project)
                    viz.args.project = self.latent.project
                    imgui.same_line()
                    if self.latent.mode:
                        self.seed_viz()
                        self.seed_osc_menu()
                    else:
                        self.vec_viz()
                        self.vec_osc_menu()

                except Exception as e:
                    self.viz.print_error(e)

                if self.latent.update_mode == 1:
                    self.update = True
                if self.latent.update_mode in [1, 2] and self.update:
                    self.update_vec()
                self.update = False
        viz.args.seeds = (self.latent.x, self.latent.y)
        viz.args.vec = self.latent.vec.pin_memory() if torch.cuda.is_available() else self.latent.vec




    def model_handler(self):
        def func(address, *args):
            try:
                assert (type(args[-1]) is str), f"OSC Message and Parameter type must align [OSC] {type(args[-1])} != [Param] {str}"
                # check if the string that is sent in the message os.exists
                print(os.getcwd() + os.sep + 'models' + os.sep + args[-1])
                if os.path.exists(args[-1]):
                    print(os.getcwd() + os.sep + 'models' + os.sep + args[-1])
                    self.viz.pickle_widget.user_pkl = args[-1]
                    self.viz.pickle_widget.load_pkl(args[-1])
                elif os.path.exists(os.getcwd() + os.sep + 'models' + os.sep + args[-1]):
                    print(os.getcwd() + os.sep + 'models' + os.sep + args[-1])
                    self.viz.pickle_widget.user_pkl = os.getcwd() + os.sep + 'models' + os.sep + args[-1]
                    self.viz.pickle_widget.load_pkl(os.getcwd() + os.sep + 'models' + os.sep + args[-1])
            except Exception as e:
                self.viz.print_error(e)
        return func

    def randomize_handler(self):
        def func(address, *args):
            """
            OSC Handler for randomizing the vector receives a boolean message and if true randomizes the vector
            :param address:
            :param args:
            :return:
            """
            try:
                if bool(args[-1]):
                    self.latent.vec = torch.randn(self.latent.vec.shape)
                    self.latent.next = torch.randn(self.latent.next.shape)
            except Exception as e:
                self.viz.print_error(e)
        return func

#----------------------------------------------------------------------------
