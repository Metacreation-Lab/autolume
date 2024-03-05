# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import contextlib
import re

import cv2

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

import copy
import os
import random

import imgui
import numpy as np
import torch
import yaml

import dnnlib
from utils.gui_utils import imgui_utils, gl_utils
from widgets import osc_menu
from assets.colors import *

# ----------------------------------------------------------------------------

transforms = np.array(
    ["translate", "rotate", "scale", "erode", "dilate", "invert", "flip-h", "flip-v", "binary-thresh",
     "scalar-multiply", "ablate"])
num_params = np.array([2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
param_type = np.array(["f", "f", "f", "ui", "ui", "b", "b", "b", "f", "f", "b"])
modes = ["all", "random", "range", "config"]


def selectable_color(r, g, b, a):
    p_min = imgui.get_item_rect_min()
    p_max = imgui.get_item_rect_max()
    imgui.get_window_draw_list().add_rect_filled(0, p_min[1], imgui.get_content_region_available_width(), p_max[1], imgui.get_color_u32_rgba(r, g, b, a))


def open_path(trans):
    try:
        if os.path.exists(trans.cluster_path):
            with open(trans.cluster_path.clusters, 'r') as stream:
                trans.cluster_config = yaml.load(stream)
        else:
            trans.cluster_config = None
    except Exception as e:
        print(e)
        trans.cluster_config = None


class LayerWidget:
    def __init__(self, viz):
        self.viz = viz
        self.prev_layers = None
        self.cur_layer = None
        self.sel_channels = 3
        self.cur_channels = 0
        self.base_channel = 0
        self.img_scale_db = 0
        self.img_normalize = False
        self.refocus = False
        self.mode = True
        self.cached_transforms = []
        self.osc_funcs = {}
        self.names = []
        self.has_transforms = {}
        self.has_osc = {}
        self.imgui_ids = set()
        self.capture_layer = "output"  # which layer should be captured and rendered
        self.cached_adjustments = {}  # cached adjustments for each layer that will be applied to latent vectors (currently not doing anything)
        self.noises = {}  # noise strength for each layer
        self.ratios = {}  # ratio of activations for each layer
        self.paths = {}  # path to vectors for adjustments (currently not doing anything)
        self.capture_channels = 0
        self.tab = False
        self.simplified = True

        funcs = {"Grayscale": self.grayscale_osc(), "Contrast": self.contrast_osc(), "Normalize": self.normalize_osc(), "Channels": self.channels_osc()}
        self.osc_menu = osc_menu.OscMenu(self.viz, funcs, None, label="##LayerOSCMenu")

        # read as rgba
        self.edit_img = cv2.imread("assets/pen.png")
        self.edit_img = cv2.cvtColor(self.edit_img, cv2.COLOR_BGR2RGBA)

        # in the alpha channel we put alpha to 0 where the image is black
        self.edit_img[:, :, 3] = (self.edit_img[:, :, 0] != 0) * 255
        self.edit_texture = gl_utils.Texture(image=self.edit_img, width=self.edit_img.shape[1],
                                             height=self.edit_img.shape[0], channels=self.edit_img.shape[2])

        self.view_img = cv2.imread("assets/eye.png")
        self.view_img = cv2.cvtColor(self.view_img, cv2.COLOR_BGR2RGBA)

        # in the alpha channel we put alpha to 0 where the image is black
        self.view_img[:, :, 3] = (self.view_img[:, :, 0] != 0) * 255
        self.view_texture = gl_utils.Texture(image=self.view_img, width=self.view_img.shape[1],
                                             height=self.view_img.shape[0], channels=self.view_img.shape[2])

        self.transform_img = cv2.imread("assets/transformation.png")
        self.transform_img = cv2.cvtColor(self.transform_img, cv2.COLOR_BGR2RGBA)
        print("UNIQUES", np.unique(self.transform_img))

        self.transform_img[:, :, 3] = np.where(self.transform_img[:, :, 0] == 255, 255, 0)
        self.transform_texture = gl_utils.Texture(image=self.transform_img, width=self.transform_img.shape[1],
                                                  height=self.transform_img.shape[0],
                                                  channels=self.transform_img.shape[2])

        self.noise_img = cv2.imread("assets/noise.png")
        self.noise_img = cv2.cvtColor(self.noise_img, cv2.COLOR_BGR2RGBA)
        self.noise_img[:, :, 3] = np.where(self.noise_img[:, :, 0] == 255, 255, 0)
        self.noise_texture = gl_utils.Texture(image=self.noise_img, width=self.noise_img.shape[1],
                                              height=self.noise_img.shape[0], channels=self.noise_img.shape[2])

    def get_params(self):
        return self.mode, self.cached_transforms, self.names, self.has_transforms, self.cached_adjustments, \
            self.noises, self.ratios, self.paths, self.imgui_ids, self.capture_layer, self.capture_channels, \
            self.tab, self.img_scale_db, self.img_normalize

    def set_params(self, param):
        self.mode, cached_transforms, self.names, self.has_transforms, cached_adjustments, noises, self.ratios, self.paths, self.imgui_ids, self.capture_layer, self.capture_channels, self.tab, self.img_scale_db, self.img_normalize = param
        for i, trans in enumerate(self.cached_transforms):
            for j in range(len(trans.params)):
                try:
                    self.viz.osc_dispatcher.unmap(f"/{trans.osc_address[j]}",
                                                  self.transform_osc(trans, param_idx=j))
                except Exception as e:
                    print(e)
                    print(f"{trans.osc_address[j]} is not mapped")

        self.cached_transforms = cached_transforms
        for i, trans in enumerate(self.cached_transforms):
            for j in range(len(trans.params)):
                self.viz.osc_dispatcher.map(f"/{trans.osc_address[j]}",
                                            self.transform_osc(trans, param_idx=j))

        for _, noise in self.noises.items():
            try:
                self.viz.osc_dispatcher.unmap(f"/{noise['osc_address']}",
                                              self.noise_osc(noise))
            except Exception as e:
                print(e)
                print(f"{noise['osc_address']} is not mapped")
        self.noises = noises
        for _, noise in self.noises.items():
            self.viz.osc_dispatcher.map(f"/{noise['osc_address']}", self.noise_osc(noise))

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_params(), f)

    def load(self, path):
        with open(path, "rb") as f:
            self.set_params(pickle.load(f))

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        layers = viz.result.get('layers', [])
        if self.prev_layers != layers:
            self.names = [layer.name for layer in layers]
            self.prev_layers = layers
            self.refocus = True
        for layer in layers:
            if not layer.name in self.has_transforms:
                self.has_transforms[layer.name] = False
            if not layer.name in self.has_osc:
                self.has_osc[layer.name] = False
        layer = ([layer for layer in layers if layer.name == self.cur_layer] + [None])[0]

        if layer is None and len(layers) > 0:
            layer = layers[-1]
            self.cur_layer = layer.name
            self.cur_channels = layer.shape[1]

        base_channel_max = max(self.capture_channels - self.sel_channels, 0)

        if show:

            # Begin list.
            width = viz.app.font_size * 28
            height = imgui.get_text_line_height_with_spacing() * 12 + viz.app.spacing
            # imgui.push_style_var(imgui.STYLE_FRAME_PADDING, [0, 0])
            imgui.push_style_color(imgui.COLOR_CHILD_BACKGROUND, *OPAQUEDARKGRAY)
            imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, *OPAQUEDARKGRAY)

            imgui.push_style_color(imgui.COLOR_HEADER, 0, 0, 0, 0)
            imgui.push_style_color(imgui.COLOR_TEXT, 1,1,1,1)
            imgui.push_style_color(imgui.COLOR_HEADER_HOVERED, 0,0,0,0)
            imgui.push_style_color(imgui.COLOR_HEADER_ACTIVE, 0,0,0,0)

            imgui.begin_child('##list', width=width, height=height, border=True, flags=imgui.WINDOW_NO_SCROLLBAR|imgui.WINDOW_NO_INPUTS)
            imgui.begin_child('##list2', width=width, height=self.viz.app.font_size * 1.75, border=False, flags=imgui.WINDOW_NO_SCROLLBAR|imgui.WINDOW_NO_SCROLL_WITH_MOUSE)
            imgui.set_cursor_pos((0, -1))
            if self.simplified:
                imgui_utils.color_button("Simple", color=GREEN, width=(width // 2))
            else:
                imgui.push_style_color(imgui.COLOR_BUTTON, *DARKGRAY)
                imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, *GRAY)
                imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, *GREEN)
                imgui.push_style_color(imgui.COLOR_BORDER, *LIGHTGRAY)

                if imgui_utils.button("Simple", width=(width // 2),
                                      enabled=not self.simplified):
                    self.simplified = True
                imgui.pop_style_color(4)
            imgui.same_line(width//2)

            if not self.simplified:
                imgui_utils.color_button("Advanced", color=GREEN, width=(width // 2) - (viz.app.spacing * 3))
            else:
                imgui.push_style_color(imgui.COLOR_BUTTON, *DARKGRAY)
                imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, *GRAY)
                imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, *GREEN)
                imgui.push_style_color(imgui.COLOR_BORDER, *LIGHTGRAY)
                if imgui_utils.button("Advanced", width=(width // 2) - (viz.app.spacing * 3), enabled=self.simplified):
                    self.simplified = False
                imgui.pop_style_color(4)
            imgui.separator()
            imgui.end_child()
            imgui.begin_child('##list3', width=width, height=height - self.viz.app.font_size * 1.5, border=False,
                              flags=imgui.WINDOW_ALWAYS_VERTICAL_SCROLLBAR)
            checkbox_size = viz.app.font_size
            draw_list = imgui.get_window_draw_list()
            # List items.
            if self.simplified:
                resolution = 4
                for layer in layers:
                    if 'conv' in layer.name or "torgb" in layer.name:
                        res = int(re.search(r'\d+', layer.name).group())
                        if res == resolution:
                            resolution *= 2
                            draw_list.channels_split(2)
                            draw_list.channels_set_current(1)
                            selected = (self.cur_layer == layer.name)
                            clicked, state = imgui_utils.img_checkbox(self.view_texture.gl_id,
                                                                      self.capture_layer == f"b{res}.torgb",
                                                                      width=checkbox_size)
                            if clicked and not self.capture_layer == layer.name:
                                self.capture_layer = f"b{res}.torgb"
                                for ltmp in layers:
                                    if ltmp.name == self.capture_layer:
                                        self.capture_channels = ltmp.shape[1]
                            # imgui.same_line(viz.app.font_size + viz.app.spacing * 1.5)
                            imgui.set_cursor_pos([imgui.get_cursor_pos()[0] + viz.app.font_size + viz.app.spacing // 2,
                                                  imgui.get_cursor_pos()[1]])
                            _opened, selected = imgui.selectable(f'##{layer.name}_selectable',
                                                                 width=width - viz.app.button_w,
                                                                 selected=selected)

                            if imgui.is_item_hovered():
                                selectable_color(*DARKGRAY)
                            if _opened or (imgui.is_item_hovered() and imgui.is_mouse_down(0)):
                                selectable_color(*GRAY)
                            if layer.name in self.has_osc:
                                if self.has_osc[layer.name]:
                                    draw_list.channels_set_current(0)
                                    selectable_color(*ACTIVE_RED)
                            imgui.same_line(checkbox_size + viz.app.spacing * 2)
                            _clicked, selected = imgui_utils.img_checkbox(self.edit_texture.gl_id,
                                                                          selected, width=checkbox_size,
                                                                          label=layer.name)
                            if selected:
                                self.cur_layer = layer.name
                                self.cur_channels = layer.shape[1]
                                if self.refocus:
                                    imgui.set_scroll_here()
                                    viz.app.skip_frame()  # Focus will change on next frame.
                                    self.refocus = False
                            draw_list.channels_merge()
                            imgui.same_line(width - viz.app.font_size * 13)
                            imgui.text('x'.join(str(x) for x in layer.shape[2:]))
                            imgui.same_line(width - viz.app.font_size * 8)
                            imgui.text(str(layer.shape[1]))
                            # display if noise or transformation is applied
                            if self.has_transforms[layer.name]:
                                imgui.same_line(width - viz.app.font_size * 5)
                                ratio = self.transform_texture.height / self.transform_texture.width
                                imgui.image(self.transform_texture.gl_id, viz.app.font_size,
                                            viz.app.font_size * ratio)
                            if layer.name in self.noises:
                                if self.noises[layer.name]["strength"] != 0:
                                    imgui.same_line(width - viz.app.font_size * 3.5)
                                    ratio = self.noise_texture.height / self.noise_texture.width
                                    imgui.image(self.noise_texture.gl_id, viz.app.font_size, viz.app.font_size * ratio)

                            self.has_transforms[layer.name] = False
                    if layer.name == "output":
                        draw_list.channels_split(2)
                        draw_list.channels_set_current(1)
                        selected = (self.cur_layer == layer.name)
                        clicked, state = imgui_utils.img_checkbox(self.view_texture.gl_id,
                                                                  self.capture_layer == layer.name,
                                                                  width=checkbox_size)
                        if clicked and not self.capture_layer == layer.name:
                            self.capture_layer = layer.name
                            self.capture_channels = layer.shape[1]
                        imgui.set_cursor_pos([imgui.get_cursor_pos()[0] + viz.app.font_size + viz.app.spacing // 2,
                                              imgui.get_cursor_pos()[1]])
                        _opened, selected = imgui.selectable(f'##{layer.name}_selectable',
                                                             width=width - viz.app.button_w,
                                                             selected=selected)

                        if imgui.is_item_hovered():
                            selectable_color(*DARKGRAY)
                        if _opened or (imgui.is_item_hovered() and imgui.is_mouse_down(0)):
                            selectable_color(*GRAY)

                        if layer.name in self.has_osc:
                            if self.has_osc[layer.name]:
                                draw_list.channels_set_current(0)
                                selectable_color(*ACTIVE_RED)
                        imgui.same_line(checkbox_size + viz.app.spacing * 2)
                        _clicked, selected = imgui_utils.img_checkbox(self.edit_texture.gl_id,
                                                                      selected, width=checkbox_size,
                                                                      label=layer.name)
                        if selected:
                            self.cur_layer = layer.name
                            self.cur_channels = layer.shape[1]
                            if self.refocus:
                                imgui.set_scroll_here()
                                viz.app.skip_frame()  # Focus will change on next frame.
                                self.refocus = False
                        draw_list.channels_merge()
                        imgui.same_line(width - viz.app.font_size * 13)
                        imgui.text('x'.join(str(x) for x in layer.shape[2:]))
                        imgui.same_line(width - viz.app.font_size * 8)
                        imgui.text(str(layer.shape[1]))

                        # display if noise or transformation is applied
                        if self.has_transforms[layer.name]:
                            imgui.same_line(width - viz.app.font_size * 5)
                            ratio = self.transform_texture.height / self.transform_texture.width
                            imgui.image(self.transform_texture.gl_id, viz.app.font_size,
                                        viz.app.font_size * ratio)
                        if layer.name in self.noises:
                            if self.noises[layer.name]["strength"] != 0:
                                imgui.same_line(width - viz.app.font_size * 3.5)
                                ratio = self.noise_texture.height / self.noise_texture.width
                                imgui.image(self.noise_texture.gl_id, viz.app.font_size, viz.app.font_size * ratio)

                        self.has_transforms[layer.name] = False
            else:
                print([layer.name for layer in layers])
                for layer in layers:
                    if 'conv' in layer.name or "torgb" in layer.name:
                        draw_list.channels_split(2)
                        draw_list.channels_set_current(1)
                        selected = (self.cur_layer == layer.name)
                        clicked, state = imgui_utils.img_checkbox(self.view_texture.gl_id,
                                                                  self.capture_layer == layer.name,
                                                                  width=checkbox_size)
                        if clicked and not self.capture_layer == layer.name:
                            self.capture_layer = layer.name
                            self.capture_channels = layer.shape[1]
                        imgui.set_cursor_pos([imgui.get_cursor_pos()[0] + viz.app.font_size + viz.app.spacing // 2,
                                              imgui.get_cursor_pos()[1]])
                        _opened, selected = imgui.selectable(f'##{layer.name}_selectable',
                                                             width=width - viz.app.button_w,
                                                             selected=selected)

                        if imgui.is_item_hovered():
                            selectable_color(*DARKGRAY)
                        if _opened or (imgui.is_item_hovered() and imgui.is_mouse_down(0)):
                            selectable_color(*GRAY)

                        if layer.name in self.has_osc:
                            if self.has_osc[layer.name]:
                                draw_list.channels_set_current(0)
                                selectable_color(*ACTIVE_RED)
                        imgui.same_line(checkbox_size + viz.app.spacing * 2)
                        _clicked, selected = imgui_utils.img_checkbox(self.edit_texture.gl_id,
                                                                      selected, width=checkbox_size,
                                                                      label=layer.name)
                        if selected:
                            self.cur_layer = layer.name
                            self.cur_channels = layer.shape[1]
                            if self.refocus:
                                imgui.set_scroll_here()
                                viz.app.skip_frame()  # Focus will change on next frame.
                                self.refocus = False
                        draw_list.channels_merge()
                        imgui.same_line(width - viz.app.font_size * 13)
                        imgui.text('x'.join(str(x) for x in layer.shape[2:]))
                        imgui.same_line(width - viz.app.font_size * 8)
                        imgui.text(str(layer.shape[1]))
                        # display if noise or transformation is applied
                        if self.has_transforms[layer.name]:
                            imgui.same_line(width - viz.app.font_size * 5)
                            ratio = self.transform_texture.height / self.transform_texture.width
                            imgui.image(self.transform_texture.gl_id, viz.app.font_size,
                                        viz.app.font_size * ratio)
                        if layer.name in self.noises:
                            if self.noises[layer.name]["strength"] != 0:
                                imgui.same_line(width - viz.app.font_size * 3.5)
                                ratio = self.noise_texture.height / self.noise_texture.width
                                imgui.image(self.noise_texture.gl_id, viz.app.font_size, viz.app.font_size * ratio)

                            self.has_transforms[layer.name] = False

                    if layer.name == "output":
                        selected = (self.cur_layer == layer.name)
                        clicked, state = imgui_utils.img_checkbox(self.view_texture.gl_id,
                                                                  self.capture_layer == layer.name,
                                                                  width=checkbox_size)
                        draw_list.channels_split(2)
                        draw_list.channels_set_current(1)
                        selected = (self.cur_layer == layer.name)
                        clicked, state = imgui_utils.img_checkbox(self.view_texture.gl_id,
                                                                  self.capture_layer == layer.name,
                                                                  width=checkbox_size)
                        if clicked and not self.capture_layer == layer.name:
                            self.capture_layer = layer.name
                            self.capture_channels = layer.shape[1]
                        imgui.set_cursor_pos([imgui.get_cursor_pos()[0] + viz.app.font_size + viz.app.spacing // 2,
                                              imgui.get_cursor_pos()[1]])
                        _opened, selected = imgui.selectable(f'##{layer.name}_selectable',
                                                             width=width - viz.app.button_w,
                                                             selected=selected)

                        if imgui.is_item_hovered():
                            selectable_color(*DARKGRAY)
                        if _opened or (imgui.is_item_hovered() and imgui.is_mouse_down(0)):
                            selectable_color(*GRAY)

                        if layer.name in self.has_osc:
                            if self.has_osc[layer.name]:
                                draw_list.channels_set_current(0)
                                selectable_color(*ACTIVE_RED)
                        imgui.same_line(checkbox_size + viz.app.spacing * 2)
                        _clicked, selected = imgui_utils.img_checkbox(self.edit_texture.gl_id,
                                                                      selected, width=checkbox_size,
                                                                      label=layer.name)
                        if selected:
                            self.cur_layer = layer.name
                            self.cur_channels = layer.shape[1]
                            if self.refocus:
                                imgui.set_scroll_here()
                                viz.app.skip_frame()  # Focus will change on next frame.
                                self.refocus = False
                        draw_list.channels_merge()
                        imgui.same_line(width - viz.app.font_size * 13)
                        imgui.text('x'.join(str(x) for x in layer.shape[2:]))
                        imgui.same_line(width - viz.app.font_size * 8)
                        imgui.text(str(layer.shape[1]))
                        # display if noise or transformation is applied
                        if self.has_transforms[layer.name]:
                            imgui.same_line(width - viz.app.font_size * 5)
                            ratio = self.transform_texture.height / self.transform_texture.width
                            imgui.image(self.transform_texture.gl_id, viz.app.font_size,
                                        viz.app.font_size * ratio)
                        if layer.name in self.noises:
                            if self.noises[layer.name]["strength"] != 0:
                                imgui.same_line(width - viz.app.font_size * 3.5)
                                ratio = self.noise_texture.height / self.noise_texture.width
                                imgui.image(self.noise_texture.gl_id, viz.app.font_size, viz.app.font_size * ratio)


                        imgui.text("")


                        self.has_transforms[layer.name] = False

            # End list.
            if len(layers) == 0:
                imgui.text_colored('No layers found', *LIGHTGRAY)
            imgui.end_child()
            imgui.end_child()
            imgui.pop_style_color(6)
            # imgui.pop_style_var(1)
            imgui.same_line()
            imgui.begin_child('##adjust', width=-1, height=height, border=True)
            tab_width = imgui.get_content_region_available_width() / 2
            if (self.tab is False) and (len(layers) > 0):
                if self.has_transforms[layer.name]:
                    tab_width = (imgui.get_content_region_available_width() - imgui.get_style().scrollbar_size) / 2

            with greened_out((self.tab is False) and (len(layers) > 0)):
                if imgui_utils.button("Transform",
                                      width=tab_width,
                                      enabled=len(layers)>0):
                    self.tab = not self.tab

            imgui.same_line(tab_width + viz.app.spacing)
            has_noise = 'torgb' in self.cur_layer or "output" in self.cur_layer if self.cur_layer is not None else True
            if has_noise and self.tab:
                self.tab = False
            with greened_out(self.tab):
                if imgui_utils.button("Noise",
                                      width=tab_width, enabled=not(has_noise)):
                    self.tab = not self.tab
            imgui.separator()
            if self.cur_layer is not None:
                ratio = self.ratios.get(self.cur_layer, (1, 1))
            if not self.tab:
                label = "Transform" if self.mode else "Adjust"
                imgui.text(label)
                imgui.same_line()
                if self.cur_layer is not None:
                    if "torgb" not in self.cur_layer and "output" not in self.cur_layer:
                        with imgui_utils.item_width(-1):
                            _, ratio = imgui.input_float2(f"Ratio##{self.cur_layer}ratio", *ratio, format='%.2f',
                                                          flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)

                if len(layers) > 0:
                    self.transform_widget(layers)

                # set all has_osc to false
                for layer in self.has_osc:
                    self.has_osc[layer] = False

                for i, trans in enumerate(self.cached_transforms):
                    if trans.layerID in self.names:
                        self.has_transforms[trans.layerID] = True
                    if trans.layerID in self.has_osc:
                        self.has_osc[trans.layerID] = trans.use_osc

                for noise in self.noises:
                    if noise in self.has_osc:
                        if self.has_osc[noise]==False:
                            self.has_osc[noise] = self.noises[noise]["use_osc"]
            else:
                self.adjust_noise()
                # set all has_osc to false
                for layer in self.has_osc:
                    self.has_osc[layer] = False
                for i, trans in enumerate(self.cached_transforms):
                    if trans.layerID in self.names:
                        self.has_transforms[trans.layerID] = True
                    if trans.layerID in self.has_osc:
                        self.has_osc[trans.layerID] = trans.use_osc

                for noise in self.noises:
                    if noise in self.has_osc:
                        if self.has_osc[noise]==False:
                            self.has_osc[noise] = self.noises[noise]["use_osc"]
            imgui.end_child()

            if self.cur_layer is not None:
                self.ratios[self.cur_layer] = ratio
            viz.args.ratios = self.ratios

            # Begin options.

            # RGB & normalize.
            grayscale = (self.sel_channels == 1)
            _clicked, grayscale = imgui.checkbox('B/W', grayscale)
            self.sel_channels = 1 if grayscale else 3
            imgui.same_line(viz.app.font_size * 4)
            _clicked, self.img_normalize = imgui.checkbox('Normalize', self.img_normalize)
            imgui.same_line()
            if imgui_utils.button('Reset##img_flags', width=viz.app.button_w,
                                  enabled=(self.sel_channels != 3 or self.img_normalize)):
                self.sel_channels = 3
                self.img_normalize = False
            imgui.same_line()
            # Image scale.
            with imgui_utils.item_width(viz.app.button_w):
                _changed, self.img_scale_db = imgui.slider_float('##scale', self.img_scale_db, min_value=-40,
                                                                 max_value=40, format='Contrast %+.1f dB')
            imgui.same_line()
            if imgui_utils.button('Reset##scale', width=viz.app.button_w, enabled=(self.img_scale_db != 0)):
                self.img_scale_db = 0

            # Base channel.
            imgui.same_line()
            self.base_channel = min(max(self.base_channel, 0), base_channel_max)
            narrow_w = imgui.get_text_line_height_with_spacing()
            with imgui_utils.grayed_out(base_channel_max == 0):
                with imgui_utils.item_width(-1 - viz.app.button_w - narrow_w * 2 - viz.app.spacing * 3):
                    _changed, self.base_channel = imgui.drag_int('##channel', self.base_channel, change_speed=0.05,
                                                                 min_value=0, max_value=base_channel_max,
                                                                 format=f'Channel %d/{self.capture_channels}')
                imgui.same_line()
                if imgui_utils.button('-##channel', width=narrow_w):
                    self.base_channel -= 1
                imgui.same_line()
                if imgui_utils.button('+##channel', width=narrow_w):
                    self.base_channel += 1
            imgui.same_line()
            self.base_channel = min(max(self.base_channel, 0), base_channel_max)
            if imgui_utils.button('Reset##channel', width=-1,
                                  enabled=(self.base_channel != 0 and base_channel_max > 0)):
                self.base_channel = 0

            self.osc_menu()
        # End options.

        self.base_channel = min(max(self.base_channel, 0), base_channel_max)
        viz.args.layer_name = self.capture_layer if len(layers) > 0 and self.names and self.capture_layer != layers[
            -1].name else None
        viz.args.update(sel_channels=self.sel_channels, base_channel=self.base_channel, img_scale_db=self.img_scale_db,
                        img_normalize=self.img_normalize)
        viz.args.latent_transforms = copy.deepcopy(self.cached_transforms)
        viz.args.noise_adjustments = copy.deepcopy(self.noises)

    def make_id(self):
        # make a unique id for imgui
        u_id = len(self.imgui_ids)
        self.imgui_ids.add(u_id)
        return u_id

    def drag(self, dx, dy, l_idx, p_idx, type="f"):
        # drag util that links button to drag to adjust params
        self.cached_transforms[l_idx].params[p_idx] += dx / self.viz.app.font_size * 4e-2
        self.cached_transforms[l_idx].params[p_idx] += (dy / self.viz.app.font_size * 4e-2) * 100
        if type == "ui":
            self.cached_transforms[l_idx].params[p_idx] = np.round(self.cached_transforms[l_idx].params[p_idx])

    @imgui_utils.scoped_by_object_id
    def transform_widget(self, layers):
        # Widget to add manipulations to the activations of the layers
        with imgui_utils.item_width(-1):
            clicked, current = imgui.combo("##Transformation", 0, transforms.tolist())
        if clicked:
            item = transforms[current].item()
            params = [0, ] * num_params[current].item()

            transformation = dnnlib.EasyDict(
                {"transformID": item, "layerID": self.cur_layer, "params": params, "indices": 0, "mode": 0,
                 "max_idx": self.cur_channels, "percentage": 0.5, "low": 0, "high": self.cur_channels,
                 "cluster_ID": 0, "cluster_config": None, "cluster_path": "",
                 "type": param_type[current].item()}, imgui_id=self.make_id(), use_osc=False,
                osc_address=["osc address", ] * num_params[current].item(),
                mapping=["x"] * num_params[current].item())
            funcs = [self.transform_osc(transformation, param_idx) for param_idx in range(len(transformation.params))]
            self.osc_funcs[transformation.imgui_id] = funcs
            self.cached_transforms.append(transformation)

        to_remove = []

        if len(layers) > 0:
            for i, trans in enumerate(self.cached_transforms):
                if trans.layerID in self.names:
                    self.has_transforms[trans.layerID] = True
                    self.has_transforms[trans.layerID] = trans.use_osc
                    if trans.layerID == self.cur_layer:
                        u_id = trans.imgui_id
                        if imgui.button(f"x##{u_id}", width=self.viz.app.font_size):
                            imgui.open_popup(f"remove-transform##{u_id}")

                        if imgui.begin_popup(f"remove-transform##{u_id}"):
                            imgui.text(f"Remove Transformation {trans.transformID}")
                            imgui.separator()
                            _, confirmed = imgui.selectable("Confirm")
                            if confirmed:
                                to_remove.append(i)
                            _, canceled = imgui.selectable("Cancel")
                            imgui.end_popup()
                        trans.update(max_idx=self.cur_channels, high=self.cur_channels)
                        imgui.same_line()
                        imgui.text(trans.transformID)
                        self.indices_widget(trans)
                        with imgui_utils.item_width(self.viz.app.font_size * 8):
                            for j in range(len(trans.params)):
                                if trans.type == "f":
                                    changed, trans.params[j] = imgui.input_float(f"##{j}_{u_id}", trans.params[j])
                                    imgui.same_line()
                                    _clicked, dragging, dx, dy = imgui_utils.drag_button(f"Drag##_{j}_{u_id}",
                                                                                         width=self.viz.app.button_w)
                                    if dragging:
                                        self.drag(dx, dy, i, j)
                                if trans.type == "ui":
                                    changed, trans.params[j] = imgui.input_int(f"##{j}_{u_id}", trans.params[j])
                                    if trans.params[j] <= 0:
                                        trans.params[j] = 1
                                    imgui.same_line()
                                    _clicked, dragging, dx, dy = imgui_utils.drag_button(f"Drag##_{j}_{u_id}",
                                                                                         width=self.viz.app.button_w)
                                    if dragging:
                                        self.drag(dx, dy, i, j, type="ui")
                                if trans.type == "b":
                                    changed, trans.params[j] = imgui.checkbox(f"##{j}_{u_id}",
                                                                              trans.params[j])
                        expanded, _visible = imgui_utils.collapsing_header('OSC', default=True)
                        if expanded:
                            _, trans.use_osc = imgui.checkbox(f"Use OSC##{u_id}", trans.use_osc)

                            with imgui_utils.item_width(self.viz.app.font_size * 5):

                                with imgui_utils.grayed_out(not (trans.use_osc)):
                                    for j in range(len(trans.params)):
                                        changed, address = imgui.input_text(f"##osc_{j}_{u_id}",
                                                                            trans.osc_address[j],
                                                                            256,
                                                                            imgui.INPUT_TEXT_CHARS_NO_BLANK | imgui.INPUT_TEXT_ENTER_RETURNS_TRUE |
                                                                            (
                                                                                imgui.INPUT_TEXT_READ_ONLY) * (
                                                                                not trans.use_osc))
                                        if j < len(trans.params) - 1:
                                            imgui.same_line()

                                        if changed:
                                            try:
                                                self.viz.osc_dispatcher.unmap(f"/{trans.osc_address[j]}",
                                                                              self.osc_funcs[trans.imgui_id][j])
                                                print(f"Unmapped", trans.osc_address[j])
                                                print(self.viz.osc_dispatcher.mappings)
                                            except Exception as e:
                                                print(f"{trans.osc_address[j]} is not mapped")
                                                print(e)
                                                print(self.viz.osc_dispatcher._map)
                                            self.viz.osc_dispatcher.map(f"/{address}",
                                                                        self.osc_funcs[trans.imgui_id][j])
                                            trans.osc_address[j] = address
                            imgui.separator()

        for idx in to_remove:
            del self.cached_transforms[idx]
        self.viz.args.latent_transforms = copy.deepcopy(self.cached_transforms)

    @imgui_utils.scoped_by_object_id
    def indices_widget(self, trans):
        # Widget that allows users to set which features should be manipulated
        imgui.begin_group()
        imgui.text("Application Style:")
        imgui.same_line()
        update = False
        if imgui_utils.button(f"{modes[trans.mode]}##{trans.imgui_id}", width=self.viz.app.button_w):
            trans.mode = (trans.mode + 1) % len(modes)
            update = True

        if modes[trans.mode] == "all":
            trans.indices = list(range(0, trans.max_idx))
        elif modes[trans.mode] == "random":
            _changed, p = imgui.input_float("%##{trans.imgui_id}", trans.percentage)
            trans.percentage = np.clip(p, 0, 1)
            imgui.same_line()
            if imgui_utils.button(f"randomize##{trans.imgui_id}", width=self.viz.app.button_w) or _changed or update:
                num_samples = int(trans.max_idx * trans.percentage)
                trans.indices = random.sample(range(0, trans.max_idx), num_samples)
        elif modes[trans.mode] == "range":
            _changed, (low, high) = imgui.input_int2("Range##{trans.imgui_id}", trans.low, trans.high)
            trans.high = np.clip(high, 0, trans.max_idx)
            trans.low = np.clip(low, 0, trans.high)
            trans.indices = list(range(trans.low, trans.high))
        elif modes[trans.mode] == "config":
            changed, trans.cluster_path = imgui_utils.input_text(f"Config Path##configpath_{trans.imgui_id}",
                                                                 trans.cluster_path, 256,
                                                                 width=self.viz.app.font_size * 10,
                                                                 flags=0,
                                                                 help_text="Config Path")

            imgui.same_line()

            if imgui_utils.button(f'Open##config_{trans.imgui_id}', width=self.viz.app.button_w):
                open_path(trans)
            if not (trans.cluster_config is None):
                indices = []
                for i, c_dict in enumerate(trans.cluster_config[trans.layerID]):
                    if c_dict['cluster_index'] == int(trans.cluster_ID):
                        indices.append(c_dict['feature_index'])
                if len(indices) == 0:
                    print("No indicies found for clusterID: " + str(trans.cluster_ID) + " on layer: " + str(
                        trans.layerID))
                trans.indices = indices
        imgui.separator()
        imgui.end_group()

    @imgui_utils.scoped_by_object_id
    def adjust_noise(self):
        # Widget that allows users to adjust the magnitude of the noise
        if self.cur_layer not in self.noises:
            self.noises[self.cur_layer] = {"strength": 0, "id": self.make_id(), "use_osc": False,
                                           "osc_address": "osc address", "mapping": "x"}

        if self.cur_layer in self.noises:
            with imgui_utils.item_width(self.viz.app.button_w * 2 - self.viz.app.spacing * 2):
                _changed, self.noises[self.cur_layer]["strength"] = imgui.slider_float(
                    f"##{self.noises[self.cur_layer]['id']}",
                    self.noises[self.cur_layer]['strength'], -5, 5,
                    format='Strength %.3f', power=3)
            imgui.same_line()
            if imgui_utils.button(f"Reset##{self.noises[self.cur_layer]['id']}", width=self.viz.app.button_w,
                                  enabled=self.noises[self.cur_layer]['strength'] != 0):
                self.noises[self.cur_layer]["strength"] = 0
            expanded, _visible = imgui_utils.collapsing_header('OSC', default=True)
            if expanded:
                noise = self.noises[self.cur_layer]
                _, noise["use_osc"] = imgui.checkbox(f"Use OSC##{noise['id']}", noise["use_osc"])

                with imgui_utils.item_width(self.viz.app.font_size * 5):

                    with imgui_utils.grayed_out(not (noise["use_osc"])):
                        changed, address = imgui.input_text(f"##osc_noise_{noise['id']}",
                                                            noise["osc_address"],
                                                            256,
                                                            imgui.INPUT_TEXT_CHARS_NO_BLANK |
                                                            (
                                                                imgui.INPUT_TEXT_READ_ONLY) * (
                                                                not noise["use_osc"]))

                        if changed:

                            try:
                                self.viz.osc_dispatcher.unmap(f"/{noise['osc_address']}",
                                                              self.noise_osc(noise))
                            except:
                                print(f"{noise['osc_address']} is not mapped")
                            self.viz.osc_dispatcher.map(f"/{address}",
                                                        self.noise_osc(noise))
                            noise["osc_address"] = address

                        changed, noise["mapping"] = imgui.input_text(f"##osc_noise_mapping_{noise['id']}",
                                                                     noise["mapping"], 256,
                                                                     imgui.INPUT_TEXT_ENTER_RETURNS_TRUE | (
                                                                             imgui.INPUT_TEXT_READ_ONLY * (
                                                                         not noise["use_osc"])))

    def transform_osc(self, trans, param_idx=-1):
        def func(address, *args):
            try:
                f = lambda x: eval(trans.mapping[param_idx])
                if trans.use_osc:
                    out = f(args[-1])
                    if isinstance(out, (int, float)):
                        trans.params[param_idx] = f(args[-1])
            except Exception as e:
                if trans.use_osc and isinstance(args[-1], (int, float)):
                    trans.params[param_idx] = args[-1]

        return func

    def noise_osc(self, noise):
        def func(address, *args):
            try:
                f = lambda x: eval(noise["mapping"])
                if noise["use_osc"]:
                    out = f(args[-1])
                    if isinstance(out, (int, float)):
                        noise["strength"] = f(args[-1])
            except Exception as e:
                if noise["use_osc"] and isinstance(args[-1], (int, float)):
                    noise["strength"] = args[-1]

        return func

    def grayscale_osc(self):
        def func(address, *args):
            try:
                if bool(args[-1]):
                    self.sel_channels = 1
                else:
                    self.sel_channels = 3


            except Exception as e:
                self.viz.print_error(e)

        return func

    def contrast_osc(self):
        def func(address, *args):
            try:
                self.img_scale_db = float(args[-1])
            except Exception as e:
                self.viz.print_error(e)

        return func
    
    def normalize_osc(self):
        def func(address, *args):
            try:
                self.img_normalize = bool(args[-1])
            except Exception as e:
                self.viz.print_error(e)

        return func

    def channels_osc(self):
        def func(address, *args):
            try:
                nec_type = type(self.base_channel)
                base_channel_max = max(self.capture_channels - self.sel_channels, 0)
                self.base_channel = min(nec_type(args[-1]), base_channel_max)
            except Exception as e:
                self.viz.print_error(e)

        return func
@contextlib.contextmanager
def greened_out(cond=True):
    if cond:
        # change color of button to green
        imgui.push_style_color(imgui.COLOR_BUTTON, *GREEN)
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, *GREEN)
        imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, *GREEN)
        imgui.push_style_color(imgui.COLOR_BORDER, *GREEN)
        yield
        imgui.pop_style_color(4)

    else:

        imgui.push_style_color(imgui.COLOR_BUTTON, *DARKGRAY)
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, *GRAY)
        imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, *GREEN)
        imgui.push_style_color(imgui.COLOR_BORDER, *GREEN)
        yield
        imgui.pop_style_color(4)