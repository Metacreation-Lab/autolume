import imgui

import dnnlib
from utils.gui_utils import imgui_utils
import numpy as np
import math
import torch


class OscMenu:
    def __init__(self, viz, funcs, use_map=None, label="##OSC"):
        self.viz = viz
        self.label = label
        self.funcs = funcs
        self.use_map = use_map
        if use_map is None:
            self.use_map = dict(zip(self.funcs.keys(),[True] * len(self.funcs)))
        print(self.use_map)
        self.use_osc = dnnlib.EasyDict(zip(funcs.keys(), [False] * len(funcs)))
        self.osc_addresses = dnnlib.EasyDict(zip(funcs.keys(), ["..."] * len(funcs)))
        self.cached_osc_addresses = dnnlib.EasyDict(self.osc_addresses)
        self.mappings = dnnlib.EasyDict(zip(funcs.keys(), ["x"] * len(funcs)))

        for key, func in self.funcs.items():  # maybe with map faster
            self.funcs[key] = self.check_osc(func, key)
        self.wrapped_funcs = self.funcs.copy()

    #TODO might only need key and no func since same as self.func[key]

    def get_params(self):
        return self.use_map, self.use_osc, self.osc_addresses, self.cached_osc_addresses, self.mappings

    def set_params(self, params):
        # TODO unmap old addresses
        self.use_map, self.use_osc, self.osc_addresses, self.cached_osc_addresses, self.mappings = params
        for key, func in self.funcs.items():  # maybe with map faster
            self.funcs[key] = self.check_osc(func, key)
        for key, func in self.funcs.items():
            self.wrapped_funcs[key] = self.map_func(self.funcs[key], key)

    def check_osc(self, func, key):
        def wrapper(*args, **kwargs):
            if self.use_osc[key]:
                try:
                    func(*args, **kwargs)
                except Exception as e:
                    print(e)

        return wrapper

    def map_func(self, func, key):
        def wrapper(*args, **kwargs):
            try:
                f = lambda x: eval(self.mappings[key])
                func(args[0], f(args[-1]))
            except Exception as e:
                print(e)
                func(*args)

        return wrapper

    @imgui_utils.scoped_by_object_id
    def osc_item(self, key):
        viz = self.viz
        _, self.use_osc[key] = imgui.checkbox(f"Use OSC##{self.label}_{key}", self.use_osc[key])
        with imgui_utils.grayed_out(not self.use_osc[key]):
            changed, self.osc_addresses[key] = imgui.input_text(f"##OSCAddress_{self.label}_{key}",
                                                                self.osc_addresses[key], 256,
                                                                imgui.INPUT_TEXT_CHARS_NO_BLANK | imgui.INPUT_TEXT_ENTER_RETURNS_TRUE | (
                                                                        imgui.INPUT_TEXT_READ_ONLY * (
                                                                    not self.use_osc[key])))
            if changed:
                viz.osc_dispatcher.map(self.osc_addresses[key], self.wrapped_funcs[key])
                try:
                    viz.osc_dispatcher.unmap(self.cached_osc_addresses[key], self.wrapped_funcs[key])
                except:
                    print(self.cached_osc_addresses[key], "is not mapped")
                self.cached_osc_addresses[key] = self.osc_addresses[key]

            if self.use_map[key]:
                changed, self.mappings[key] = imgui.input_text(f"##Mappings_{self.label}_{key}",
                                                               self.mappings[key], 256,
                                                               imgui.INPUT_TEXT_ENTER_RETURNS_TRUE | (
                                                                       imgui.INPUT_TEXT_READ_ONLY * (
                                                                   not self.use_osc[key])))

                if changed:
                    try:
                        viz.osc_dispatcher.unmap(self.osc_addresses[key], self.wrapped_funcs[key])
                    except:
                        print(self.cached_osc_addresses[key], "is not mapped")
                    self.wrapped_funcs[key] = self.map_func(self.funcs[key], key)
                    viz.osc_dispatcher.map(self.osc_addresses[key], self.wrapped_funcs[key])

    @imgui_utils.scoped_by_object_id
    def __call__(self):
        viz = self.viz
        imgui.begin_child(self.label, viz.pane_w, viz.app.font_size*1.5,
                          flags=imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_MENU_BAR)
        if imgui.begin_menu_bar():
            imgui.text("Osc Menu |")
            for key in self.funcs.keys():
                if imgui.begin_menu(key, True):
                    self.osc_item(key)
                    imgui.end_menu()
            imgui.end_menu_bar()

        imgui.end_child()
