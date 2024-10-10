import os

import imgui
import numpy as np
import torch
import torch.nn.functional as F

from assets import ACTIVE_RED

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle
import dnnlib
from utils.gui_utils import imgui_utils
from widgets.browse_widget import BrowseWidget

class AdjusterWidget:

    def __init__(self, viz):
        self.viz = viz
        self.dirs = torch.randn(6, 512)
        self.all_dir = ""
        self.weights = torch.zeros(len(self.dirs))
        self.paths = [""] * len(self.weights)
        self.vslide_use_osc = [False] * len(self.weights)
        self.vslide_address = [""] * len(self.weights)
        self.vslide_mappings = ["x"] * len(self.weights)
        self.file_dialogs = [BrowseWidget(viz, f"Browse##vec{i}", os.path.abspath(os.getcwd()), ["*",".pth", ".pt"], width=self.viz.app.button_w, multiple=False, traverse_folders=False) for i in range(len(self.weights))]
        self.all_dialog = BrowseWidget(viz, f"Browse##vecalldialog", os.path.abspath(os.getcwd()), ["*",".pth", ".pt"], width=self.viz.app.button_w, multiple=False, traverse_folders=False)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_params(), f)

    def load(self, path):
        with open(path, "rb") as f:
            self.set_params(pickle.load(f))

    def get_params(self):
        return self.dirs, self.paths, self.all_dir,self.weights, self.vslide_use_osc, self.vslide_address, self.vslide_mappings

    def set_params(self, params):
        self.dirs, self.paths, self.all_dir, self.weights, self.vslide_use_osc, new_addresses, self.vslide_mappings = params
        for i,old_address in enumerate(self.vslide_address):
            try:
                self.viz.osc_dispatcher.unmap(old_address,
                                                self.vec_handler(i))
            except:
                print(f"{old_address} is not mapped")
        for i, new_address in enumerate(new_addresses):
            self.viz.osc_dispatcher.map(new_address,
                                          self.vec_handler(i))
        self.vslide_address = new_addresses
        self.file_dialogs = [BrowseWidget(self.viz, f"Vector##vec{i}", os.path.abspath(os.getcwd()), ["*",".pth", ".pt"], width=self.viz.app.button_w, multiple=False, traverse_folders=False) for i in range(len(self.weights))]



    def vec_handler(self, idx):
        def func(address, *args):
            try:
                print(self.weights[idx], args[-1])
                if self.vslide_use_osc[idx]:
                    print("in")
                    f = lambda x: eval(self.vslide_mappings[idx])
                    out = f(args[-1])
                    if isinstance(out, (int, float)):
                        self.weights[idx] = f(args[-1])
                    print("assigned")
            except Exception as e:
                if type(args[-1]) is type(self.weights[idx]):
                    self.weights[idx] = f(args[-1])
                print(e)
                #self.viz.print_error(e)

        return func


    def open_vecs(self):
        try:
            vecs = torch.from_numpy(np.load(self.all_dir)).squeeze()
            reshaped_vecs = vecs
            if len(reshaped_vecs.shape) == 1:
                reshaped_vecs = reshaped_vecs.unsqueeze(0)
            assert len(vecs.shape) <= 2 or vecs.shape[-1] != self.dirs.shape[-1], f"Trying to load vecs expected shape [None, self.dirs.shape[-1]] got {vecs.shape}"

            self.dirs = reshaped_vecs.to(self.dirs.dtype)
            self.weights = torch.zeros(len(self.dirs))
            self.paths = [""] * len(self.weights)
            self.vslide_use_osc = [False] * len(self.weights)
            self.vslide_address = [""] * len(self.weights)
            self.vslide_mappings = ["x"] * len(self.weights)
            self.file_dialogs = [BrowseWidget(self.viz, f"Vector##vec{i}", os.path.abspath(os.getcwd()), ["*",".pth", ".pt"], width=self.viz.app.button_w, multiple=False, traverse_folders=False) for i in range(len(self.weights))]

        except Exception as e:
            print(e)

    def open_vec(self, idx):
        try:
            vec = torch.from_numpy(np.load(self.paths[idx])).squeeze()

            assert vec.shape == self.dirs[idx].shape, f"The Tensor you are loading has a different shape, Loaded Shape {vec.shape} != Target Shape {self.dirs[idx].shape}"
            self.dirs[idx] = vec
        except Exception as e:
            print(e)

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        if show:
            remove_idx = None
            for i in range(len(self.weights)):
                imgui.begin_group()
                s = imgui.get_style()

                draw_list = imgui.get_window_draw_list()

                draw_list.channels_split(2)
                draw_list.channels_set_current(1)
                changed, self.vslide_use_osc[i] = imgui.checkbox(f"Use OSC##vslide_use_osc{i}", self.vslide_use_osc[i])
                use_osc = self.vslide_use_osc[i]

                # if use_osc draw red box under checkbox label
                if use_osc:

                    draw_list.channels_set_current(0)
                    p_min = imgui.get_item_rect_min()
                    p_max = imgui.get_item_rect_max()
                    draw_list.add_rect_filled(p_min.x + (self.viz.app.font_size * 1.5), p_min.y , p_min.x + (self.viz.app.button_w) , p_max.y, imgui.get_color_u32_rgba(*ACTIVE_RED))

                draw_list.channels_merge()

                with imgui_utils.grayed_out(not use_osc):
                    changed, new_address = imgui_utils.input_text(f"##vslide_osc{i}", self.vslide_address[i], 256,
                                                                  imgui.INPUT_TEXT_CHARS_NO_BLANK,
                                                                  width=self.viz.app.button_w,
                                                                  help_text="osc address")
                    s.item_spacing = [self.viz.app.spacing, self.viz.app.spacing]
                    s.item_inner_spacing = [self.viz.app.spacing, self.viz.app.spacing]
                    if changed:
                        self.viz.osc_dispatcher.map(f"/{new_address}",
                                                      self.vec_handler(i))
                        try:
                            self.viz.osc_dispatcher.unmap(self.vslide_address[i],
                                                            self.vec_handler(i))
                        except:
                            print(f"{self.vslide_address[i]} is not mapped")

                        self.vslide_address[i] = new_address
                    changed, self.vslide_mappings[i] = imgui_utils.input_text(f"##vslide_mapping{i}",
                                                                              self.vslide_mappings[i], 256,
                                                                              flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE,
                                                                              width=self.viz.app.button_w,
                                                                              help_text="mapping")
                changed, self.weights[i] = imgui.v_slider_float(f"##slider{i}", self.viz.app.button_w,
                                                                self.viz.app.content_height / 10, self.weights[i], -5, 5,
                                                                format='%.3f', power=3)
                changed, self.paths[i] = imgui_utils.input_text(f"##vec_path{i}", self.paths[i], 256,
                                                                imgui.INPUT_TEXT_CHARS_NO_BLANK,
                                                                width=self.viz.app.button_w, help_text="filepath")
                _clicked, files = self.file_dialogs[i](self.viz.app.button_w)
                if _clicked:
                    self.paths[i] = files[0]
                if imgui_utils.button(f"Load Vec##{i}", (self.viz.app.button_w * (6 / 8) - (self.viz.app.spacing / 2))):
                    self.open_vec(i)
                imgui.same_line()
                if imgui_utils.button(f"-##vecs{i}", self.viz.app.button_w * (2 / 8) - (self.viz.app.spacing / 2)):
                    remove_idx = i
                if imgui_utils.button(f"Randomize##{i}", self.viz.app.button_w):
                    self.dirs[i] = torch.randn(self.dirs[i].shape)

                imgui.end_group()
                if (self.viz.pane_w - imgui.get_item_rect_max()[0]) >= self.viz.app.button_w:
                    imgui.same_line()

            imgui.begin_group()
            if imgui_utils.button("+##vecs", width=self.viz.app.button_w):
                self.weights = torch.concat((self.weights, torch.tensor([0])))
                self.dirs = torch.concat((self.dirs, torch.randn(1, 512)), dim=0)
                self.paths.append("")
                self.vslide_use_osc.append(False)
                self.vslide_address.append("")
                self.vslide_mappings.append("x")
                self.file_dialogs.append(BrowseWidget(self.viz, f"Vector##vec{len(self.weights)-1}", os.path.abspath(os.getcwd()), ["*",".pth", ".pt"], width=self.viz.app.button_w, multiple=False, traverse_folders=False))
            if imgui_utils.button(f"Randomize##all", self.viz.app.button_w):
                self.dirs = torch.randn(self.dirs.shape)
            _clicked, files = self.all_dialog()
            if _clicked:
                    self.all_dir = files[0]
            if imgui_utils.button(f"Load Vecs##all", self.viz.app.button_w):
                self.open_vecs()
            imgui.end_group()

            if not remove_idx is None:
                idxs = list(range(len(self.weights)))
                idxs.pop(remove_idx)
                self.weights = self.weights[idxs]
                self.dirs = self.dirs[idxs]
                self.paths = [self.paths[i] for i in idxs]
                self.vslide_use_osc = [self.vslide_use_osc[i] for i in idxs]
                self.vslide_address = [self.vslide_address[i] for i in idxs]
                self.vslide_mappings = [self.vslide_mappings[i] for i in idxs]

        self.viz.args.update(direction=self.weights @ self.dirs)
