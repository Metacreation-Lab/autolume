import copy

import imgui
import numpy as np

import dnnlib
from utils.gui_utils import imgui_utils

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

import glob
import os
import re

from widgets import browse_widget


def _locate_results(pattern):
    return pattern

def extract_conv_names(model):
    model_names = [name for name, weight in model.named_parameters() if "mapping" not in name]
    return model_names

def extract_mapping_names(model):
    model_names = [name for name, weight in model.named_parameters() if "mapping" in name]
    return model_names

def resolve_pkl(pattern):
        print("RESOLVE", pattern)
        assert isinstance(pattern, str)
        assert pattern != ''

        # URL => return as is.
        if dnnlib.util.is_url(pattern):
            return pattern

        # Short-hand pattern => locate.
        path = _locate_results(pattern)

        # Run dir => pick the last saved snapshot.
        if os.path.isdir(path):
            pkl_files = sorted(glob.glob(os.path.join(path, 'network-snapshot-*.pkl')))
            if len(pkl_files) == 0:
                raise IOError(f'No network pickle found in "{path}"')
            path = pkl_files[-1]

        # Normalize.
        path = os.path.abspath(path)
        return path

class MixingWidget:
    def __init__(self, viz):
        self.viz = viz
        self.output_name = ""
        self.model_pth = ""
        self.main_model = ""
        self.browse_refocus = False
        self.layer1 = []
        self.layer2 = []
        self.output_name = ""
        self._save = False

        self.models = []
        self.combined_layers = []
        self.collapsed = []
        self.cached_layers = []
        self.mixing = False

        self.browser = browse_widget.BrowseWidget(viz, "Find", ".", [".pkl"], width=self.viz.app.button_w,
                                                  multiple=False, traverse_folders=False)

        for pkl in os.listdir("./models"):
            if pkl.endswith(".pkl"):
                self.models.append(os.path.join(os.getcwd(),"models",pkl))

    @imgui_utils.scoped_by_object_id
    def __call__(self, show):
        if show:
            _, self.mixing = imgui.checkbox("##mixingwidget_ckb", self.mixing)
            imgui.same_line()
            model_changed, self.model_pth = imgui_utils.input_text(f'##network_mixing_widget', self.model_pth , 1024,
                                                        flags=(
                                                                    imgui.INPUT_TEXT_AUTO_SELECT_ALL | imgui.INPUT_TEXT_ENTER_RETURNS_TRUE),
                                                        width=(100),
                                                        help_text='<PATH> | <URL> | <RUN_DIR> | <RUN_ID> | <RUN_ID>/<KIMG>.pkl')


            imgui.same_line()
            if imgui_utils.button(f'Models##mixingWidget', enabled=len(self.models) > 0):
                imgui.open_popup(f'browse_pkls_popup##mixingWidget')
                self.browse_refocus = True

            imgui.same_line()
            _clicked, pkl = self.browser()
            if _clicked:
                print("SELECTED", pkl)
                self.model_pth = resolve_pkl(pkl[0])
                model_changed = True

            if imgui.begin_popup(f'browse_pkls_popup##mixingWidget'):
                for pkl in self.models:
                    clicked, _state = imgui.menu_item(pkl)
                    if clicked:
                        self.model_pth = resolve_pkl(pkl)
                        model_changed = True
                if self.browse_refocus:
                    imgui.set_scroll_here()
                    self.browse_refocus = False

                imgui.end_popup()
            
            imgui.same_line()
            with imgui_utils.item_width(self.viz.app.button_w):
                _, self.output_name = imgui_utils.input_text("##network_mixing_pkl", self.output_name, 1024, help_text="Name of the output model",
                                    flags=(imgui.INPUT_TEXT_AUTO_SELECT_ALL ))
            imgui.same_line()
            imgui.text(".pkl")
            imgui.same_line()
            if imgui_utils.button("Save##mixing widget", enabled=self.output_name != ""):
                print("saving at", self.output_name)
                self._save = True
            
            imgui.separator()
            if "g2_layers" in self.viz.result and "g1_layers" in self.viz.result:
                layers1 = self.viz.result.g1_layers
                layers2 = self.viz.result.g2_layers

                if self.viz.args.pkl != self.main_model or model_changed or layers1 != self.layer1 or layers2 != self.layer2:
                    self.layer1 = layers1
                    self.layer2 = layers2
                    print("reinitatilzation")
                    self.main_model = self.viz.args.pkl
                    self.combined_layers = ["A"] * len(layers1)
                    if len(layers2) > len(layers1):
                        self.combined_layers = self.combined_layers + ["B"] * (len(layers2) - len(layers1))

                    self.cached_layers = copy.deepcopy(self.combined_layers)

                    self.collapsed = [">"] * len(self.combined_layers)
                if len(layers1) > len(layers2):
                    layers2 = layers2 + [''] * (len(layers1) - len(layers2))
                elif len(layers2) > len(layers1):
                    layers1 = layers1 + [''] * (len(layers2) - len(layers1))

                self.display_layers(layers1, layers2)
        self.viz.args.combined_layers = self.combined_layers
        self.viz.args.pkl2 = self.model_pth
        self.viz.args.mixing = self.mixing
        self.viz.args.save_model = self._save
        self.viz.args.save_path = self.output_name
        self._save = False
    
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_params(), f)

    def load(self, path):
        with open(path, "rb") as f:
            self.set_params(pickle.load(f))
    
    def get_params(self):
        return self.output_name, self.model_pth, self.layer1, self.layer2, self.combined_layers, self.collapsed, self.cached_layers, self.mixing  
    
    def set_params(self, params):
        self.output_name, self.model_pth, self.layer1, self.layer2, self.combined_layers, self.collapsed, self.cached_layers, self.mixing = params

    def display_layers(self, layer1, layer2):
        imgui.begin_group()
        imgui.set_cursor_pos((imgui.get_content_region_available_width() // 3, imgui.get_cursor_pos()[1]))
        imgui.text(os.path.basename(self.viz.args.pkl))
        imgui.same_line(imgui.get_content_region_available_width() // 3 * 2)
        imgui.text(os.path.basename(self.model_pth))
        imgui.separator()
        res_exp = 2
        for i, (l1, l2) in enumerate(zip(layer1, layer2)):
            resolution = 2 ** res_exp
            l1_res = 0
            l2_res = 0
            if l1:
                l1_res = int(re.search(r'\d+', l1).group())
            if l2:
                l2_res = int(re.search(r'\d+', l2).group())

            if l1_res == resolution or l2_res == resolution:
                imgui.begin_child(f"##{resolution}_global", 0, 14 * 16 if self.collapsed[i] == "v" else 14 * 2.3, border=True,
                                  flags= (imgui.WINDOW_NO_SCROLLBAR|imgui.WINDOW_NO_SCROLL_WITH_MOUSE) if self.collapsed[i] == ">" else 0)
                imgui.text(self.collapsed[i])
                if imgui.is_item_clicked():
                    self.collapsed[i] = ">" if self.collapsed[i] == "v" else "v"
                imgui.same_line()
                imgui.text(f"Resolution {resolution} x {resolution}")
                if imgui.is_item_clicked():
                    self.collapsed[i] = ">" if self.collapsed[i] == "v" else "v"
                if self.collapsed[i] == ">":
                    # Check if all the layers with the res in the name are already set to Model A or Model B and display if so otherwise display mixed
                    layer1_matches, layer2_matches = [], []
                    for k, (layer1tmp, layer2_tmp) in enumerate(zip(layer1, layer2)):
                        if layer1tmp:
                            if int(re.search(r'\d+', layer1tmp).group()) == resolution:
                                layer1_matches += [k]
                        if layer2_tmp:
                            if int(re.search(r'\d+', layer2_tmp).group()) == resolution:
                                layer2_matches += [k]
                    ckb_display = "A"
                    if all(np.array(self.combined_layers)[layer1_matches] == "A") and len(np.array(self.combined_layers)[layer1_matches]):
                        ckb_display = "A"
                    elif all(np.array(self.combined_layers)[layer2_matches] == "B") and len(np.array(self.combined_layers)[layer2_matches]):
                        ckb_display = "B"
                    elif all(np.array(self.combined_layers)[layer2_matches] == "X") or all(np.array(self.combined_layers)[layer1_matches] == "X"):
                        ckb_display = "X"
                    else:
                        ckb_display = "Mixed"
                    imgui.same_line(imgui.get_content_region_available_width() // 3)
                    
                    with imgui_utils.grayed_out(l1 == '' or ckb_display == "X"):
                        clicked, _ = imgui.checkbox(f"##layer1{i}", (ckb_display == "A" or ckb_display == "Mixed") and layer1[i] != '')
                    if clicked and layer1[i] != '' and ckb_display != "X":
                        print("clicked1")
                        self.combined_layers[i] = "A"
                        for j in range(i + 1, len(self.combined_layers)):
                            if layer1[j]:
                                res = int(re.search(r'\d+', layer1[j]).group())
                                if res == resolution and layer1[j]:
                                    self.combined_layers[j] = "A"
                    imgui.same_line(imgui.get_content_region_available_width() // 3 * 2)
                    with imgui_utils.grayed_out(l2=='' or ckb_display == "X"):
                        clicked, _ = imgui.checkbox(f"##layer2{i}", ckb_display == "B" or ckb_display == "Mixed" and layer2[i] != '')
                    if clicked and layer2[i] != ''and ckb_display != "X":
                        print("clicked2")
                        self.combined_layers[i] = "B"
                        for j in range(i + 1, len(self.combined_layers)):
                            if layer2[j]:
                                res = int(re.search(r'\d+', layer2[j]).group())
                                if res == resolution and layer2[j]:
                                    self.combined_layers[j] = "B"
                    imgui.same_line(imgui.get_content_region_available_width() -self.viz.app.button_w)
                    if self.combined_layers[i] == "X":
                        if imgui.button(f"Recover##{i}"):
                            # find last entry in l1 or l2 that has the same resolution as resolution and copy all the layers from self.cached to self.combined up to that point
                            for j, (l1, l2) in enumerate(zip(layer1, layer2)):
                                res1 = 0
                                res2 = 0
                                if l1:
                                    res1 = int(re.search(r'\d+', l1).group())
                                if l2:
                                    res2 = int(re.search(r'\d+', l2).group())
                                if res1 == resolution * 2 or res2 == resolution * 2:
                                    self.combined_layers[:j] = copy.deepcopy(self.cached_layers[:j])
                                    break
                                # deal with the last resolution
                                if j == len(layer1) - 1:
                                    self.combined_layers = copy.deepcopy(self.cached_layers)
                    else:
                        if imgui.button(f"X##{i}", width=self.viz.app.font_size * 1.5, height=self.viz.app.font_size * 1.5):
                            self.cached_layers[:i] = copy.deepcopy(self.combined_layers[:i])
                            self.combined_layers[i] = "X"
                            self.combined_layers[i + 1:] = ["X"] * (len(self.combined_layers) - i - 1)
                else:
                    imgui.same_line(imgui.get_content_region_available_width() - self.viz.app.button_w + imgui.get_style().scrollbar_size)
                    if self.combined_layers[i] ==  "X":
                        if imgui.button(f"Recover##{i}"):
                            # find last entry in l1 or l2 that has the same resolution as resolution and copy all the layers from self.cached to self.combined up to that point
                            for j, (l1, l2) in enumerate(zip(layer1, layer2)):
                                res1 = 0
                                res2 = 0
                                if l1:
                                    res1 = int(re.search(r'\d+', l1).group())
                                if l2:
                                    res2 = int(re.search(r'\d+', l2).group())
                                if res1 == resolution * 2 or res2 == resolution * 2:
                                    self.combined_layers[:j] = copy.deepcopy(self.cached_layers[:j])
                                    break
                                # deal with the last resolution
                                if j == len(layer1) - 1:
                                    self.combined_layers = copy.deepcopy(self.cached_layers)
                    else:
                        if imgui.button(f"X##{i}", width=self.viz.app.font_size * 1.5, height=self.viz.app.font_size * 1.5):
                            self.cached_layers[:i] = copy.deepcopy(self.combined_layers[:i])
                            self.combined_layers[i] = "X"
                            self.combined_layers[i + 1:] = ["X"] * (len(self.combined_layers) - i - 1)
                    for it, (l1t, l2t) in enumerate(zip(layer1, layer2)):
                        l1t_res = 0
                        l2t_res = 0
                        if l1t:
                            l1t_res = int(re.search(r'\d+', l1t).group())
                        if l2t:
                            l2t_res = int(re.search(r'\d+', l2t).group())
                        if l1t_res == resolution or l2t_res == resolution:
                            imgui.text(l1t if l1t else l2t)
                            imgui.same_line((imgui.get_content_region_available_width() // 3) + imgui.get_style().scrollbar_size - imgui.get_style().item_spacing[0] * 2)
                            with imgui_utils.grayed_out(l1t == '' or self.combined_layers[it]=="X"):
                                clicked, _ = imgui.checkbox(f"##layer1{i}{it}", self.combined_layers[it] == "A")
                            if clicked and l1t != '' and not(self.combined_layers[it]=="X"):
                                print("clicked1")
                                self.combined_layers[it] = "A"
                            imgui.same_line((imgui.get_content_region_available_width() // 3 * 2) + imgui.get_style().scrollbar_size - imgui.get_style().item_spacing[0])
                            with imgui_utils.grayed_out(l2t == '' or self.combined_layers[it]=="X"):
                                clicked, _ = imgui.checkbox(f"##layer2{i}{it}", self.combined_layers[it]=="B")
                            if clicked and l2t != '' and not(self.combined_layers[it]=="X"):
                                print("clicked2")
                                self.combined_layers[it] = "B"

                imgui.end_child()
                res_exp += 1
        imgui.end_group()
