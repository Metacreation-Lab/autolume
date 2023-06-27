import copy

import imgui
import numpy as np

import dnnlib
from torch_utils import legacy
from architectures import custom_stylegan2
from utils.gui_utils import imgui_utils
import pickle

import glob
import os
import re


def _locate_results(pattern):
    return pattern


def extract_conv_names(model):
    model_names = [name for name, weight in model.named_parameters() if "mapping" not in name]
    return model_names


def extract_mapping_names(model):
    model_names = [name for name, weight in model.named_parameters() if "mapping" in name]
    return model_names


class MixingModule:

    def __init__(self, menu):
        self.output_name = ""
        self._networks = dict()
        self._pkl_data = dict()
        self.menu = menu
        self.app = menu.app
        self.model1 = ""
        self.pkl1 = None
        self.data1 = None
        self.model2 = ""
        self.pkl2 = None
        self.data2 = None

        self.browse_refocus = False

        self.models = []
        self.combined_layers = []
        self.collapsed = []
        self.cached_layers = []

        for pkl in os.listdir("./models"):
            if pkl.endswith(".pkl"):
                self.models.append(os.path.join(os.getcwd(), "models", pkl))

    def load_pkl(self, pkl, m, ignore_errors=False):
        print("loading---------")
        menu = self.menu
        try:
            resolved = self.resolve_pkl(pkl)
            name = resolved.replace('\\', '/').split('/')[-1]
            if m == 1:
                self.model1 = resolved
            else:
                self.model2 = resolved
        except Exception as e:
            print("error", e)
            self.cur_pkl = None
            self.user_pkl = pkl
            if pkl != '':
                imgui.open_popup('Error##pkl')

            if imgui.begin_popup('Error##pkl'):
                imgui.text(f'Failed to load network pickle because of the following error: {e}')
                imgui.end_popup()

            if not ignore_errors:
                raise

            return

        imgui.set_next_window_position(self.menu.app.content_width // 4, self.menu.app.content_height // 8)
        imgui.begin("Loading Window", False,
                    flags=imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_COLLAPSE)
        imgui.text(f"Loading model...{name}")
        imgui.end()
        print("loading model starting get function, should be showing a window", name)
        net, data = self.get_network(resolved, 'G_ema')

        # if net is an exception then show a popup with the error
        if isinstance(net, Exception):
            imgui.open_popup('Error##pkl')
            print("error", net)
            if imgui.begin_popup('Error##pkl'):
                imgui.text(f'Failed to load network pickle because of the following error: {net}')
                imgui.end_popup()
            return

        print("Net is:", net)
        if m == 1:
            self.pkl1 = net
            self.data1 = data
        else:
            self.pkl2 = net
            self.data2 = data

    def model_selection_gui(self, title, m=1):
        imgui.begin_group()
        imgui.text(title)
        imgui.separator()
        if m == 1:
            model = self.model1
        else:
            model = self.model2
        changed, model = imgui_utils.input_text(f'##surgery{m}', model, 1024,
                                                flags=(
                                                        imgui.INPUT_TEXT_AUTO_SELECT_ALL | imgui.INPUT_TEXT_ENTER_RETURNS_TRUE),
                                                width=(100),
                                                help_text='<PATH> | <URL> | <RUN_DIR> | <RUN_ID> | <RUN_ID>/<KIMG>.pkl')
        if changed:
            if m == 1:
                self.model1 = model
            else:
                self.model2 = model
            self.load_pkl(self.model1 if m == 1 else self.model2, m, ignore_errors=True)

        imgui.same_line()
        if imgui_utils.button(f'Browse...##{m}', enabled=len(self.models) > 0, width=self.app.button_w):
            imgui.open_popup(f'browse_pkls_popup##{m}')
            self.browse_refocus = True

        if imgui.begin_popup(f'browse_pkls_popup##{m}'):
            for pkl in self.models:
                clicked, _state = imgui.menu_item(pkl)
                if clicked:
                    self.load_pkl(pkl, m, ignore_errors=True)
            if self.browse_refocus:
                imgui.set_scroll_here()
                self.browse_refocus = False

            imgui.end_popup()

        imgui.end_group()

    def get_network(self, pkl, key, **tweak_kwargs):
        data = self._pkl_data.get(pkl, None)
        if data is None:
            print(f'Loading "{pkl}"... ', end='', flush=True)
            try:
                with dnnlib.util.open_url(pkl, verbose=False) as f:
                    data = legacy.load_network_pkl(f, custom=True)
            except Exception as e:
                return e
        orig_net = data[key]
        cache_key = (orig_net, tuple(sorted(tweak_kwargs.items())))
        net = self._networks.get(cache_key, None)
        if net is None:
            try:
                net = copy.deepcopy(orig_net)
                net = net.eval().requires_grad_(False)
            except Exception as e:
                return e
            self._networks[cache_key] = net
        return net, data

    def resolve_pkl(self, pattern):
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

    @imgui_utils.scoped_by_object_id
    def __call__(self):
        self.model_selection_gui("Model 1", 1)
        imgui.same_line()
        self.model_selection_gui("Model 2", 2)
        if imgui_utils.button("Combine", enabled=self.pkl1 and self.pkl2,
                              width=imgui.get_content_region_available_width()):
            imgui.set_next_window_size(800, 800)
            imgui.open_popup("Combine")

            layers1 = extract_conv_names(self.pkl1)
            layers2 = extract_conv_names(self.pkl2)

            self.combined_layers = ["A"] * len(layers1)
            if len(layers2) > len(layers1):
                self.combined_layers = self.combined_layers + ["B"] * (len(layers2) - len(layers1))

            self.cached_layers = copy.deepcopy(self.combined_layers)

            self.collapsed = [">"] * len(self.combined_layers)

        if imgui.begin_popup_modal("Combine", flags=imgui.WINDOW_NO_RESIZE)[0]:
            layer1 = extract_conv_names(self.pkl1)
            layer2 = extract_conv_names(self.pkl2)

            # pad the smaller list with empty strings
            if len(layer1) > len(layer2):
                layer2 = layer2 + [''] * (len(layer1) - len(layer2))
            elif len(layer2) > len(layer1):
                layer1 = layer1 + [''] * (len(layer2) - len(layer1))

            self.display_layers(layer1, layer2)

            imgui.separator()

            _, self.output_name = imgui_utils.input_text("##network_mixing_pkl", self.output_name, 1024,
                                                         help_text="Name of the output model",
                                                         flags=(imgui.INPUT_TEXT_AUTO_SELECT_ALL))
            imgui.same_line()
            imgui.text(".pkl")
            imgui.same_line()
            if imgui_utils.button("Combine", enabled=self.output_name != ""):
                self.combine_models()
                imgui.close_current_popup()
            if imgui.button("Close"):
                imgui.close_current_popup()

            imgui.end_popup()

    def display_layers(self, layer1, layer2):
        imgui.begin_group()
        imgui.set_cursor_pos((imgui.get_content_region_available_width() // 3, imgui.get_cursor_pos()[1]))
        imgui.text(os.path.basename(self.model1))
        imgui.same_line(imgui.get_content_region_available_width() // 3 * 2)
        imgui.text(os.path.basename(self.model2))
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
                imgui.begin_child(f"##{resolution}_global", 0, 14 * 16 if self.collapsed[i] == "v" else 14 * 2.3,
                                  border=True,
                                  flags=imgui.WINDOW_NO_SCROLLBAR if self.collapsed[i] == ">" else 0)
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
                    if all(np.array(self.combined_layers)[layer1_matches] == "A"):
                        ckb_display = "A"
                    elif all(np.array(self.combined_layers)[layer2_matches] == "B") and len(
                            np.array(self.combined_layers)[layer2_matches]):
                        ckb_display = "B"
                    elif all(np.array(self.combined_layers)[layer2_matches] == "X") or all(
                            np.array(self.combined_layers)[layer1_matches] == "X"):
                        ckb_display = "X"
                    else:
                        ckb_display = "Mixed"
                    imgui.same_line(imgui.get_content_region_available_width() // 3)

                    with imgui_utils.grayed_out(l1 == '' or ckb_display == "X"):
                        clicked, _ = imgui.checkbox(f"##layer1{i}", ckb_display == "A" or ckb_display == "Mixed")
                    if clicked and layer1[i] != '' and ckb_display != "X":
                        print("clicked1")
                        self.combined_layers[i] = "A"
                        for j in range(i + 1, len(self.combined_layers)):
                            if layer1[j]:
                                res = int(re.search(r'\d+', layer1[j]).group())
                                if res == resolution and layer1[j]:
                                    self.combined_layers[j] = "A"
                    imgui.same_line(imgui.get_content_region_available_width() // 3 * 2)
                    with imgui_utils.grayed_out(l2 == '' or ckb_display == "X"):
                        clicked, _ = imgui.checkbox(f"##layer2{i}", ckb_display == "B" or ckb_display == "Mixed")
                    if clicked and layer2[i] != '' and ckb_display != "X":
                        print("clicked2")
                        self.combined_layers[i] = "B"
                        for j in range(i + 1, len(self.combined_layers)):
                            if layer2[j]:
                                res = int(re.search(r'\d+', layer2[j]).group())
                                if res == resolution and layer2[j]:
                                    self.combined_layers[j] = "B"
                    imgui.same_line(imgui.get_window_width() - self.app.button_w)
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
                        if imgui.button(f"X##{i}"):
                            self.cached_layers[:i] = copy.deepcopy(self.combined_layers[:i])
                            self.combined_layers[i] = "X"
                            self.combined_layers[i + 1:] = ["X"] * (len(self.combined_layers) - i - 1)
                else:
                    imgui.same_line(imgui.get_window_width() - self.app.button_w)
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
                        if imgui.button(f"X##{i}"):
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
                            imgui.same_line(imgui.get_content_region_available_width() // 3)
                            with imgui_utils.grayed_out(l1t == '' or self.combined_layers[it] == "X"):
                                clicked, _ = imgui.checkbox(f"##layer1{i}{it}", self.combined_layers[it] == "A")
                            if clicked and l1t != '' and not (self.combined_layers[it] == "X"):
                                print("clicked1")
                                self.combined_layers[it] = "A"
                            imgui.same_line(imgui.get_content_region_available_width() // 3 * 2)
                            with imgui_utils.grayed_out(l2t == '' or self.combined_layers[it] == "X"):
                                clicked, _ = imgui.checkbox(f"##layer2{i}{it}", self.combined_layers[it] == "B")
                            if clicked and l2t != '' and not (self.combined_layers[it] == "X"):
                                print("clicked2")
                                self.combined_layers[it] = "B"

                imgui.end_child()
                res_exp += 1
        imgui.end_group()

    def combine_models(self):
        # find last entry in list that is not "" or "X" which might not be the last entry of the list
        last_index = 0
        for i, entry in enumerate(self.combined_layers):
            if entry != "" and entry != "X":
                last_index = i
        last_entry = {self.combined_layers[last_index]: last_index}

        layer1 = extract_conv_names(self.pkl1)
        layer2 = extract_conv_names(self.pkl2)
        if last_entry.keys() == {"A"}:
            # get resolution through regex from last entry
            img_resolution = int(re.search(r'\d+', layer1[last_entry["A"]]).group())
        elif last_entry.keys() == {"B"}:
            img_resolution = int(re.search(r'\d+', layer2[last_entry["B"]]).group())
        else:
            raise ValueError("Last entry should be either A or B but is: ", last_entry)

        model_out = custom_stylegan2.Generator(z_dim=self.pkl1.z_dim, w_dim=self.pkl1.w_dim, c_dim=self.pkl1.c_dim,
                                               img_channels=self.pkl1.img_channels,
                                               img_resolution=img_resolution)

        dict_dest = model_out.state_dict()
        # depending on what model is used in the first entry extract the mapping layers from the corresponding model and copy them to the new model
        if self.combined_layers[0] == "A":
            print("MAPPING A")
            mapping_names = extract_mapping_names(self.pkl1)
            for name in mapping_names:
                dict_dest[name] = self.pkl1.state_dict()[name]
        elif self.combined_layers[0] == "B":
            print("MAPPING B")
            mapping_names = extract_mapping_names(self.pkl2)
            for name in mapping_names:
                dict_dest[name] = self.pkl2.state_dict()[name]

        # iterate over self.combine_channels and copy weights from self.pkl1 or self.pkl2 depending on the value
        for i, entry in enumerate(self.combined_layers):
            if entry == "A":
                print("A")
                dict_dest[layer1[i]] = self.pkl1.state_dict()[layer1[i]]
            elif entry == "B":
                print("B")
                dict_dest[layer2[i]] = self.pkl2.state_dict()[layer2[i]]

        model_out_dict = model_out.state_dict()
        model_out_dict.update(dict_dest)
        model_out.load_state_dict(dict_dest)
        print("Saving model...")
        data = dict([('G', None), ('D', None), ('G_ema', None)])

        with open(os.path.join(os.getcwd(), "models", self.output_name + ".pkl"), 'wb') as f:
            data['G_ema'] = model_out
            data['G'] = model_out
            data['D'] = self.data2['D']
            pickle.dump(data, f)

