import copy

import imgui

import dnnlib
import legacy
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

class SurgeryModule:
    """
    Allows users to combine two networks by copying layers of either model to create a new model.
    """
    def __init__(self, menu):
        self.output_name = ""
        self._networks = dict()
        self._pkl_data = dict()
        self.menu = menu
        self.model1 = ""
        self.pkl1 = None
        self.data1 = None
        self.model2 = ""
        self.pkl2 = None
        self.data2 = None

        self.browse_refocus = False

        self.models = []
        self.combined_layers = []
        self.cached_layers = []

        self.show_interface = False

        for pkl in os.listdir("./models"):
            if pkl.endswith(".pkl"):
                self.models.append(os.path.join(os.getcwd(),"models",pkl))


    def load_pkl(self, pkl, m,ignore_errors=False):
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



    def model_selection_gui(self, title, m = 1):
        imgui.begin_group()
        imgui.text(title)
        imgui.separator()
        if m == 1:
            model = self.model1
        else:
            model = self.model2
        changed, model = imgui_utils.input_text(f'##surgery{m}', model , 1024,
                                                        flags=(
                                                                    imgui.INPUT_TEXT_AUTO_SELECT_ALL | imgui.INPUT_TEXT_ENTER_RETURNS_TRUE),
                                                        width=(100),
                                                        help_text='<PATH> | <URL> | <RUN_DIR> | <RUN_ID> | <RUN_ID>/<KIMG>.pkl')
        if changed:
            if m == 1:
                self.model1 = model
            else:
                self.model2 = model
            self.load_pkl(self.model1 if m == 1 else self.model2, m,ignore_errors=True)

        imgui.same_line()
        if imgui_utils.button(f'Browse...##{m}', enabled=len(self.models) > 0):
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
        imgui.same_line()
        if imgui_utils.button("Combine", enabled=self.pkl1 and self.pkl2):
            self.show_interface = True

            layers1 = extract_conv_names(self.pkl1)
            layers2 = extract_conv_names(self.pkl2)

            self.combined_layers = ["Model A"] * len(layers1)
            if len(layers2) > len(layers1):
                self.combined_layers = self.combined_layers + ["Model B"] * (len(layers2) - len(layers1))

            self.cached_layers = copy.deepcopy(self.combined_layers)

        if self.show_interface:
            imgui.set_next_window_position(self.menu.app.content_width//4, self.menu.app.content_height// 8)
            imgui.set_next_window_size(self.menu.app.content_width//2, (self.menu.app.content_height * 3) // 4)
            imgui.begin("Combine Models", False, flags=imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_COLLAPSE)

            layer1 = extract_conv_names(self.pkl1)
            layer2 = extract_conv_names(self.pkl2)

            # pad the smaller list with empty strings
            if len(layer1) > len(layer2):
                layer2 = layer2 + [''] * (len(layer1) - len(layer2))
            elif len(layer2) > len(layer1):
                layer1 = layer1 + [''] * (len(layer2) - len(layer1))


            for i, (l1, l2) in enumerate(zip(layer1, layer2)):
                if self.combined_layers[i] != "" and self.combined_layers[i] != "X":
                    imgui.text_colored(l1, 0.0, 1.0, 0.0, 1.0)
                    imgui.same_line()
                    if imgui.button(f">##{i}") and l1:
                        self.combined_layers[i] = "Model A"
                    imgui.same_line()
                    if self.combined_layers[i] == "Model A":
                        imgui.text_colored("Model A", 0.0, 1.0, 0.0, 1.0)
                    elif self.combined_layers[i] == "Model B":
                        imgui.text_colored("Model B", 0.0, 0.0, 1.0, 1.0)
                    else:
                        imgui.text_colored("None", 1.0, 0.0, 0.0, 1.0)
                    imgui.same_line()
                    if imgui.button(f"<##{i}") and l2:
                        self.combined_layers[i] = "Model B"
                    imgui.same_line()
                    imgui.text_colored(l2, 0.0, 0.0, 1.0, 1.0)
                    if "torgb.affine.bias" in l1 and i < len(self.combined_layers) - 1:
                        imgui.same_line()
                        if self.combined_layers[i+1] == "X":
                            if imgui.button(f"Recover##{i}"):
                                self.combined_layers = copy.deepcopy(self.cached_layers)
                        else:
                            if imgui.button(f"X##{i}"):
                                self.cached_layers[:i] = copy.deepcopy(self.combined_layers[:i])
                                self.combined_layers[i+1] = "X"
                                self.combined_layers[i+2:] = [""] * (len(self.combined_layers) - i - 2)

            if imgui.button("Close"):
                self.show_interface = False

            _, self.output_name = imgui_utils.input_text("Output Name", self.output_name, 1024, help_text="Name of the output model",
                                   flags=(imgui.INPUT_TEXT_AUTO_SELECT_ALL ))
            imgui.same_line()
            if imgui_utils.button("Combine", enabled=self.output_name != ""):
                self.combine_models()
                self.show_interface = False

            imgui.end()

    def combine_models(self):
        # find last entry in list that is not "" or "X" which might not be the last entry of the list
        last_index = 0
        for i, entry in enumerate(self.combined_layers):
            if entry != "" and entry != "X":
                last_index = i
        last_entry = {self.combined_layers[last_index]: last_index}


        layer1 = extract_conv_names(self.pkl1)
        layer2 = extract_conv_names(self.pkl2)
        print(last_entry)
        if last_entry.keys() == {"Model A"}:
            # get resolution through regex from last entry
            img_resolution = int(re.search(r'\d+', layer1[last_entry["Model A"]]).group())
        elif last_entry.keys() == {"Model B"}:
            img_resolution = int(re.search(r'\d+', layer2[last_entry["Model B"]]).group())
        else:
            raise ValueError("Last entry should be either Model A or Model B but is: ", last_entry)

        model_out = custom_stylegan2.Generator(z_dim=self.pkl1.z_dim, w_dim=self.pkl1.w_dim, c_dim=self.pkl1.c_dim, img_channels=self.pkl1.img_channels,
                                       img_resolution=img_resolution)

        dict_dest = model_out.state_dict()
        # depending on what model is used in the first entry extract the mapping layers from the corresponding model and copy them to the new model
        if self.combined_layers[0] == "Model A":
            print("MAPPING A")
            mapping_names = extract_mapping_names(self.pkl1)
            for name in mapping_names:
                dict_dest[name] = self.pkl1.state_dict()[name]
        elif self.combined_layers[0] == "Model B":
            print("MAPPING B")
            mapping_names = extract_mapping_names(self.pkl2)
            for name in mapping_names:
                dict_dest[name] = self.pkl2.state_dict()[name]

        # iterate over self.combine_channels and copy weights from self.pkl1 or self.pkl2 depending on the value
        print(self.combined_layers)
        for i, entry in enumerate(self.combined_layers):
            if entry == "Model A":
                print("A")
                dict_dest[layer1[i]] = self.pkl1.state_dict()[layer1[i]]
            elif entry == "Model B":
                print("B")
                dict_dest[layer2[i]] = self.pkl2.state_dict()[layer2[i]]

        model_out_dict = model_out.state_dict()
        model_out_dict.update(dict_dest)
        model_out.load_state_dict(dict_dest)
        print("Saving model...")
        data = dict([('G', None), ('D', None), ('G_ema', None)])

        with open(os.path.join(os.getcwd(),"models",self.output_name), 'wb') as f:
            data['G_ema'] = model_out
            data['G'] = model_out
            data['D'] = self.data2['D']
            pickle.dump(data, f)
