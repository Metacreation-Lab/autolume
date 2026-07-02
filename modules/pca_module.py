import glob
import os
from pathlib import Path
import multiprocessing as mp

import imgui
import numpy as np
import torch.cuda
import pandas as pd

import dnnlib
from torch_utils import legacy
from utils import device_utils
from utils.app_logging import LoggedProcess
from utils.gui_utils import imgui_utils
from ganspace.extract_pca import fit
from widgets.model_download_widget import ModelDropdownButton
from widgets.native_browser_widget import NativeBrowserWidget
from widgets.help_icon_widget import HelpIconWidget
from utils.user_data import data_path


def _locate_results(pattern):
    return pattern


pca_modes = ['pca', 'ipca', 'fbpca', "ica", 'spca']


class PCA_Module:
    def __init__(self, menu):
        self.help_icon = HelpIconWidget()
        self.help_texts, self.help_urls = self.help_icon.load_help_texts("pca")
        self.save_path = str(data_path("ganspace_features"))

        self.menu = menu
        self.app = menu.app
        self.G = None
        self.user_pkl = ''
        self.pca_mode = 0
        self.num_features = 0
        self.alpha = 1
        self.running = False
        self.queue = mp.Queue()
        self.reply = mp.Queue()
        self.message = ""
        self.pca_process = LoggedProcess(target=fit, args=(self.queue, self.reply),
                                         daemon=True, name='ganspace-pca')

        self.save_path_browser = NativeBrowserWidget()
        self.X_comp, self.Z_comp = None, None
        self.done = False
        self.model_dropdown = ModelDropdownButton(menu.model_downloader)

    @imgui_utils.scoped_by_object_id
    def __call__(self):
        help_width = imgui.calc_text_size("(?)").x + 10
        button_width = self.menu.app.button_w
        spacing = self.menu.app.spacing
        
        input_width = -(button_width + spacing + help_width + 30)

        text = "Extract Meaningful Directions from a Model"
        text_width = imgui.calc_text_size(text).x
        window_width = imgui.get_window_width()
        help_icon_size = imgui.get_font_size()
        style = imgui.get_style()

        imgui.text(text)
        
        spacing = window_width - (style.window_padding[0] * 2) - text_width - help_icon_size - style.item_spacing[0] - 10
        
        imgui.same_line()
        imgui.dummy(spacing, 0)
        self.help_icon.render_with_url(self.help_texts.get("pca_module"), self.help_urls.get("pca_module"), "Read More")

        imgui.separator()

        if not self.reply.empty():
            self.message, (self.X_comp, self.Z_comp), self.done = self.reply.get()
            while not self.reply.empty():
                self.message, (self.X_comp, self.Z_comp), self.done = self.reply.get()

        if self.done:
            self.running = False
            filename = Path(self.user_pkl).stem
            np.save(os.path.join(self.save_path,f"{filename}_xcomp.npy"), self.X_comp)
            np.save(os.path.join(self.save_path,f"{filename}_zcomp.npy"), self.Z_comp)

        changed, self.user_pkl = imgui_utils.input_text('##pkl', self.user_pkl, 1024,
                                                        flags=(imgui.INPUT_TEXT_AUTO_SELECT_ALL |
                                                               imgui.INPUT_TEXT_ENTER_RETURNS_TRUE),
                                                        width=input_width,
                                                        help_text='<PATH> | <URL> | <RUN_DIR> | <RUN_ID> | <RUN_ID>/<KIMG>.pkl')
        if changed:
            self.load(self.user_pkl)

        imgui.same_line()
        picked = self.model_dropdown(width=button_width)
        if picked is not None:
            self.user_pkl = picked
            self.load(picked)

        help_width = imgui.calc_text_size("(?)").x + 10
        input_width = -(self.app.button_w + self.app.spacing + help_width)

        imgui.text("PCA Estimator")
        imgui.same_line()
        with imgui_utils.item_width(input_width):
            _, self.pca_mode = imgui.combo("##pca_mode", self.pca_mode, pca_modes)

        max_features = 0 if self.G is None else self.G.w_dim
        imgui.text("Features")
        imgui.same_line()
        with imgui_utils.item_width(input_width):
            _, self.num_features = imgui.input_int("##num_features", self.num_features)

        if self.num_features > max_features:
            self.num_features = max_features
        if self.num_features < 0:
            self.num_features = 0

        imgui.text("Sparsity")
        imgui.same_line()
        with imgui_utils.item_width(input_width):
            _, self.alpha = imgui.slider_float(
                "##alpha", self.alpha,
                min_value=0.0, max_value=1.0,
                format='%.3f', power=3)

        imgui.text("Save Path")
        _, self.save_path = imgui_utils.input_text("##save_path", self.save_path, 1024, 0,
                                                    width=imgui.get_window_width() - self.menu.app.button_w - imgui.calc_text_size("Browse")[0])
        
        imgui.same_line()
        if imgui.button("Browse##pca_save_path", width=button_width):
            directory_path = self.save_path_browser.select_directory("Select Save Directory", initial_dir=self.save_path)
            if directory_path:
                self.save_path = directory_path.replace('\\', '/')

        if imgui_utils.button("Get Salient Features", width=imgui.get_content_region_available_width(), enabled=self.G is not None):
            imgui.open_popup("PCA-popup")
            self.running = True
            self.X_comp, self.Z_comp = None, None
            os.makedirs(self.save_path, exist_ok=True)
            device = device_utils.get_device()
            # MPS tensors cannot be sent across processes; ship the model on CPU
            # and let the worker move it to the device.
            G = self.G.cpu() if device.type == 'mps' else self.G
            self.queue.put((pca_modes[self.pca_mode], self.num_features, G, device.type, True, self.alpha))
            self.pca_process.start()


        if imgui.begin_popup_modal("PCA-popup")[0]:
            imgui.text(f"Extracting Salient Directions in Latent Space of: {self.user_pkl}")
            imgui.text(f"PCA Method {pca_modes[self.pca_mode]} | Num Features {self.num_features} | Sparsity {self.alpha}" )
            filename = Path(self.user_pkl).stem
            direct=os.path.join(self.save_path,f"{filename}_xcomp.npy")
            imgui.text(f"Saving Extracted Directions at: {direct}")
            if self.message != "":
                imgui.text(self.message)
            if imgui_utils.button("Done", width=self.menu.app.button_w, enabled=not self.running):
                imgui.close_current_popup()
            imgui.end_popup()



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

    def load(self, user_pkl):
        path = self.resolve_pkl(user_pkl)
        with dnnlib.util.open_url(path, verbose=False) as f:
            data = legacy.load_network_pkl(f, custom=True)
        self.G = data["G"].to(device_utils.get_device())
