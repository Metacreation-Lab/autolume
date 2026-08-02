import logging
import os
from pathlib import Path
import multiprocessing as mp

import imgui
import numpy as np

import dnnlib
from torch_utils import legacy
from utils import device_utils
from utils.app_logging import LoggedProcess
from utils.gui_utils import imgui_utils
from ganspace.extract_pca import fit
from utils.model_dir import resolve_pkl
from widgets.model_input_widget import ModelInputWidget
from widgets.native_browser_widget import NativeBrowserWidget
from widgets.help_icon_widget import HelpIconWidget
from utils.user_data import data_path


logger = logging.getLogger(__name__)

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
        self.num_features = 4
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
        self.model_input = ModelInputWidget(menu.app)

    @imgui_utils.scoped_by_object_id
    def __call__(self):
        help_width = imgui.calc_text_size("(?)").x + 10
        button_width = self.menu.app.button_w

        input_width = -(help_width + 30)

        text = "Extract Meaningful Directions from a Model"
        imgui.text(text)
        self.help_icon.render(self.help_texts.get("pca_module"),
                              url=self.help_urls.get("pca_module"),
                              align_right=True)

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

        changed, self.user_pkl = self.model_input(self.user_pkl, width=input_width)
        if changed and self.user_pkl:
            try:
                self.load(self.user_pkl)
            except Exception:
                logger.exception("Could not load model %s", self.user_pkl)

        help_width = imgui.calc_text_size("(?)").x + 10
        input_width = -(self.app.button_w + self.app.spacing + help_width)

        imgui.text("PCA Estimator")
        imgui.same_line()
        with imgui_utils.item_width(input_width):
            _, self.pca_mode = imgui.combo("##pca_mode", self.pca_mode, pca_modes)

        imgui.text("Features")
        imgui.same_line()
        with imgui_utils.item_width(input_width):
            _, self.num_features = imgui.input_int("##num_features", self.num_features)

        if self.G is not None and self.num_features > self.G.w_dim:
            self.num_features = self.G.w_dim
        if self.num_features < 1:
            self.num_features = 1

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



    def load(self, user_pkl):
        path = resolve_pkl(user_pkl)
        with dnnlib.util.open_url(path, verbose=False) as f:
            data = legacy.load_network_pkl(f, custom=True)
        self.G = data["G"].to(device_utils.get_device())
