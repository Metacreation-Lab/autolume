import glob
import os
from pathlib import Path
import multiprocessing as mp

import imgui
import numpy as np
import torch.cuda

import dnnlib
from torch_utils import legacy
from utils.gui_utils import imgui_utils
from ganspace.extract_pca import fit
from widgets.browse_widget import BrowseWidget


def _locate_results(pattern):
    return pattern


pca_modes = ['pca', 'ipca', 'fbpca', "ica", 'spca']


class PCA_Module:
    def __init__(self, menu):
        cwd = os.getcwd()
        self.save_path = os.path.join(cwd,"ganspace_features")

        self.menu = menu
        self.app = menu.app
        self.G = None
        self.user_pkl = ''
        self.pca_mode = 0
        self.num_features = 0
        self.alpha = 1
        self.browse_cache = []
        self.running = False
        self.queue = mp.Queue()
        self.reply = mp.Queue()
        self.message = ""
        self.pca_process = mp.Process(target=fit, args=(self.queue, self.reply),
                                      daemon=True)

        self.save_path_dialog = BrowseWidget(self, "Save Path", os.path.abspath(os.getcwd()), [""], multiple=False,
                                             traverse_folders=False, add_folder_button=True, width=self.menu.app.button_w)
        self.X_comp, self.Z_comp = None, None
        self.done = False
        for pkl in os.listdir("./models"):
            if pkl.endswith(".pkl"):
                print(pkl, os.path.join(os.getcwd(), "models", pkl))
                self.browse_cache.append(os.path.join(os.getcwd(), "models", pkl))

    @imgui_utils.scoped_by_object_id
    def __call__(self):
        if self.reply.qsize() > 0:
            self.message, (self.X_comp, self.Z_comp), self.done = self.reply.get()
            while self.reply.qsize() > 0:
                self.message, (self.X_comp, self.Z_comp), self.done = self.reply.get()

        if self.done:
            self.running = False
            filename = Path(self.user_pkl).stem
            np.save(os.path.join(self.save_path,f"{filename}_xcomp.npy"), self.X_comp)
            np.save(os.path.join(self.save_path,f"{filename}_zcomp.npy"), self.Z_comp)

        changed, self.user_pkl = imgui_utils.input_text('##pkl', self.user_pkl, 1024,
                                                        flags=(imgui.INPUT_TEXT_AUTO_SELECT_ALL |
                                                               imgui.INPUT_TEXT_ENTER_RETURNS_TRUE),
                                                        width=-1 - self.menu.app.button_w - self.menu.app.spacing,
                                                        help_text='<PATH> | <URL> | <RUN_DIR> | <RUN_ID> | <RUN_ID>/<KIMG>.pkl')
        if changed:
            self.load(self.user_pkl)
        imgui.same_line()
        if imgui_utils.button('Models', enabled=len(self.browse_cache) > 0, width=self.menu.app.button_w):
            imgui.open_popup('browse_pkls_popup')
            self.browse_refocus = True

        if imgui.begin_popup('browse_pkls_popup'):
            for pkl in self.browse_cache:
                clicked, _state = imgui.menu_item(pkl)
                if clicked:
                    self.user_pkl = pkl
                    self.load(pkl)

            if self.browse_refocus:
                imgui.set_scroll_here()
                self.menu.app.skip_frame()  # Focus will change on next frame.
                self.browse_refocus = False
            imgui.end_popup()
        with imgui_utils.item_width(-1 - self.menu.app.button_w - self.menu.app.spacing):
            _, self.pca_mode = imgui.combo(
                "PCA Estimator", self.pca_mode, pca_modes
            )
        max_features = 0 if self.G is None else self.G.w_dim

        with imgui_utils.item_width(-1 - self.menu.app.button_w - self.menu.app.spacing):
            _, self.num_features = imgui.input_int(
                f"Features", self.num_features
            )
        if self.num_features > max_features:
            self.num_features = max_features
        if self.num_features < 0:
            self.num_features = 0
        with imgui_utils.item_width(-1 - self.menu.app.button_w - self.menu.app.spacing):
            _, self.alpha = imgui.slider_float(
                "Sparsity", self.alpha,
                min_value=0.0, max_value=1.0,
                format='%.3f', power=3)

        changed, self.save_path = imgui_utils.input_text('##save_path', self.save_path, 1024,
                                                        flags=(imgui.INPUT_TEXT_AUTO_SELECT_ALL |
                                                               imgui.INPUT_TEXT_ENTER_RETURNS_TRUE),
                                                        width=(-1 - self.menu.app.button_w - self.menu.app.spacing),
                                                        help_text='Dir to Folder to save GANSPACE')
        imgui.same_line()
        _clicked, save_path = self.save_path_dialog(self.menu.app.button_w)
        if _clicked:
            if len(save_path) > 0:
                self.result_path = save_path[0]
                print(self.result_path)
            else:
                self.result_path = ""
                print("No path selected")

        if imgui_utils.button("Get Salient Features", width=imgui.get_content_region_available_width(), enabled=self.G is not None):
            imgui.open_popup("PCA-popup")
            self.running = True
            self.X_comp, self.Z_comp = None, None
            os.makedirs(self.save_path, exist_ok=True)
            self.queue.put((pca_modes[self.pca_mode], self.num_features, self.G, "cuda" if torch.cuda.is_available() else "cpu", True, self.alpha))
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
        self.G = data["G"].to("cuda" if torch.cuda.is_available() else "cpu")
