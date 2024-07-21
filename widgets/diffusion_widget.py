import imgui
import librosa
import numpy as np
import torch
from pythonosc.udp_client import SimpleUDPClient
import os
import dnnlib
from assets import ACTIVE_RED
from audio import audio_stream
from utils.gui_utils import imgui_utils
from widgets.browse_widget import BrowseWidget
import NDIlib as ndi


class DiffusionWidget:

    def __init__(self, viz):
        self.viz = viz

        self.ndi_find = ndi.find_create_v2()
        self.ndi_sources = []
        self.current_used_ndi_source = None

        self.input_path = ""
        self.model_id = "stabilityai/sd-turbo"
        self.lora_dict = None
        self.prompt = "1girl with brown dog ears, thick frame glasses"
        self.scale = 1.0
        self.acceleration = 1  # 0: none, 1: xformers, 2: tensorrt
        self.use_denoising_batch = True
        self.enable_similar_image_filter = True
        self.seed = 2

        self.file_dialog = BrowseWidget(viz, "Browse", os.path.abspath(os.getcwd()),
                                        ["*", ".mp4", ],
                                        width=self.viz.app.button_w, multiple=False, traverse_folders=False)

    def get_params(self):
        return (self.n_fft, self.osc_addresses, self.mappings, self.use_osc)

    def set_params(self, params):
        self.n_fft, self.osc_addresses, self.mappings, self.use_osc = params

    def callback(self, data):
        self.fft.data = np.abs(librosa.stft(data, n_fft=self.n_fft * 2 - 1))

    # def start_processing(self, ignore_errors=False):
    #     viz = self.viz
    #     viz.app.skip_frame()  # The input field will change on next frame.
    #     print(os.getcwd())
    #     try:
    #         resolved = self.resolve_pkl(pkl)
    #         name = resolved.replace('\\', '/').split('/')[-1]
    #         self.cur_pkl = resolved
    #         self.user_pkl = resolved
    #         viz.result.message = f'Loading {name}...'
    #         viz.defer_rendering()
    #         if resolved in self.recent_pkls:
    #             self.recent_pkls.remove(resolved)
    #         self.recent_pkls.insert(0, resolved)
    #     except:
    #         self.cur_pkl = None
    #         self.user_pkl = pkl
    #         if pkl == '':
    #             viz.result = dnnlib.EasyDict(message='No network pickle loaded')
    #         else:
    #             viz.result = dnnlib.EasyDict(error=renderer.CapturedException())
    #         if not ignore_errors:
    #             raise
    def refresh_ndi_sources(self):
        ndi_find = ndi.find_create_v2()
        ndi.find_wait_for_sources(ndi_find, 1000)
        self.ndi_sources = ndi.find_get_current_sources(ndi_find)

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        if show:
            if len(self.ndi_sources) == 0:
                imgui.text("No NDI sources found. Please start a NDI source and try again.")
            else:
                sources_names = [source.ndi_name for source in self.ndi_sources]
                current_index = sources_names.index(
                    self.current_used_ndi_source.ndi_name) if self.current_used_ndi_source is not None else -1
                changed, current_index = imgui.combo("NDI Sources", current_index, sources_names)

                if changed:
                    self.current_used_ndi_source = self.ndi_sources[current_index]
                    self.viz.set_ndi_source(self.current_used_ndi_source)

            if imgui.button("Refresh NDI Sources"):
                self.refresh_ndi_sources()

            if self.current_used_ndi_source is None:
                self.viz.result = dnnlib.EasyDict(
                    message='No NDI sources found. Please start a NDI source and try again.')

            model_ids = ["stabilityai/sd-turbo", "KBlueLeaf/kohaku-v2.1"]
            current_model_index = model_ids.index(self.model_id)
            changed, current_model_index = imgui.combo("Model ID", current_model_index, model_ids)
            if changed:
                self.model_id = model_ids[current_model_index]

            changed, self.prompt = imgui_utils.input_text("Prompt", self.prompt, 1024,
                                                          flags=imgui.INPUT_TEXT_AUTO_SELECT_ALL,
                                                          help_text='Prompt to be used for the model',
                                                          width=-self.viz.app.button_w - self.viz.app.spacing, )

            changed, self.scale = imgui.slider_float(
                "Scale", float(self.scale), 0.1, 2.0
            )

            self.use_denoising_batch = imgui.checkbox("Use Denoising Batch", self.use_denoising_batch)

            self.enable_similar_image_filter = imgui.checkbox("Enable Similar Image Filter",
                                                              self.enable_similar_image_filter)

            changed, self.seed = imgui.input_int("Seed", self.seed)

            if imgui_utils.button("Start Processing",
                                  enabled=(self.model_id != "" and self.current_used_ndi_source is not None)):
                self.viz.is_processing = True

            if imgui_utils.button("Stop Processing"):
                self.viz.is_processing = False

        self.viz.args.model_id = self.model_id
        self.viz.args.prompt = self.prompt
        self.viz.args.scale = self.scale
        self.viz.args.use_denoising_batch = self.use_denoising_batch
        self.viz.args.enable_similar_image_filter = self.enable_similar_image_filter
        self.viz.args.seed = self.seed
