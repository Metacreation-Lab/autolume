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

        self.model_id = "stabilityai/sd-turbo"
        self.prompt = "1girl with brown dog ears, thick frame glasses"

        self.model_params = {
            "stabilityai/sd-turbo": {
                "frame_buffer_size": 1,
                "warmup": 10,
                "acceleration": "xformers",
                "mode": "img2img",
                "t_index_list": [35, 45],
                "output_type": "np",
                "cfg_type": "none",
                "use_lcm_lora": False,
            },
            "KBlueLeaf/kohaku-v2.1": {
                "lora_dict": None,
                "t_index_list": [35, 45],
                "frame_buffer_size": 1,
                "warmup": 10,
                "acceleration": "xformers",
                "do_add_noise": False,
                "mode": "img2img",
                "output_type": "np",
                "enable_similar_image_filter": True,
                "similar_image_filter_threshold": 0.98,
                "seed": 2,
            }
        }
        self.current_params = self.model_params[self.model_id]

    def get_params(self):
        return (self.n_fft, self.osc_addresses, self.mappings, self.use_osc)

    def set_params(self, params):
        self.n_fft, self.osc_addresses, self.mappings, self.use_osc = params

    def callback(self, data):
        self.fft.data = np.abs(librosa.stft(data, n_fft=self.n_fft * 2 - 1))

    def refresh_ndi_sources(self):
        ndi_find = ndi.find_create_v2()
        ndi.find_wait_for_sources(ndi_find, 1000)
        self.ndi_sources = ndi.find_get_current_sources(ndi_find)

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        if show:

            # NDI settings
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

            # Model selection
            model_ids = list(self.model_params.keys())
            current_model_index = model_ids.index(self.model_id)
            changed, current_model_index = imgui.combo("Model ID", current_model_index, model_ids)
            if changed:
                self.model_id = model_ids[current_model_index]
                self.current_params = self.model_params[self.model_id]

            # Display and update parameters based on the current model
            for param, value in self.current_params.items():
                if param == "seed":
                    changed, self.current_params[param] = imgui.input_int("Seed", value)
                elif param in ["enable_similar_image_filter"]:
                    changed, self.current_params[param] = imgui.checkbox(param.replace("_", " ").title(), value)

            changed, self.prompt = imgui_utils.input_text("Prompt", self.prompt, 1024,
                                                          flags=imgui.INPUT_TEXT_AUTO_SELECT_ALL,
                                                          help_text='Prompt to be used for the model',
                                                          width=-self.viz.app.button_w - self.viz.app.spacing, )

            imgui.text(f'{self.viz.fps:.1f} FPS' if self.viz.fps > 0 else 'N/A')

            if imgui_utils.button("Start Processing",
                                  enabled=(self.model_id != "" and self.current_used_ndi_source is not None)):
                self.viz.is_processing = True
                self.viz.result = dnnlib.EasyDict(
                    message='Loading model, please wait...')

            if imgui_utils.button("Stop Processing"):
                self.viz.is_processing = False

        self.viz.args.clear()
        self.viz.args.model_id_or_path = self.model_id
        for param, value in self.current_params.items():
            setattr(self.viz.args, param, value)
        self.viz.args.prompt = self.prompt
