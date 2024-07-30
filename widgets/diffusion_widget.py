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

        self.t_index_min = 35
        self.t_index_max = 45

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
                "use_tiny_vae": True,
                "vae_id": None,
                "use_denoising_batch": True,
                "use_safety_checker": False,
                "engine_dir": 'engines',
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
        self.default_params = self.model_params.copy()


    def refresh_ndi_sources(self):
        ndi_find = ndi.find_create_v2()
        ndi.find_wait_for_sources(ndi_find, 1000)
        self.ndi_sources = ndi.find_get_current_sources(ndi_find)

    def reset_params(self):
        current_prompt = self.prompt
        self.current_params = self.default_params[self.model_id].copy()
        self.t_index_min, self.t_index_max = self.current_params["t_index_list"]
        self.prompt = current_prompt

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
            with imgui_utils.item_width(-(self.viz.app.button_w + self.viz.app.spacing)):
                changed, current_model_index = imgui.combo("Model ID", current_model_index, model_ids)
            if changed:
                self.model_id = model_ids[current_model_index]
                self.current_params = self.model_params[self.model_id]

            # Display and update parameters based on the current model
            with imgui_utils.item_width(-(self.viz.app.button_w + self.viz.app.spacing)):
                for param, value in self.current_params.items():
                    if param == "seed":
                        changed, self.current_params[param] = imgui.input_int("Seed", value)
                    elif param in ["enable_similar_image_filter"]:
                        changed, self.current_params[param] = imgui.checkbox(param.replace("_", " ").title(), value)
                    elif param in ["warmup"]:
                        changed, self.current_params[param] = imgui.input_int(param.replace("_", " ").title(), value)
                    elif param == "similar_image_filter_threshold":
                        changed, self.current_params[param] = imgui.slider_float(
                            "Similar Image Filter Threshold", value, 0.0, 1.0
                        )
                    elif param == "t_index_list":
                        imgui.text("T Index List")
                        imgui.same_line()
                        changed_min, self.t_index_min = imgui.input_int("Min", self.t_index_min)
                        imgui.text("T Index List")
                        imgui.same_line()
                        changed_max, self.t_index_max = imgui.input_int("Max", self.t_index_max)
                        if changed_min or changed_max:
                            self.current_params[param] = [self.t_index_min, self.t_index_max]

                # Acceleration selection
                acceleration_options = ["none", "xformers", "tensorrt"]
                current_acceleration_index = acceleration_options.index(self.current_params["acceleration"])
                with imgui_utils.item_width(-(self.viz.app.button_w + self.viz.app.spacing)):
                    changed, current_acceleration_index = imgui.combo("Acceleration", current_acceleration_index,
                                                                  acceleration_options)
                if changed:
                    self.current_params["acceleration"] = acceleration_options[current_acceleration_index]

            changed, self.prompt = imgui_utils.input_text("Prompt", self.prompt, 1024,
                                                          flags=imgui.INPUT_TEXT_AUTO_SELECT_ALL,
                                                          help_text='Prompt to be used for the model',
                                                          width=-self.viz.app.button_w - self.viz.app.spacing, )

            imgui.text(f'{self.viz.fps:.1f} FPS' if self.viz.fps > 0 else 'N/A')

            if imgui_utils.button("Reset Parameters"):
                self.reset_params()
            imgui.same_line()
            if imgui_utils.button("Start Processing",
                                  enabled=(self.model_id != "" and self.current_used_ndi_source is not None)):
                self.viz.is_processing = True
                self.viz.result = dnnlib.EasyDict(
                    message='Loading model, please wait...')
            imgui.same_line()
            if imgui_utils.button("Stop Processing"):
                self.viz.is_processing = False

        self.viz.args.clear()
        self.viz.args.model_id_or_path = self.model_id
        for param, value in self.current_params.items():
            setattr(self.viz.args, param, value)
        self.viz.args.prompt = self.prompt
