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



class DiffusionWidget:

    def __init__(self, viz):
        self.viz = viz
        self.input_path = ""
        self.model_id = "KBlueLeaf/kohaku-v2.1"
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

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        if show:
            # width = viz.app.font_size
            # height = imgui.get_text_line_height_with_spacing()
            imgui_utils.input_text("##SRINPUT", self.input_path, 1024, flags=imgui.INPUT_TEXT_READ_ONLY,
                                   width=-(self.viz.app.button_w + self.viz.app.spacing), help_text="Input File")
            imgui.same_line()

            _clicked, input = self.file_dialog(self.viz.app.button_w)
            if _clicked:
                print(input)
                self.input_path = input[0]
                print(self.input_path)

            changed, self.model_id = imgui_utils.input_text("Model ID", self.model_id, 1024,
                                                            flags=imgui.INPUT_TEXT_AUTO_SELECT_ALL,
                                                            help_text='Model ID',
                                                            width=-self.viz.app.button_w - self.viz.app.spacing, )

            changed, self.prompt = imgui_utils.input_text("Prompt", self.prompt, 1024,
                                                          flags=imgui.INPUT_TEXT_AUTO_SELECT_ALL,
                                                          help_text='Prompt to be used for the model',
                                                          width=-self.viz.app.button_w - self.viz.app.spacing, )

            changed, self.scale = imgui.slider_float(
                "Scale", float(self.scale), 0.1, 2.0
            )

            # acceleration_options = ["none", "xformers", "tensorrt"]
            # self.acceleration = imgui.combo(
            #     "Acceleration", self.acceleration, acceleration_options
            # )

            self.use_denoising_batch = imgui.checkbox("Use Denoising Batch", self.use_denoising_batch)

            self.enable_similar_image_filter = imgui.checkbox("Enable Similar Image Filter",
                                                              self.enable_similar_image_filter)

            changed, self.seed = imgui.input_int("Seed", self.seed)

        self.viz.args.input = self.input_path
        self.viz.args.model_id = self.model_id
        self.viz.args.prompt = self.prompt
        self.viz.args.scale = self.scale
        self.viz.args.use_denoising_batch = self.use_denoising_batch
        self.viz.args.enable_similar_image_filter = self.enable_similar_image_filter
        self.viz.args.seed = self.seed
