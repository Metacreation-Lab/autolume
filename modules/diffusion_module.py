# diffusion_module.py

import os
import sys
import threading
import torch
from torchvision.io import read_video, write_video
from tqdm import tqdm
import imgui

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.wrapper import StreamDiffusionWrapper
from utils.gui_utils import imgui_utils

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class DiffusionModule:
    def __init__(self, menu):
        self.menu = menu
        self.app = menu.app
        self.input_path = os.path.join(CURRENT_DIR, "images", "inputs", "input.mp4")
        self.output_path = os.path.join(CURRENT_DIR, "images", "outputs", "output.mp4")
        self.model_id = "KBlueLeaf/kohaku-v2.1"
        self.lora_dict = None
        self.prompt = "1girl with brown dog ears, thick frame glasses"
        self.scale = 1.0
        self.acceleration = 1  # 0: none, 1: xformers, 2: tensorrt
        self.use_denoising_batch = True
        self.enable_similar_image_filter = True
        self.seed = 2
        self.progress = 0.0
        self.is_processing = False

    def process_video_threaded(self):
        self.is_processing = True
        self.progress = 0.0
        threading.Thread(target=self._process_video).start()

    def _process_video(self):
        self.is_processing = True
        self.progress = 0.0

        video_info = read_video(self.input_path)
        video = video_info[0] / 255
        fps = video_info[2]["video_fps"]
        height = int(video.shape[1] * self.scale)
        width = int(video.shape[2] * self.scale)

        acceleration_options = ["none", "xformers", "tensorrt"]
        stream = StreamDiffusionWrapper(
            model_id_or_path=self.model_id,
            lora_dict=self.lora_dict,
            t_index_list=[35, 45],
            frame_buffer_size=1,
            width=width,
            height=height,
            warmup=10,
            acceleration=acceleration_options[self.acceleration],
            do_add_noise=False,
            mode="img2img",
            output_type="pt",
            enable_similar_image_filter=self.enable_similar_image_filter,
            similar_image_filter_threshold=0.98,
            use_denoising_batch=self.use_denoising_batch,
            seed=self.seed,
        )

        stream.prepare(
            prompt=self.prompt,
            num_inference_steps=50,
        )

        video_result = torch.zeros(video.shape[0], height, width, 3)

        for _ in range(stream.batch_size):
            stream(image=video[0].permute(2, 0, 1))

        for i in range(video.shape[0]):
            output_image = stream(video[i].permute(2, 0, 1))
            video_result[i] = output_image.permute(1, 2, 0)
            self.progress = (i + 1) / video.shape[0]

        video_result = video_result * 255
        write_video(self.output_path, video_result[2:], fps=fps)

        self.is_processing = False

    @imgui_utils.scoped_by_object_id
    def __call__(self):
        imgui.begin("Diffusion Module", True)

        _, self.input_path = imgui_utils.input_text(
            "Input Path", self.input_path, 256,
            width=-(self.app.button_w + self.app.spacing)
        )

        _, self.output_path = imgui_utils.input_text(
            "Output Path", self.output_path, 256,
            width=-(self.app.button_w + self.app.spacing)
        )

        _, self.model_id = imgui_utils.input_text(
            "Model ID", self.model_id, 256,
            width=-(self.app.button_w + self.app.spacing)
        )

        _, self.prompt = imgui_utils.input_text(
            "Prompt", self.prompt, 256,
            width=-(self.app.button_w + self.app.spacing)
        )

        _, self.scale = imgui.slider_float(
            "Scale", self.scale, 0.1, 2.0
        )

        acceleration_options = ["none", "xformers", "tensorrt"]
        _, self.acceleration = imgui.combo(
            "Acceleration", self.acceleration, acceleration_options
        )

        _, self.use_denoising_batch = imgui.checkbox(
            "Use Denoising Batch", self.use_denoising_batch
        )

        _, self.enable_similar_image_filter = imgui.checkbox(
            "Enable Similar Image Filter", self.enable_similar_image_filter
        )

        _, self.seed = imgui.input_int(
            "Seed", self.seed
        )

        if imgui_utils.button("Process Video", width=imgui.get_content_region_available_width()):
            self.process_video_threaded()

        if self.is_processing:
            imgui.text("Processing...")
            imgui.progress_bar(self.progress, (0, 0))

        imgui.end()