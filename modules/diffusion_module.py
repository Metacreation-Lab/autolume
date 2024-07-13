import os
import sys
import threading
import torch
from torchvision.io import read_video, write_video
from tqdm import tqdm
import imgui

from utils.wrapper import StreamDiffusionWrapper
from utils.gui_utils import imgui_utils
from widgets.browse_widget import BrowseWidget
from dnnlib import EasyDict

args = EasyDict(output_path="", input_path="", model_id="KBlueLeaf/kohaku-v2.1",
                lora_dict=None, prompt="1girl with brown dog ears, thick frame glasses", scale=1.0, acceleration=1,
                use_denoising_batch=True, enable_similar_image_filter=True, seed=2)


class DiffusionModule:
    def __init__(self, menu):
        self.menu = menu
        self.app = menu.app
        self.input_path = args.input_path
        self.output_path = args.output_path
        self.model_id = args.model_id
        self.lora_dict = args.lora_dict
        self.prompt = args.prompt
        self.scale = args.scale
        self.acceleration = args.acceleration  # 0: none, 1: xformers, 2: tensorrt
        self.use_denoising_batch = args.use_denoising_batch
        self.enable_similar_image_filter = args.enable_similar_image_filter
        self.seed = int(args.seed)
        self.progress = 0.0
        self.running = False
        self.file_dialog = BrowseWidget(self, "Browse", os.path.abspath(os.getcwd()),
                                        [".mp4"], multiple=False, traverse_folders=False,
                                        width=self.app.button_w)
        self.save_path_dialog = BrowseWidget(self, "Save Path", os.path.abspath(os.getcwd()), [""], multiple=False,
                                             traverse_folders=False, add_folder_button=True, width=self.app.button_w)

    def display_progress(self):
        imgui.begin("Diffusion start", False)
        imgui.text("Processing...")
        imgui.progress_bar(self.progress, (0, 0))
        imgui.end()

    def start_process_thread(self):
        process_thread = threading.Thread(target=self.start_process)
        process_thread.start()

    def start_process(self):
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

        self.running = False

    @imgui_utils.scoped_by_object_id
    def __call__(self):
        if self.running:
            self.display_progress()

        joined = '\n'.join(self.input_path)

        imgui_utils.input_text("##SRINPUT", joined, 1024, flags=imgui.INPUT_TEXT_READ_ONLY,
                               width=-(self.app.button_w + self.app.spacing), help_text="Input File")
        imgui.same_line()

        _clicked, input = self.file_dialog(self.app.button_w)
        if _clicked:
            self.input_path = input[0]
            print(self.input_path)

        imgui_utils.input_text("##SRRESULT", self.output_path, 1024, flags=imgui.INPUT_TEXT_READ_ONLY,
                               width=-(self.app.button_w + self.app.spacing), help_text="Output Path")
        imgui.same_line()

        _clicked, save_path = self.save_path_dialog(self.app.button_w)
        if _clicked:
            if len(save_path) > 0:
                self.output_path = save_path[0]
                print(self.output_path)
            else:
                self.output_path = ""
                print("No path selected")

        changed, self.model_id = imgui_utils.input_text("Model ID", self.model_id, 1024,
                                               flags=imgui.INPUT_TEXT_AUTO_SELECT_ALL,
                                               help_text='Model ID',
                                               width=-self.app.button_w - self.app.spacing, )

        changed, self.prompt = imgui_utils.input_text("Prompt", self.prompt, 1024,
                                             flags=imgui.INPUT_TEXT_AUTO_SELECT_ALL,
                                             help_text='Prompt to be used for the model',
                                             width=-self.app.button_w - self.app.spacing, )

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

        try:
            if imgui.button("Process Video", width=imgui.get_content_region_available_width()) and not self.running:
                self.running = True
                print("Process Video using Diffusion model")
                args.input_path = self.input_path
                args.output_path = self.output_path
                args.model_id = self.model_id
                args.prompt = self.prompt
                args.scale = self.scale
                args.acceleration = self.acceleration
                args.use_denoising_batch = self.use_denoising_batch
                args.enable_similar_image_filter = self.enable_similar_image_filter
                args.seed = int(self.seed)
                self.args = args
                print("Starting Diffusion Process...")
                self.start_process_thread()

        except Exception as e:
            print("SRR ERROR", e)
