import os
import sys
import threading
import torch
from torchvision.io import read_video, write_video
from tqdm import tqdm
import imgui
import datetime
from utils.wrapper import StreamDiffusionWrapper
from utils.gui_utils import imgui_utils
from widgets.browse_widget import BrowseWidget
from dnnlib import EasyDict
from streamdiffusion.text_to_image_generator_lora import ImageGenerator


class DiffusionModule:
    def __init__(self, menu):
        self.menu = menu
        self.app = menu.app
        self.args = EasyDict()
        self.input_path = ""
        self.output_path = ""
        self.model_id = "KBlueLeaf/kohaku-v2.1"
        self.progress = 0.0
        self.running = False
        self.file_dialog = BrowseWidget(self, "Browse", os.path.abspath(os.getcwd()),
                                        ["*", ".mp4"], multiple=False, traverse_folders=False,
                                        width=self.app.button_w)

        self.save_path_dialog = BrowseWidget(self, "Save Path", os.path.abspath(os.getcwd()), [""], multiple=False,
                                             traverse_folders=False, add_folder_button=True, width=self.app.button_w)
        self.prompt = "girl with brown dog ears, thick frame glasses"
        self.model_params = {
            "stabilityai/sd-turbo": {
                "frame_buffer_size": 1,
                "warmup": 10,
                "acceleration": "xformers",
                "mode": "img2img",
                "t_index_list": [35, 45],
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
                "enable_similar_image_filter": True,
                "similar_image_filter_threshold": 0.98,
                "seed": 2,
                "use_lcm_lora": False,
            }
        }
        self.t_index_min = 35
        self.t_index_max = 45
        self.lcm_lora_id = "latent-consistency/lcm-lora-sdv1-5"
        self.current_params = self.model_params[self.model_id]
        self.default_params = self.model_params.copy()

        self.text2image_args = EasyDict()
        self.text2image_file_dialog = BrowseWidget(self, "Browse", os.path.abspath(os.getcwd()),
                                                   ["*", ".safetensors"], multiple=False, traverse_folders=False,
                                                   width=self.app.button_w)

        self.text2image_save_path_dialog = BrowseWidget(self, "Save Path", os.path.abspath(os.getcwd()), [""],
                                                        multiple=False,
                                                        traverse_folders=False, add_folder_button=True,
                                                        width=self.app.button_w)

        self.use_model_name = True
        self.text2image_args.pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"
        self.text2image_args.prompt = "Joker from Batman"
        self.text2image_args.model_path = ""
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.text2image_args.save_path = os.path.join(os.path.abspath(os.getcwd()), f"output_{timestamp}.png")
        self.text2image_model_path_dialog = BrowseWidget(self, "Browse", os.path.abspath(os.getcwd()),
                                                         ["*", ".safetensors"], multiple=False, traverse_folders=False,
                                                         width=self.app.button_w)

    def display_progress(self):
        imgui.text("Processing...")
        imgui.progress_bar(self.progress, (0, 0))

    def start_process_thread(self):
        process_thread = threading.Thread(target=self.start_process)
        process_thread.start()

    def start_process(self):
        self.progress = 0.0

        video_info = read_video(self.input_path)
        video = video_info[0] / 255
        fps = video_info[2]["video_fps"]
        height = int(video.shape[1])
        width = int(video.shape[2])

        self.args.output_type = "pt"
        self.args.width = width
        self.args.height = height
        self.args.use_denoising_batch = True
        self.args.use_lcm_lora = True
        self.args.do_add_noise = False
        stream = StreamDiffusionWrapper(**self.args)
        # stream = StreamDiffusionWrapper(
        #     model_id_or_path="KBlueLeaf/kohaku-v2.1",
        #     lora_dict=None,
        #     t_index_list=[35, 45],
        #     frame_buffer_size=1,
        #     width=int(video.shape[2] * 1.0),
        #     height=int(video.shape[1] * 1.0),
        #     warmup=10,
        #     acceleration="xformers",
        #     do_add_noise=False,
        #     mode="img2img",
        #     output_type="pt",
        #     enable_similar_image_filter=True,
        #     similar_image_filter_threshold=0.98,
        #     use_denoising_batch=True,
        #     seed=2,
        # )

        stream.prepare(
            prompt=self.prompt,
            num_inference_steps=50,
        )

        video_result = torch.zeros(video.shape[0], height, width, 3)

        for _ in range(stream.batch_size):
            stream(image=video[0].permute(2, 0, 1))

        for i in tqdm(range(video.shape[0])):
            output_image = stream(video[i].permute(2, 0, 1))
            video_result[i] = output_image.permute(1, 2, 0)
            self.progress = (i + 1) / video.shape[0]

        video_result = video_result * 255
        write_video(self.output_path, video_result[2:], fps=fps)

        self.running = False

    def reset_params(self):
        self.current_params = self.default_params[self.model_id].copy()
        self.t_index_min, self.t_index_max = self.current_params["t_index_list"]

    def start_image_process_thread(self):
        process_thread = threading.Thread(target=self.start_image_process)
        process_thread.start()

    def start_image_process(self):
        print("Generate images with LoRA")
        generator = ImageGenerator()
        generator.predict(**self.text2image_args)
        print("Images saved to", self.text2image_args.save_path)
        self.running = False

    @imgui_utils.scoped_by_object_id
    def __call__(self):
        if self.running:
            self.display_progress()

        joined = '\n'.join(self.input_path)

        imgui_utils.input_text("##SRINPUT", self.input_path, 1024, flags=imgui.INPUT_TEXT_READ_ONLY,
                               width=-(self.app.button_w + self.app.spacing), help_text="Input File")
        imgui.same_line()

        _clicked, input = self.file_dialog(self.app.button_w)
        if _clicked:
            print(input)
            self.input_path = input[0]
            print(self.input_path)

        imgui_utils.input_text("##SRRESULT", self.output_path, 1024, flags=imgui.INPUT_TEXT_READ_ONLY,
                               width=-(self.app.button_w + self.app.spacing), help_text="Output Path")
        imgui.same_line()

        _clicked, save_path = self.save_path_dialog(self.app.button_w)
        if _clicked:
            if len(save_path) > 0:
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                self.output_path = os.path.join(save_path[0], f"output_{timestamp}.mp4")
                print(self.output_path)
            else:
                self.output_path = ""
                print("No path selected")

        # Model selection
        model_ids = list(self.model_params.keys())
        current_model_index = model_ids.index(self.model_id)
        with imgui_utils.item_width(-(self.app.button_w + self.app.spacing)):
            changed, current_model_index = imgui.combo("Model ID", current_model_index, model_ids)
        if changed:
            self.model_id = model_ids[current_model_index]
            self.current_params = self.model_params[self.model_id]

        # Acceleration selection
        acceleration_options = ["none", "xformers", "tensorrt"]
        current_acceleration_index = acceleration_options.index(self.current_params["acceleration"])
        with imgui_utils.item_width(-(self.app.button_w + self.app.spacing)):
            changed, current_acceleration_index = imgui.combo("Acceleration", current_acceleration_index,
                                                              acceleration_options)
        if changed:
            self.current_params["acceleration"] = acceleration_options[current_acceleration_index]

        # Display and update parameters based on the current model
        with imgui_utils.item_width(-(self.app.button_w + self.app.spacing)):
            for param, value in self.current_params.items():
                if param == "seed":
                    changed, self.current_params[param] = imgui.input_int("Seed", value)
                elif param in ["enable_similar_image_filter"]:
                    changed, self.current_params[param] = imgui.checkbox(param.replace("_", " ").title(), value)
                elif param in ["use_lcm_lora"]:
                    changed, self.current_params[param] = imgui.checkbox(param.replace("_", " ").title(), value)
                    if self.current_params[param]:
                        lcm_lora_options = ["latent-consistency/lcm-lora-sdv1-5"]
                        current_lcm_lora_index = lcm_lora_options.index(self.lcm_lora_id)
                        with imgui_utils.item_width(-(self.app.button_w + self.app.spacing)):
                            changed, current_lcm_lora_index = imgui.combo("LCM LoRA ID", current_lcm_lora_index,
                                                                          lcm_lora_options)
                        if changed:
                            self.lcm_lora_id = lcm_lora_options[current_lcm_lora_index]
                elif param in ["warmup"]:
                    changed, self.current_params[param] = imgui.input_int(param.replace("_", " ").title(), value)
                elif param == "similar_image_filter_threshold":
                    with imgui_utils.item_width(-(self.app.button_w + self.app.spacing) * 2):
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

        changed, prompt = imgui_utils.input_text("Prompt", self.prompt, 1024,
                                                 flags=imgui.INPUT_TEXT_AUTO_SELECT_ALL,
                                                 help_text='Prompt to be used for the model',
                                                 width=-self.app.button_w - self.app.spacing, )
        if changed:
            self.prompt = prompt
        if imgui_utils.button("Reset Parameters"):
            self.reset_params()

        try:
            if imgui.button("Process Video", width=imgui.get_content_region_available_width()) and not self.running:
                self.running = True
                print("Process Video using Diffusion model")

                self.args.clear()
                self.args.model_id_or_path = self.model_id
                for param, value in self.current_params.items():
                    setattr(self.args, param, value)
                if self.args.use_lcm_lora:
                    self.args.lcm_lora_id = self.lcm_lora_id

                print("Starting Diffusion Process...")
                self.start_process_thread()

        except Exception as e:
            print("SRR ERROR", e)

        imgui.text("Generate images with LoRA")

        # Save path
        imgui_utils.input_text("##SRRESULTFILE", self.text2image_args.save_path, 1024, flags=imgui.INPUT_TEXT_READ_ONLY,
                               width=-(self.app.button_w + self.app.spacing), help_text="Output Path")
        imgui.same_line()

        _clicked, save_path = self.text2image_save_path_dialog(self.app.button_w)
        if _clicked:
            if len(save_path) > 0:
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                self.text2image_args.save_path = os.path.join(save_path[0], f"output_{timestamp}.png")
                print(self.text2image_args.save_path)
            else:
                self.text2image_args.save_path = ""
                print("No path selected")

        # Model name
        imgui.text("Use Model Name or Model Path")
        imgui.same_line()
        if imgui.radio_button("Use Name", self.use_model_name):
            self.use_model_name = True

        imgui.same_line()

        if imgui.radio_button("Use Path", not self.use_model_name):
            self.use_model_name = False

        if self.use_model_name:
            with imgui_utils.item_width(-(self.app.button_w + self.app.spacing) * 1.7):
                _, self.text2image_args.pretrained_model_name_or_path = imgui.input_text("Pretrained Model Name",
                                                                                         self.text2image_args.pretrained_model_name_or_path,
                                                                                         1024)
        else:
            imgui_utils.input_text("##SRModel Path", "", 1024,
                                   flags=imgui.INPUT_TEXT_READ_ONLY,
                                   width=-(self.app.button_w + self.app.spacing), help_text="Model Path")
            imgui.same_line()

            _clicked, model_path = self.text2image_model_path_dialog(self.app.button_w)
            if _clicked and len(model_path) > 0:
                self.text2image_args.pretrained_model_name_or_path = model_path[0]
                print(self.text2image_args.pretrained_model_name_or_path)

        # Model LoRA file selection
        imgui_utils.input_text("##SRINPUTFILE", self.text2image_args.model_path, 1024,
                               flags=imgui.INPUT_TEXT_READ_ONLY,
                               width=-(self.app.button_w + self.app.spacing), help_text="LoRA safetensors file, can be empty")
        imgui.same_line()

        _clicked, model_path = self.text2image_file_dialog(self.app.button_w)
        if _clicked:
            self.text2image_args.model_path = model_path[0]
            print(self.text2image_args.model_path)

        # Prompt
        changed, prompt = imgui_utils.input_text("##SRPrompt", self.text2image_args.prompt, 1024,
                                                 flags=imgui.INPUT_TEXT_AUTO_SELECT_ALL,
                                                 help_text='Prompt to be used for the image generation',
                                                 width=-self.app.button_w - self.app.spacing, )
        if changed:
            self.text2image_args.prompt = prompt

        if imgui.button("Generate Images", width=imgui.get_content_region_available_width()) and not self.running:
            self.running = True
            print("Starting Image Diffusion Process...")
            try:
                self.start_image_process_thread()
            except Exception as e:
                print("SRR ERROR", e)
