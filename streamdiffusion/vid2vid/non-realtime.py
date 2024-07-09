import os
import sys
from typing import Literal, Dict, Optional

import torch
from torchvision.io import read_video, write_video
from tqdm import tqdm
import imgui
import glfw
from imgui.integrations.glfw import GlfwRenderer
from OpenGL import GL as gl
import threading
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.wrapper import StreamDiffusionWrapper

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class VideoProcessor:
    def __init__(self):
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

def impl_glfw_init():
    width, height = 1280, 720
    window_name = "Video Processor"

    if not glfw.init():
        print("Could not initialize OpenGL context")
        exit(1)

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)

    window = glfw.create_window(int(width), int(height), window_name, None, None)
    glfw.make_context_current(window)

    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        exit(1)

    return window

def main():
    imgui.create_context()
    window = impl_glfw_init()

    impl = GlfwRenderer(window)

    io = imgui.get_io()
    io.fonts.add_font_default()
    io.display_size = 1280, 720

    processor = VideoProcessor()

    acceleration_options = ["none", "xformers", "tensorrt"]
    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()

        imgui.new_frame()

        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("File", True):
                clicked_quit, selected_quit = imgui.menu_item(
                    "Quit", 'Cmd+Q', False, True
                )

                if clicked_quit:
                    exit(0)

                imgui.end_menu()
            imgui.end_main_menu_bar()

        imgui.begin("Video Processor", True)

        changed, processor.input_path = imgui.input_text(
            "Input Path", processor.input_path, 256
        )

        changed, processor.output_path = imgui.input_text(
            "Output Path", processor.output_path, 256
        )

        changed, processor.model_id = imgui.input_text(
            "Model ID", processor.model_id, 256
        )

        changed, processor.prompt = imgui.input_text(
            "Prompt", processor.prompt, 256
        )

        changed, processor.scale = imgui.slider_float(
            "Scale", processor.scale, 0.1, 2.0
        )

        changed, processor.acceleration = imgui.combo(
            "Acceleration", processor.acceleration, acceleration_options
        )

        changed, processor.use_denoising_batch = imgui.checkbox(
            "Use Denoising Batch", processor.use_denoising_batch
        )

        changed, processor.enable_similar_image_filter = imgui.checkbox(
            "Enable Similar Image Filter", processor.enable_similar_image_filter
        )

        changed, processor.seed = imgui.input_int(
            "Seed", processor.seed
        )

        if imgui.button("Process Video"):
            processor.process_video_threaded()

        if processor.is_processing:
            imgui.text("Processing...")
            imgui.progress_bar(processor.progress, (0, 0))

        imgui.end()

        gl.glClearColor(0.1, 0.1, 0.1, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

    impl.shutdown()
    glfw.terminate()

if __name__ == "__main__":
    main()