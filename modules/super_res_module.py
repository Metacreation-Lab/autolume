import os
import time

import cv2
import imgui
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np

from utils.gui_utils import imgui_utils
from super_res.super_res import main as super_res_main, load_model, get_resolution, check_width_height, get_audio, Reader, Writer

from dnnlib import EasyDict
import multiprocessing as mp

import gc

from widgets.browse_widget import BrowseWidget
from widgets.native_browser_widget import NativeBrowserWidget
from widgets.help_icon_widget import HelpIconWidget
import pandas as pd

args = EasyDict(result_path="", input_path=[""], model_type="Balance",
                outscale=3, width=4096, height=4096, sharpen_scale=1, scale_mode=0)
scale_factor = ['1', '2', '3', '4', '5', '6', '7', '8']


class SuperResModule:
    def __init__(self, menu):
        self.result_path = args.result_path
        self.input_path = args.input_path
        self.models = ['Quality','Balance','Fast']
        self.model_selected = 0
        self.model_type = self.models[self.model_selected]
        self.width = args.width
        self.height = args.height
        self.out_scale = args.outscale
        self.sharpen = args.sharpen_scale
        self.menu = menu
        self.app = menu.app
        # self.show_help = False  
        self.file_dialog = BrowseWidget(self, "Browse", os.path.abspath(os.getcwd()), ["*", ".mp4", ".avi", ".jpg", ".png", ".jpeg", ".bmp"], traverse_folders=True, width=self.app.button_w)
        self.save_path_browser = NativeBrowserWidget()
        self.scale_mode = 0
        self.running = False
        self.writer = None
        self.reader = None
        self.files = []
        self.file_idx = 0
        self.super_res_idx = 0
        self.total_frames = -1
        self.super_res_model = None
        self.start_time = 0
        self.eta = -1
        self.video_width = 0
        self.video_height = 0
        self.help_icon = HelpIconWidget()
        self.help_texts, self.help_urls = self.help_icon.load_help_texts("super_res")


    def display_progress(self):
        imgui.begin("Super Resolution", False)
        imgui.text('Super Resolution...')
        imgui.text("Files: " + str(self.file_idx + 1) + "/" + str(len(self.files)))
        imgui.text("Current File: " + self.files[self.file_idx])
        imgui.text("Progress: " + "#"*int((self.super_res_idx/self.total_frames*10) + 1) + " " + str((self.super_res_idx+1)/self.total_frames*100) + "%")
        # self.eta is in seconds so we convert it to hours minutes and seconds if not -1
        if self.eta != -1:
            hours = int(self.eta/3600)
            minutes = int((self.eta - hours*3600)/60)
            seconds = int(self.eta - hours*3600 - minutes*60)
            imgui.text("ETA: " + str(hours) + "h " + str(minutes) + "m " + str(seconds) + "s")
        imgui.text(str(self.super_res_idx) + "/" + str(self.total_frames) + " frames")
        imgui.end()
        self.perform_super_res()


    @imgui_utils.scoped_by_object_id
    def __call__(self):
        help_width = imgui.calc_text_size("(?)").x + 10
        button_width = self.app.button_w
        spacing = self.app.spacing
        input_width = -(button_width + spacing + help_width + 30)

        text = "Use AI to upscale your images and videos"
        text_width = imgui.calc_text_size(text).x
        window_width = imgui.get_window_width()
        help_icon_size = imgui.get_font_size()
        style = imgui.get_style()

        imgui.text(text)
        
        spacing = window_width - (style.window_padding[0] * 2) - text_width - help_icon_size - style.item_spacing[0] - 10
        
        imgui.same_line()
        imgui.dummy(spacing, 0)
        self.help_icon.render_with_url(self.help_texts.get("super_res_module"), self.help_urls.get("super_res_module"), "Read More")

        imgui.separator()

        if self.running:
            self.display_progress()

        # Input path
        joined = '\n'.join(self.input_path)
        imgui_utils.input_text("##SRINPUT", joined, 1024, 
                              flags=imgui.INPUT_TEXT_READ_ONLY, 
                              width=input_width, 
                              help_text="Input Files")
        
        imgui.same_line()
        _clicked, input = self.file_dialog(button_width)
        if _clicked:
            self.input_path = input
            print(self.input_path)

        # Result path
        imgui.text("Save Path")
        _, self.result_path = imgui_utils.input_text("##save_path", self.result_path, 1024, 0,
                                                     width=imgui.get_window_width() - self.menu.app.button_w - imgui.calc_text_size("Browse")[0])
        
        imgui.same_line()
        if imgui.button("Browse##super_res_result_path", width=button_width):
            directory_path = self.save_path_browser.select_directory("Select Save Directory")
            if directory_path:
                self.result_path = directory_path.replace('\\', '/')
            else:
                print("No save path selected")
        self.models = ['Quality','Balance','Fast']
        if len(self.models) > 0:
            # Model selection
            imgui.text("Model")
            imgui.same_line()
            with imgui_utils.item_width(input_width):
                _, self.model_selected = imgui.combo("##model", self.model_selected, self.models)
            self.model_type = self.models[self.model_selected]

        # Scale mode
        imgui.text("Scale Mode")
        imgui.same_line()
        with imgui_utils.item_width(input_width):
            clicked, self.scale_mode = imgui.combo("##scale_mode", self.scale_mode, ["Custom", "Scale"])
        if clicked:
            print(self.scale_mode)

        # Scale factor or custom resolution
        if self.scale_mode:
            imgui.text("Scale Factor")
            imgui.same_line()
            with imgui_utils.item_width(input_width):
                _, self.out_scale = imgui.combo("##scale_factor", self.out_scale, scale_factor)
        else:
            imgui.text("Height")
            imgui.same_line()
            with imgui_utils.item_width(input_width):
                _, self.height = imgui.input_int("##height", self.height)
            
            imgui.text("Width")
            imgui.same_line()
            with imgui_utils.item_width(input_width):
                _, self.width = imgui.input_int("##width", self.width)

        # Sharpening
        imgui.text("Sharpening")
        imgui.same_line()
        with imgui_utils.item_width(input_width):
            _, self.sharpen = imgui.input_int("##sharpening", self.sharpen)
        if self.sharpen < 1:
            self.sharpen = 1


        try:
            if imgui.button("Super Resolution", width=imgui.get_content_region_available_width()) and not self.running:
                self.running = True
                print("Super Resolution")
                args.result_path = self.result_path
                args.input_path = self.input_path
                args.model_type = self.model_type
                args.outscale = self.out_scale + 1
                args.out_height = self.height
                args.out_width = self.width
                args.sharpen_scale = self.sharpen
                args.scale_mode = self.scale_mode
                self.args = args
                print("Starting Super Resolution")
                self.start_super_res()

        except Exception as e:
            print("SRR ERROR", e)




    def start_super_res(self):
        self.start_time = time.time()
        if self.model_type == "Quality":
            model_path = "./sr_models/Quality.pth"
        elif self.model_type == "Balance":
            model_path = "./sr_models/Balance.pth"
        elif self.model_type == "Fast":
            model_path = "./sr_models/Fast.pt"

        self.super_res_model = load_model(self.model_type, model_path)
        self.files = self.input_path

        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        self.file_idx = 0
        self.super_res_idx = 0

        if len(self.files) == 0:
            self.running = False


    # def perform_super_res(self):
    #     self.start_time = time.time()

    #     # 确保只有当super_res_idx为0时，才开始处理新的文件
    #     if self.super_res_idx == 0:
    #         file = self.files[self.file_idx]
    #         self.start_time = time.time()
    #         head, tail = os.path.split(file)
            
    #         # 检查文件是否为图像或视频
    #         if file.lower().endswith(('jpg', 'png', 'jpeg', 'bmp')):
    #             data_transformer = transforms.Compose([transforms.ToTensor()])
    #             image = cv2.imread(file)
    #             input_height, input_width = image.shape[0], image.shape[1]
    #             image = data_transformer(image).to('cuda')
    #             input = torch.unsqueeze(image, 0)

    #             # 处理图像
    #             with torch.inference_mode():
    #                 output = self.super_res_model(input)
    #                 output = F.adjust_sharpness(output, self.args.sharpen_scale) * 255
    #                 output = output[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    
    #                 if self.args.scale_mode:
    #                     if self.args.outscale != 4:
    #                         output = cv2.resize(output, 
    #                                             (int(input_width * self.args.outscale), 
    #                                             int(input_height * self.args.outscale)), 
    #                                             interpolation=cv2.INTER_LINEAR)
    #                 else:
    #                     output = cv2.resize(output, 
    #                                         (int(self.args.out_width), 
    #                                         int(self.args.out_height)), 
    #                                         interpolation=cv2.INTER_LINEAR)

    #                 path = os.path.join(self.args.result_path, 
    #                                     tail[:-4] + f'_result_{self.args.model_type}_{int(input_width * self.args.outscale)}x{int(input_height * self.args.outscale)}_Sharpness{self.args.sharpen_scale}.jpg')
    #                 cv2.imwrite(path, output)
    #             self.file_idx += 1  # 递增file_idx

    #         elif file.lower().endswith(('mp4', 'avi', 'mov')):
    #             # 处理视频文件
    #             audio = get_audio(file)
    #             self.video = cv2.VideoCapture(file)
    #             self.fps = self.video.get(cv2.CAP_PROP_FPS)
    #             self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
    #             self.video_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
    #             self.video_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #             # 设置 video_save_path
    #             if self.args.scale_mode:
    #                 output_width = int(self.video_width * self.args.outscale)
    #                 output_height = int(self.video_height * self.args.outscale)
    #             else:
    #                 output_width = int(self.args.out_width)
    #                 output_height = int(self.args.out_height)

    #             # 生成视频保存路径
    #             self.video_save_path = os.path.join(self.args.result_path,
    #                                                 tail[:-4] + f'_result_{self.args.model_type}_{output_width}x{output_height}_Sharpness{self.args.sharpen_scale}.mp4')

    #             print(f"Saving video to {self.video_save_path}")


    #             self.writer = Writer(self.args, audio, self.video_height, self.video_width, 
    #                                 video_save_path=self.video_save_path, fps=self.fps)
    #             self.reader = Reader(self.video_width, self.video_height, file)

    #             # 处理视频的每一帧
    #             if self.super_res_idx < self.total_frames:
    #                 img = self.reader.get_frame()
    #                 if img is not None:
    #                     sr_input = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().to('cuda') / 255
    #                     with torch.inference_mode():
    #                         sr_output = self.super_res_model(sr_input)
    #                         sr_output = F.adjust_sharpness(sr_output, self.args.sharpen_scale) * 255
    #                         sr_output = sr_output[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    #                         # 缩放和保存
    #                         if self.args.scale_mode:
    #                             if self.args.outscale != 4:
    #                                 sr_output = cv2.resize(sr_output, 
    #                                                     (int(self.video_width * self.args.outscale), 
    #                                                         int(self.video_height * self.args.outscale)), 
    #                                                     interpolation=cv2.INTER_LINEAR)
    #                         else:
    #                             sr_output = cv2.resize(sr_output, 
    #                                                 (int(self.args.out_width), 
    #                                                     int(self.args.out_height)), 
    #                                                 interpolation=cv2.INTER_LINEAR)

    #                         self.writer.write_frame(sr_output)
    #                         self.super_res_idx += 1
    #             else:
    #                 if self.writer is not None:
    #                     self.writer.close()
    #                 self.super_res_idx = 0
    #                 self.file_idx += 1

    #     torch.cuda.empty_cache()
    #     gc.collect()

    #     # 当所有文件处理完后，停止运行
    #     if self.file_idx >= len(self.files):
    #         self.running = False

   


    def perform_super_res(self):
        self.start_time = time.time()

        # 确保只有当super_res_idx为0时，才开始处理新的文��
        if self.super_res_idx == 0:
            file = self.files[self.file_idx]
            self.start_time = time.time()
            head, tail = os.path.split(file)
            
            # 检查文件是否为图像或视频
            if file.lower().endswith(('jpg', 'png', 'jpeg', 'bmp')):
                data_transformer = transforms.Compose([transforms.ToTensor()])
                image = cv2.imread(file)
                input_height, input_width = image.shape[0], image.shape[1]
                image = data_transformer(image).to('cuda')
                input = torch.unsqueeze(image, 0)

                # 处理图像
                with torch.inference_mode():
                    output = self.super_res_model(input)
                    output = F.adjust_sharpness(output, self.args.sharpen_scale) * 255
                    output = output[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    
                    if self.args.scale_mode:
                        if self.args.outscale != 4:
                            output = cv2.resize(output, 
                                                (int(input_width * self.args.outscale), 
                                                int(input_height * self.args.outscale)), 
                                                interpolation=cv2.INTER_LINEAR)
                    else:
                        output = cv2.resize(output, 
                                            (int(self.args.out_width), 
                                            int(self.args.out_height)), 
                                            interpolation=cv2.INTER_LINEAR)

                    path = os.path.join(self.args.result_path, 
                                        tail[:-4] + f'_result_{self.args.model_type}_{int(input_width * self.args.outscale)}x{int(input_height * self.args.outscale)}_Sharpness{self.args.sharpen_scale}.jpg')
                    cv2.imwrite(path, output)

                # 在图像处理后，递增 file_idx 和 super_res_idx
                self.file_idx += 1
                self.super_res_idx = 0

            elif file.lower().endswith(('mp4', 'avi', 'mov')):
                # 视频处理逻辑...
                self.process_video(file)  # 调用视频处理方法
                self.file_idx += 1

        # 清理内存
        torch.cuda.empty_cache()
        gc.collect()

        # 当所有文件处理完后停止运行
        if self.file_idx >= len(self.files):
            self.running = False
    
    def process_video(self, file):
        """处理视频文件的逻辑"""
        audio = get_audio(file)  # 获取音频信息
        video = cv2.VideoCapture(file)
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        head, tail = os.path.split(file)

        # 设置视频保存路径
        if self.args.scale_mode:
            video_save_path = os.path.join(self.args.result_path, 
                                        tail[:-4] + f'_result_{self.args.model_type}_{int(video_width * self.args.outscale)}x{int(video_height * self.args.outscale)}_Sharpness{self.args.sharpen_scale}.mp4')
        else:
            video_save_path = os.path.join(self.args.result_path, 
                                        tail[:-4] + f'_result_{self.args.model_type}_{int(self.args.out_width)}x{int(self.args.out_height)}_Sharpness{self.args.sharpen_scale}.mp4')

        print(f"Saving video to {video_save_path}")

        # 初始化 Writer 和 Reader
        writer = Writer(self.args, audio, video_height, video_width, 
                        video_save_path=video_save_path, fps=fps)
        reader = Reader(video_width, video_height, file)
        super_res_idx = 0  # 初始化帧计数

        # 处理每一帧
        while super_res_idx < total_frames:
            print(f"Processing frame {super_res_idx}/{total_frames}")
            img = reader.get_frame()
            if img is not None:
                with torch.inference_mode():
                    sr_input = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().to('cuda') / 255
                    sr_output = self.super_res_model(sr_input)
                    sr_output = F.adjust_sharpness(sr_output, self.args.sharpen_scale) * 255
                    sr_output = sr_output[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)

                    # 根据 scale_mode 进行缩放
                    if self.args.scale_mode:
                        sr_output = cv2.resize(sr_output, 
                                            (int(video_width * self.args.outscale), 
                                                int(video_height * self.args.outscale)), 
                                            interpolation=cv2.INTER_LINEAR)
                    else:
                        sr_output = cv2.resize(sr_output, 
                                            (int(self.args.out_width), 
                                                int(self.args.out_height)), 
                                            interpolation=cv2.INTER_LINEAR)

                    print(f"Saving frame {super_res_idx} to {video_save_path}")
                    writer.write_frame(sr_output)
                    super_res_idx += 1  # 处理下一帧

        # 完成后关闭 writer
        writer.close()

        # 清理内存
        torch.cuda.empty_cache()
        gc.collect()


    
