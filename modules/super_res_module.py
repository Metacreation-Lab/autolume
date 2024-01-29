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

args = EasyDict(result_path="", input_path=[""], model_type="Balance",
                outscale=3, width=4096, height=4096, sharpen_scale=1, scale_mode=0)
scale_factor = ['1', '2', '3', '4', '5', '6', '7', '8']


class SuperResModule:
    def __init__(self, menu):
        self.result_path=args.result_path
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
        self.file_dialog = BrowseWidget(self, "Browse", os.path.abspath(os.getcwd()), ["*", ".mp4", ".avi", ".jpg", ".png", ".jpeg", ".bmp"], traverse_folders=True, width=self.app.button_w)
        self.save_path_dialog = BrowseWidget(self, "Save Path", os.path.abspath(os.getcwd()), [""], multiple=False, traverse_folders=False, add_folder_button=True,  width=self.app.button_w)
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
        if self.running:
            self.display_progress()

        joined = '\n'.join(self.input_path)
        imgui_utils.input_text("##SRINPUT", joined, 1024, flags=imgui.INPUT_TEXT_READ_ONLY, width=-(self.app.button_w + self.app.spacing), help_text="Input Files")
        imgui.same_line()
        _clicked, input = self.file_dialog(self.app.button_w)
        if _clicked:
            self.input_path = input
            print(self.input_path)

        imgui_utils.input_text("##SRRESULT", self.result_path, 1024, flags=imgui.INPUT_TEXT_READ_ONLY, width=-(self.app.button_w + self.app.spacing), help_text="Result Path")
        imgui.same_line()
        _clicked, save_path = self.save_path_dialog(self.app.button_w)
        if _clicked:
            if len(save_path) > 0:
                self.result_path = save_path[0]
                print(self.result_path)
            else:
                self.result_path = ""
                print("No path selected")
        self.models = ['Quality','Balance','Fast']
        if len(self.models) > 0:
            with imgui_utils.item_width(-(self.app.button_w + self.app.spacing)):
                _, self.model_selected = imgui.combo("Model", self.model_selected, self.models)
            self.model_type = self.models[self.model_selected]
        with imgui_utils.item_width(-(self.app.button_w + self.app.spacing)):
            clicked, self.scale_mode = imgui.combo("Scale Mode", self.scale_mode, ["Custom", "Scale"])
        if clicked:
            print(self.scale_mode)
        if self.scale_mode:

            with imgui_utils.item_width(-(self.app.button_w + self.app.spacing)):
                _, self.out_scale = imgui.combo("Scale Factor", self.out_scale, scale_factor)
        else:

            with imgui_utils.item_width(-(self.app.button_w + self.app.spacing)):
                _, self.height = imgui.input_int("Height", self.height)
                _, self.width = imgui.input_int("Width", self.width)

        with imgui_utils.item_width(-(self.app.button_w + self.app.spacing)):
            _, self.sharpen = imgui.input_int("Sharpening", self.sharpen)
        if self.sharpen<1:
            self.sharpen=1
        if imgui.is_item_hovered():
            imgui.set_tooltip("Additional sharpening performed after super resolution")
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


    def perform_super_res(self):
        self.start_time = time.time()
        if self.super_res_idx == 0:
            file = self.files[self.file_idx]
            self.start_time = time.time()
            head, tail = os.path.split(file)
            if file[-3:] == 'jpg' or file[-3:] == 'png':
                data_transformer = transforms.Compose([transforms.ToTensor()])
                image = cv2.imread(file)
                input_height, input_width = image.shape[0], image.shape[1]
                print("INPUT DIMENSIONS", input_width, input_height, image.shape)
                image = data_transformer(image).to('cuda')
                input = torch.unsqueeze(image, 0)

                with torch.inference_mode():
                    output = self.super_res_model(input)
                    print("OUTPUT DIMENSIONS", output.shape)
                    output = F.adjust_sharpness(output, self.args.sharpen_scale) * 255

                    output = output[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    print('Output shape:' + str(output.shape))
                    if self.args.scale_mode:
                        if self.args.outscale != 4:
                            output = cv2.resize(
                                output, (
                                    int(input_width * self.args.outscale),
                                    int(input_height * self.args.outscale),
                                ), interpolation=cv2.INTER_LINEAR)


                    else:
                        output = cv2.resize(
                            output, (
                                int(self.args.out_width),
                                int(self.args.out_height),
                            ), interpolation=cv2.INTER_LINEAR)

                if self.args.scale_mode:
                    print("USING these params", input_width, input_height, self.args.outscale)
                    path = os.path.join(self.args.result_path,
                                        tail[
                                        :-4] + f'_result_{self.args.model_type}_{int(input_width * self.args.outscale)}x{int(input_height * self.args.outscale)}_Sharpness{self.args.sharpen_scale}.jpg')

                else:
                    path = os.path.join(self.args.result_path,
                                        tail[
                                        :-4] + f'_result_{self.args.model_type}_{int(self.args.out_width)}x{int(self.args.out_height)}_Sharpness{self.args.sharpen_scale}.jpg')

                print("Saving image to {}".format(path))
                cv2.imwrite(path, output)
                self.file_idx += 1
            if file[-3:] == 'mp4' or file[-3:] == 'avi' or file[-3:] == 'mov':
                audio = get_audio(file)
                self.video = cv2.VideoCapture(file)
                self.fps = self.video.get(cv2.CAP_PROP_FPS)
                self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
                self.video_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.video_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if args.scale_mode:
                    self.video_save_path = os.path.join(self.args.result_path, tail[:-4] + f'_result_{self.args.model_type}_{int(self.video_width * args.outscale)}x{int(self.video_height * self.args.outscale)}_Sharpness{self.args.sharpen_scale}.mp4')
                else:
                    self.video_save_path = os.path.join(self.args.result_path, tail[:-4] + f'_result_{self.args.model_type}_x{int(self.args.out_width)}x{int(self.args.out_height)}_Sharpness{self.args.sharpen_scale}.mp4')


                self.writer = Writer(self.args, audio, self.video_height, self.video_width, video_save_path=self.video_save_path, fps=self.fps)
                self.reader = Reader(self.video_width, self.video_height, file)

                self.super_res_idx = 0

        if self.super_res_idx < self.total_frames:
            print(self.super_res_idx, self.total_frames)
            img = self.reader.get_frame()
            if img is not None:

                with torch.inference_mode():
                    sr_input = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().to('cuda')/255
                    sr_output = self.super_res_model(sr_input)
                    sr_output = F.adjust_sharpness(sr_output, self.args.sharpen_scale) * 255

                    sr_output = sr_output[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)

                    if self.args.scale_mode:
                            if self.args.outscale != 4:
                                sr_output = cv2.resize(
                                    sr_output, (
                                        int(self.video_width * self.args.outscale),
                                        int(self.video_height * self.args.outscale),
                                    ), interpolation=cv2.INTER_LINEAR)


                    else:
                        sr_output = cv2.resize(
                            sr_output, (
                                int(self.args.out_width),
                                int(self.args.out_height),
                            ), interpolation=cv2.INTER_LINEAR)
                    print("Saving frame {} to {}".format(self.super_res_idx, self.video_save_path))
                    self.writer.write_frame(sr_output)
                    ret, img = self.video.read()
                    self.super_res_idx += 1
            self.eta = (time.time() - self.start_time) * (self.total_frames - self.super_res_idx)

        else:
            if self.writer is not None:
                self.writer.close()
            self.super_res_idx = 0
            self.file_idx += 1
            # torch clear cuda cache
            torch.cuda.empty_cache()
            gc.collect()
        if self.file_idx >= len(self.files):
            self.running = False

