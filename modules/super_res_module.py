import os
import time

import cv2
import imgui
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np

from utils import device_utils
from utils.device_utils import get_device
from utils.gui_utils import imgui_utils
from super_res.super_res import main as super_res_main, load_model, get_resolution, check_width_height, get_audio, Reader, Writer, run_super_res

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
        self.queue = mp.Queue()
        self.reply = mp.Queue()
        self.sr_process = None
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
        if self.file_idx < len(self.files):
            imgui.text("Current File: " + self.files[self.file_idx])
        if self.total_frames > 0:
            percent = (self.super_res_idx + 1) / self.total_frames * 100
            bar = "#" * int(self.super_res_idx / self.total_frames * 10 + 1)
            imgui.text("Progress: " + bar + " " + f"{percent:.1f}%")
        else:
            imgui.text("Progress: preparing...")
        # self.eta is in seconds so we convert it to hours minutes and seconds if not -1
        if self.eta != -1:
            hours = int(self.eta/3600)
            minutes = int((self.eta - hours*3600)/60)
            seconds = int(self.eta - hours*3600 - minutes*60)
            imgui.text("ETA: " + str(hours) + "h " + str(minutes) + "m " + str(seconds) + "s")
        imgui.text(str(self.super_res_idx) + "/" + str(self.total_frames) + " frames")
        imgui.end()


    @imgui_utils.scoped_by_object_id
    def __call__(self):
        if not self.reply.empty():
            msg = self.reply.get()
            while not self.reply.empty():
                msg = self.reply.get()
            self.file_idx, self.super_res_idx, self.total_frames, self.eta, done = msg
            if done:
                self.running = False
                if self.sr_process is not None:
                    self.sr_process.join()
                    self.sr_process = None
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
        self.files = self.input_path

        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        self.file_idx = 0
        self.super_res_idx = 0
        self.total_frames = -1
        self.eta = -1

        if len(self.files) == 0:
            self.running = False
            return

        # Run upscaling in a separate process so it gets the GPU at full speed and the UI stays responsive
        self.queue = mp.Queue()
        self.reply = mp.Queue()
        self.sr_process = mp.Process(target=run_super_res, args=(self.queue, self.reply), daemon=True)
        self.sr_process.start()
        self.queue.put(self.args)
