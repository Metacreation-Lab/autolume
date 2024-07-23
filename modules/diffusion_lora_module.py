import os
import zipfile

import imgui
import multiprocessing as mp

import dnnlib
from dnnlib import EasyDict
from utils.gui_utils import imgui_utils
from streamdiffusion.train_text_to_image_lora import main as train_main
from utils import dataset_tool
from widgets.browse_widget import BrowseWidget
import cv2
from utils.gui_utils import gl_utils

from PIL import Image
import PIL
import os

augs = ["ADA", "DiffAUG"]
modes = ["crop", "expand", "nearest", "box", "bilinear", "hamming", "bicubic", "lanczos"]
ada_pipes = ['blit', 'geom', 'color', 'filter', 'noise', 'cutout', 'bg', 'bgc', 'bgcf', 'bgcfn', 'bgcfnc']
diffaug_pipes = ['color,translation,cutout', 'color,translation', 'color,cutout', 'color',
                 'translation', 'cutout,translation', 'cutout']
configs = ['auto', 'stylegan2', 'paper256', 'paper512', 'paper1024', 'cifar']


class DiffusionLoraModule:
    def __init__(self, menu):
        cwd = os.getcwd()
        self.args = EasyDict(
            pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4",
            dataset_name="lambdalabs/naruto-blip-captions",
            use_ema=True,
            resolution=512,
            center_crop=True,
            random_flip=True,
            train_batch_size=1,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            max_train_steps=1,
            learning_rate=1e-05,
            max_grad_norm=1,
            lr_scheduler="constant",
            lr_warmup_steps=0,
            output_dir="sd-naruto-model",
        )

        self.args.output_dir = os.path.join(cwd, "training-runs")
        # create data folder if not exists
        if not os.path.exists(os.path.abspath(os.path.join(os.getcwd(), "data"))):
            os.makedirs(os.path.abspath(os.path.join(os.getcwd(), "data")))
        self.file_dialog = BrowseWidget(menu, "Dataset", os.path.abspath(os.path.join(os.getcwd(), "data")), ["*", ""],
                                        multiple=False, traverse_folders=False, width=menu.app.button_w)
        self.app = menu.app

        self.output_dir_dialog = BrowseWidget(self, "Save Path", os.path.abspath(os.getcwd()), [""], multiple=False,
                                              traverse_folders=False, add_folder_button=True, width=menu.app.button_w)

        self.menu = menu

        self.queue = mp.Queue()
        self.reply = mp.Queue()
        self.message = ""
        self.done = False
        self.training_process = mp.Process(target=train_main, args=(self.queue, self.reply), name='TrainingProcess')
        self.fps = 10
        self.found_video = False
        self._zipfile = None

        self.mirror = True
        self.done_button = False

    @imgui_utils.scoped_by_object_id
    def __call__(self):
        if self.reply.qsize() > 0:
            self.message, self.done = self.reply.get()
            while self.reply.qsize() > 0:
                self.message, self.done = self.reply.get()

            print(self.message, self.done)

        # imgui.text("Train a diffusion model on your own data with LoRA")

        _, self.args.pretrained_model_name_or_path = imgui.input_text("Pretrained Model Name or Path",
                                                                      self.args.pretrained_model_name_or_path, 1024)

        _, self.args.output_dir = imgui.input_text("Save Path", self.args.output_dir, 1024)
        imgui.same_line()
        _clicked, output_dir = self.output_dir_dialog(self.menu.app.button_w)
        if _clicked:
            if len(output_dir) > 0:
                self.output_dir = output_dir[0]
                print(self.output_dir)
            else:
                self.save_path = ""
                print("No path selected")

        _, self.args.dataset_name = imgui.input_text("Dataset Name or Path", self.args.dataset_name, 1024)
        imgui.same_line()
        _clicked, dataset_name = self.file_dialog(self.menu.app.button_w)
        if _clicked:
            self.args.dataset_name = dataset_name[0]

        _, self.args.use_ema = imgui.checkbox("Use EMA", self.args.use_ema)

        _, self.args.resolution = imgui.input_int("Resolution", self.args.resolution)
        _, self.args.train_batch_size = imgui.input_int("Train Batch Size", self.args.train_batch_size)
        _, self.args.gradient_accumulation_steps = imgui.input_int("Gradient Accumulation Steps",
                                                                   self.args.gradient_accumulation_steps)
        _, self.args.max_train_steps = imgui.input_int("Max Train Steps", self.args.max_train_steps)
        _, self.args.learning_rate = imgui.input_float("Learning Rate", self.args.learning_rate)
        _, self.args.max_grad_norm = imgui.input_float("Max Grad Norm", self.args.max_grad_norm)
        _, self.args.lr_warmup_steps = imgui.input_int("LR Warmup Steps", self.args.lr_warmup_steps)

        _, self.args.center_crop = imgui.checkbox("Center Crop", self.args.center_crop)
        _, self.args.random_flip = imgui.checkbox("Random Flip", self.args.random_flip)
        _, self.args.gradient_checkpointing = imgui.checkbox("Gradient Checkpointing", self.args.gradient_checkpointing)

        lr_scheduler_options = ["constant", "linear", "cosine", "cosine_with_restarts", "polynomial",
                                "constant_with_warmup"]
        _, selected_lr_scheduler = imgui.combo("LR Scheduler", lr_scheduler_options.index(self.args.lr_scheduler),
                                               lr_scheduler_options)
        self.args.lr_scheduler = lr_scheduler_options[selected_lr_scheduler]

        imgui.set_next_window_size(self.menu.app.content_width // 4, (self.menu.app.content_height // 4), imgui.ONCE)

        if imgui.button("Train", width=-1):
            imgui.open_popup("Training")
            print("training")

            if self.done == True:
                self.queue = mp.Queue()
                self.reply = mp.Queue()
                self.training_process = mp.Process(target=train_main, args=(self.queue, self.reply),
                                                   name='TrainingProcess')
                self.done = False
            print(self.args)
            self.queue.put(self.args)
            self.training_process.start()

        imgui.set_next_window_size(self.menu.app.content_width // 2, (self.menu.app.content_height // 2), imgui.ONCE)

        if imgui.begin_popup_modal("Training")[0]:
            imgui.text("Training...")
            if self.message != "":
                imgui.text(self.message)
            if imgui_utils.button("Done", enabled=1):
                self.queue.put('done')
                self.done_button = True
            if self.done:
                self.training_process.terminate()
                self.training_process.join()
                if self.done_button == True:
                    imgui.close_current_popup()
                    self.message = ''
                    self.done_button = False
            imgui.end_popup()
