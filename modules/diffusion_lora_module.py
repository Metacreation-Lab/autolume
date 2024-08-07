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


class DiffusionLoraModule:
    def __init__(self, menu):
        cwd = os.getcwd()
        self.args = EasyDict(
            pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4",
            train_data_dir=os.path.abspath(os.path.join(cwd, "data")),
            caption_column="text",
            resolution=512,
            random_flip=True,
            train_batch_size=1,
            num_train_epochs=100,
            checkpointing_steps=5000,
            learning_rate=0.001,
            lr_scheduler="constant",
            lr_warmup_steps=0,
            seed=42,
            output_dir="sd-model-finetuned-lora",
            validation_prompt="cute dragon creature",
        )

        self.args.output_dir = os.path.join(cwd, "sd-model-finetuned-lora")
        # create data folder if not exists
        if not os.path.exists(os.path.abspath(os.path.join(os.getcwd(), "data"))):
            os.makedirs(os.path.abspath(os.path.join(os.getcwd(), "data")))

        self.use_model_name = True
        self.model_path_dialog = BrowseWidget(self, "Browse", os.path.abspath(os.getcwd()),
                                              ["*"], multiple=False, traverse_folders=False,
                                              width=menu.app.button_w)

        self.dataset_dir_dialog = BrowseWidget(menu, "Dataset", os.path.abspath(os.path.join(os.getcwd(), "data")),
                                               ["*", ""],
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

        self.model_path = ""
        self.output_path = ""
        self.use_dataset_name = True
        self.validation_prompt = "cute dragon creature"
        self.caption_column = "text"

    @imgui_utils.scoped_by_object_id
    def __call__(self):
        if self.reply.qsize() > 0:
            self.message, self.done = self.reply.get()
            while self.reply.qsize() > 0:
                self.message, self.done = self.reply.get()

            print(self.message, self.done)

        # pretrained_model_name_or_path

        imgui.text("Use Model Name or Model Path")
        imgui.same_line()
        if imgui.radio_button("Use Name", self.use_model_name):
            self.use_model_name = True

        imgui.same_line()

        if imgui.radio_button("Use Path", not self.use_model_name):
            self.use_model_name = False

        if self.use_model_name:
            with imgui_utils.item_width(-(self.app.button_w + self.app.spacing) * 1.7):
                _, self.args.pretrained_model_name_or_path = imgui.input_text("Pretrained Model Name",
                                                                              self.args.pretrained_model_name_or_path,
                                                                              1024)
        else:
            imgui_utils.input_text("##SRModel Path", "", 1024,
                                   flags=imgui.INPUT_TEXT_READ_ONLY,
                                   width=-(self.app.button_w + self.app.spacing), help_text="Model Path")
            imgui.same_line()

            _clicked, model_path = self.model_path_dialog(self.app.button_w)
            if _clicked and len(model_path) > 0:
                self.args.pretrained_model_name_or_path = model_path[0]
                print(self.args.pretrained_model_name_or_path)

        # output_dir
        with imgui_utils.item_width(-(self.app.button_w + self.app.spacing) * 1.7):
            imgui_utils.input_text("##SROUTPUT PATH", self.args.output_dir, 1024, flags=imgui.INPUT_TEXT_READ_ONLY,
                                   width=-(self.app.button_w + self.app.spacing) * 1.7, help_text="Output Path")
        imgui.same_line()

        _clicked, save_path = self.output_dir_dialog(self.menu.app.button_w)
        if _clicked:
            if len(save_path) > 0:
                self.args.output_dir = save_path[0]
                print(self.args.output_dir)
            else:
                self.args.output_dir = ""
                print("No path selected")

        # dataset_path
        imgui_utils.input_text("##SRDATASET PATH", self.args.train_data_dir, 1024,
                               flags=imgui.INPUT_TEXT_READ_ONLY,
                               width=-(self.app.button_w + self.app.spacing) * 1.7, help_text="Dataset Path")
        imgui.same_line()

        _clicked, dataset_path = self.dataset_dir_dialog(self.app.button_w)
        if _clicked and len(dataset_path) > 0:
            self.args.train_data_dir = dataset_path[0]
            print(self.args.train_data_dir)

        with imgui_utils.item_width(-(self.app.button_w + self.app.spacing) * 1.7):
            changed, caption_column = imgui_utils.input_text("Caption Column", self.caption_column, 1024,
                                                             flags=imgui.INPUT_TEXT_AUTO_SELECT_ALL,
                                                             help_text='The column of the dataset containing a caption or a list of captions.',
                                                             width=-(self.app.button_w + self.app.spacing) * 1.7, )
            if changed:
                self.args.caption_column = caption_column
            _, self.args.resolution = imgui.input_int("Resolution", self.args.resolution)
            _, self.args.random_flip = imgui.checkbox("Random Flip", self.args.random_flip)
            _, self.args.train_batch_size = imgui.input_int("Train Batch Size", self.args.train_batch_size)
            _, self.args.num_train_epochs = imgui.input_int("Num Train Epochs", self.args.num_train_epochs)
            _, self.args.checkpointing_steps = imgui.input_int("Checkpointing Steps", self.args.checkpointing_steps)
            _, self.args.learning_rate = imgui.input_float("Learning Rate", self.args.learning_rate)
            lr_scheduler_options = ["constant", "linear", "cosine", "cosine_with_restarts", "polynomial",
                                    "constant_with_warmup"]
            _, selected_lr_scheduler = imgui.combo("LR Scheduler", lr_scheduler_options.index(self.args.lr_scheduler),
                                                   lr_scheduler_options)
            self.args.lr_scheduler = lr_scheduler_options[selected_lr_scheduler]
            _, self.args.lr_warmup_steps = imgui.input_int("LR Warmup Steps", self.args.lr_warmup_steps)

            _, self.args.seed = imgui.input_int("Seed", self.args.seed)
            changed, validation_prompt = imgui_utils.input_text("Validation Prompt", self.validation_prompt, 1024,
                                                                flags=imgui.INPUT_TEXT_AUTO_SELECT_ALL,
                                                                help_text='A prompt that is sampled during training for inference.',
                                                                width=-(self.app.button_w + self.app.spacing) * 1.7, )
            if changed:
                self.args.validation_prompt = validation_prompt

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

            # Check if the process is already running before starting it
            if self.training_process.is_alive():
                self.training_process.terminate()
                self.training_process.join()

            # Create a new process object and start it
            self.training_process = mp.Process(target=train_main, args=(self.queue, self.reply), name='TrainingProcess')
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
            if self.done_button:
                self.training_process.terminate()
                self.training_process.join()
                imgui.close_current_popup()
                self.message = ''
                self.done_button = False
            imgui.end_popup()
