import logging
from pathlib import Path
import zipfile

import imgui
import multiprocessing as mp
import psutil

import dnnlib
from utils.app_logging import LoggedProcess
from utils.gui_utils import imgui_utils
from train import main as train_main
from widgets.native_browser_widget import NativeBrowserWidget
from utils.user_data import data_path
from widgets.help_icon_widget import HelpIconWidget
from widgets.model_dropdown_widget import ModelDropdownButton

import cv2
from utils.gui_utils import gl_utils

augs = ["ADA", "DiffAUG"]
ada_pipes = ['blit', 'geom', 'color', 'filter', 'noise', 'cutout', 'bg', 'bgc', 'bgcf', 'bgcfn', 'bgcfnc']
diffaug_pipes = ['color,translation,cutout', 'color,translation', 'color,cutout', 'color',
                 'translation', 'cutout,translation', 'cutout']
configs = ['auto', 'stylegan2', 'paper256', 'paper512', 'paper1024', 'cifar']
resize_mode = ['stretch','center crop']
MBSTD_GROUP = 2
BATCH_SIZE_CHOICES = [MBSTD_GROUP * x for x in range(1, 33)]

logger = logging.getLogger(__name__)

class TrainingModule:
    def __init__(self, menu):
        self.save_path = str(data_path("training-runs"))
        self.data_path = str(data_path("datasets"))
        self.app = menu.app
        self.config = 1
        self.resume_pkl = ""
        self.aug = 0
        self.ada_pipe = 7
        self.diffaug_pipe = 0
        self.batch_size = 8
        if self.batch_size not in BATCH_SIZE_CHOICES:
            self.batch_size = min(BATCH_SIZE_CHOICES, key=lambda x: abs(x - self.batch_size))

        # Native browsers for main training paths
        self.data_path_browser = NativeBrowserWidget()
        self.save_path_browser = NativeBrowserWidget()

        self.model_dropdown = ModelDropdownButton(label='Browse', include_models=False)

        self.menu = menu

        self.help_icon = HelpIconWidget()
        self.help_texts, self.help_urls = self.help_icon.load_help_texts("training")

        self.queue = mp.Queue()
        self.reply = mp.Queue()
        self.message = ""
        self.done = False
        self.training_process = LoggedProcess(target=train_main, args=(self.queue, self.reply), name='training')
        self.found_video = False
        self._zipfile = None
        self.gamma = 10
        self.glr = 0.002
        self.dlr = 0.002
        self.snap = 4
        self.mirror = False # Mirror only accesible in preprocessing module
        self.done_button = False
        self.image_path = ''
        self.resize_mode = 0
        self.fps = 10

    @property
    def is_training(self):
        return self.training_process.is_alive()

    @staticmethod
    def _file_ext(fname):
        return Path(fname).suffix.lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self.data_path)
        return self._zipfile

    def _start_training(self):
        target_data_path = self.data_path

        # Manipulate resolution training parameter based on dataset resolution (for now)
        # Read resolution from first image in dataset
        detected_resolution = None
        target_path = Path(target_data_path)
        if target_path.is_dir():
            image_files = [f for f in target_path.iterdir()
                        if f.is_file() and f.suffix.lower() == '.png']
            if image_files:
                first_image_path = str(image_files[0])
                img = cv2.imread(first_image_path)
                if img is not None:
                    height, width = img.shape[:2]
                    detected_resolution = (width, height)
                    logger.debug("Detected image resolution from dataset: %s", detected_resolution)

        kwargs = dnnlib.EasyDict(
            outdir=self.save_path,
            data=str(target_data_path),
            cfg=configs[self.config],
            batch=self.batch_size,
            topk=None,
            gpus=1,
            gamma=self.gamma,
            z_dim=512,
            w_dim=512,
            cond=False,
            mirror=self.mirror,
            resolution=detected_resolution,
            resize_mode = resize_mode[self.resize_mode],
            aug="ada" if augs[self.aug] == "ADA" else "noaug",
            augpipe=ada_pipes[self.ada_pipe],
            resume=self.resume_pkl if self.resume_pkl != "" else None,
            freezed=0,
            p=0.2,
            target=0.6,
            batch_gpu=self.batch_size//1, #gpus param?
            cbase=32768,
            cmax=512,
            glr=self.glr,
            dlr=self.dlr,
            map_depth=8,
            mbstd_group=MBSTD_GROUP,
            initstrength=None,
            projected=False,
            diffaugment= diffaug_pipes[self.diffaug_pipe] if self.aug == 1 else None,
            desc="",
            metrics=[],
            kimg=25000,
            nkimg=0,
            tick=4,
            snap=self.snap,
            seed=0,
            nobench=False,
            dry_run=False,
            fp32=False,
            workers=4,
            kd_l1_lambda=0.0,
            kd_lpips_lambda=0.0,
            kd_mode="Output_Only",
            content_aware_kd=False,
            teacher = None,
            custom=True,
            lpips_image_size=256,
            fps=self.fps if self.found_video else 10,
        )

        if self.training_process.pid is not None:
            self.queue = mp.Queue()
            self.reply = mp.Queue()
            self.training_process = LoggedProcess(target=train_main, args=(self.queue, self.reply), name='training')
        self.done = False
        self.queue.put(kwargs)
        self.training_process.start()

    @imgui_utils.scoped_by_object_id
    def __call__(self):
        if not self.reply.empty():
            self.message, self.done = self.reply.get()
            while not self.reply.empty():
                self.message, self.done = self.reply.get()

            # avoid re-printing stats lines
            if not self.message.strip().startswith('tick '):
                logger.debug("Training update: %s", self.message)

        training_active = self.is_training
        left_pane_width = self.menu.app.content_width // 4
        available_height = imgui.get_content_region_available()[1]

        imgui.begin_child("##train_settings_pane", left_pane_width, available_height, border=True)

        with imgui_utils.grayed_out(training_active):
            text = "Train a model with your dataset"
            text_width = imgui.calc_text_size(text).x
            pane_width = imgui.get_window_width()
            help_icon_size = imgui.get_font_size()
            style = imgui.get_style()

            imgui.text(text)

            spacing = pane_width - (style.window_padding[0] * 2) - text_width - help_icon_size - style.item_spacing[0] - 10

            imgui.same_line()
            imgui.dummy(spacing, 0)
            training_hyperlinks = []
            training_url = self.help_urls.get("training_module")
            if training_url:
                training_hyperlinks.append((training_url, "Read More"))
            augmentation_guide_url = self.help_urls.get("training_augmentation_guide")
            if augmentation_guide_url:
                training_hyperlinks.append((augmentation_guide_url, "How to choose training augmentation"))

            if training_hyperlinks:
                self.help_icon.render_with_urls(self.help_texts.get("training_module"), training_hyperlinks)
            else:
                self.help_icon.render(self.help_texts.get("training_module"))
            imgui.separator()

            imgui.text("Save Path")
            current_y = imgui.get_cursor_pos_y()
            imgui.set_cursor_pos_y(current_y - 3)
            if not training_active:
                _, self.save_path = imgui_utils.input_text("##Save Path", self.save_path, 1024, 0,
                    width=pane_width - imgui.calc_text_size("Browse##main_save")[0] - style.window_padding[0] * 2)
            else:
                imgui_utils.input_text("##Save Path", self.save_path, 1024, imgui.INPUT_TEXT_READ_ONLY,
                    width=pane_width - imgui.calc_text_size("Browse##main_save")[0] - style.window_padding[0] * 2)

            imgui.same_line()
            if imgui.button("Browse##main_save", width=self.menu.app.button_w) and not training_active:
                directory_path = self.save_path_browser.select_directory("Select Training Results Save Path", initial_dir=self.save_path)
                if directory_path:
                    self.save_path = directory_path

            imgui.text("Dataset Path")
            current_y = imgui.get_cursor_pos_y()
            imgui.set_cursor_pos_y(current_y - 3)
            if not training_active:
                _, self.data_path = imgui_utils.input_text("##Dataset Path", self.data_path, 1024, 0,
                    width=pane_width - imgui.calc_text_size("Browse##main_data")[0] - style.window_padding[0] * 2)
            else:
                imgui_utils.input_text("##Dataset Path", self.data_path, 1024, imgui.INPUT_TEXT_READ_ONLY,
                    width=pane_width - imgui.calc_text_size("Browse##main_data")[0] - style.window_padding[0] * 2)

            imgui.same_line()
            if imgui.button("Browse##main_data", width=self.menu.app.button_w) and not training_active:
                directory_path = self.data_path_browser.select_directory("Select Training Dataset Directory", initial_dir=self.data_path)
                if directory_path:
                    self.data_path = directory_path

            imgui.text("Resume Pkl")
            current_y = imgui.get_cursor_pos_y()
            imgui.set_cursor_pos_y(current_y - 3)
            if not training_active:
                _, self.resume_pkl = imgui_utils.input_text("##Resume Pkl", self.resume_pkl, 1024, 0,
                    width=pane_width - imgui.calc_text_size("Browse##Resume Pkl")[0] - style.window_padding[0] * 2)
            else:
                imgui_utils.input_text("##Resume Pkl", self.resume_pkl, 1024, imgui.INPUT_TEXT_READ_ONLY,
                    width=pane_width - imgui.calc_text_size("Browse##Resume Pkl")[0] - style.window_padding[0] * 2)

            imgui.same_line()
            picked = self.model_dropdown(width=self.menu.app.button_w)
            if picked is not None and not training_active:
                self.resume_pkl = picked

            imgui.text("Training Augmentation")
            imgui.same_line()
            if not training_active:
                _, self.aug = imgui.combo("##Training Augmentation", self.aug, augs)
            else:
                imgui.combo("##Training Augmentation", self.aug, augs)
            if self.aug == 0:
                imgui.text("Augmentation Pipeline")
                imgui.same_line()
                if not training_active:
                    _, self.ada_pipe = imgui.combo("##Augmentation Pipeline", self.ada_pipe, ada_pipes)
                else:
                    imgui.combo("##Augmentation Pipeline", self.ada_pipe, ada_pipes)
            else:
                imgui.text("Augmentation Pipeline")
                imgui.same_line()
                if not training_active:
                    _, self.diffaug_pipe = imgui.combo("##Augmentation Pipeline", self.diffaug_pipe, diffaug_pipes)
                else:
                    imgui.combo("##Augmentation Pipeline", self.diffaug_pipe, diffaug_pipes)

            imgui.text("Batch Size")
            imgui.same_line()
            input_width = int(self.menu.app.font_size * 6)
            button_width = self.menu.app.font_size * 1.2
            with imgui_utils.item_width(input_width):
                imgui.input_text("##Batch Size", str(self.batch_size), 32, flags=imgui.INPUT_TEXT_READ_ONLY)
            imgui.same_line()
            if self.batch_size not in BATCH_SIZE_CHOICES:
                self.batch_size = min(BATCH_SIZE_CHOICES, key=lambda x: abs(x - self.batch_size))
            batch_idx = BATCH_SIZE_CHOICES.index(self.batch_size)
            if imgui.button("-##batch_size", width=button_width) and not training_active:
                batch_idx = max(0, batch_idx - 1)
                self.batch_size = BATCH_SIZE_CHOICES[batch_idx]
            imgui.same_line()
            if imgui.button("+##batch_size", width=button_width) and not training_active:
                batch_idx = min(len(BATCH_SIZE_CHOICES) - 1, batch_idx + 1)
                self.batch_size = BATCH_SIZE_CHOICES[batch_idx]

            imgui.text("Configuration")
            imgui.same_line()
            if not training_active:
                _, self.config = imgui.combo("##Configuration", self.config, configs)
            else:
                imgui.combo("##Configuration", self.config, configs)

            imgui.set_next_window_size(self.menu.app.content_width // 4, self.menu.app.content_height // 4, imgui.ONCE)

            if imgui_utils.button("Advanced...", width=-1, enabled=not training_active):
                imgui.open_popup("Advanced...")

            if imgui.begin_popup_modal("Advanced...")[0]:
                imgui.text("Advanced Training Options")
                imgui.text("Generator Learning Rate")
                _, self.glr = imgui_utils.input_float("##Generator Learning Rate", self.glr, format='%.6f')

                imgui.text("Discriminator Learning Rate")
                _, self.dlr = imgui_utils.input_float("##Discriminator Learning Rate", self.dlr, format='%.6f')

                imgui.text("Gamma")
                _, self.gamma = imgui.input_int("##Gamma", self.gamma)

                imgui.text("Number of ticks between snapshots")
                _, self.snap = imgui.input_int("##Number of ticks between snapshots", self.snap)

                if imgui_utils.button("Close", enabled=1):
                    imgui.close_current_popup()

                imgui.end_popup()

        if self.done_button:
            imgui_utils.button("Stopping...", width=-1, enabled=False)
        elif training_active:
            if imgui.button("Stop Training", width=-1):
                self._kill_training_process()
                self.done_button = True
        else:
            if imgui.button("Start Training", width=-1):
                self._start_training()

        imgui.end_child()

        imgui.same_line()

        imgui.begin_child("##train_output_pane", 0, available_height, border=True)

        if training_active or self.message != '' or self.image_path != '':
            imgui.text("Training...")
            # Replies are either progress text or the path of a snapshot grid.
            # Only stat plausible paths: arbitrary text (e.g. the multi-line
            # config JSON) raises ENAMETOOLONG from Path.exists() on macOS.
            message_is_image = False
            if self.message and '\n' not in self.message and self.message.lower().endswith('.png'):
                try:
                    message_is_image = Path(self.message).exists()
                except OSError:
                    message_is_image = False
            if message_is_image and self.image_path != self.message:
                self.image_path = self.message
                self.grid = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
                self.grid = cv2.cvtColor(self.grid, cv2.COLOR_BGRA2RGBA)
                self.grid_texture = gl_utils.Texture(image=self.grid, width=self.grid.shape[1],
                                               height=self.grid.shape[0], channels=self.grid.shape[2])
            elif self.message != "":
                imgui.text(self.message)
            if self.image_path != '':
                imgui.text("Current sample of fake imagery")
                avail_w = imgui.get_content_region_available_width()
                avail_h = imgui.get_content_region_available()[1] - imgui.get_frame_height_with_spacing() * 2
                aspect = self.grid.shape[1] / self.grid.shape[0]
                display_h = min(avail_h, avail_w / aspect)
                display_w = display_h * aspect
                imgui.image(self.grid_texture.gl_id, display_w, display_h)
            if (self.done or self.done_button) and not self.training_process.is_alive():
                self.training_process.join()
                self.message = ''
                self.done = False
                self.done_button = False
                self.image_path = ''

        imgui.end_child()

    def _kill_training_process(self):
        try:
            parent = psutil.Process(self.training_process.pid)
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()
        except psutil.NoSuchProcess:
            pass
