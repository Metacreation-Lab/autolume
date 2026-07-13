import json
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
from training.dataset_validator import validate_dataset
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
# Mirrors augpipe_specs in train.py; used to recover the pipe name from
# the augment_kwargs flags stored in a run's training_options.json.
augpipe_specs = {
    'blit': dict(xflip=1, rotate90=1, xint=1),
    'geom': dict(scale=1, rotate=1, aniso=1, xfrac=1),
    'color': dict(brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
    'filter': dict(imgfilter=1),
    'noise': dict(noise=1),
    'cutout': dict(cutout=1),
    'bg': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1),
    'bgc': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1,
                lumaflip=1, hue=1, saturation=1),
    'bgcf': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1,
                 lumaflip=1, hue=1, saturation=1, imgfilter=1),
    'bgcfn': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1,
                  lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1),
    'bgcfnc': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1,
                   lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1, cutout=1),
}
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

        self.model_dropdown = ModelDropdownButton(label='Browse')

        self.menu = menu

        self.help_icon = HelpIconWidget()
        self.help_texts, self.help_urls = self.help_icon.load_help_texts("training")

        self.queue = mp.Queue()
        self.reply = mp.Queue()
        self.message = ""
        self.done = False
        self.training_process = LoggedProcess(target=train_main, args=(self.queue, self.reply), name='training')
        self._zipfile = None
        self.gamma = 10
        self.glr = 0.002
        self.dlr = 0.002
        self.snap = 4
        self.mirror = False # Mirror only accesible in preprocessing module
        self.done_button = False
        self.image_path = ''

        # Pre-training dataset validation state
        self._open_validation_popup = False
        self.validation_queue = mp.Queue()
        self.validation_reply = mp.Queue()
        self.validation_process = None
        self.is_validating = False
        self.validation_done = False
        self.validation_is_valid = False
        self.validation_errors = []
        self.validation_consistency = []
        self.validation_current = 0
        self.validation_total = 0
        self.validation_percentage = 0.0
        self.validation_current_file = ""

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

    def _apply_run_settings(self, pkl_path):
        # Restore the UI settings of the training run the snapshot came from,
        # using the training_options.json saved next to it. Every field is
        # optional: anything missing or unrecognized is skipped.
        options_path = Path(pkl_path).parent / "training_options.json"
        try:
            with open(options_path, encoding="utf-8") as f:
                options = json.load(f)
        except (OSError, json.JSONDecodeError):
            logger.debug("No readable training_options.json next to %s", pkl_path)
            return

        dataset_path = options.get("training_set_kwargs", {}).get("path")
        if dataset_path and Path(dataset_path).exists():
            self.data_path = str(dataset_path)

        batch = options.get("batch_size")
        if isinstance(batch, int):
            self.batch_size = min(BATCH_SIZE_CHOICES, key=lambda x: abs(x - batch))

        diffaug = options.get("loss_kwargs", {}).get("diffaugment")
        if diffaug in diffaug_pipes:
            self.aug = augs.index("DiffAUG")
            self.diffaug_pipe = diffaug_pipes.index(diffaug)
        elif "augment_kwargs" in options:
            flags = {k: v for k, v in options["augment_kwargs"].items() if k != "class_name"}
            for name, spec in augpipe_specs.items():
                if spec == flags:
                    self.aug = augs.index("ADA")
                    self.ada_pipe = ada_pipes.index(name)
                    break

        run_dir = options.get("run_dir")
        if run_dir:
            tokens = Path(run_dir).name.split("-")
            if len(tokens) > 1 and tokens[1] in configs:
                self.config = configs.index(tokens[1])

        gamma = options.get("loss_kwargs", {}).get("r1_gamma")
        if isinstance(gamma, (int, float)):
            self.gamma = int(round(gamma))

        glr = options.get("G_opt_kwargs", {}).get("lr")
        if isinstance(glr, (int, float)):
            self.glr = float(glr)

        dlr = options.get("D_opt_kwargs", {}).get("lr")
        if isinstance(dlr, (int, float)):
            self.dlr = float(dlr)

        snap = options.get("network_snapshot_ticks")
        if isinstance(snap, int):
            self.snap = snap

    def _start_training(self):
        target_data_path = self.data_path

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
            nkimg=None,
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

            imgui.text("Resume Pkl (optional)")
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
                self._apply_run_settings(picked)

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

            advanced_header_open = imgui.collapsing_header("Advanced Options")[0]
            self.help_icon.render_aligned("Advanced Options", pane_width,
                                          self.help_texts.get("advanced_options"),
                                          url=self.help_urls.get("advanced_options"),
                                          trailing_offset=35)

            if advanced_header_open:
                imgui.text("Generator Learning Rate")
                changed, value = imgui_utils.input_float("##Generator Learning Rate", self.glr, format='%.6f')
                if changed and not training_active:
                    self.glr = value

                imgui.text("Discriminator Learning Rate")
                changed, value = imgui_utils.input_float("##Discriminator Learning Rate", self.dlr, format='%.6f')
                if changed and not training_active:
                    self.dlr = value

                imgui.text("Gamma")
                changed, value = imgui.input_int("##Gamma", self.gamma)
                if changed and not training_active:
                    self.gamma = value

                imgui.text("Number of ticks between snapshots")
                changed, value = imgui.input_int("##Number of ticks between snapshots", self.snap)
                if changed and not training_active:
                    self.snap = value

        if self.done_button:
            imgui_utils.button("Stopping...", width=-1, enabled=False)
        elif training_active:
            if imgui.button("Stop Training", width=-1):
                self._kill_training_process()
                self.done_button = True
        else:
            if imgui_utils.button("Start Training", width=-1, enabled=not self.is_validating):
                self._start_dataset_validation(self.data_path)

        if self._open_validation_popup:
            imgui.open_popup("Validating Dataset")
            self._open_validation_popup = False
        self._render_validation_popup()

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

    def _start_dataset_validation(self, dataset_path):
        """Spawn the dataset validation subprocess and open the validation popup."""
        self._cleanup_validation_process()

        self.validation_done = False
        self.validation_is_valid = False
        self.validation_errors = []
        self.validation_consistency = []
        self.validation_current = 0
        self.validation_total = 0
        self.validation_percentage = 0.0
        self.validation_current_file = ""

        self.validation_queue = mp.Queue()
        self.validation_reply = mp.Queue()
        self.validation_queue.put({'path': dataset_path})
        self.validation_process = mp.Process(
            target=validate_dataset,
            args=(self.validation_queue, self.validation_reply),
            name='DatasetValidationProcess',
        )
        self.is_validating = True
        self.validation_process.start()
        self._open_validation_popup = True

    def _cleanup_validation_process(self):
        """Tear down the validation subprocess and clear its queues."""
        if self.validation_process is not None and self.validation_process.is_alive():
            try:
                self.validation_process.terminate()
                self.validation_process.join(timeout=2)
                if self.validation_process.is_alive():
                    self.validation_process.kill()
                    self.validation_process.join()
            except Exception as e:
                logger.debug("Error cleaning up validation process: %s", e)
        self.validation_process = None

        while not self.validation_queue.empty():
            try:
                self.validation_queue.get_nowait()
            except Exception:
                break
        while not self.validation_reply.empty():
            try:
                self.validation_reply.get_nowait()
            except Exception:
                break

        self.is_validating = False

    def _drain_validation_replies(self):
        """Pull every pending message off the validation reply queue."""
        while not self.validation_reply.empty():
            try:
                data = self.validation_reply.get_nowait()
            except Exception:
                break
            if not isinstance(data, dict):
                continue
            if data.get('type') == 'progress':
                self.validation_current = int(data.get('current', 0))
                self.validation_total = int(data.get('total', 0))
                self.validation_percentage = float(data.get('percentage', 0.0))
                self.validation_current_file = str(data.get('current_file', ''))
            elif data.get('type') == 'completed':
                self.validation_done = True
                self.validation_is_valid = bool(data.get('is_valid', False))
                self.validation_errors = list(data.get('errors', []))
                self.validation_consistency = list(data.get('consistency', []))
                self.validation_total = int(data.get('total', self.validation_total))
                self.validation_current = int(data.get('checked', self.validation_current))

    def _render_validation_popup(self):
        """Render the 'Validating Dataset' modal."""
        popup_width = self.menu.app.content_width // 1.5
        popup_height = self.menu.app.content_height // 1.5
        imgui.set_next_window_size(popup_width, popup_height, imgui.ONCE)

        if not imgui.begin_popup_modal("Validating Dataset")[0]:
            return

        self._drain_validation_replies()
        progress_width = popup_width - 20

        if not self.validation_done:
            if self.validation_total > 0:
                imgui.text(f"Checked: {self.validation_current}/{self.validation_total} images")
                if self.validation_current_file:
                    imgui.text(f"Current file: {self.validation_current_file}")
                imgui.progress_bar(self.validation_percentage / 100.0, (progress_width, 20))
                pct_label = f"{self.validation_percentage:.1f}%"
                text_width = imgui.calc_text_size(pct_label)[0]
                imgui.set_cursor_pos_x((progress_width - text_width) / 2)
                imgui.text(pct_label)
            else:
                imgui.text("Scanning dataset for images...")

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            if imgui_utils.button("Cancel", width=progress_width):
                try:
                    self.validation_queue.put('cancel')
                except Exception:
                    pass
                self._cleanup_validation_process()
                imgui.close_current_popup()

        elif self.validation_is_valid:
            self._cleanup_validation_process()
            imgui.close_current_popup()
            imgui.end_popup()
            self._start_training()
            return

        else:
            imgui.text_colored("Dataset validation failed", 1.0, 0.3, 0.3, 1.0)
            imgui.text("Please verify the path or use the Data Preparation tool to prepare your dataset.")
            imgui.separator()
            imgui.text(f"Checked {self.validation_current}/{self.validation_total} images.")

            list_height = max(popup_height * 0.75, 120)
            imgui.begin_child(
                "##validation_errors_list",
                progress_width,
                list_height,
                True,
            )
            # Entries with a filename describe one invalid image each. Entries
            # without one are dataset-level notices (no images found, scan
            # stopped early, ...) and must not count as invalid images.
            image_entries = []
            general_entries = []
            for entry in self.validation_errors:
                if isinstance(entry, dict) and entry.get('filename'):
                    image_entries.append(entry)
                else:
                    general_entries.append(entry)

            for entry in general_entries:
                message = entry.get('message', str(entry)) if isinstance(entry, dict) else str(entry)
                imgui.push_text_wrap_pos(progress_width - 20)
                imgui.text(message)
                imgui.pop_text_wrap_pos()
                imgui.spacing()

            if image_entries:
                imgui.text_colored(
                    f"Invalid images ({len(image_entries)}):",
                    1.0, 0.5, 0.0, 1.0,
                )
                imgui.spacing()
                for entry in image_entries:
                    imgui.text_colored(entry['filename'], 1.0, 0.8, 0.4, 1.0)
                    imgui.push_text_wrap_pos(progress_width - 20)
                    imgui.text(entry.get('message', ''))
                    imgui.pop_text_wrap_pos()
                    imgui.spacing()

            if self.validation_consistency:
                if self.validation_errors:
                    imgui.separator()
                    imgui.spacing()
                imgui.text_colored("Dataset consistency issues:", 1.0, 0.5, 0.0, 1.0)
                imgui.spacing()
                for block in self.validation_consistency:
                    imgui.push_text_wrap_pos(progress_width - 20)
                    imgui.text(str(block))
                    imgui.pop_text_wrap_pos()
                    imgui.spacing()
            imgui.end_child()

            if imgui_utils.button("Open Data Preparation Module", width=progress_width):
                self._cleanup_validation_process()
                self.validation_done = False
                self.validation_errors = []
                self.validation_consistency = []
                imgui.close_current_popup()
                from modules.autolume_live import States
                self.app.navigate_to(States.PREPROCESSING)
            if imgui_utils.button("Close", width=progress_width):
                self._cleanup_validation_process()
                self.validation_done = False
                self.validation_errors = []
                self.validation_consistency = []
                imgui.close_current_popup()

        imgui.end_popup()

    def _kill_training_process(self):
        try:
            parent = psutil.Process(self.training_process.pid)
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()
        except psutil.NoSuchProcess:
            pass
