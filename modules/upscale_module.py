import logging
import os
import threading
import time
from pathlib import Path

import imgui

from utils.app_logging import LoggedProcess
from utils.gui_utils import imgui_utils
from upscale import ensure_weight, required_weights, weight_path
from upscale.batch import (BATCH_DEFAULT_MODEL, BATCH_LABELS, BATCH_MODELS,
                           run_batch_upscale)

from dnnlib import EasyDict
import multiprocessing as mp


from widgets.native_browser_widget import NativeBrowserWidget
from widgets.help_icon_widget import HelpIconWidget

args = EasyDict(result_path="", input_path=[""], model_type=BATCH_DEFAULT_MODEL)

logger = logging.getLogger(__name__)


class UpscaleModule:
    def __init__(self, menu):
        self.result_path = args.result_path
        self.input_path = args.input_path
        self.models = [BATCH_LABELS[key] for key in BATCH_MODELS]
        self.model_selected = BATCH_MODELS.index(BATCH_DEFAULT_MODEL)
        self.model_type = BATCH_MODELS[self.model_selected]
        self.menu = menu
        self.app = menu.app
        self.browser = NativeBrowserWidget()
        self.running = False
        self.writer = None
        self.reader = None
        self.queue = mp.Queue()
        self.reply = mp.Queue()
        self.upscale_process = None
        self.files = []
        self.file_idx = 0
        self.upscale_idx = 0
        self.total_frames = -1
        
        self.start_time = 0
        self.eta = -1
        self.video_width = 0
        self.video_height = 0
        self.help_icon = HelpIconWidget()
        self.help_texts, self.help_urls = self.help_icon.load_help_texts("upscale")
        # First-run weight download state.
        self.downloading = False
        self.download_thread = None
        self.download_cancel = None
        self.download_status = None  # None while running, then "ok"/"cancelled"/"error: ..."
        self.dl_done = 0
        self.dl_total = 0
        self.pending_start = False


    def display_progress(self):
        width = imgui.get_font_size() * 22
        label = f"Processing file {min(self.file_idx + 1, len(self.files))} of {len(self.files)}"
        if self.file_idx < len(self.files):
            label += f": {os.path.basename(self.files[self.file_idx])}"
        imgui.text(label)
        if self.total_frames > 0:
            frac = min(self.upscale_idx / self.total_frames, 1.0)
            imgui.progress_bar(frac, (width, 0.0), f"{self.upscale_idx}/{self.total_frames}")
        else:
            imgui.progress_bar(0.0, (width, 0.0), "preparing...")
        if self.eta != -1:
            hours = int(self.eta / 3600)
            minutes = int((self.eta - hours * 3600) / 60)
            seconds = int(self.eta - hours * 3600 - minutes * 60)
            if hours:
                eta_str = f"{hours}h {minutes}m {seconds}s"
            elif minutes:
                eta_str = f"{minutes}m {seconds}s"
            else:
                eta_str = f"{seconds}s"
        else:
            eta_str = "estimating..."
        imgui.text(f"ETA: {eta_str}")
        imgui.spacing()
        if imgui.button("Cancel", width=width):
            self.cancel_upscale()

    def cancel_upscale(self):
        # Stop the worker process; the partial output file is left as-is.
        if self.upscale_process is not None:
            self.upscale_process.terminate()
            self.upscale_process.join(timeout=1)
            self.upscale_process = None
        self.running = False


    @imgui_utils.scoped_by_object_id
    def __call__(self):
        if not self.reply.empty():
            msg = self.reply.get()
            while not self.reply.empty():
                msg = self.reply.get()
            self.file_idx, self.upscale_idx, self.total_frames, self.eta, done = msg
            if done:
                self.running = False
                if self.upscale_process is not None:
                    self.upscale_process.join()
                    self.upscale_process = None
        button_width = self.app.button_w
        spacing = self.app.spacing
        # One label column for every row, sized by the widest label, and one
        # right edge for every field, so the form reads as a grid.
        labels = ("Input Files", "Save Path", "Model")
        label_col = max(imgui.calc_text_size(l)[0] for l in labels) + spacing * 2
        field_width = -(button_width + spacing)

        text = "Use AI to upscale your images and videos"
        imgui.text(text)
        self.help_icon.render(self.help_texts.get("upscale_module"),
                              url=self.help_urls.get("upscale_module"),
                              align_right=True)

        imgui.separator()

        # Input path
        imgui.text("Input Files")
        imgui.same_line(label_col)
        joined = '\n'.join(self.input_path)
        imgui_utils.input_text("##upscale_input_files", joined, 1024,
                               flags=imgui.INPUT_TEXT_READ_ONLY,
                               width=field_width,
                               help_text="Select images or videos")
        imgui.same_line()
        if imgui.button("Browse##upscale_input", width=button_width):
            files = self.browser.select_media_files(initial_dir=self.input_path[0] if self.input_path else "")
            if files:
                self.input_path = [str(f) for f in files]
                if not self.result_path:
                    self.result_path = Path(self.input_path[0]).parent.as_posix()

        # Result path
        imgui.text("Save Path")
        imgui.same_line(label_col)
        _, self.result_path = imgui_utils.input_text("##upscale_save_path", self.result_path, 1024, 0,
                                                     width=field_width)
        imgui.same_line()
        if imgui.button("Browse##upscale_result_path", width=button_width):
            directory_path = self.browser.select_directory("Select Save Directory", initial_dir=self.result_path)
            if directory_path:
                self.result_path = directory_path.replace('\\', '/')

        # Model selection
        imgui.text("Model")
        imgui.same_line(label_col)
        with imgui_utils.item_width(field_width):
            _, self.model_selected = imgui.combo("##upscale_model", self.model_selected, self.models)
        self.model_type = BATCH_MODELS[self.model_selected]

        imgui.text_disabled("The output is 4 times the input resolution.")

        try:
            if imgui.button("Upscale", width=imgui.get_content_region_available_width()) and not self.running and not self.downloading:
                args.result_path = self.result_path
                args.input_path = self.input_path
                args.model_type = self.model_type
                self.args = args
                missing = [key for key in required_weights(self.model_type)
                           if not os.path.exists(weight_path(key))]
                if not missing:
                    self.running = True
                    logger.info("Starting upscaling: input=%s output=%s model=%s",
                                self.input_path, self.result_path, self.model_type)
                    self.start_upscale()
                    imgui.open_popup("Upscaling")
                else:
                    self._begin_download(missing)
                    imgui.open_popup("Downloading Model")

        except Exception:
            logger.exception("Super resolution failed to start")

        if imgui.begin_popup_modal("Downloading Model", flags=imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_ALWAYS_AUTO_RESIZE)[0]:
            self._display_download()
            imgui.end_popup()

        if self.pending_start:
            self.pending_start = False
            imgui.open_popup("Upscaling")

        if imgui.begin_popup_modal("Upscaling", flags=imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_ALWAYS_AUTO_RESIZE)[0]:
            self.display_progress()
            if not self.running:
                imgui.close_current_popup()
            imgui.end_popup()




    def _begin_download(self, weight_keys):
        self.download_cancel = threading.Event()
        self.download_status = None
        self.dl_done = 0
        self.dl_total = 0
        self.downloading = True
        self.download_thread = threading.Thread(
            target=self._download_weights, args=(weight_keys,), daemon=True)
        self.download_thread.start()

    def _download_weights(self, weight_keys):
        def progress(done, total):
            self.dl_done, self.dl_total = done, total
        try:
            for key in weight_keys:
                if ensure_weight(key, progress_cb=progress,
                                 cancel_event=self.download_cancel) is None:
                    self.download_status = "cancelled"
                    return
            self.download_status = "ok"
        except Exception as e:
            self.download_status = f"error: {e}"

    def _join_download_thread(self):
        if self.download_thread is not None:
            self.download_thread.join(timeout=1)
            self.download_thread = None

    def _display_download(self):
        width = imgui.get_font_size() * 22
        status = self.download_status

        if isinstance(status, str) and status.startswith("error"):
            imgui.text("Model download failed:")
            imgui.text_wrapped(status[7:] if status.startswith("error: ") else status)
            imgui.spacing()
            if imgui.button("Close", width=width):
                self.downloading = False
                self._join_download_thread()
                self.running = False
                imgui.close_current_popup()
            return

        imgui.text(f"Downloading {BATCH_LABELS[self.model_type]} model weights...")
        if self.dl_total > 0:
            fraction = min(self.dl_done / self.dl_total, 1.0)
            label = f"{self.dl_done / (1024 * 1024):.1f} / {self.dl_total / (1024 * 1024):.1f} MB"
            imgui.progress_bar(fraction, (width, 0.0), label)
        else:
            imgui.progress_bar(0.0, (width, 0.0), "connecting...")
        imgui.spacing()

        if status is None:
            if imgui.button("Cancel", width=width):
                self.download_cancel.set()
            return

        # Download finished: tear down and either launch or bail out.
        self.downloading = False
        self._join_download_thread()
        imgui.close_current_popup()
        if status == "ok":
            self.running = True
            self.start_upscale()
            self.pending_start = True
        else:  # cancelled
            self.running = False

    def start_upscale(self):
        self.start_time = time.time()
        self.files = self.input_path

        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        self.file_idx = 0
        self.upscale_idx = 0
        self.total_frames = -1
        self.eta = -1

        if len(self.files) == 0:
            self.running = False
            return

        # Run upscaling in a separate process so it gets the GPU at full speed and the UI stays responsive
        self.queue = mp.Queue()
        self.reply = mp.Queue()
        self.upscale_process = LoggedProcess(target=run_batch_upscale, args=(self.queue, self.reply), daemon=True, name='upscale')
        self.upscale_process.start()
        self.queue.put(self.args)
