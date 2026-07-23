import logging
import os
import threading
import time
from pathlib import Path

import imgui

from utils.app_logging import LoggedProcess
from utils.gui_utils import imgui_utils
from super_res.super_res import run_super_res, sr_weight_path, ensure_sr_weight

from dnnlib import EasyDict
import multiprocessing as mp


from widgets.native_browser_widget import NativeBrowserWidget
from widgets.help_icon_widget import HelpIconWidget

args = EasyDict(result_path="", input_path=[""], model_type="Balance",
                outscale=3, width=4096, height=4096, sharpen_scale=1, scale_mode=1)
scale_factor = ['1', '2', '3', '4', '5', '6', '7', '8']

logger = logging.getLogger(__name__)


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
        self.browser = NativeBrowserWidget()
        self.scale_mode = args.scale_mode
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
            frac = min(self.super_res_idx / self.total_frames, 1.0)
            imgui.progress_bar(frac, (width, 0.0), f"{self.super_res_idx}/{self.total_frames}")
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
            self.cancel_super_res()

    def cancel_super_res(self):
        # Stop the worker process; the partial output file is left as-is.
        if self.sr_process is not None:
            self.sr_process.terminate()
            self.sr_process.join(timeout=1)
            self.sr_process = None
        self.running = False


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
        imgui.text(text)
        self.help_icon.render(self.help_texts.get("super_res_module"),
                              url=self.help_urls.get("super_res_module"),
                              align_right=True)

        imgui.separator()

        # Input path
        joined = '\n'.join(self.input_path)
        imgui_utils.input_text("##SRINPUT", joined, 1024, 
                              flags=imgui.INPUT_TEXT_READ_ONLY, 
                              width=input_width, 
                              help_text="Input Files")
        
        imgui.same_line()
        if imgui.button("Browse##super_res_input", width=button_width):
            files = self.browser.select_media_files(initial_dir=self.input_path[0] if self.input_path else "")
            if files:
                self.input_path = [str(f) for f in files]
                if not self.result_path:
                    self.result_path = Path(self.input_path[0]).parent.as_posix()

        # Result path
        imgui.text("Save Path")
        _, self.result_path = imgui_utils.input_text("##save_path", self.result_path, 1024, 0,
                                                     width=imgui.get_window_width() - self.menu.app.button_w - imgui.calc_text_size("Browse")[0])
        
        imgui.same_line()
        if imgui.button("Browse##super_res_result_path", width=button_width):
            directory_path = self.browser.select_directory("Select Save Directory", initial_dir=self.result_path)
            if directory_path:
                self.result_path = directory_path.replace('\\', '/')
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
            if imgui.button("Super Resolution", width=imgui.get_content_region_available_width()) and not self.running and not self.downloading:
                args.result_path = self.result_path
                args.input_path = self.input_path
                args.model_type = self.model_type
                args.outscale = self.out_scale + 1
                args.out_height = self.height
                args.out_width = self.width
                args.sharpen_scale = self.sharpen
                args.scale_mode = self.scale_mode
                self.args = args
                if os.path.exists(sr_weight_path(self.model_type)):
                    self.running = True
                    logger.info("Starting super resolution: input=%s output=%s model=%s",
                                self.input_path, self.result_path, self.model_type)
                    self.start_super_res()
                    imgui.open_popup("Super Resolution")
                else:
                    self._begin_download(self.model_type)
                    imgui.open_popup("Downloading Model")

        except Exception:
            logger.exception("Super resolution failed to start")

        if imgui.begin_popup_modal("Downloading Model", flags=imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_ALWAYS_AUTO_RESIZE)[0]:
            self._display_download()
            imgui.end_popup()

        if self.pending_start:
            self.pending_start = False
            imgui.open_popup("Super Resolution")

        if imgui.begin_popup_modal("Super Resolution", flags=imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_ALWAYS_AUTO_RESIZE)[0]:
            self.display_progress()
            if not self.running:
                imgui.close_current_popup()
            imgui.end_popup()




    def _begin_download(self, model_type):
        self.download_cancel = threading.Event()
        self.download_status = None
        self.dl_done = 0
        self.dl_total = 0
        self.downloading = True
        self.download_thread = threading.Thread(
            target=self._download_weight, args=(model_type,), daemon=True)
        self.download_thread.start()

    def _download_weight(self, model_type):
        def progress(done, total):
            self.dl_done, self.dl_total = done, total
        try:
            result = ensure_sr_weight(model_type, progress_cb=progress,
                                      cancel_event=self.download_cancel)
            self.download_status = "ok" if result is not None else "cancelled"
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

        imgui.text(f"Downloading {self.model_type} model weights...")
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
            self.start_super_res()
            self.pending_start = True
        else:  # cancelled
            self.running = False

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
        self.sr_process = LoggedProcess(target=run_super_res, args=(self.queue, self.reply), daemon=True, name='super-res')
        self.sr_process.start()
        self.queue.put(self.args)
