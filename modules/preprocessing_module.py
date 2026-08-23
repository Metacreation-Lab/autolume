import logging
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import imgui
import PIL.Image
from utils.gui_utils import imgui_utils
import multiprocessing as mp

from upscale import ensure_weight, required_weights, weight_path
from utils import video_io
from utils.app_logging import LoggedProcess
from widgets.native_browser_widget import NativeBrowserWidget
from widgets.thumbnail_widget import ThumbnailWidget
from widgets.image_preview_widget import ImagePreviewWidget
from widgets.loading_widget import LoadingOverlayManager
from widgets.help_icon_widget import HelpIconWidget
from utils.dataset_preprocessing_utils import DatasetPreprocessingUtils

resize_mode = ['stretch','center crop']
padding_color = ['black', 'white', 'bleeding']

logger = logging.getLogger(__name__)


class _VideoInfoProber:
    """Probes video media info on worker threads.

    Media I/O never runs on the render thread: a probe that stalls on a network
    share or a corrupt file would freeze the UI (issue #62).
    """

    WORKERS = 2

    def __init__(self):
        self._executor = None
        self._done = queue.Queue()   # (generation, path, info)
        self._generation = 0         # bumped when the pending set is discarded

    def start(self, paths):
        """Queue probes for ``paths``, discarding anything still in flight."""
        self._generation += 1
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.WORKERS,
                                                thread_name_prefix='video-probe')
        for path in paths:
            self._executor.submit(self._probe, path, self._generation)

    def poll(self):
        """Return {path: MediaInfo} for the probes that finished since last call."""
        infos = {}
        while True:
            try:
                generation, path, info = self._done.get_nowait()
            except queue.Empty:
                return infos
            if generation == self._generation:
                infos[path] = info

    def cancel(self):
        """Drop the results of any probe still in flight."""
        self._generation += 1

    def shutdown(self):
        self.cancel()
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    def _probe(self, path, generation):
        try:
            info = video_io.probe(path)
        except video_io.VideoIOError as e:
            logger.warning("Could not probe video %s: %s", path, e)
            info = video_io.MediaInfo(duration=0.0, width=0, height=0, fps=0.0,
                                      has_audio=False)
        self._done.put((generation, path, info))

def _format_interval(value):
    """Seconds value with trailing zeros trimmed but at least one decimal."""
    text = f"{value:.3f}".rstrip('0')
    return text + '0' if text.endswith('.') else text


class DataPreprocessing:
    """Data Preprocessing UI"""

    # Image headers read per frame while a large import is being probed.
    PROBE_CHUNK = 200

    def __init__(self, app):
        self.app = app

        self.settings = DatasetPreprocessingUtils()

        self.data_browser = NativeBrowserWidget()
        self.thumbnail_widget = ThumbnailWidget(padding_value=26)  # Imported image thumbnails
        self.video_thumbnail_widget = ThumbnailWidget()  # Video popup thumbnails (black padding)
        self.image_preview_widget = ImagePreviewWidget()
        self.loading_widget = LoadingOverlayManager(app)  # Enhanced loading overlay manager

        self.imported_files = [] 
        self.selected_video_files = [] 
        self.current_duplicates = []  # Store current image duplicates for popup display
        self.selected_files = []  # Store selected files when duplicates are found
        
        self.preview_original = False

        # Video frame extraction
        self.extraction_interval = self.settings.extraction_interval
        self.extraction_interval_text = _format_interval(self.extraction_interval)
        self.video_infos = {}
        self.video_prober = _VideoInfoProber()
        self.last_video_count = None  # gates video thumbnail refreshes
        self.video_extraction_queue = mp.Queue()
        self.video_extraction_reply = mp.Queue()
        self.is_processing_video = False

        # Upscaling section state
        self.image_dims = {}                 # path -> (width, height) or None
        self._pending_dims = []              # paths still waiting to be probed
        self._below_count = 0                # cached below-target image count
        self.upscale_download_thread = None
        self.upscale_download_cancel = None
        self.upscale_download_kind = None    # WEIGHTS key being downloaded
        self.upscale_download_queue = []     # WEIGHTS keys waiting to download
        self.upscale_download_result = None
        self.upscale_download_active = False
        self.upscale_download_cancelling = False
        self.upscale_settings_revert = None  # snapshot restored if a download fails

        self.min_res = 8
        self.max_res = 1024
        self.img_res = self.settings.size # current image resolution

        self.square = True # non-square framing settings (image is square or not)
        
        # Track changes for preview updates
        self.last_selected_file = None
        self.last_settings_hash = None

        # Dataset processing
        self.processing_queue = mp.Queue()
        self.processing_reply = mp.Queue()
        self.cancel_processing = False
        self.processing_process = LoggedProcess(target=DatasetPreprocessingUtils.create_training_dataset, args=(self.processing_queue, self.processing_reply), name='dataset-build')
        
        # Processing popup control
        self.is_processing_dataset = False
        self.folder_exists_warning = False  
        # Progress tracking
        self.progress_current = 0
        self.progress_total = 0
        self.progress_percentage = 0
        self.progress_file = ""
        self.processing_completed = False
        
        self.save_path = str(self.settings.output_path) 

        self.help_icon = HelpIconWidget()
        self.help_texts, self.help_urls = self.help_icon.load_help_texts("preprocessing")

    def __call__(self):
        """Preprocessing content"""
        imgui_utils.set_default_style()

        self._poll_upscale_download()
        self._probe_pending_dims()

        navbar_h = self.app.navbar_height
        available_height = self.app.content_height - navbar_h

        # Calculate column widths
        first_column_width = self.app.content_width * 0.2
        remaining_width = self.app.content_width - first_column_width
        second_column_width = remaining_width // 2
        third_column_width = remaining_width - second_column_width

        button_width = self.app.font_size
        
        # --- Column 1: Dataset Parameter and Options ---
        imgui.set_next_window_position(0, navbar_h)
        imgui.set_next_window_size(first_column_width, available_height)
        imgui.begin('Parameters##Preprocessing', closable=False, flags=(
            imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS))
        
        parameter_column_width = first_column_width - 20 

        imgui.text("Import Data")

        import_hyperlinks = []
        import_url = self.help_urls.get("import_data")
        if import_url:
            import_hyperlinks.append((import_url, "Supported Media Formats"))
        tutorial_video_url = "https://www.youtube.com/watch?v=7Pc5-ULeXkM&feature=youtu.be"
        import_hyperlinks.append((tutorial_video_url, "Tutorial Video"))

        self.help_icon.render(self.help_texts.get("import_data"),
                              hyperlinks=import_hyperlinks, align_right=True)

        imgui.separator()
 
        if imgui.button("Import Images", width=parameter_column_width, height=30):
            selected_images = self.data_browser.select_images_from_folder()
            duplicates = self.check_for_duplicates(selected_images)
            if duplicates:
                self.current_duplicates = duplicates  # Store duplicate filenames for popup
                self.selected_files = selected_images  # Store full file paths for processing
                imgui.open_popup("Duplicates")
            else:
                self.imported_files.extend(selected_images)
                self._refresh_imported_thumbnails()
                self.last_selected_file = None

        imgui.set_next_window_size(self.app.content_width // 2.5, self.app.content_height // 2.5, imgui.ONCE)
        if imgui.begin_popup_modal("Duplicates")[0]:
            imgui.text_colored("Duplicates found:", 1.0, 1.0, 0.0, 1.0) 
            imgui.separator()
            
            if self.current_duplicates:
                imgui.text(f"Found {len(self.current_duplicates)} duplicate file(s):")
                imgui.spacing()
                for duplicate in self.current_duplicates:
                    imgui.bullet_text(duplicate)
                
                imgui.spacing()
                imgui.separator()
                imgui.text("What would you like to do?")
                imgui.spacing()
                
                # Action buttons
                imgui.begin_group()
                if imgui.button("Add Duplicates", width=150):
                    self.handle_duplicate_files("add")
                    self.current_duplicates = []
                    self.selected_files = []
                    imgui.close_current_popup()
                
                imgui.same_line()
                if imgui.button("Skip Duplicates", width=150):
                    self.handle_duplicate_files("skip")
                    self.current_duplicates = []
                    self.selected_files = []
                    imgui.close_current_popup()
                imgui.end_group()
            else:
                imgui.text("No duplicates found.")
                imgui.spacing()
                if imgui.button("Close", width=100):
                    self.current_duplicates = []
                    imgui.close_current_popup()
            imgui.end_popup()

        if imgui.button("Import Videos", width=parameter_column_width, height=30):
            self.selected_video_files = self.data_browser.select_video_files()
            if self.selected_video_files:
                self._reset_video_selection_state()
                self.video_prober.start(self.selected_video_files)
                imgui.open_popup("Video Frame Extraction")

        # Video Frame Extraction Popup
        imgui.set_next_window_position(self.app.content_width / 2, self.app.content_height / 2,
                                       imgui.APPEARING, 0.5, 0.5)
        imgui.set_next_window_size(840, 520, imgui.APPEARING)
        popup_opened, popup_visible = imgui.begin_popup_modal(
            "Video Frame Extraction", True,
            flags=imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_RESIZE)
        if popup_opened:
            self.video_infos.update(self.video_prober.poll())

            left_popup_width = imgui.get_window_width() // 3

            # --- LEFT SECTION: Controls ---
            # Bordered panes, as in the training module's settings/output split.
            imgui.begin_child("VideoPopupLeft", width=left_popup_width, height=0, border=True)
            pane_width = imgui.get_content_region_available()[0]
            imgui.text("Frame Extraction Option")

            imgui.separator()
            imgui.spacing()

            # --- Frame Extraction Interval ---
            imgui.text("Seconds Between Frames:")
            self.help_icon.render(self.help_texts.get("interval_video_extraction"))

            # Text backed like the training learning rate field: the buffer
            # keeps exactly what was typed, so no display format can lie
            # about the stored value.
            with imgui_utils.item_width(pane_width):
                changed, self.extraction_interval_text = imgui_utils.input_text(
                    "##interval_input", self.extraction_interval_text,
                    buffer_length=16, flags=imgui.INPUT_TEXT_CHARS_DECIMAL)
                if changed:
                    try:
                        self.extraction_interval = min(max(
                            float(self.extraction_interval_text), 0.0), 3600.0)
                    except ValueError:
                        pass

            imgui.spacing()

            # Note
            imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + pane_width)
            imgui.text_colored("Note: All selected videos use the same settings. If you want different settings for individual videos, please go back and extract them one by one.", 0.8, 0.8, 0.8)
            imgui.pop_text_wrap_pos()

            imgui.spacing()

            if not self.is_processing_video:
                if imgui.button("Extract Frames", width=pane_width):
                    self.is_processing_video = True
                    # A cancel token the previous worker never consumed would
                    # stop this run at its first poll.
                    self._drain_queue(self.video_extraction_queue)
                    self._drain_extraction_replies()
                    self.loading_widget.show_simple("Extracting frames...", show_progress=True,
                                                    on_cancel=self._cancel_video_extraction)
                    self.loading_widget.update_progress(0, len(self.selected_video_files))
                    LoggedProcess(
                        target=DatasetPreprocessingUtils.extract_videos,
                        args=(self.selected_video_files, self.extraction_interval, self.video_extraction_queue, self.video_extraction_reply),
                        name='video-extract'
                    ).start()
            else:
                self.loading_widget.render()

                # Check for video extraction progress updates
                if not self.video_extraction_reply.empty():
                    try:
                        progress_data = self.video_extraction_reply.get_nowait()
                        
                        if progress_data.get('type') == 'progress':
                            # Update progress
                            current = progress_data.get('current', 0)
                            total = progress_data.get('total', 0)
                            current_file = progress_data.get('current_file', '')
                            percentage = progress_data.get('percentage', None)
                            self.loading_widget.update_progress(current, total, current_file, percentage=percentage)
                            
                        elif progress_data.get('type') == 'completed':
                            # Extraction completed
                            frames_paths = progress_data.get('results', [])
                            
                            for frames_dir in frames_paths:
                                frame_path = Path(frames_dir)
                                frame_files = [str(f) for f in sorted(frame_path.iterdir())]
                                self.imported_files.extend(frame_files)

                            self._refresh_imported_thumbnails()

                            # Reset variables
                            self.selected_video_files = []
                            self._reset_video_selection_state()
                            self.is_processing_video = False
                            self.loading_widget.hide()
                            self.extraction_interval = self.settings.extraction_interval
                            self.extraction_interval_text = _format_interval(self.extraction_interval)
                            self.last_selected_file = None
                            
                            self._drain_extraction_replies()

                            imgui.close_current_popup()
                            
                    except:
                        pass

            imgui.end_child()

            imgui.same_line()

            # --- RIGHT SECTION: Video Thumbnails ---
            imgui.begin_child("VideoPopupRight", width=0, height=0, border=True)
            imgui.text("Selected Videos")
            imgui.separator()
            imgui.spacing()

            scroll_height = imgui.get_content_region_available()[1] - 40

            imgui.begin_child("VideoThumbnailsScroll", width=0, height=scroll_height, border=False)

            video_available_width, video_available_height = imgui.get_content_region_available()

            if self.selected_video_files:
                # Only update thumbnails if video list changed
                if len(self.selected_video_files) != self.last_video_count:
                    self.video_thumbnail_widget.update_thumbnails(self.selected_video_files)
                    self.last_video_count = len(self.selected_video_files)

                self.video_thumbnail_widget.poll()

                self.video_thumbnail_widget.render_thumbnails(video_available_width, video_available_height)
            else:
                imgui.text("No videos selected.")
            
            imgui.end_child() # Thumbnail Video Scroll

            infos = [self.video_infos.get(p) for p in self.selected_video_files]
            expected = sum(self.settings.estimate_extracted_frames(info,
                                                                   self.extraction_interval)
                           for info in infos if info)
            pending = " (calculating...)" if any(info is None for info in infos) else ""
            imgui.text(f"Expected frames extracted: up to {expected}{pending}")

            imgui.same_line(position=imgui.get_window_width() - 110)
            if imgui.button("Remove", width=100):
                selected_videos = self.video_thumbnail_widget.get_selected_indices()

                for idx in sorted(selected_videos, reverse=True):
                    if 0 <= idx < len(self.selected_video_files):
                        removed_path = self.selected_video_files[idx]
                        self.video_infos.pop(removed_path, None)
                        del self.selected_video_files[idx]

                self.video_thumbnail_widget.update_thumbnails(self.selected_video_files)
                self.video_thumbnail_widget.clear_selected()
                self.last_video_count = len(self.selected_video_files)

            imgui.end_child() # Video popup right

            # Title bar close icon pressed
            if not popup_visible:
                self.selected_video_files = []
                self._reset_video_selection_state()
                imgui.close_current_popup()

            imgui.end_popup() # Video popup end

        imgui.spacing()

        input_width = int(parameter_column_width * 0.25)

        # Image options
        header_opened = imgui.collapsing_header("Image Options", flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]

        self.help_icon.render(self.help_texts.get("image_options"),
                              url=self.help_urls.get("image_options"),
                              align_right=True)

        if header_opened:
            imgui.text("Resize Mode")
            imgui.same_line()
            changed_resize, new_resize_mode = imgui.combo("##resize_mode", self.settings.resizeMode, resize_mode)
            if changed_resize:
                self.settings.resizeMode = new_resize_mode

            imgui.text("Resolution")
            imgui.same_line()
            
            with imgui_utils.item_width(input_width):
                imgui.input_text("##res_w", str(self.img_res), 512, flags=imgui.INPUT_TEXT_READ_ONLY)
            imgui.same_line()
            imgui.text("x")
            imgui.same_line()
            with imgui_utils.item_width(input_width):
                imgui.input_text("##res_h", str(self.img_res), 512, flags=imgui.INPUT_TEXT_READ_ONLY)
            
            imgui.same_line()
            if imgui.button("-##img_res", width=button_width):
                self.img_res = max(self.img_res // 2, self.min_res)
                self.settings.size = self.img_res
                self._refresh_badges()

            imgui.same_line()
            if imgui.button("+##img_res", width=button_width):
                self.img_res = min(self.img_res * 2, self.max_res)
                self.settings.size = self.img_res
                self._refresh_badges()
            
            # Non-square settings checkbox
            clicked, non_square = imgui.checkbox("Non-square Framing", not self.square)
            if clicked:
                self.square = not non_square
                self.settings.nonSquare = not self.square
                self._refresh_badges()


            if not self.square:
                imgui.text("Aspect Ratio:")
                
                imgui.text("Width Ratio")
                imgui.same_line()
                changed_width, new_width_ratio = imgui.input_int("##width_ratio", self.settings.nonSquareSettings["widthRatio"])
                if changed_width and new_width_ratio >= 1:
                    self.settings.nonSquareSettings["widthRatio"] = new_width_ratio
                    self._refresh_badges()

                imgui.text("Height Ratio")
                imgui.same_line()
                changed_height, new_height_ratio = imgui.input_int("##height_ratio", self.settings.nonSquareSettings["heightRatio"])
                if changed_height and new_height_ratio >= 1:
                    self.settings.nonSquareSettings["heightRatio"] = new_height_ratio
                    self._refresh_badges()
                
                base_size = self.img_res
                ratio = self.settings.nonSquareSettings["heightRatio"] / self.settings.nonSquareSettings["widthRatio"]

                if ratio <= 1:
                    actual_width = base_size
                    actual_height = int(base_size * ratio)
                else:
                    actual_height = base_size
                    actual_width = int(base_size / ratio)
                
                imgui.text(f"Actual resolution: {actual_width}x{actual_height}")

                imgui.text("Padding Options")
                imgui.same_line()
                changed_color, new_padding_color = imgui.combo("##padding_options", self.settings.nonSquareSettings["paddingMode"], padding_color)
                if changed_color:
                    self.settings.nonSquareSettings["paddingMode"] = new_padding_color      

        # End of Image options

        upscaling_header_opened = imgui.collapsing_header("Upscaling", flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]

        self.help_icon.render(self.help_texts.get("upscaling"), align_right=True)

        if upscaling_header_opened:
            ai_on = self.settings.upscaleSettings["aiUpscale"]
            total_count = len(self.imported_files)
            imgui.text(f"{self._below_count} of {total_count} images will be upscaled.")

            if imgui.radio_button("Standard##upscale_method", not ai_on) and ai_on:
                self.settings.upscaleSettings["aiUpscale"] = False
                ai_on = False
            imgui.same_line(spacing=self.app.spacing * 2)
            if imgui.radio_button("AI Enhanced##upscale_method", ai_on) and not ai_on:
                previous = dict(self.settings.upscaleSettings)
                self.settings.upscaleSettings["aiUpscale"] = True
                self._request_upscale_weights(previous)
                ai_on = True

            control_width = parameter_column_width - imgui.calc_text_size("Denoise")[0] - 20
            with imgui_utils.grayed_out(not ai_on):
                imgui.text("Denoise")
                imgui.same_line()
                with imgui_utils.item_width(control_width):
                    denoise_changed, new_denoise = imgui.slider_float(
                        "##upscale_denoise", self.settings.upscaleSettings["denoise"],
                        0.0, 1.0, "%.2f")
            if denoise_changed and ai_on:
                previous = dict(self.settings.upscaleSettings)
                self.settings.upscaleSettings["denoise"] = new_denoise
                self._request_upscale_weights(previous)
        # End of Upscaling options

        augmentation_header_opened = imgui.collapsing_header("Augmentation", flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]

        self.help_icon.render(self.help_texts.get("augmentation"), align_right=True)

        if augmentation_header_opened:
            xflip_clicked, new_xflip = imgui.checkbox("X-Flip", self.settings.augmentationSettings["xFlip"])
            if xflip_clicked:
                self.settings.augmentationSettings["xFlip"] = new_xflip

            yflip_clicked, new_yflip = imgui.checkbox("Y-Flip", self.settings.augmentationSettings["yFlip"])
            if yflip_clicked:
                self.settings.augmentationSettings["yFlip"] = new_yflip
        
        imgui.spacing()

        imgui.separator()
        
        imgui.text("Folder Name")
        total_width = parameter_column_width - 20
        folder_name_width = total_width * 0.75  
        
        with imgui_utils.item_width(folder_name_width):
            changed, new_folder_name = imgui.input_text("##folder_name", self.settings.folder_name, 1024)
            if changed:
                self.settings.folder_name = new_folder_name
        imgui.same_line()
        imgui.text(f"_{self.settings.size}x{self.settings.size}")

        imgui.text("Save Path")
        
        _, new_save_path = imgui_utils.input_text("##save_path", str(self.save_path), 1024, 0, 
        width=parameter_column_width - imgui.calc_text_size("Browse##save_path")[0] + 8)
        if new_save_path != self.save_path:
            self.save_path = new_save_path.replace('\\', '/')
        
        imgui.same_line()
        if imgui.button("Browse##save_path", width=self.app.button_w, height=25):
            directory_path = self.data_browser.select_directory("Select Save Path", initial_dir=self.save_path)
            if directory_path:
                self.save_path = directory_path.replace('\\', '/')
        
        imgui.spacing()
        
        if imgui.button("Process & Save Data", width=parameter_column_width, height=30):
            if not self.imported_files:
                logger.warning("No images to process")
            else:
                self.settings.images = self.imported_files

                proposed_output_path = self._construct_output_path()
                self.settings.output_path = proposed_output_path
                
                if Path(proposed_output_path).exists():
                    self.folder_exists_warning = True
                else:
                    self.folder_exists_warning = False
                    self.is_processing_dataset = True
                    self.process_dataset()
                
                imgui.open_popup("Processing Dataset")

        # Batch Preprocessing Popup ------------------------------------------------------------
        imgui.set_next_window_size(self.app.content_width // 2.5, self.app.content_height // 2.5, imgui.ONCE)
        if imgui.begin_popup_modal("Processing Dataset")[0]:
            
            # Show folder exists warning first (if applicable)
            if self.folder_exists_warning and not self.is_processing_dataset and not self.processing_completed:
                imgui.text_colored("Warning: Folder Already Exists!", 1.0, 0.5, 0.0, 1.0)
                imgui.separator()
                
                imgui.push_text_wrap_pos(self.app.content_width // 2.5 - 40)
                imgui.text("The folder already exists at:")
                imgui.spacing()
                imgui.text_colored(self.settings.output_path, 1.0, 1.0, 0.0, 1.0)
                imgui.spacing()
                imgui.text("If you continue, existing files in this folder may be overwritten.")
                imgui.pop_text_wrap_pos()
                
                imgui.spacing()
                
                button_width = (self.app.content_width // 2.5 - 60) / 2
                imgui.begin_group()
                if imgui.button("Overwrite & Process", width=button_width):
                    self.folder_exists_warning = False
                    self.process_dataset()
                
                imgui.same_line()
                if imgui.button("Cancel", width=button_width):
                    self.folder_exists_warning = False
                    imgui.close_current_popup()
                imgui.end_group()
            
            # Show processing progress
            elif self.is_processing_dataset:
                imgui.text("Preprocessing Dataset for Training...")
                imgui.separator()
                
                imgui.text_colored("Settings:", 1.0, 1.0, 0.0, 1.0)

                imgui.text(f"Images: {len(self.settings.images)} files")
                imgui.text(f"Resolution: {self.settings.size}x{self.settings.size}")
                imgui.text(f"Resize Mode: {resize_mode[self.settings.resizeMode]}")
                
                if self.settings.nonSquare:
                    imgui.text("Non-square settings:")
                    imgui.indent(20)
                    imgui.text(f"  Width Ratio: {self.settings.nonSquareSettings['widthRatio']}")
                    imgui.text(f"  Height Ratio: {self.settings.nonSquareSettings['heightRatio']}")
                    imgui.text(f"  Padding Mode: {padding_color[self.settings.nonSquareSettings['paddingMode']]}")
                    imgui.unindent(20)
                
                # Augmentation settings
                if any(self.settings.augmentationSettings.values()):
                    imgui.text("Augmentations:")
                    imgui.indent(20)
                    if self.settings.augmentationSettings['xFlip']:
                        imgui.text("X-Flip: Yes")
                    if self.settings.augmentationSettings['yFlip']:
                        imgui.text("Y-Flip: Yes")
                    imgui.unindent(20)

                if self.settings.upscaleSettings["aiUpscale"]:
                    imgui.text(f"AI Upscaling: {self._below_count} of {len(self.settings.images)} images")
                    imgui.text(f"Denoise: {self.settings.upscaleSettings['denoise']:.2f}")

                imgui.text(f"Output: {self.settings.output_path}")

                imgui.spacing()
                imgui.separator()
                imgui.spacing()

                latest_progress = None
                while not self.processing_reply.empty():
                    try:
                        data = self.processing_reply.get_nowait()
                        if data.get('type') == 'completed':
                            self.processing_completed = True
                            self.is_processing_dataset = False
                            if hasattr(self, 'processing_process') and self.processing_process.is_alive():
                                self.processing_process.terminate()
                                self.processing_process.join(timeout=5)
                        elif data.get('type') == 'progress':
                            latest_progress = data
                    except:
                        pass
                
                if latest_progress:
                    self.progress_current = latest_progress.get('current', 0)
                    self.progress_total = latest_progress.get('total', 0)
                    self.progress_percentage = latest_progress.get('percentage', 0)
                    self.progress_file = latest_progress.get('current_file', '')
                
                imgui.text(f"Processing: {self.progress_current}/{self.progress_total} images")
                if self.progress_file:
                    imgui.text(f"Current File: {self.progress_file}")
                
                # Progress bar
                progress_width = self.app.content_width // 2.5 - 40
                imgui.progress_bar(self.progress_percentage / 100.0, (progress_width, 20))
                text_width = imgui.calc_text_size(f"{self.progress_percentage:.1f}%")[0]
                imgui.set_cursor_pos_x((progress_width - text_width)/2)
                imgui.text(f"{self.progress_percentage:.1f}%")
                
                # Cancel button (only show while processing)
                if imgui_utils.button("Cancel", width=progress_width):
                    self.processing_queue.put('cancel')
                    self.cancel_processing = True
                    self.is_processing_dataset = False

                    self.reset_progress_variables()
                    imgui.close_current_popup()
            
            # Completion message
            elif self.processing_completed:
                imgui.text_colored("Settings:", 1.0, 1.0, 0.0, 1.0)
                imgui.text(f"Images: {len(self.settings.images)} files")
                imgui.text(f"Resolution: {self.settings.size}x{self.settings.size}")
                imgui.text(f"Resize Mode: {resize_mode[self.settings.resizeMode]}")
                
                if self.settings.nonSquare:
                    imgui.indent(20)
                    imgui.text("Non-square settings:")
                    imgui.text(f"  Width Ratio: {self.settings.nonSquareSettings['widthRatio']}")
                    imgui.text(f"  Height Ratio: {self.settings.nonSquareSettings['heightRatio']}")
                    imgui.text(f"  Padding Mode: {padding_color[self.settings.nonSquareSettings['paddingMode']]}")
                    imgui.unindent(20)
                
                if any(self.settings.augmentationSettings.values()):
                    imgui.text("Augmentations:")
                    imgui.indent(20)
                    if self.settings.augmentationSettings['xFlip']:
                        imgui.text("X-Flip: Yes")
                    if self.settings.augmentationSettings['yFlip']:
                        imgui.text("Y-Flip: Yes")
                    imgui.unindent(20)

                if self.settings.upscaleSettings["aiUpscale"]:
                    imgui.text(f"AI Upscaling: {self._below_count} of {len(self.settings.images)} images")
                    imgui.text(f"Denoise: {self.settings.upscaleSettings['denoise']:.2f}")

                imgui.text(f"Output: {self.settings.output_path}")

                imgui.spacing()
                imgui.separator()
                imgui.spacing()

                imgui.text_colored("Processing completed successfully!", 0.0, 1.0, 0.0, 1.0)
                imgui.spacing()
                if imgui_utils.button("Close", width=self.app.content_width // 2.5 - 20):
                    self.processing_completed = False
                    self.folder_exists_warning = False
 
                    self.reset_progress_variables()
                    imgui.close_current_popup()

            imgui.end_popup()
        # End of Batch Preprocessing Popup ------------------------------------------------------------

        imgui.end()

        # --- Column 2: Image Thumbnails ---
        imgui.set_next_window_position(first_column_width, navbar_h)
        imgui.set_next_window_size(second_column_width, available_height)
        imgui.begin('Thumbnails##Preprocessing', closable=False, flags=(
            imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS))

        # Sticky Header (outside scrollable area)
        imgui.text("Imported Images")
        imgui.same_line()
        imgui.text(f"({len(self.imported_files)} images)")
        
        imgui.same_line(position=imgui.get_window_width() - imgui.calc_text_size("Select All")[0] - 50)

        if imgui.button("Select All", width=self.app.button_w):
            self.thumbnail_widget.select_all()

        imgui.separator()

        # --- Scrollable Thumbnails Grid ---
        scroll_height = available_height - 150
        imgui.begin_child("ThumbnailsScroll", width=0, height=scroll_height, border=False)

        available_width = second_column_width - 35
        self.thumbnail_widget.render_thumbnails(available_width, scroll_height)

        imgui.end_child()
        # End Thumbnails Scroll

        self.thumbnail_widget.poll()

        remove_selected = imgui.button("Remove Selected Images", width=available_width + 18, height=30)
        remove_selected = remove_selected or self.thumbnail_widget.is_delete_pressed()
        
        if remove_selected:
            selected_indices = self.thumbnail_widget.get_selected_indices()

            for idx in sorted(selected_indices, reverse=True):
                if 0 <= idx < len(self.imported_files):
                    del self.imported_files[idx]

            self._refresh_imported_thumbnails()
            self.thumbnail_widget.clear_selected()
            self.last_selected_file = None

        imgui.end() 
        # --- End of column 2 ---

        # --- Column 3: Image Preview ---
        imgui.set_next_window_position(first_column_width + second_column_width, navbar_h)
        imgui.set_next_window_size(third_column_width, available_height)
        imgui.begin('Preview##Preprocessing', closable=False, flags=(
            imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS))
        
        imgui.text("Image Preview")

        imgui.same_line(position=imgui.get_window_width() - imgui.calc_text_size("Preview Original")[0] - 55)
        # Preview Original button
        preview_original_clicked, preview_original_new = imgui.checkbox("Preview Original", self.preview_original)
        if preview_original_clicked:
            self.preview_original = preview_original_new
            
        imgui.separator()

        # Space for image preview
        preview_width = third_column_width - 25
        preview_height = (available_height - 100)/2

        selected_indices = self.thumbnail_widget.get_selected_indices()
        selected_file = None
        if selected_indices:
            # Use the first selected index for preview
            selected_idx = selected_indices[0]
            if 0 <= selected_idx < len(self.thumbnail_widget.selected_files):
                selected_file = self.thumbnail_widget.selected_files[selected_idx]
        
        if self._should_update_preview(selected_file):
            if selected_file:
                self.image_preview_widget.update_preview(selected_file, self.settings, self.preview_original)
            else:
                self.image_preview_widget.update_preview(None, self.settings, self.preview_original)
        
        imgui.begin_child("Preview", width=preview_width, height=preview_height, border=False)
        self.image_preview_widget.render(preview_width, preview_height)
        imgui.end_child()


        if selected_file:
            image_data = self.settings.get_image_data(selected_file)
            
            if image_data['error']:
                imgui.text(f"Error loading file: {image_data['error']}")
            else:
                imgui.text(f"Selected File: {image_data['filename']}")
        else:
            imgui.text("No image selected")
        
        imgui.separator()

        if imgui.collapsing_header("Image Details")[0]:
            if selected_file and not image_data.get('error'):
                imgui.text(f"Full Path: {image_data['full_path']}")
                
                if image_data['file_size'] is not None:
                    imgui.text(f"Size: {image_data['file_size']} MB")
                else:
                    imgui.text("Size: Unable to determine")
                
                if image_data['width'] and image_data['height']:
                    imgui.text(f"Original Resolution: {image_data['width']}x{image_data['height']}")
                else:
                    imgui.text("Resolution: Unable to determine")
                    
                if image_data['format']:
                    imgui.text(f"Image Format: {image_data['format']}")
                    
                if image_data['mode']:
                    imgui.text(f"Color Mode: {image_data['mode']}")
                    
                if image_data['orientation']:
                    imgui.text(f"Image Orientation: {image_data['orientation']}")
            else:
                imgui.text("No file selected")
        
        imgui.end()

        # End of __call__ preprocessing module content

    # Check imported files for duplicates
    def check_for_duplicates(self, selected_files):
        """Check if files in selected_files have duplicate full paths in the imported files list"""
        duplicates = []

        imported_paths = set(self.imported_files)

        for file_path in selected_files:
            if file_path in imported_paths:
                duplicates.append(file_path)
        
        return duplicates
    
    def clean_duplicate_files(self, selected_files):
        """Clean duplicate files from the selected files list"""
        imported_paths = set(self.imported_files)
        
        cleaned_files = []
        for file_path in selected_files:
            if file_path not in imported_paths:
                cleaned_files.append(file_path)
        
        return cleaned_files
    
    def handle_duplicate_files(self, status):
        """Handle duplicate files based on user choice"""
        if not self.selected_files:
            return
        
        if status == "add":
            self.imported_files.extend(self.selected_files)
        elif status == "skip":
            files_to_add = self.clean_duplicate_files(self.selected_files)
            self.imported_files.extend(files_to_add)
        
        self._refresh_imported_thumbnails()
        self.last_selected_file = None
    # ------------------------------

    # --- Upscaling ---
    def _probe_image_dims(self, paths):
        """Header-only read of image dimensions, EXIF orientation aware."""
        for path in paths:
            if path in self.image_dims:
                continue
            try:
                with PIL.Image.open(path) as img:
                    width, height = img.size
                    orientation = img.getexif().get(0x0112, 1)
                if orientation in (5, 6, 7, 8):
                    width, height = height, width
                self.image_dims[path] = (width, height)
            except Exception:
                logger.debug("Could not read dimensions for %s", path, exc_info=True)
                self.image_dims[path] = None

    def _probe_pending_dims(self):
        """Probe one bounded chunk of pending paths per frame.

        Opening every file of a large import in a single frame stalls the UI
        for seconds, so the work is spread over frames instead.
        """
        if not self._pending_dims:
            return
        chunk = self._pending_dims[:self.PROBE_CHUNK]
        del self._pending_dims[:self.PROBE_CHUNK]
        self._probe_image_dims(chunk)
        self._refresh_badges()

    def _target_dims(self):
        """Output dimensions of the current settings, as the Image Options show them."""
        base = self.settings.size
        if not self.settings.nonSquare:
            return base, base
        width_ratio = max(self.settings.nonSquareSettings["widthRatio"], 1)
        height_ratio = max(self.settings.nonSquareSettings["heightRatio"], 1)
        ratio = height_ratio / width_ratio
        if ratio <= 1:
            return base, int(base * ratio)
        return int(base / ratio), base

    def _below_target_paths(self):
        target_width, target_height = self._target_dims()
        below = []
        for path in self.imported_files:
            dims = self.image_dims.get(path)
            if not dims or not dims[0] or not dims[1]:
                continue
            if dims[0] < target_width or dims[1] < target_height:
                below.append(path)
        return below

    def _refresh_badges(self):
        target_width, target_height = self._target_dims()
        badges = {}
        for path in self._below_target_paths():
            width, height = self.image_dims[path]
            badges[path] = (f"This {width}x{height} image will be upscaled "
                            f"to {target_width}x{target_height}.")
        self.thumbnail_widget.set_badges(badges)
        self._below_count = len(badges)

    def _refresh_imported_thumbnails(self):
        self.thumbnail_widget.update_thumbnails(self.imported_files)
        self._pending_dims = [p for p in self.imported_files if p not in self.image_dims]
        self._refresh_badges()

    def _request_upscale_weights(self, previous_settings):
        """Queue downloads for every weight the current settings need.

        previous_settings is the upscaleSettings snapshot taken before the
        change. It is restored if a queued download fails or is cancelled.
        """
        current = self.settings.upscaleSettings
        needed = [key for key in required_weights(current["denoise"])
                  if not os.path.exists(weight_path(key))
                  and key != self.upscale_download_kind
                  and key not in self.upscale_download_queue]
        if not needed:
            return
        if self.upscale_settings_revert is None:
            self.upscale_settings_revert = dict(previous_settings)
        self.upscale_download_queue.extend(needed)
        if not self.upscale_download_active:
            self._start_upscale_download(self.upscale_download_queue.pop(0))

    def _start_upscale_download(self, weight_key):
        """Download one weight in a thread with a progress overlay."""
        self.upscale_download_kind = weight_key
        self.upscale_download_result = None
        self.upscale_download_cancel = threading.Event()
        self.upscale_download_active = True
        self.upscale_download_cancelling = False
        self.loading_widget.show_simple("Downloading upscaling model...", show_progress=True,
                                        on_cancel=self._cancel_upscale_download,
                                        count_label=None, file_label="Downloaded")

        def on_progress(done, total):
            # The overlay counts items by default. This one counts bytes, so
            # the size goes in the detail line and the percentage is explicit.
            if total:
                text = f"{done / 1048576:.1f} of {total / 1048576:.1f} MB"
                percentage = done / total * 100.0
            else:
                text = f"{done / 1048576:.1f} MB"
                percentage = 0.0
            self.loading_widget.update_progress(done, total, current_file=text,
                                                percentage=percentage)

        def run():
            try:
                self.upscale_download_result = ensure_weight(
                    weight_key, on_progress, self.upscale_download_cancel)
            except Exception:
                logger.exception("Upscaling weight download failed")
                self.upscale_download_result = None

        self.upscale_download_thread = threading.Thread(target=run, daemon=True,
                                                        name='upscale-weights')
        self.upscale_download_thread.start()

    def _cancel_upscale_download(self):
        """Close the overlay at once and ask the worker to stop.

        The worker only sees the cancel token on its next chunk, so polling
        continues until it exits and the setting is reverted there.
        """
        self.upscale_download_cancelling = True
        self.upscale_download_cancel.set()
        self.loading_widget.hide()

    def _poll_upscale_download(self):
        """Render the overlay while downloading. Revert settings on failure."""
        if not self.upscale_download_active:
            return
        if self.upscale_download_thread is not None and self.upscale_download_thread.is_alive():
            if not self.upscale_download_cancelling:
                self.loading_widget.render()
            return
        if self.upscale_download_result is None:
            # Cancelled or failed: drop the queue and restore the settings
            # from before the change that started these downloads.
            if self.upscale_settings_revert is not None:
                self.settings.upscaleSettings.update(self.upscale_settings_revert)
            self._finish_upscale_downloads()
            self._refresh_badges()
            return
        while self.upscale_download_queue:
            next_type = self.upscale_download_queue.pop(0)
            if not os.path.exists(weight_path(next_type)):
                self._start_upscale_download(next_type)
                return
        self._finish_upscale_downloads()

    def _finish_upscale_downloads(self):
        self.upscale_download_queue = []
        self.upscale_download_active = False
        self.upscale_download_kind = None
        self.upscale_settings_revert = None
        if not self.upscale_download_cancelling:
            self.loading_widget.hide()
        self.upscale_download_cancelling = False
    # ------------------------------

    def _reset_video_selection_state(self):
        """Drop probed media info and thumbnail state of the video popup."""
        self.video_prober.cancel()
        self.video_infos = {}
        self.last_video_count = None

    @staticmethod
    def _drain_queue(q):
        while not q.empty():
            try:
                q.get_nowait()
            except:
                break

    def _drain_extraction_replies(self):
        self._drain_queue(self.video_extraction_reply)

    def _cancel_video_extraction(self):
        """Stop the extraction worker and return to the settings popup.

        Frames already written stay on disk but are not imported: a cancelled
        worker never sends the completed message.
        """
        self.video_extraction_queue.put("cancel")
        self.is_processing_video = False
        self.loading_widget.hide()
        self._drain_extraction_replies()

    # ---Helper functions for preview updates
    def _get_settings_hash(self):
        """Create a hash of current settings for change detection."""
        return (
            self.settings.size,
            self.settings.resizeMode,
            self.settings.nonSquare,
            self.settings.nonSquareSettings.get("widthRatio", 16),
            self.settings.nonSquareSettings.get("heightRatio", 9),
            self.settings.nonSquareSettings.get("paddingMode", 0),
            self.preview_original 
        )

    def _should_update_preview(self, selected_file):
        """Check if preview should be updated based on changes."""
        current_settings_hash = self._get_settings_hash()

        # Initial state or deletion handling
        if self.last_selected_file is None:
            self.last_selected_file = selected_file
            self.last_settings_hash = current_settings_hash
            return True
        
        # Check if file or settings changed
        file_changed = selected_file != self.last_selected_file
        settings_changed = current_settings_hash != self.last_settings_hash
        
        if file_changed or settings_changed:
            self.last_selected_file = selected_file
            self.last_settings_hash = current_settings_hash
            return True
        
        return False
    # ------------------------------

    # --- Process Dataset ---
    def _construct_output_path(self):
        """Construct output path from parent directory + folder name + resolution"""
        resolution_suffix = f"_{self.settings.size}x{self.settings.size}"
        folder_name_with_resolution = self.settings.folder_name + resolution_suffix
        return str(Path(self.save_path) / folder_name_with_resolution)
    
    def process_dataset(self):
        """Start the dataset processing in a separate process"""
        # Update output path
        self.settings.output_path = self._construct_output_path()
        
        self.is_processing_dataset = True

        self.processing_queue.put(self.settings)
        
        self.processing_process = LoggedProcess(
            target=DatasetPreprocessingUtils.create_training_dataset,
            args=(self.processing_queue, self.processing_reply),
            name='dataset-build')
        self.processing_process.start()
    # ------------------------------

    # --- Cleanup ---
    def cleanup(self):
        """Clean up multiprocessing resources before destroying the object"""
        try:
            self.reset_progress_variables()

            self.video_prober.shutdown()

            if hasattr(self, 'thumbnail_widget') and self.thumbnail_widget is not None:
                self.thumbnail_widget.cleanup()
            if hasattr(self, 'video_thumbnail_widget') and self.video_thumbnail_widget is not None:
                self.video_thumbnail_widget.cleanup()
            if hasattr(self, 'image_preview_widget') and self.image_preview_widget is not None:
                self.image_preview_widget.cleanup()
            if hasattr(self, 'data_browser') and self.data_browser is not None:
                self.data_browser.cleanup()
            if hasattr(self, 'loading_widget') and self.loading_widget is not None:
                self.loading_widget.cleanup()
            if hasattr(self, 'help_icon') and self.help_icon is not None:
                self.help_icon.cleanup()
                
        except Exception as e:
            logger.warning("Error during cleanup: %s", e)

    def reset_progress_variables(self):
        """Reset progress tracking variables to default values"""
        self.progress_current = 0
        self.progress_total = 0
        self.progress_percentage = 0
        self.progress_file = ""
        self.processing_completed = False
        self.is_processing_dataset = False
        self.cancel_processing = False
        self.folder_exists_warning = False
        
        # Terminate the background process if it's still running
        if hasattr(self, 'processing_process') and self.processing_process.is_alive():
            self.processing_process.terminate()
            self.processing_process.join(timeout=5)  # Wait up to 5 seconds before fully closing
        
        # Clear any remaining items in queues
        while not self.processing_queue.empty():
            try:
                self.processing_queue.get_nowait()
            except:
                break
    # ------------------------------