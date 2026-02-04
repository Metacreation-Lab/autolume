from pathlib import Path
import pandas as pd
import imgui
from utils.gui_utils import imgui_utils
import multiprocessing as mp

from widgets.native_browser_widget import NativeBrowserWidget
from widgets.thumbnail_widget import ThumbnailWidget
from widgets.image_preview_widget import ImagePreviewWidget
from widgets.loading_widget import LoadingOverlayManager
from widgets.help_icon_widget import HelpIconWidget
from utils.dataset_preprocessing_utils import DatasetPreprocessingUtils

resize_mode = ['stretch','center crop']
padding_color = ['black', 'white', 'bleeding']

class DataPreprocessing:
    """Data Preprocessing UI"""
    def __init__(self, app):
        self.app = app

        self.settings = DatasetPreprocessingUtils()

        self.data_browser = NativeBrowserWidget()
        self.thumbnail_widget = ThumbnailWidget() # Imported image thumbnails
        self.video_thumbnail_widget = ThumbnailWidget()  # Video popup thumbnails
        self.video_thumbnail_widget.generate_thumbnails = True  # Always render video thumbnails
        self.image_preview_widget = ImagePreviewWidget()
        self.loading_widget = LoadingOverlayManager(app)  # Enhanced loading overlay manager
        
        # Background thumbnail processing
        self.thumbnail_queue = mp.Queue()
        self.thumbnail_reply = mp.Queue()
        self.thumbnail_process = None
        self.is_processing_thumbnails = False
        self.thumbnail_process_started = False
        
        # Video thumbnail processing
        self.video_thumbnail_queue = mp.Queue()
        self.video_thumbnail_reply = mp.Queue()
        self.video_thumbnail_process = None
        self.is_processing_video_thumbnails = False 

        self.imported_files = [] 
        self.selected_video_files = [] 
        self.current_duplicates = []  # Store current image duplicates for popup display
        self.selected_files = []  # Store selected files when duplicates are found
        
        self.preview_original = False

        # Video frame extraction
        self.fps = self.settings.fps
        self.video_extraction_queue = mp.Queue()
        self.video_extraction_reply = mp.Queue()
        self.is_processing_video = False
        
        self.res_factor = 0
        self.start_res = self.settings.size 
        self.img_res = self.start_res * (2 ** self.res_factor) # current image resolution

        self.square = True # non-square framing settings (image is square or not)
        
        # Track changes for preview updates
        self.last_selected_file = None
        self.last_settings_hash = None

        # Dataset processing
        self.processing_queue = mp.Queue()
        self.processing_reply = mp.Queue()
        self.cancel_processing = False
        self.processing_process = mp.Process(target=DatasetPreprocessingUtils.create_training_dataset, args=(self.processing_queue, self.processing_reply))
        
        # Processing popup control
        self.is_processing_dataset = False
        self.folder_exists_warning = False  
        # Progress tracking
        self.progress_current = 0
        self.progress_total = 0
        self.progress_percentage = 0
        self.progress_file = ""
        self.processing_completed = False
        
        self.save_path = self.settings.output_path 

        self.help_icon = HelpIconWidget()
        self.help_texts, self.help_urls = self.help_icon.load_help_texts("preprocessing")

    def __call__(self):
        """Preprocessing content"""
        imgui_utils.set_default_style()
        
        # Calculate column widths
        first_column_width = self.app.content_width * 0.2
        remaining_width = self.app.content_width - first_column_width
        second_column_width = remaining_width // 2
        third_column_width = remaining_width - second_column_width

        button_width = self.app.font_size
        
        # --- Column 1: Dataset Parameter and Options ---
        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(first_column_width, self.app.content_height)
        imgui.begin('Parameters##Preprocessing', closable=False, flags=(
            imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS))
        
        parameter_column_width = first_column_width - 20 

        text = "Import Data"
        text_width = imgui.calc_text_size(text).x
        help_icon_size = imgui.get_font_size()
        style = imgui.get_style()

        imgui.text(text)

        spacing = first_column_width - (style.window_padding[0] * 2) - text_width - help_icon_size - style.item_spacing[0] - 10
            
        imgui.same_line()
        imgui.dummy(spacing, 0)
        import_hyperlinks = []
        import_url = self.help_urls.get("import_data")
        if import_url:
            import_hyperlinks.append((import_url, "Supported Media Formats"))
        tutorial_video_url = "https://www.youtube.com/watch?v=7Pc5-ULeXkM&feature=youtu.be"
        import_hyperlinks.append((tutorial_video_url, "Tutorial Video"))
        
        if import_hyperlinks:
            self.help_icon.render_with_urls(self.help_texts.get("import_data"), import_hyperlinks)
        else:
            self.help_icon.render(self.help_texts.get("import_data"))

        imgui.separator()
 
        if imgui.button("Import Images", width=parameter_column_width, height=30):
            selected_images = self.data_browser.select_image_files()
            duplicates = self.check_for_duplicates(selected_images)
            if duplicates:
                self.current_duplicates = duplicates  # Store duplicate filenames for popup
                self.selected_files = selected_images  # Store full file paths for processing
                imgui.open_popup("Duplicates")
            else:
                self.imported_files.extend(selected_images)
                self.thumbnail_widget.update_thumbnails(self.imported_files)
                
                if self.thumbnail_widget.generate_thumbnails and self.imported_files:
                    self._start_background_thumbnail_generation()
                
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
                # Reset cache variables when opening popup with new videos
                if hasattr(self, 'last_video_count'):
                    delattr(self, 'last_video_count')
                if hasattr(self, 'last_fps'):
                    delattr(self, 'last_fps')
                if hasattr(self, 'last_expected_frames'):
                    delattr(self, 'last_expected_frames')
                # Start background thumbnail generation for video thumbnails
                self._start_video_thumbnail_generation()
                imgui.open_popup("Video Frame Extraction")

        # Video Frame Extraction Popup
        imgui.set_next_window_size(self.app.content_width // 2.5, self.app.content_height // 2.5, imgui.ONCE)
        if imgui.begin_popup_modal("Video Frame Extraction", flags=imgui.WINDOW_NO_SCROLLBAR)[0]:
            popup_width = imgui.get_window_width()
            popup_height = imgui.get_window_height() - 50

            left_popup_width = popup_width // 3 
            right_popup_width = popup_width - left_popup_width

            # --- LEFT SECTION: Controls ---
            imgui.begin_child("VideoPopupLeft", width=left_popup_width, height=popup_height, border=False)
            imgui.text("Frame Extraction Option")

            imgui.separator() 

            # --- Frame Extraction FPS ---
            imgui.text("FPS for Video Extraction:")
            self.help_icon.render(self.help_texts.get("fps_video_extraction"))
            
            min_fps = 1
            max_fps = 120
            with imgui_utils.item_width(left_popup_width - 10):
                _, self.fps = imgui.input_int("##fps_input", self.fps)
                if self.fps < min_fps:
                    self.fps = 1
                elif self.fps > max_fps:
                    self.fps = 120

            imgui.spacing()

            # Note
            imgui.push_text_wrap_pos(left_popup_width - 20)
            imgui.text_colored("Note: All selected videos will be extracted using the same frame rate you choose here.\nIf you want to use different frame rates for individual videos, please go back and extract them one by one.", 0.8, 0.8, 0.8)
            imgui.pop_text_wrap_pos()

            imgui.spacing()

            if not self.is_processing_video:
                if imgui.button("Extract Frames", width=left_popup_width - 10):
                    self.is_processing_video = True
                    self.loading_widget.show_simple("Extracting frames...", show_progress=True)
                    self.loading_widget.update_progress(0, len(self.selected_video_files))
                    mp.Process(
                        target=DatasetPreprocessingUtils.extract_videos,
                        args=(self.selected_video_files, self.fps, self.video_extraction_queue, self.video_extraction_reply)
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
                            self.loading_widget.update_progress(current, total, current_file)
                            
                        elif progress_data.get('type') == 'completed':
                            # Extraction completed
                            frames_paths = progress_data.get('results', [])
                            
                            for frames_dir in frames_paths:
                                frame_path = Path(frames_dir)
                                frame_files = [str(f) for f in frame_path.iterdir()]
                                self.imported_files.extend(frame_files)

                            self.thumbnail_widget.update_thumbnails(self.imported_files)
                            
                            if self.thumbnail_widget.generate_thumbnails and self.imported_files:
                                self._start_background_thumbnail_generation()
                            
                            # Reset variables
                            self.selected_video_files = [] 
                            self.is_processing_video = False
                            self.loading_widget.hide()
                            self.fps = self.settings.fps 
                            self.last_selected_file = None
                            
                            # Clear the queue
                            while not self.video_extraction_reply.empty():
                                try:
                                    self.video_extraction_reply.get_nowait()
                                except:
                                    break
                            
                            imgui.close_current_popup() 
                            
                    except:
                        pass

            imgui.spacing()

            if imgui_utils.button("Back to Main Menu", width=left_popup_width - 10, enabled=True):
                self.selected_video_files = []  
                self._stop_video_thumbnail_generation()
                # Reset cache variables when closing popup
                if hasattr(self, 'last_video_count'):
                    delattr(self, 'last_video_count')
                if hasattr(self, 'last_fps'):
                    delattr(self, 'last_fps')
                if hasattr(self, 'last_expected_frames'):
                    delattr(self, 'last_expected_frames')
                imgui.close_current_popup()
            imgui.end_child()

            imgui.same_line()

            # --- RIGHT SECTION: Video Thumbnails ---
            imgui.begin_child("VideoPopupRight", width=right_popup_width - 30, height=popup_height, border=False)
            imgui.text("Selected Videos")
            imgui.separator()

            scroll_height = popup_height - 80
            
            imgui.begin_child("VideoThumbnailsScroll", width=0, height=scroll_height, border=False)

            video_available_width = right_popup_width - 50
            video_available_height = scroll_height - 60 

            if self.selected_video_files:
                # Only update thumbnails if video list changed
                if not hasattr(self, 'last_video_count') or len(self.selected_video_files) != self.last_video_count:
                    self.video_thumbnail_widget.update_thumbnails(self.selected_video_files)
                    self.last_video_count = len(self.selected_video_files)

                self._check_video_thumbnail_results() 
                
                self.video_thumbnail_widget.render_thumbnails(video_available_width, video_available_height)
            else:
                imgui.text("No videos selected.")
            
            imgui.end_child() # Thumbnail Video Scroll
                        
            # Only recalculate frames if FPS or video list changed
            if (not hasattr(self, 'last_fps') or self.fps != self.last_fps or 
                not hasattr(self, 'last_video_count') or len(self.selected_video_files) != self.last_video_count):
                expected_frames = 0
                for video_path in self.selected_video_files:
                    expected_frames = expected_frames + DatasetPreprocessingUtils.calculate_expected_video_frames(video_path, self.fps)
                self.last_expected_frames = expected_frames
                self.last_fps = self.fps
                self.last_video_count = len(self.selected_video_files)
            
            imgui.text(f"Expected frames extracted: {getattr(self, 'last_expected_frames', 0)}")

            imgui.same_line(position=video_available_width - 50)
            if imgui.button("Remove"):
                selected_videos = self.video_thumbnail_widget.get_selected_indices()

                for idx in sorted(selected_videos, reverse=True):
                    if 0 <= idx < len(self.selected_video_files):
                        del self.selected_video_files[idx]

                self.video_thumbnail_widget.update_thumbnails(self.selected_video_files)
                self.video_thumbnail_widget.clear_selected()

            imgui.end_child() # Video popup right

            imgui.end_popup() # Video popup end

        imgui.spacing()

        input_width = int(parameter_column_width * 0.25)

        # Image options
        header_opened = imgui.collapsing_header("Image Options", flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]
        
        imgui.same_line()
        help_icon_size = imgui.get_font_size()
        style = imgui.get_style()
        header_text_width = imgui.calc_text_size("Image Options").x
        spacing = first_column_width - (style.window_padding[0] * 2) - header_text_width - help_icon_size - style.item_spacing[0] - 40
        imgui.dummy(spacing, 0)
        self.help_icon.render_with_url(self.help_texts.get("image_options"), self.help_urls.get("image_options"), "Read More")
        
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
                self.res_factor = max(self.res_factor - 1, 0)   
                self.img_res = self.start_res * (2 ** self.res_factor)
                self.settings.size = self.img_res
            
            imgui.same_line()
            if imgui.button("+##img_res", width=button_width):
                self.res_factor = self.res_factor + 1
                self.img_res = self.start_res * (2 ** self.res_factor)
                self.settings.size = self.img_res
            
            # Non-square settings checkbox
            clicked, non_square = imgui.checkbox("Non-square Framing", not self.square)
            if clicked:
                self.square = not non_square 
                self.settings.nonSquare = not self.square
                       
            if not self.square:
                imgui.text("Aspect Ratio:")
                
                imgui.text("Width Ratio")
                imgui.same_line()
                changed_width, new_width_ratio = imgui.input_int("##width_ratio", self.settings.nonSquareSettings["widthRatio"])
                if changed_width and new_width_ratio >= 1:
                    self.settings.nonSquareSettings["widthRatio"] = new_width_ratio

                imgui.text("Height Ratio")
                imgui.same_line()
                changed_height, new_height_ratio = imgui.input_int("##height_ratio", self.settings.nonSquareSettings["heightRatio"])
                if changed_height and new_height_ratio >= 1:
                    self.settings.nonSquareSettings["heightRatio"] = new_height_ratio
                
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

        augmentation_header_opened = imgui.collapsing_header("Augmentation", flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]
        
        imgui.same_line()
        help_icon_size = imgui.get_font_size()
        style = imgui.get_style()
        header_text_width = imgui.calc_text_size("Augmentation").x
        spacing = first_column_width - (style.window_padding[0] * 2) - header_text_width - help_icon_size - style.item_spacing[0] - 40
        imgui.dummy(spacing, 0)
        self.help_icon.render(self.help_texts.get("augmentation"))
        
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
        self.help_icon.render(self.help_texts.get("save_path"))
        
        _, new_save_path = imgui_utils.input_text("##save_path", self.save_path, 1024, 0, 
        width=parameter_column_width - imgui.calc_text_size("Browse##save_path")[0] + 8)
        if new_save_path != self.save_path:
            self.save_path = new_save_path.replace('\\', '/')
        
        imgui.same_line()
        if imgui.button("Browse##save_path", width=self.app.button_w, height=25):
            directory_path = self.data_browser.select_directory("Select Save Path")
            if directory_path:
                self.save_path = directory_path.replace('\\', '/')
            else:
                print("No save path selected")
        
        imgui.spacing()
        
        if imgui.button("Process & Save Data", width=parameter_column_width, height=30):
            if not self.imported_files:
                print("No images to process.")
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

        imgui.spacing()
        imgui.spacing()
        
        imgui.separator()
        if imgui.button("Back to Menu", width=parameter_column_width, height=30):
            self.cleanup()
            self.app.set_visible_menu()

        imgui.end()
        
        # --- Column 2: Image Thumbnails ---
        imgui.set_next_window_position(first_column_width, 0)
        imgui.set_next_window_size(second_column_width, self.app.content_height)
        imgui.begin('Thumbnails##Preprocessing', closable=False, flags=(
            imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS))

        # Sticky Header (outside scrollable area)
        imgui.text("Imported Images")
        imgui.same_line()
        imgui.text(f"({len(self.imported_files)} images)")
        
        imgui.same_line(position=imgui.get_window_width() - imgui.calc_text_size("Render Thumbnail")[0] - imgui.calc_text_size("Select All")[0] - 70)

        prev_thumbnail_mode = self.thumbnail_widget.generate_thumbnails
        generate_thumbnails_clicked, self.thumbnail_widget.generate_thumbnails = imgui.checkbox(
            "Render Thumbnail", self.thumbnail_widget.generate_thumbnails
        )

        imgui.same_line()
        
        if imgui.button("Select All"):
            self.thumbnail_widget.select_all()
        
        if generate_thumbnails_clicked:
            new_thumbnail_mode = self.thumbnail_widget.generate_thumbnails
            self.thumbnail_widget.set_thumbnail_mode(new_thumbnail_mode, prev_thumbnail_mode)
            if new_thumbnail_mode and self.imported_files:
                self._start_background_thumbnail_generation()
            else:
                self._stop_background_thumbnail_generation()

        imgui.separator()

        # --- Scrollable Thumbnails Grid ---
        scroll_height = self.app.content_height - 150  
        imgui.begin_child("ThumbnailsScroll", width=0, height=scroll_height, border=False)

        available_width = second_column_width - 35
        available_height = scroll_height
        self.thumbnail_widget.render_thumbnails(available_width, available_height)

        imgui.end_child()  
        # End Thumbnails Scroll
        
        self._check_background_thumbnail_results()

        remove_selected = imgui.button("Remove Selected Images", width=available_width, height=30)
        remove_selected = remove_selected or self.thumbnail_widget.is_delete_pressed()
        
        if remove_selected:
            selected_indices = self.thumbnail_widget.get_selected_indices()

            for idx in sorted(selected_indices, reverse=True):
                if 0 <= idx < len(self.imported_files):
                    del self.imported_files[idx]

            self.thumbnail_widget.update_thumbnails(self.imported_files)
            self.thumbnail_widget.clear_selected()
            self.last_selected_file = None

        imgui.end() 
        # --- End of column 2 ---

        # --- Column 3: Image Preview ---
        imgui.set_next_window_position(first_column_width + second_column_width, 0)
        imgui.set_next_window_size(third_column_width, self.app.content_height)
        imgui.begin('Preview##Preprocessing', closable=False, flags=(
            imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS))
        
        imgui.text("Image Preview")

        imgui.same_line(position=imgui.get_window_width() - imgui.calc_text_size("Preview Original")[0] - 50)
        # Preview Original button
        preview_original_clicked, preview_original_new = imgui.checkbox("Preview Original", self.preview_original)
        if preview_original_clicked:
            self.preview_original = preview_original_new
            
        imgui.separator()

        # Space for image preview
        preview_width = third_column_width - 25
        preview_height = (self.app.content_height - 100)/2

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
        
        self.thumbnail_widget.update_thumbnails(self.imported_files)
        
        if self.thumbnail_widget.generate_thumbnails and self.imported_files:
            self._start_background_thumbnail_generation()
        
        self.last_selected_file = None
    # ------------------------------

     # --- Background Thumbnail Generation Helper Functions ---
    def _start_background_thumbnail_generation(self):
        """Start background thumbnail generation process"""
        if not self.thumbnail_process_started or (self.thumbnail_process is not None and not self.thumbnail_process.is_alive()):
            self.thumbnail_process = mp.Process(
                target=ThumbnailWidget.process_thumbnails_background,
                args=(self.thumbnail_queue, self.thumbnail_reply)
            )
            self.thumbnail_process.start()
            self.thumbnail_process_started = True

        self.is_processing_thumbnails = True
        
        request = {
            'type': 'generate_thumbnails',
            'file_paths': self.imported_files,
            'thumbnail_size': self.thumbnail_widget.thumbnail_size
        }
        self.thumbnail_queue.put(request)
    
    def _start_video_thumbnail_generation(self):
        """Start background thumbnail generation process for video thumbnails"""
        if self.is_processing_video_thumbnails:
            return  
        
        # Start process if not started or if it died
        if self.video_thumbnail_process is None or not self.video_thumbnail_process.is_alive():
            self.video_thumbnail_process = mp.Process(
                target=ThumbnailWidget.process_thumbnails_background,
                args=(self.video_thumbnail_queue, self.video_thumbnail_reply)
            )
            self.video_thumbnail_process.start()
        
        # Send thumbnail generation request
        self.is_processing_video_thumbnails = True
        
        # Send request with file paths and thumbnail size
        request = {
            'type': 'generate_thumbnails',
            'file_paths': self.selected_video_files,
            'thumbnail_size': self.video_thumbnail_widget.thumbnail_size
        }
        self.video_thumbnail_queue.put(request)
    
    def _stop_background_thumbnail_generation(self):
        """Stop background thumbnail generation process"""
        if self.thumbnail_process is not None and self.thumbnail_process.is_alive():
            try:
                # Send shutdown signal
                self.thumbnail_queue.put({'type': 'shutdown'})
                
                self.thumbnail_process.join(timeout=1.0)
                
                if self.thumbnail_process.is_alive():
                    self.thumbnail_process.terminate()
                    self.thumbnail_process.join(timeout=1.0)
                    
                    if self.thumbnail_process.is_alive():
                        self.thumbnail_process.kill()
                        self.thumbnail_process.join()
                        
            except Exception as e:
                print(f"Error stopping thumbnail process: {e}")
            finally:
                self.thumbnail_process = None
        
        try:
            while not self.thumbnail_queue.empty():
                self.thumbnail_queue.get_nowait()
        except:
            pass
            
        try:
            while not self.thumbnail_reply.empty():
                self.thumbnail_reply.get_nowait()
        except:
            pass
        
        # Reset processing state
        self.is_processing_thumbnails = False
        self.thumbnail_process_started = False
    
    def _check_background_thumbnail_results(self):
        """Check for background thumbnail generation results"""
        if not self.is_processing_thumbnails or self.thumbnail_reply.empty():
            return
        
        try:
            while not self.thumbnail_reply.empty():
                result = self.thumbnail_reply.get_nowait()
                
                if result['type'] == 'thumbnail':
                    # Update thumbnail widget with processed data
                    file_path = result['file_path']
                    thumbnail_data = result['thumbnail_data']
                    self.thumbnail_widget.update_thumbnail_from_data(file_path, thumbnail_data)
                elif result['type'] == 'completed':
                    if self.thumbnail_queue.empty():
                        self.is_processing_thumbnails = False
                    
        except Exception as e:
            print(f"Error processing background thumbnail results: {e}")
    
    def _check_video_thumbnail_results(self):
        """Check for video thumbnail generation results"""
        if not self.is_processing_video_thumbnails or self.video_thumbnail_reply.empty():
            return
        
        try:
            while not self.video_thumbnail_reply.empty():
                result = self.video_thumbnail_reply.get_nowait()
                
                if result['type'] == 'thumbnail':
                    # Update video thumbnail widget with processed data
                    file_path = result['file_path']
                    thumbnail_data = result['thumbnail_data']
                    self.video_thumbnail_widget.update_thumbnail_from_data(file_path, thumbnail_data)
                elif result['type'] == 'completed':
                    self.is_processing_video_thumbnails = False
                    
        except Exception as e:
            print(f"Error processing video thumbnail results: {e}")
    
    def _stop_video_thumbnail_generation(self):
        """Stop background video thumbnail generation process"""
        if self.video_thumbnail_process is not None and self.video_thumbnail_process.is_alive():
            try:
                # Send shutdown signal
                self.video_thumbnail_queue.put({'type': 'shutdown'})
                
                # Wait for graceful shutdown
                self.video_thumbnail_process.join(timeout=1.0)
                
                # Force terminate if still alive
                if self.video_thumbnail_process.is_alive():
                    self.video_thumbnail_process.terminate()
                    self.video_thumbnail_process.join(timeout=1.0)
                    
                    # Kill if still alive
                    if self.video_thumbnail_process.is_alive():
                        self.video_thumbnail_process.kill()
                        self.video_thumbnail_process.join()
                        
            except Exception as e:
                print(f"Error stopping video thumbnail process: {e}")
            finally:
                self.video_thumbnail_process = None
        
        # Clear queues
        try:
            while not self.video_thumbnail_queue.empty():
                self.video_thumbnail_queue.get_nowait()
        except:
            pass
            
        try:
            while not self.video_thumbnail_reply.empty():
                self.video_thumbnail_reply.get_nowait()
        except:
            pass
        
        # Reset processing state
        self.is_processing_video_thumbnails = False
    # ------------------------------
    
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
        return (Path(self.save_path) / folder_name_with_resolution).as_posix()
    
    def process_dataset(self):
        """Start the dataset processing in a separate process"""
        # Update output path
        self.settings.output_path = self._construct_output_path()
        
        self.is_processing_dataset = True

        self.processing_queue.put(self.settings)
        
        self.processing_process = mp.Process(
            target=DatasetPreprocessingUtils.create_training_dataset,
            args=(self.processing_queue, self.processing_reply))
        self.processing_process.start()
    # ------------------------------

    # --- Cleanup ---
    def cleanup(self):
        """Clean up multiprocessing resources before destroying the object"""
        try:
            self._stop_background_thumbnail_generation()
            self._stop_video_thumbnail_generation()
            
            self.reset_progress_variables()

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
            print(f"Warning: Error during cleanup: {e}")

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