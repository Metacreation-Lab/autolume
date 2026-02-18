from pathlib import Path
import zipfile
import io

import imgui
import multiprocessing as mp
import PIL.Image

import dnnlib
from utils.gui_utils import imgui_utils
from train import main as train_main
from widgets.native_browser_widget import NativeBrowserWidget
from utils.dataset_preprocessing_utils import DatasetPreprocessingUtils
from widgets.help_icon_widget import HelpIconWidget

import cv2
from utils.gui_utils import gl_utils
import pandas as pd

augs = ["ADA", "DiffAUG"]
ada_pipes = ['blit', 'geom', 'color', 'filter', 'noise', 'cutout', 'bg', 'bgc', 'bgcf', 'bgcfn', 'bgcfnc']
diffaug_pipes = ['color,translation,cutout', 'color,translation', 'color,cutout', 'color',
                 'translation', 'cutout,translation', 'cutout']
configs = ['auto', 'stylegan2', 'paper256', 'paper512', 'paper1024', 'cifar']
resize_mode = ['stretch','center crop']

class TrainingModule:
    def __init__(self, menu):
        cwd = Path.cwd()
        self.save_path = (cwd / "training-runs")
        self.data_path = (cwd / "data")
        # create data folder if not exists
        data_dir = (cwd / "data").resolve()
        if not data_dir.exists():
            data_dir.mkdir(parents=True, exist_ok=True)
        self.app = menu.app
        self.config = 1
        self.resume_pkl = ""
        self.browse_cache = []
        self.aug = 0
        self.ada_pipe = 7
        self.diffaug_pipe = 0
        self.batch_size = 8
        
        # Native browsers for main training paths
        self.data_path_browser = NativeBrowserWidget()
        self.save_path_browser = NativeBrowserWidget()

        models_dir = Path.cwd() / "models"
        for pkl in models_dir.iterdir() if models_dir.exists() else []:
            if pkl.suffix == ".pkl":
                pkl_path = str(pkl)
                print(pkl.name, pkl_path)
                self.browse_cache.append(pkl_path)

        self.menu = menu
        
        self.help_icon = HelpIconWidget()
        self.help_texts, self.help_urls = self.help_icon.load_help_texts("training")

        self.queue = mp.Queue()
        self.reply = mp.Queue()
        self.message = ""
        self.done = False
        self.training_process = mp.Process(target=train_main, args=(self.queue, self.reply), name='TrainingProcess')
        self.found_video = False
        self._zipfile = None
        self.gamma = 10
        self.glr = 0.002
        self.dlr = 0.002
        self.snap = 4
        self.mirror = False # Mirror only accesible in preprocessing module
        self.done_button = False
        self.image_path = ''

        # Preprocessing settings toggle
        self.preprocessing_settings  = False
        self.preprocessing_settings_obj = DatasetPreprocessingUtils()
        self.resize_mode = self.preprocessing_settings_obj.resizeMode
        self.start_res = self.preprocessing_settings_obj.size
        self.res_factor = 0
        self.img_size = self.start_res * (2 ** self.res_factor)  # current image resolution (square)
        self.fps = self.preprocessing_settings_obj.fps
                
        self.preprocessing_data_browser = NativeBrowserWidget()
        self.preprocessing_save_browser = NativeBrowserWidget()
        self.preprocessing_save_path = self.preprocessing_settings_obj.output_path  
        self.preprocessing_folder_name = self.preprocessing_settings_obj.folder_name  
        self.preprocessing_data_path = Path.home() / "Desktop"
        self.data_path_has_videos = False  
        self.video_files_list = [] 
        
        # Dataset creation processing
        self.dataset_queue = mp.Queue()
        self.dataset_reply = mp.Queue()
        self.dataset_process = None
        self.is_creating_dataset = False
        self.dataset_message = ""
        self.dataset_done = False
        self.dataset_progress_current = 0
        self.dataset_progress_total = 0
        self.dataset_progress_percentage = 0
        self.dataset_progress_file = ""
        self.dataset_image_count = 0
        self.dataset_processed_count = 0
        # Video extraction progress tracking
        self.video_extraction_in_progress = False
        self.video_extraction_current = 0
        self.video_extraction_total = 0
        self.video_extraction_file = ""
        
        # Temp variables to store image files and extracted frame directories for dataset creation workflow
        self.temp_image_files = []
        self.extracted_frame_directories = []
        
        # Popup control flags
        self._open_dataset_popup = False
        self._open_training_popup = False
        self.folder_exists_warning = False 
        
    # preprocessing window
    def launch_preprocessing_window(self):
        """Launch the DataPreprocessingWindow by switching app state"""
        self.menu.app.start_preprocessing()

    @staticmethod
    def _file_ext(fname):
        return Path(fname).suffix.lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self.data_path)
        return self._zipfile

    @imgui_utils.scoped_by_object_id
    def __call__(self):

        if self.reply.qsize() > 0:
            self.message, self.done = self.reply.get()
            while self.reply.qsize() > 0:
                self.message, self.done = self.reply.get()
            print(self.message, self.done)

        # Create fixed regions to seperate training and preprocessing regions.
        # Max 94% height, or scroll bar would show up
        available_height = imgui.get_window_height()
        preprocessing_height = available_height * 0.47  
        training_height = available_height * 0.47    
        
        # Data Preprocessing Section (Top Region)
        if imgui.begin_child("DataPrepRegion", 0, preprocessing_height, True):

            text = "Prepare your data for training"
            text_width = imgui.calc_text_size(text).x
            window_width = imgui.get_window_width()
            help_icon_size = imgui.get_font_size()
            style = imgui.get_style()

            imgui.text(text)
            
            spacing = window_width - (style.window_padding[0] * 2) - text_width - help_icon_size - style.item_spacing[0] - 10
            
            imgui.same_line()
            imgui.dummy(spacing, 0)
            data_preprocessing_hyperlinks = []
            data_preprocessing_url = self.help_urls.get("data_preprocessing")
            if data_preprocessing_url:
                data_preprocessing_hyperlinks.append((data_preprocessing_url, "Read More"))
            tutorial_video_url = "https://www.youtube.com/watch?v=7Pc5-ULeXkM&feature=youtu.be"
            data_preprocessing_hyperlinks.append((tutorial_video_url, "Tutorial Video"))
            
            if data_preprocessing_hyperlinks:
                self.help_icon.render_with_urls(self.help_texts.get("data_preprocessing"), data_preprocessing_hyperlinks)
            else:
                self.help_icon.render(self.help_texts.get("data_preprocessing"))

            imgui.separator()

            if imgui.button("Open Data Preparation Module", width=-1):
                self.launch_preprocessing_window()

            clicked, preprocessing_settings = imgui.checkbox("Quick Data Preparation", self.preprocessing_settings)
            if clicked:
                self.preprocessing_settings = preprocessing_settings

            if self.preprocessing_settings: 
                imgui.text("Data Path")
                current_y = imgui.get_cursor_pos_y()
                imgui.set_cursor_pos_y(current_y - 3)  
                _, new_data_path = imgui_utils.input_text("##preprocessing_data_path", str(self.preprocessing_data_path), 1024, 0, 
                width=imgui.get_window_width() - self.menu.app.button_w - imgui.calc_text_size("Browse")[0])
                if new_data_path != self.preprocessing_data_path:
                    self.preprocessing_data_path = new_data_path
                
                imgui.same_line()
                if imgui.button("Browse##preprocessing_data", width=self.menu.app.button_w):
                    directory_path, has_videos, video_files = self.preprocessing_data_browser.select_directory_with_video_check("Select Data Directory")
                    if directory_path:
                        self.preprocessing_data_path = directory_path
                        self.data_path_has_videos = has_videos  
                        self.video_files_list = video_files  
                        print(f"Data path selected: {self.preprocessing_data_path}")
                        if has_videos:
                            print(f"Found {len(video_files)} video files in directory")
                            for video in video_files:
                                print(f"  - {video}")
                        else:
                            print("No video files found in directory")
                    else:
                        print("No data path selected")

                imgui.spacing()
                
                imgui.text("Resize Mode")
                imgui.same_line()   
                _, self.resize_mode = imgui.combo("##Resize Mode", self.resize_mode, resize_mode)

                imgui.text("Resolution")
                imgui.same_line()
                input_width = int(self.menu.app.font_size * 6)  
                button_width = self.menu.app.font_size * 1.2
                
                with imgui_utils.item_width(input_width):
                    imgui.input_text("##res_w", str(self.img_size), 512, flags=imgui.INPUT_TEXT_READ_ONLY)
                
                imgui.same_line()
                imgui.text("x")
                
                imgui.same_line()
                with imgui_utils.item_width(input_width):
                    imgui.input_text("##res_h", str(self.img_size), 512, flags=imgui.INPUT_TEXT_READ_ONLY)
                
                imgui.same_line()
                if imgui.button("-##img_res", width=button_width):
                    self.res_factor = max(self.res_factor - 1, 0)   
                    self.img_size = self.start_res * (2 ** self.res_factor)
                
                imgui.same_line()
                if imgui.button("+##img_res", width=button_width):
                    self.res_factor = self.res_factor + 1
                    self.img_size = self.start_res * (2 ** self.res_factor)
                
                # FPS input for video files (only show if videos were found)
                if hasattr(self, 'data_path_has_videos') and self.data_path_has_videos:
                    imgui.text("FPS for Video Extraction")
                    imgui.same_line()
                    with imgui_utils.item_width(imgui.get_window_width() - self.menu.app.button_w - imgui.calc_text_size("FPS for Video Extraction")[0]):
                        _, self.fps = imgui.input_int("##fps", self.fps)
                    if self.fps < 1:
                        self.fps = 1

                imgui.text("Folder Name")
                current_y = imgui.get_cursor_pos_y()
                imgui.set_cursor_pos_y(current_y - 3)  
                total_width = imgui.get_window_width() - 50
                folder_name_width = total_width * 0.75  
                
                with imgui_utils.item_width(folder_name_width):
                    changed, new_folder_name = imgui.input_text("##preprocessing_folder_name", self.preprocessing_folder_name, 1024)
                    if changed:
                        self.preprocessing_folder_name = new_folder_name
                imgui.same_line()
                imgui.text(f"_{self.img_size}x{self.img_size}")

                imgui.text("Save Path")
                current_y = imgui.get_cursor_pos_y()
                imgui.set_cursor_pos_y(current_y - 3)  
                _, new_save_path = imgui_utils.input_text("##preprocessing_save", str(self.preprocessing_save_path), 1024, 0, 
                width=imgui.get_window_width() - self.menu.app.button_w - imgui.calc_text_size("Browse")[0])
                if new_save_path != self.preprocessing_save_path:
                    self.preprocessing_save_path = new_save_path
                
                imgui.same_line()
                if imgui.button("Browse##preprocessing_save", width=self.menu.app.button_w):
                    directory_path = self.preprocessing_save_browser.select_directory("Select Save Path")
                    if directory_path:
                        self.preprocessing_save_path = directory_path
                    else:
                        print("No save path selected")

                imgui.spacing()

                if imgui.button("Process & Save Data", width=-3):
                    if not self.preprocessing_save_path or not self.preprocessing_folder_name:
                        print("Please specify both parent directory and folder name")
                    else:
                        self.create_preprocessing_dataset()
        imgui.end_child()
        
        # Training Section (Bottom Region)
        if imgui.begin_child("TrainingRegion", 0, training_height, True):
            text = "Train a model with your dataset"
            text_width = imgui.calc_text_size(text).x
            window_width = imgui.get_window_width()
            help_icon_size = imgui.get_font_size()
            style = imgui.get_style()

            imgui.text(text)
            
            spacing = window_width - (style.window_padding[0] * 2) - text_width - help_icon_size - style.item_spacing[0] - 10
            
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
            _, self.save_path = imgui_utils.input_text("##Save Path", str(self.save_path), 1024, 0, 
            width=imgui.get_window_width() - imgui.calc_text_size("Browse##main_save")[0])
            
            imgui.same_line()
            if imgui.button("Browse##main_save", width=self.menu.app.button_w):
                directory_path = self.save_path_browser.select_directory("Select Training Results Save Path")
                if directory_path:
                    self.save_path = directory_path
                else:
                    self.save_path = self.save_path

            imgui.text("Dataset Path")
            current_y = imgui.get_cursor_pos_y()
            imgui.set_cursor_pos_y(current_y - 3)  
            _, self.data_path = imgui_utils.input_text("##Dataset Path", str(self.data_path), 1024, 0, 
            width=imgui.get_window_width() - imgui.calc_text_size("Browse##main_data")[0])
            
            imgui.same_line()
            if imgui.button("Browse##main_data", width=self.menu.app.button_w):
                directory_path = self.data_path_browser.select_directory("Select Training Dataset Directory")
                if directory_path:
                    self.data_path = directory_path

                    # Check for PKL files in the directory
                    pkl_files = []
                    data_path = Path(self.data_path)
                    if data_path.is_dir():
                        for pkl_path in data_path.rglob("*.pkl"):
                            if pkl_path.is_file():
                                pkl_path_str = str(pkl_path)
                                pkl_files.append(pkl_path_str)
                                if pkl_path not in self.browse_cache:
                                    self.browse_cache.append(pkl_path)
                    
                    if pkl_files:
                        print(f"Found {len(pkl_files)} PKL files in directory:")
                        for pkl in pkl_files:
                            print(f"  - {pkl}")
                    else:
                        print("No PKL files found in directory")
                else:
                    print("No data path selected")
                
            imgui.text("Resume Pkl")
            current_y = imgui.get_cursor_pos_y()
            imgui.set_cursor_pos_y(current_y - 3)  
            _, self.resume_pkl = imgui_utils.input_text("##Resume Pkl", self.resume_pkl, 1024, 0, 
            width=imgui.get_window_width() - imgui.calc_text_size("Browse##Resume Pkl")[0])
            
            imgui.same_line()
            if imgui_utils.button('Browse##Resume Pkl', enabled=len(self.browse_cache) > 0, width=self.menu.app.button_w):
                imgui.open_popup('browse_pkls_popup_training')

            if imgui.begin_popup('browse_pkls_popup_training'):
                for pkl in self.browse_cache:
                    clicked, _state = imgui.menu_item(pkl)
                    if clicked:
                        self.resume_pkl = pkl
                imgui.end_popup()

            imgui.text("Training Augmentation")
            imgui.same_line()
            _, self.aug = imgui.combo("##Training Augmentation", self.aug, augs)
            if self.aug == 0:
                imgui.text("Augmentation Pipeline")
                imgui.same_line()
                _, self.ada_pipe = imgui.combo("##Augmentation Pipeline", self.ada_pipe, ada_pipes)
            else:
                imgui.text("Augmentation Pipeline")
                imgui.same_line()
                _, self.diffaug_pipe = imgui.combo("##Augmentation Pipeline", self.diffaug_pipe, diffaug_pipes)

            imgui.text("Batch Size")
            imgui.same_line()
            _, self.batch_size = imgui.input_int("##Batch Size", self.batch_size)
            if self.batch_size < 1:
                self.batch_size = 1
            
            imgui.text("Configuration")
            imgui.same_line()
            _, self.config = imgui.combo("##Configuration", self.config, configs)
            

            imgui.set_next_window_size( self.menu.app.content_width // 4, (self.menu.app.content_height // 4), imgui.ONCE)

            if imgui.button("Advanced...", width=-1):
                imgui.open_popup("Advanced...")

            if imgui.begin_popup_modal("Advanced...")[0]:
                imgui.text("Advanced Training Options")
                imgui.text("Generator Learning Rate")
                _, self.glr = imgui.input_float("##Generator Learning Rate", self.glr)

                imgui.text("Discriminator Learning Rate")
                _, self.dlr = imgui.input_float("##Discriminator Learning Rate", self.dlr)

                imgui.text("Gamma")
                _, self.gamma = imgui.input_int("##Gamma", self.gamma)

                imgui.text("Number of ticks between snapshots")
                _, self.snap = imgui.input_int("##Number of ticks between snapshots", self.snap)

                if imgui_utils.button("Close", enabled=1):
                    imgui.close_current_popup()


                imgui.end_popup()


            if imgui.button("Train", width=-1):
                detected_resolution = None
                target_dataset_path = Path(self.data_path)
                if target_dataset_path.is_dir():
                    image_files = [f for f in target_dataset_path.iterdir()
                                if f.is_file()]
                    if image_files:
                        first_image_path = str(image_files[0])
                        img = PIL.Image.open(first_image_path)
                        width, height = img.size
                        detected_resolution = (width, height)

                self._open_training_popup = True
                print("training")

                kwargs = dnnlib.EasyDict(
                        outdir=str(self.save_path),
                        data=str(target_dataset_path),
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
                        mbstd_group=2,
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
                if self.done == True:
                    self.queue = mp.Queue()
                    self.reply = mp.Queue()
                    self.training_process = mp.Process(target=train_main, args=(self.queue, self.reply), name='TrainingProcess')
                    self.done = False
                self.queue.put(kwargs)
                self.training_process.start()
        imgui.end_child()

        #------------------------------------------------------------------------------------------------
        
        # Open popups if flags are set (need to open at root window level, not inside child windows)
        if hasattr(self, '_open_dataset_popup') and self._open_dataset_popup:
            imgui.open_popup("Processing Data")
            self._open_dataset_popup = False
        
        if hasattr(self, '_open_training_popup') and self._open_training_popup:
            imgui.open_popup("Training")
            self._open_training_popup = False
        
        # Quick Dataset Creation Popup Modal
        imgui.set_next_window_size(self.menu.app.content_width // 2.5, self.menu.app.content_height // 2.5, imgui.ONCE)
        if imgui.begin_popup_modal("Processing Data")[0]:
            
            if self.folder_exists_warning and not self.is_creating_dataset and not self.dataset_done:
                imgui.text_colored("Warning: Folder Already Exists!", 1.0, 0.5, 0.0, 1.0)
                imgui.separator()
                
                imgui.push_text_wrap_pos(self.menu.app.content_width // 2.5 - 40)
                imgui.text("The folder already exists at:")
                imgui.spacing()
                imgui.text_colored(self.dataset_output_path, 1.0, 1.0, 0.0, 1.0)
                imgui.spacing()
                imgui.text("If you continue, existing files in this folder may be overwritten.")
                imgui.pop_text_wrap_pos()
                
                imgui.spacing()
                
                button_width = (self.menu.app.content_width // 2.5 - 60) / 2
                imgui.begin_group()
                if imgui.button("Overwrite & Process", width=button_width):
                    self.folder_exists_warning = False
                    self.is_creating_dataset = True
                    self.start_processing_workflow()
                
                imgui.same_line()
                if imgui.button("Cancel", width=button_width):
                    self.folder_exists_warning = False
                    imgui.close_current_popup()
                imgui.end_group()
            
            # Show processing progress
            elif self.is_creating_dataset:
                imgui.text("Preparing data for training...")
                imgui.separator()
                
                imgui.text_colored("Settings:", 1.0, 1.0, 0.0, 1.0)
                
                if hasattr(self, 'dataset_image_count'):
                    imgui.text(f"Images: {self.dataset_image_count} files")

                if hasattr(self, 'video_extraction_total') and self.video_extraction_total > 0:
                    imgui.text(f"Videos: {self.video_extraction_total} videos")

                imgui.text(f"Resolution: {self.img_size}x{self.img_size}")
                imgui.text(f"Resize Mode: {resize_mode[self.resize_mode]}")
                
                if self.data_path_has_videos:
                    imgui.text(f"Frame Extraction FPS: {self.fps}")
                
                imgui.text(f"Output: {self.dataset_output_path}")
                
                imgui.spacing()
                imgui.separator()
                imgui.spacing()

                latest_progress = None
                completion_data = None
                while not self.dataset_reply.empty():
                    try:
                        data = self.dataset_reply.get_nowait()
                        if data.get('type') == 'completed':
                            completion_data = data
                        elif data.get('type') == 'progress':
                            latest_progress = data
                    except:
                        pass
                
                if completion_data:
                    if self.video_extraction_in_progress:
                        self.extracted_frame_directories = completion_data.get('results', [])
                        if hasattr(self, 'dataset_process') and self.dataset_process and self.dataset_process.is_alive():
                            self.dataset_process.terminate()
                            self.dataset_process.join(timeout=2)
                        self.start_image_processing()
                    else:
                        self.dataset_done = True
                        self.is_creating_dataset = False
                        self.dataset_processed_count = completion_data.get('processed_count', 0)
                        print(f"Dataset created! Processed {self.dataset_processed_count} images")
                        if hasattr(self, 'dataset_output_path'):
                            self.data_path = self.dataset_output_path
                        if hasattr(self, 'dataset_process') and self.dataset_process and self.dataset_process.is_alive():
                            self.dataset_process.terminate()
                            self.dataset_process.join(timeout=2)
                
                if latest_progress:
                    if self.video_extraction_in_progress:
                        self.video_extraction_current = latest_progress.get('current', 0)
                        self.video_extraction_file = latest_progress.get('current_file', '')
                    else:
                        self.dataset_progress_current = latest_progress.get('current', 0)
                        self.dataset_progress_total = latest_progress.get('total', 0)
                        self.dataset_progress_percentage = latest_progress.get('percentage', 0)
                        self.dataset_progress_file = latest_progress.get('current_file', '')
                
                progress_width = self.menu.app.content_width // 2.5 - 40
                
                if self.video_extraction_in_progress:
                    imgui.text_colored("Extracting Video Frames:", 1.0, 0.8, 0.0, 1.0)
                    imgui.text(f"Processing: {self.video_extraction_current + 1}/{self.video_extraction_total} videos")
                    if self.video_extraction_file:
                        imgui.text(f"Current video: {self.video_extraction_file}")
                    
                    # Video extraction progress bar
                    video_progress = (self.video_extraction_current / self.video_extraction_total) if self.video_extraction_total > 0 else 0
                    imgui.progress_bar(video_progress, (progress_width, 20))
                    video_percentage = video_progress * 100.0
                    text_width = imgui.calc_text_size(f"{video_percentage:.1f}%")[0]
                    imgui.set_cursor_pos_x((progress_width - text_width) / 2)
                    imgui.text(f"{video_percentage:.1f}%")
                    
                    imgui.spacing()
                    imgui.separator()
                    imgui.spacing()
                
                if not self.video_extraction_in_progress and self.dataset_progress_total > 0:
                    imgui.text_colored("Processing Images:", 1.0, 0.8, 0.0, 1.0)
                    imgui.text(f"Processing: {self.dataset_progress_current}/{self.dataset_progress_total} images")
                    if self.dataset_progress_file:
                        imgui.text(f"Current file: {self.dataset_progress_file}")
                    
                    # Progress bar
                    imgui.progress_bar(self.dataset_progress_percentage / 100.0, (progress_width, 20))
                    text_width = imgui.calc_text_size(f"{self.dataset_progress_percentage:.1f}%")[0]
                    imgui.set_cursor_pos_x((progress_width - text_width) / 2)
                    imgui.text(f"{self.dataset_progress_percentage:.1f}%")
                    
                    imgui.spacing()
                
                if imgui_utils.button("Cancel", width=progress_width):
                    self.dataset_queue.put('cancel')
                    print("Dataset creation cancelled by user")
                    self.cleanup_dataset_process()
                    self.folder_exists_warning = False
                    self.video_extraction_in_progress = False
                    
                    # Clear temporary storage
                    self.temp_image_files = []
                    self.extracted_frame_directories = []
                    
                    imgui.close_current_popup()
            
            # Completion message
            elif self.dataset_done:
                imgui.text_colored("Settings:", 1.0, 1.0, 0.0, 1.0)
                
                if hasattr(self, 'dataset_image_count'):
                    imgui.text(f"Images: {self.dataset_image_count} files")

                if hasattr(self, 'video_extraction_total') and self.video_extraction_total > 0:
                    imgui.text(f"Videos: {self.video_extraction_total} videos")

                imgui.text(f"Resolution: {self.img_size}x{self.img_size}")
                imgui.text(f"Resize Mode: {resize_mode[self.resize_mode]}")
                
                if self.data_path_has_videos:
                    imgui.text(f"Frame Extraction FPS: {self.fps}")
                
                imgui.text(f"Output: {self.dataset_output_path}")
                
                imgui.spacing()
                imgui.separator()
                imgui.spacing()
                
                imgui.text_colored("Processing completed successfully!", 0.0, 1.0, 0.0, 1.0)
                imgui.spacing()
                if hasattr(self, 'dataset_processed_count'):
                    imgui.text(f"{self.dataset_processed_count} images processed and saved")
                imgui.spacing()

                button_width = self.menu.app.content_width // 2.5 - 40
                if imgui_utils.button("Close", width=button_width):
                    self.dataset_done = False
                    self.folder_exists_warning = False
                    self.dataset_progress_current = 0
                    self.dataset_progress_total = 0
                    self.dataset_progress_percentage = 0
                    self.dataset_progress_file = ""
                    self.video_extraction_in_progress = False
                    self.video_extraction_current = 0
                    self.video_extraction_total = 0
                    self.video_extraction_file = ""
                    
                    # Clear temporary storage
                    self.temp_image_files = []
                    self.extracted_frame_directories = []
                    
                    imgui.close_current_popup()
            
            imgui.end_popup()

            # End of Quick Dataset Creation Popup Modal

        # Training Popup Modal
        training_popup_width = self.menu.app.content_width // 1.5
        training_popup_height = self.menu.app.content_height // 1.5
        imgui.set_next_window_size(training_popup_width, training_popup_height, imgui.ONCE)

        if imgui.begin_popup_modal("Training")[0]:
            imgui.text("Training...")
            if Path(self.message).exists() and self.image_path != self.message:
                self.image_path = self.message
                self.grid = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
                self.grid = cv2.cvtColor(self.grid, cv2.COLOR_BGRA2RGBA)
                self.grid_texture = gl_utils.Texture(image=self.grid, width=self.grid.shape[1],
                                               height=self.grid.shape[0], channels=self.grid.shape[2])
            elif self.message != "":
                if self.done:
                    imgui.text_colored("Error:", 1.0, 0.3, 0.3, 1.0)
                    imgui.text_wrapped(self.message)
                else:
                    imgui.text(self.message)
            if self.image_path != '':
                imgui.text("Current sample of fake imagery")
                fake_display_height = training_popup_height - 200
                fake_display_width = int((self.grid.shape[1] / self.grid.shape[0]) * fake_display_height)
                imgui.image(self.grid_texture.gl_id, fake_display_width, fake_display_height)
            if imgui_utils.button("Stop Training", enabled=1):
                self.queue.put('done')
                self.done_button = True
            if self.done:
                self.training_process.terminate()
                self.training_process.join()
                if self.done_button == True:
                    imgui.close_current_popup()
                    self.message = ''
                    self.done_button = False
                    self.image_path = ''
            imgui.end_popup()
            # End of Training Popup Modal

    # collect image files from directory to later be appened with extracted frames from videos
    def collect_image_files(self, directory_path):
        """Collect all image files from selected directory path (quick dataset creation)"""
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}
        image_files = []
        
        dir_path = Path(directory_path)
        if dir_path.is_dir():
            for path in dir_path.rglob("*"):
                if path.is_file() and path.suffix.lower() in image_extensions:
                    image_files.append(str(path))
        
        return image_files
    
    def create_preprocessing_dataset(self):
        """Start the preprocessing dataset creation process"""
        # Validate paths
        if not Path(self.preprocessing_data_path).exists():
            print(f"Error: Data path does not exist: {self.preprocessing_data_path}")
            self.dataset_message = f"Data path does not exist:\n{self.preprocessing_data_path}"
            return
        
        # Collect image files from directory
        image_files = self.collect_image_files(self.preprocessing_data_path)
        
        if not image_files and not self.video_files_list:
            print("Error: No images or videos found in directory")
            self.dataset_message = "No images or videos found in the selected directory"
            return
        
        self.dataset_image_count = len(image_files) + len(self.video_files_list)
        
        # Add resolution suffix to folder name and construct full path
        resolution_suffix = f"_{self.img_size}x{self.img_size}"
        folder_name_with_resolution = self.preprocessing_folder_name + resolution_suffix
        self.dataset_output_path = (Path(self.preprocessing_save_path) / folder_name_with_resolution)
        
        self.temp_image_files = list(image_files)
        self.extracted_frame_directories = []
        
        if Path(self.dataset_output_path).exists():
            self.folder_exists_warning = True
        else:
            self.folder_exists_warning = False
            self.is_creating_dataset = True
            self.start_processing_workflow()
        
        self._open_dataset_popup = True
        self.dataset_done = False
    
    def start_processing_workflow(self):
        """Start the video extraction and image processing workflow"""
        # Clear queues before starting
        while not self.dataset_queue.empty():
            try:
                self.dataset_queue.get_nowait()
            except:
                break
        while not self.dataset_reply.empty():
            try:
                self.dataset_reply.get_nowait()
            except:
                break
        
        if self.video_files_list:
            print(f"Extracting frames from {len(self.video_files_list)} video(s)...")
            self.video_extraction_in_progress = True
            self.video_extraction_current = 0
            self.video_extraction_total = len(self.video_files_list)
            self.video_extraction_file = ""
            self.dataset_message = "Extracting video frames..."
            
            video_process = mp.Process(
                target=DatasetPreprocessingUtils.extract_videos,
                args=(self.video_files_list, self.fps, self.dataset_queue, self.dataset_reply)
            )
            video_process.start()
            self.dataset_process = video_process
        else:
            self.start_image_processing()

    def start_image_processing(self):
        """Start the image processing after video extraction (if any) is complete"""
        # Collect all images including extracted frames
        all_images = list(self.temp_image_files)
        
        # Add extracted frames
        for frame_dir in self.extracted_frame_directories:
            frame_path = Path(frame_dir)
            if frame_path.exists():
                frames = [str(f) for f in frame_path.iterdir() 
                         if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
                all_images.extend(frames)
                print(f"Added {len(frames)} frames from {frame_path.name}")
        
        # Prepare settings with all collected images
        settings = DatasetPreprocessingUtils()
        settings.images = all_images
        settings.size = self.img_size
        settings.resizeMode = self.resize_mode
        settings.fps = self.fps
        settings.output_path = self.dataset_output_path
        
        print(f"Processing {len(all_images)} images...")
        self.dataset_message = f"Processing {len(all_images)} images..."
        
        # Clear queues before starting
        while not self.dataset_queue.empty():
            try:
                self.dataset_queue.get_nowait()
            except:
                break
        while not self.dataset_reply.empty():
            try:
                self.dataset_reply.get_nowait()
            except:
                break
        
        self.dataset_queue.put(settings)
        self.dataset_process = mp.Process(
            target=DatasetPreprocessingUtils.create_training_dataset,
            args=(self.dataset_queue, self.dataset_reply)
        )
        self.dataset_process.start()
        self.video_extraction_in_progress = False
    
    def cleanup_dataset_process(self):
        """Clean up dataset creation process"""
        if self.dataset_process and self.dataset_process.is_alive():
            try:
                print("Terminating dataset creation process...")
                self.dataset_process.terminate()
                self.dataset_process.join(timeout=2)
                
                if self.dataset_process.is_alive():
                    print("Force killing dataset creation process...")
                    self.dataset_process.kill()
                    self.dataset_process.join()
            except Exception as e:
                print(f"Error cleaning up dataset process: {e}")
            finally:
                self.dataset_process = None
        
        while not self.dataset_queue.empty():
            try:
                self.dataset_queue.get_nowait()
            except:
                break
        while not self.dataset_reply.empty():
            try:
                self.dataset_reply.get_nowait()
            except:
                break
        
        self.is_creating_dataset = False
        self.dataset_done = False
        self.folder_exists_warning = False
        self.dataset_progress_current = 0
        self.dataset_progress_total = 0
        self.dataset_progress_percentage = 0
        self.dataset_progress_file = ""
        self.video_extraction_in_progress = False
        self.video_extraction_current = 0
        self.video_extraction_total = 0
        self.video_extraction_file = ""
        
        # Clear temporary storage
        self.temp_image_files = []
        self.extracted_frame_directories = []

