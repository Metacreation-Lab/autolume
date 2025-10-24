import os
import zipfile

import imgui
import multiprocessing as mp

import dnnlib
from utils.gui_utils import imgui_utils
from train import main as train_main
from utils import dataset_tool
from widgets.native_browser_widget import NativeBrowserWidget
from utils.dataset_preprocessing_utils import DatasetPreprocessingUtils

import cv2
from utils.gui_utils import gl_utils
import pandas as pd

augs = ["ADA", "DiffAUG"]
ada_pipes = ['blit', 'geom', 'color', 'filter', 'noise', 'cutout', 'bg', 'bgc', 'bgcf', 'bgcfn', 'bgcfnc']
diffaug_pipes = ['color,translation,cutout', 'color,translation', 'color,cutout', 'color',
                 'translation', 'cutout,translation', 'cutout']
configs = ['auto', 'stylegan2', 'paper256', 'paper512', 'paper1024', 'cifar']
resize_mode = ['stretch','center crop']

# Add constants for help texts
DEFAULT_HELP_TEXTS = {
    "save_path_training": "Path to save training results\nModel checkpoints and generated images will be saved here during training",
    "data_path_training": "Path to the training dataset\nSupported formats:\n- Image folder\n- ZIP archive\n- Video file (.mp4/.avi)",
    "resume_pkl_training": "Model checkpoint file (.pkl) to resume training\nLeave empty to start from scratch",
    "augmentation_training": "Data augmentation method:\nADA - Adaptive Discriminator Augmentation\nDiffAUG - Differential Augmentation",
    "aug_pipeline_training": "Specific configuration of the data augmentation pipeline\nDifferent augmentation methods have different options",
    "resize_mode_training": "Method to resize images:\nstretch - Stretch to target size\ncenter crop - Center crop",
    "batch_size_training": "Number of images per training batch\nLarger batches require more VRAM",
    "config_training": "Preset training configurations:\nauto - Automatic configuration\nstylegan2 - Standard StyleGAN2\npaper256/512/1024 - Paper configurations\ncifar - CIFAR dataset configuration",
    "advanced_training": "Advanced training options:\nGenerator LR - Generator learning rate\nDiscriminator LR - Discriminator learning rate\nGamma - Training stability parameter\nSnapshot - Checkpoint saving interval\nMirror - Horizontal flip of dataset",
    "generator_lr_training": "Learning rate for the generator network",
    "discriminator_lr_training": "Learning rate for the discriminator network",
    "gamma_training": "Training stability parameter",
    "snapshot_training": "Checkpoints saving interval",
    "data_preprocessing": "Preprocessing module", 
    "quick_data_path": "Quick settings data path\nPath to the training dataset for quick settings\nSupported formats:\n- Image folder\n- ZIP archive\n- Video file (.mp4/.avi)",
    "quick_save_path": "Quick settings save path\nPath to save training results for quick settings\nGenerated images will be saved here during training",
}
# 尝试从Excel加载帮助文本
HELP_TEXTS = DEFAULT_HELP_TEXTS.copy()
try:
    excel_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets","help_contents.xlsx")
    if os.path.exists(excel_path):
        df = pd.read_excel(excel_path, engine='openpyxl')
        for _, row in df.iterrows():
            if pd.notna(row['key']) and pd.notna(row['text']):
                HELP_TEXTS[str(row['key'])] = str(row['text'])
        print(f"Successfully loaded help texts from: {excel_path}")
except Exception as e:
    print(f"Warning: Using default help texts. Error: {e}")

class TrainingModule:
    def __init__(self, menu):
        cwd = os.getcwd()
        self.save_path = os.path.join(cwd, "training-runs").replace('\\', '/')
        self.data_path = os.path.join(cwd, "data").replace('\\', '/')
        # self.show_help = False 
        # create data folder if not exists
        if not os.path.exists(os.path.abspath(os.path.join(os.getcwd(),"data")).replace('\\', '/')):
            os.makedirs(os.path.abspath(os.path.join(os.getcwd(),"data")).replace('\\', '/'))
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

        for pkl in os.listdir("./models"):
            if pkl.endswith(".pkl"):
                print(pkl, os.path.join(os.getcwd(),"models",pkl))
                self.browse_cache.append(os.path.join(os.getcwd(),"models",pkl))

        self.menu = menu

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

        # Quick preprocessingsettings toggle
        self.quick_settings  = False
        # Quick dataset preprocessing attributes
        self.quick_preprocessing_settings = DatasetPreprocessingUtils()
        self.resize_mode = self.quick_preprocessing_settings.resizeMode
        self.start_res = self.quick_preprocessing_settings.size
        self.res_factor = 0
        self.img_size = self.start_res * (2 ** self.res_factor)  # current image resolution (square)
        self.fps = self.quick_preprocessing_settings.fps
                
        # Native browser widgets for quick settings
        self.quick_data_browser = NativeBrowserWidget()
        self.quick_save_browser = NativeBrowserWidget()
        self.quick_save_path = self.quick_preprocessing_settings.output_path  
        self.quick_folder_name = self.quick_preprocessing_settings.folder_name  
        self.quick_data_path = os.path.join(os.path.expanduser("~"), "Desktop").replace('\\', '/') 
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
        return os.path.splitext(fname)[1].lower()

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

        # imgui.begin_group()
        # imgui.text("Train a model on your own data")
        # imgui.same_line()
        # remaining_width = imgui.get_content_region_available_width()
        # imgui.dummy(remaining_width - 60, 0)  
        # imgui.same_line()
        # if imgui_utils.button("Help", width=50):
        #     self.show_help = not self.show_help
        # imgui.end_group()

        # Create fixed regions to seperate training and preprocessing regions.
        # Max 94% height, or scroll bar would show up
        available_height = imgui.get_window_height()
        preprocessing_height = available_height * 0.47  
        training_height = available_height * 0.47    
        
        # Data Preprocessing Section (Top Region)
        if imgui.begin_child("DataPrepRegion", 0, preprocessing_height, True):
            imgui.text("Prepare your data for training")

            imgui.separator()

            if imgui.button("Open Data Preprocessing Module", width=-1):
                self.launch_preprocessing_window()
            if self.menu.show_help and imgui.is_item_hovered(): 
                imgui.set_tooltip(HELP_TEXTS["data_preprocessing"])

            clicked, quick_settings = imgui.checkbox("Quick Settings", self.quick_settings)
            if clicked:
                self.quick_settings = quick_settings

            if self.quick_settings:  # Show image options when quick settings is checked
                # Quick settings data path
                imgui.text("Data Path")
                # Reduce spacing by setting cursor position closer to the label
                current_y = imgui.get_cursor_pos_y()
                imgui.set_cursor_pos_y(current_y - 3)  
                _, new_data_path = imgui_utils.input_text("##quick_data", self.quick_data_path, 1024, 0, 
                width=imgui.get_window_width() - imgui.calc_text_size("Browse##quick_data")[0])
                if new_data_path != self.quick_data_path:
                    # Normalize to forward slashes
                    self.quick_data_path = new_data_path.replace('\\', '/')
                if self.menu.show_help and imgui.is_item_hovered():
                    imgui.set_tooltip(HELP_TEXTS["quick_data_path"])
                
                imgui.same_line()
                if imgui.button("Browse##quick_data", width=self.menu.app.button_w):
                    # Use directory selection with video check for data path
                    directory_path, has_videos, video_files = self.quick_data_browser.select_directory_with_video_check("Select Data Directory")
                    if directory_path:
                        # Normalize to forward slashes
                        self.quick_data_path = directory_path.replace('\\', '/')
                        self.data_path_has_videos = has_videos  # Store video detection result
                        self.video_files_list = video_files  # Store video files list
                        print(f"Data path selected: {self.quick_data_path}")
                        if has_videos:
                            print(f"Found {len(video_files)} video files in directory")
                            for video in video_files:
                                print(f"  - {video}")
                        else:
                            print("No video files found in directory")
                    else:
                        print("No data path selected")
                if self.menu.show_help and imgui.is_item_hovered():
                    imgui.set_tooltip(HELP_TEXTS["quick_data_path"])

                imgui.spacing()
                
                # Resize Mode
                imgui.text("Resize Mode")
                imgui.same_line()
                # current_y = imgui.get_cursor_pos_y()
                # imgui.set_cursor_pos_y(current_y - 3)  
                _, self.resize_mode = imgui.combo("##Resize Mode", self.resize_mode, resize_mode)
                if self.menu.show_help and imgui.is_item_hovered():
                    imgui.set_tooltip(HELP_TEXTS["resize_mode_training"])


                imgui.text("Resolution")
                imgui.same_line()
                # current_y = imgui.get_cursor_pos_y()
                # imgui.set_cursor_pos_y(current_y - 3)  
                input_width = int(self.menu.app.font_size * 6)  # Make input fields wider
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
                    with imgui_utils.item_width(imgui.get_window_width() - imgui.calc_text_size("FPS for Video Extraction")[0] - 150):
                        _, self.fps = imgui.input_int("##fps", self.fps)
                    if self.fps < 1:
                        self.fps = 1
                    if self.menu.show_help and imgui.is_item_hovered():
                        imgui.set_tooltip("Frames per second to extract from video files")

                # Folder Name for processed dataset
                imgui.text("Folder Name")
                current_y = imgui.get_cursor_pos_y()
                imgui.set_cursor_pos_y(current_y - 3)  
                total_width = imgui.get_window_width() - 50
                folder_name_width = total_width * 0.75  
                suffix_width = total_width * 0.25 
                
                with imgui_utils.item_width(folder_name_width):
                    changed, new_folder_name = imgui.input_text("##quick_folder_name", self.quick_folder_name, 1024)
                    if changed:
                        self.quick_folder_name = new_folder_name
                imgui.same_line()
                imgui.text(f"_{self.img_size}x{self.img_size}")
                if self.menu.show_help and imgui.is_item_hovered():
                    imgui.set_tooltip("Name of the folder where the processed dataset will be saved")

                # Parent Directory Path (quick_save_path)
                imgui.text("Directory Path")
                current_y = imgui.get_cursor_pos_y()
                imgui.set_cursor_pos_y(current_y - 3)  
                _, new_save_path = imgui_utils.input_text("##quick_save", self.quick_save_path, 1024, 0, 
                width=imgui.get_window_width() - imgui.calc_text_size("Browse##quick_save")[0])
                if new_save_path != self.quick_save_path:
                    self.quick_save_path = new_save_path.replace('\\', '/')
                if self.menu.show_help and imgui.is_item_hovered():
                    imgui.set_tooltip("Parent directory where the dataset folder will be created")
                
                imgui.same_line()
                if imgui.button("Browse##quick_save", width=self.menu.app.button_w):
                    directory_path = self.quick_save_browser.select_directory("Select Parent Directory")
                    if directory_path:
                        self.quick_save_path = directory_path.replace('\\', '/')
                    else:
                        print("No save path selected")
                if self.menu.show_help and imgui.is_item_hovered():
                    imgui.set_tooltip("Browse to select parent directory where the dataset folder will be created")

                imgui.spacing()

                # Preprocess data button
                if imgui.button("Process & Save Data", width=-3):
                    if not self.quick_save_path or not self.quick_folder_name:
                        print("Please specify both parent directory and folder name")
                    else:
                        self.create_quick_dataset()
                
                if self.menu.show_help and imgui.is_item_hovered():
                    imgui.set_tooltip("Create a new dataset using the specified paths and settings")
        imgui.end_child()
        
        # Training Section (Bottom Region)
        if imgui.begin_child("TrainingRegion", 0, training_height, True):
            # Training options
            imgui.text("Train a model on your prepared dataset")

            imgui.separator()

            imgui.text("Save Path")
            current_y = imgui.get_cursor_pos_y()
            imgui.set_cursor_pos_y(current_y - 3)  
            _, self.save_path = imgui_utils.input_text("##Save Path", self.save_path, 1024, 0, 
            width=imgui.get_window_width() - imgui.calc_text_size("Browse##main_save")[0])
            if self.menu.show_help and imgui.is_item_hovered():
                imgui.set_tooltip(HELP_TEXTS["save_path_training"])
            
            imgui.same_line()
            if imgui.button("Browse##main_save", width=self.menu.app.button_w):
                directory_path = self.save_path_browser.select_directory("Select Training Results Save Path")
                if directory_path:
                    self.save_path = directory_path.replace('\\', '/')
                else:
                    print("No save path selected")
            if self.menu.show_help and imgui.is_item_hovered():
                imgui.set_tooltip(HELP_TEXTS["save_path_training"])

            imgui.text("Dataset Path")
            current_y = imgui.get_cursor_pos_y()
            imgui.set_cursor_pos_y(current_y - 3)  
            _, self.data_path = imgui_utils.input_text("##Data Path", self.data_path, 1024, 0, 
            width=imgui.get_window_width() - imgui.calc_text_size("Browse##main_data")[0])
            if self.menu.show_help and imgui.is_item_hovered():
                imgui.set_tooltip(HELP_TEXTS["data_path_training"])
            
            imgui.same_line()
            if imgui.button("Browse##main_data", width=self.menu.app.button_w):
                directory_path = self.data_path_browser.select_directory("Select Training Dataset Directory")
                if directory_path:
                    self.data_path = directory_path.replace('\\', '/')
                    print(f"Data path selected: {self.data_path}")
                    
                    # Check for PKL files in the directory
                    pkl_files = []
                    for root, dirs, files in os.walk(self.data_path):
                        for file in files:
                            if file.endswith('.pkl'):
                                pkl_path = os.path.join(root, file)
                                pkl_files.append(pkl_path)
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
            if self.menu.show_help and imgui.is_item_hovered():
                imgui.set_tooltip(HELP_TEXTS["data_path_training"])
                
            imgui.text("Resume Pkl")
            current_y = imgui.get_cursor_pos_y()
            imgui.set_cursor_pos_y(current_y - 3)  
            _, self.resume_pkl = imgui_utils.input_text("##Resume Pkl", self.resume_pkl, 1024, 0, 
            width=imgui.get_window_width() - imgui.calc_text_size("Browse##Resume Pkl")[0])
            if self.menu.show_help and imgui.is_item_hovered():
                imgui.set_tooltip(HELP_TEXTS["resume_pkl_training"])
            
            imgui.same_line()
            if imgui_utils.button('Browse##Resume Pkl', enabled=len(self.browse_cache) > 0, width=self.menu.app.button_w):
                imgui.open_popup('browse_pkls_popup_training')
            if self.menu.show_help and imgui.is_item_hovered():
                imgui.set_tooltip(HELP_TEXTS["resume_pkl_training"])

            if imgui.begin_popup('browse_pkls_popup_training'):
                for pkl in self.browse_cache:
                    clicked, _state = imgui.menu_item(pkl)
                    if clicked:
                        self.resume_pkl = pkl
                imgui.end_popup()

            imgui.text("Training Augmentation")
            imgui.same_line()
            _, self.aug = imgui.combo("##Training Augmentation", self.aug, augs)
            if self.menu.show_help and imgui.is_item_hovered():
                imgui.set_tooltip(HELP_TEXTS["augmentation_training"])
            if self.aug == 0:
                imgui.text("Augmentation Pipeline")
                imgui.same_line()
                _, self.ada_pipe = imgui.combo("##Augmentation Pipeline", self.ada_pipe, ada_pipes)
            else:
                imgui.text("Augmentation Pipeline")
                imgui.same_line()
                _, self.diffaug_pipe = imgui.combo("##Augmentation Pipeline", self.diffaug_pipe, diffaug_pipes)
            if self.menu.show_help and imgui.is_item_hovered():
                imgui.set_tooltip(HELP_TEXTS["aug_pipeline_training"])

            imgui.text("Batch Size")
            imgui.same_line()
            _, self.batch_size = imgui.input_int("##Batch Size", self.batch_size)
            if self.menu.show_help and imgui.is_item_hovered():
                imgui.set_tooltip(HELP_TEXTS["batch_size_training"])
            if self.batch_size < 1:
                self.batch_size = 1
            
            imgui.text("Configuration")
            imgui.same_line()
            _, self.config = imgui.combo("##Configuration", self.config, configs)
            if self.menu.show_help and imgui.is_item_hovered():
                imgui.set_tooltip(HELP_TEXTS["config_training"])
            

            imgui.set_next_window_size( self.menu.app.content_width // 4, (self.menu.app.content_height // 4), imgui.ONCE)

            if imgui.button("Advanced...", width=-1):
                imgui.open_popup("Advanced...")
            if self.menu.show_help and imgui.is_item_hovered():
                imgui.set_tooltip(HELP_TEXTS["advanced_training"])

            if imgui.begin_popup_modal("Advanced...")[0]:
                imgui.text("Advanced Training Options")
                _, self.glr = imgui.input_float("Generator Learning Rate", self.glr)
                if self.menu.show_help and imgui.is_item_hovered():
                    imgui.set_tooltip(HELP_TEXTS["generator_lr_training"])

                _, self.dlr = imgui.input_float("Discriminator Learning Rate", self.dlr)
                if self.menu.show_help and imgui.is_item_hovered():
                    imgui.set_tooltip(HELP_TEXTS["discriminator_lr_training"])

                _, self.gamma = imgui.input_int("Gamma", self.gamma)
                if self.menu.show_help and imgui.is_item_hovered():
                    imgui.set_tooltip(HELP_TEXTS["gamma_training"])

                _, self.snap = imgui.input_int("Number of ticks between snapshots", self.snap)
                if self.menu.show_help and imgui.is_item_hovered():
                    imgui.set_tooltip(HELP_TEXTS["snapshot_training"])

                if imgui_utils.button("Close", enabled=1):
                    imgui.close_current_popup()


                imgui.end_popup()


            if imgui.button("Train", width=-1):
                self._open_training_popup = True
                print("training")
                
                target_data_path = self.data_path

                # Manipulate resolution training parameter based on dataset resolution (for now)
                # Read resolution from first image in dataset
                detected_resolution = None
                if os.path.isdir(target_data_path):
                    # Get list of PNG image files
                    image_files = [f for f in os.listdir(target_data_path) 
                                if f.lower().endswith('.png')]
                    if image_files:
                        first_image_path = os.path.join(target_data_path, image_files[0])
                        img = cv2.imread(first_image_path)
                        if img is not None:
                            height, width = img.shape[:2]
                            detected_resolution = (width, height)
                            print(f"Detected image resolution from dataset: {detected_resolution}")

                kwargs = dnnlib.EasyDict(
                    outdir=self.save_path,
                    data=target_data_path,
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
        
        # Quick Settings Dataset Creation Popup Modal
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
                
                # Check for progress updates
                if not self.dataset_reply.empty():
                    try:
                        progress_data = self.dataset_reply.get_nowait()
                        
                        if progress_data.get('type') == 'completed':
                            # Check if this is video extraction completion or image processing completion
                            if self.video_extraction_in_progress:
                                # Video extraction completed
                                frame_directories = progress_data.get('results', [])
                                self.extracted_frame_directories = frame_directories

                                # Terminate video process
                                if hasattr(self, 'dataset_process') and self.dataset_process and self.dataset_process.is_alive():
                                    self.dataset_process.terminate()
                                    self.dataset_process.join(timeout=2)
                                
                                # Start image processing
                                self.start_image_processing()
                            else:
                                # Image processing completed
                                self.dataset_done = True
                                self.is_creating_dataset = False
                                self.dataset_processed_count = progress_data.get('processed_count', 0)
                                print(f"Dataset created! Processed {self.dataset_processed_count} images")
                                
                                # Automatically populate the main training data path with the completed dataset path
                                if hasattr(self, 'dataset_output_path'):
                                    self.data_path = self.dataset_output_path
                                                                    
                                if hasattr(self, 'dataset_process') and self.dataset_process and self.dataset_process.is_alive():
                                    self.dataset_process.terminate()
                                    self.dataset_process.join(timeout=2)
                        elif progress_data.get('type') == 'progress':
                            # Check if this is video extraction or image processing progress
                            if self.video_extraction_in_progress:
                                # Video extraction progress
                                self.video_extraction_current = progress_data.get('current', 0)
                                self.video_extraction_file = progress_data.get('current_file', '')
                            else:
                                # Image processing progress
                                self.dataset_progress_current = progress_data.get('current', 0)
                                self.dataset_progress_total = progress_data.get('total', 0)
                                self.dataset_progress_percentage = progress_data.get('percentage', 0)
                                self.dataset_progress_file = progress_data.get('current_file', '')
                    except:
                        pass
                
                progress_width = self.menu.app.content_width // 2.5 - 40
                
                # Video extraction progress bar (shown first if videos are being extracted)
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
                
                # Image processing progress bar (shown after video extraction or immediately if no videos)
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

            # End of Quick Settings Creating Dataset Popup Modal

        # Training Popup Modal
        training_popup_width = self.menu.app.content_width // 1.5
        training_popup_height = self.menu.app.content_height // 1.5
        imgui.set_next_window_size(training_popup_width, training_popup_height, imgui.ONCE)

        if imgui.begin_popup_modal("Training")[0]:
            imgui.text("Training...")
            if os.path.exists(self.message) and self.image_path != self.message:
                self.image_path = self.message
                self.grid = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
                self.grid = cv2.cvtColor(self.grid, cv2.COLOR_BGRA2RGBA)
                self.grid_texture = gl_utils.Texture(image=self.grid, width=self.grid.shape[1],
                                               height=self.grid.shape[0], channels=self.grid.shape[2])
            elif self.message != "":
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
        """Collect all image files from selected directory path (quick settings)"""
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}
        image_files = []
        
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if os.path.splitext(file)[1].lower() in image_extensions:
                    image_files.append(os.path.join(root, file))
        
        return image_files
    
    def create_quick_dataset(self):
        """Start the quick dataset creation process"""
        # Validate paths
        if not os.path.exists(self.quick_data_path):
            print(f"Error: Data path does not exist: {self.quick_data_path}")
            self.dataset_message = f"Data path does not exist:\n{self.quick_data_path}"
            return
        
        # Collect image files from directory
        image_files = self.collect_image_files(self.quick_data_path)
        
        if not image_files and not self.video_files_list:
            print("Error: No images or videos found in directory")
            self.dataset_message = "No images or videos found in the selected directory"
            return
        
        self.dataset_image_count = len(image_files) + len(self.video_files_list)
        
        # Add resolution suffix to folder name and construct full path
        resolution_suffix = f"_{self.img_size}x{self.img_size}"
        folder_name_with_resolution = self.quick_folder_name + resolution_suffix
        self.dataset_output_path = os.path.join(self.quick_save_path, folder_name_with_resolution).replace('\\', '/')
        
        self.temp_image_files = list(image_files)
        self.extracted_frame_directories = []
        
        if os.path.exists(self.dataset_output_path):
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
            if os.path.exists(frame_dir):
                frames = [os.path.join(frame_dir, f) for f in os.listdir(frame_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                all_images.extend(frames)
                print(f"Added {len(frames)} frames from {os.path.basename(frame_dir)}")
        
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

