import os
import zipfile

import imgui
import multiprocessing as mp


import dnnlib
from utils.gui_utils import imgui_utils
from train import main as train_main
from utils import dataset_tool
from widgets.browse_widget import BrowseWidget
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
    "mirror_training": "Horizontal flip of dataset",
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
        self.save_path = os.path.join(cwd, "training-runs")
        self.data_path = os.path.join(cwd, "data")
        # self.show_help = False 
        # create data folder if not exists
        if not os.path.exists(os.path.abspath(os.path.join(os.getcwd(),"data"))):
            os.makedirs(os.path.abspath(os.path.join(os.getcwd(),"data")))
        self.file_dialog = BrowseWidget(menu, "Dataset", os.path.abspath(os.path.join(os.getcwd(),"data")), ["*",""], multiple=False, traverse_folders=False, width=menu.app.button_w)
        self.app = menu.app
        self.config = 1
        self.resume_pkl = ""
        self.start_res = [4,4]
        self.browse_cache = []
        self.aug = 0
        self.ada_pipe = 7
        self.resize_mode = 0
        self.diffaug_pipe = 0
        self.img_factor = 1
        self.height_factor = 1
        self.img_size = self.start_res[0] * (2 ** self.img_factor)
        self.square = True
        self.height = self.start_res[1] * (2 ** self.height_factor)
        self.batch_size = 8
        self.save_path_dialog = BrowseWidget(self, "Save Path", os.path.abspath(os.path.join(os.getcwd())), [""], multiple=False,
                                             traverse_folders=False, add_folder_button=True, width=menu.app.button_w)

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
        self.fps = 10
        self.found_video = False
        self._zipfile = None
        self.gamma = 10
        self.glr = 0.002
        self.dlr = 0.002
        self.snap = 4
        self.mirror = True
        self.done_button = False
        self.image_path = ''

        # Initialize non-square cropping attributes
        self.crop_width_ratio = 16  # Default 16:9 ratio
        self.crop_height_ratio = 9
        self.padding_color = 0  # 0=black padding, 1=white padding

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

        imgui.begin_group()
        imgui.text("Train a model on your own data")
        imgui.end_group()

        _, self.save_path = imgui.input_text("Save Path", self.save_path, 1024)
        if self.menu.show_help and imgui.is_item_hovered():
            imgui.set_tooltip(HELP_TEXTS["save_path_training"])
        
        imgui.same_line()
        _clicked, save_path = self.save_path_dialog(self.menu.app.button_w)
        if _clicked:
            if len(save_path) > 0:
                self.save_path = save_path[0]
                print(self.save_path)
            else:
                self.save_path = ""
                print("No path selected")
        if self.menu.show_help and imgui.is_item_hovered():
            imgui.set_tooltip(HELP_TEXTS["save_path_training"])

        _, self.data_path = imgui.input_text("Data Path", self.data_path, 1024)
        if self.menu.show_help and imgui.is_item_hovered():
            imgui.set_tooltip(HELP_TEXTS["data_path_training"])
        
        imgui.same_line()
        _clicked, data_path = self.file_dialog(self.menu.app.button_w)
        if _clicked:
            self.data_path = data_path[0]
            # find all files in the directory and then check if any are videos
            if os.path.isdir(self.data_path):
                self._type = 'dir'
                self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self.data_path) for root, _dirs, files
                                    in os.walk(self.data_path) for fname in files}
            elif self._file_ext(self.data_path) == '.zip':
                self._type = 'zip'
                self._all_fnames = set(self._get_zipfile().namelist())
            elif self._file_ext(self.data_path) == '.mp4' or self._file_ext(self.data_path) == '.avi':
                self._type = 'video'
                self._all_fnames = {self.data_path}
            else:
                raise IOError('Path must point to a directory or zip')
            self.found_video = False
            # if any file in self__all_fnames is a video create a new subfolder where we save the frames based on fps using ffmpeg
            for fname in self._all_fnames:
                print(str(fname))
                if fname.endswith('.mp4') or fname.endswith('.avi'):
                    self.found_video = True
                    break
        if self.menu.show_help and imgui.is_item_hovered():
            imgui.set_tooltip(HELP_TEXTS["data_path_training"])
        _, self.resume_pkl = imgui.input_text("Resume Pkl", self.resume_pkl, 1024)
        if self.menu.show_help and imgui.is_item_hovered():
            imgui.set_tooltip(HELP_TEXTS["resume_pkl_training"])
        
        imgui.same_line()
        if imgui_utils.button('Browse...', enabled=len(self.browse_cache) > 0, width=-1):
            imgui.open_popup('browse_pkls_popup_training')
        if self.menu.show_help and imgui.is_item_hovered():
            imgui.set_tooltip(HELP_TEXTS["resume_pkl_training"])

        if imgui.begin_popup('browse_pkls_popup_training'):
            for pkl in self.browse_cache:
                clicked, _state = imgui.menu_item(pkl)
                if clicked:
                    self.resume_pkl = pkl
            imgui.end_popup()

        _, self.aug = imgui.combo("Augmentation", self.aug, augs)
        if self.menu.show_help and imgui.is_item_hovered():
            imgui.set_tooltip(HELP_TEXTS["augmentation_training"])
        if self.aug == 0:
            _, self.ada_pipe = imgui.combo("Augmentation Pipeline", self.ada_pipe, ada_pipes)
        else:
            _, self.diffaug_pipe = imgui.combo("Augmentation Pipeline", self.diffaug_pipe, diffaug_pipes)
        if self.menu.show_help and imgui.is_item_hovered():
            imgui.set_tooltip(HELP_TEXTS["aug_pipeline_training"])

        # _changed, start_res = imgui.input_int2("Start Resolution", *self.start_res)
        # if _changed:
        #     if start_res[0] < 1:
        #         start_res[0] = 1
        #     if start_res[1] < 1:
        #         start_res[1] = 1
        #     self.start_res = start_res
        #     self.img_size = start_res[0] * (2 ** self.img_factor)
        #     self.height = start_res[1] * (2 ** self.height_factor)
        _, self.resize_mode = imgui.combo("Resize Mode", self.resize_mode, resize_mode)
        if self.menu.show_help and imgui.is_item_hovered():
            imgui.set_tooltip(HELP_TEXTS["resize_mode_training"])

        imgui.input_text("Width", str(self.img_size), 512,flags=imgui.INPUT_TEXT_READ_ONLY)


        imgui.same_line()
        if imgui.button("-##img_size", width=self.menu.app.font_size):
            self.img_factor = max(self.img_factor - 1, 1)   
            self.img_size = self.start_res[0] * (2 ** self.img_factor)
            self.height = self.img_size


        imgui.same_line()
        if imgui.button("+##img_size", width=self.menu.app.font_size):
            self.img_factor = self.img_factor + 1
            self.img_size = self.start_res[0] * (2 ** self.img_factor)
            self.height = self.img_size


        imgui.input_text("Height", str(self.height), 512, flags=imgui.INPUT_TEXT_READ_ONLY)


        imgui.same_line()
        if imgui.button("-##height", width=self.menu.app.font_size):
            self.img_factor = max(self.img_factor - 1, 1)   
            self.height = self.start_res[0] * (2 ** self.img_factor)
            self.img_size = self.height


        imgui.same_line()
        if imgui.button("+##height", width=self.menu.app.font_size):
            self.img_factor = self.img_factor + 1
            self.height = self.start_res[0] * (2 ** self.img_factor)
            self.img_size = self.height


        if self.found_video:
            _, self.fps = imgui.input_int("FPS for frame extraction", self.fps)
            if self.fps < 1:
                self.fps = 1

        _, self.batch_size = imgui.input_int("Batch Size", self.batch_size)
        if self.menu.show_help and imgui.is_item_hovered():
            imgui.set_tooltip(HELP_TEXTS["batch_size_training"])
        if self.batch_size < 1:
            self.batch_size = 1
        
        _, self.config = imgui.combo("Configuration", self.config, configs)
        if self.menu.show_help and imgui.is_item_hovered():
            imgui.set_tooltip(HELP_TEXTS["config_training"])
        
        clicked, non_square = imgui.checkbox("non-square settings", not self.square)
        if clicked:
            self.square = not non_square  # 当checkbox被点击时更新状态
            if not self.square:  # 如果启用了non-square设置
                if not hasattr(self, 'crop_width_ratio'):
                    self.crop_width_ratio = 16  # 默认宽高比
                if not hasattr(self, 'crop_height_ratio'):
                    self.crop_height_ratio = 9   # 默认宽高比
                if not hasattr(self, 'padding_color'):
                    self.padding_color = 0  # 默认黑色填充

        if not self.square:  # 当选中 checkbox 时显示 non-square 选项
            imgui.indent(20)
            imgui.text("aspect ratio:")
            changed_width, new_width_ratio = imgui.input_int("Width Ratio", self.crop_width_ratio)
            if changed_width and new_width_ratio >= 1:  
                self.crop_width_ratio = new_width_ratio


            changed_height, new_height_ratio = imgui.input_int("Height Ratio", self.crop_height_ratio)
            if changed_height and new_height_ratio >= 1:  
                self.crop_height_ratio = new_height_ratio

            
            base_size = self.img_size 
            ratio = self.crop_height_ratio / self.crop_width_ratio

            if ratio <= 1:  
                actual_width = base_size
                actual_height = int(base_size * ratio)
            else:  
                actual_height = base_size
                actual_width = int(base_size / ratio)
                
            imgui.text(f"Actual resolution: {actual_width}x{actual_height}")
            changed_color, new_padding_color = imgui.combo("Padding Options", self.padding_color, ["Black", "White", "Bleeding"])
            if changed_color:
                self.padding_color = new_padding_color

            imgui.unindent(20)

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

            _, self.mirror = imgui.checkbox('Mirror', self.mirror)
            if self.menu.show_help and imgui.is_item_hovered():
                imgui.set_tooltip(HELP_TEXTS["mirror_training"])

            if imgui_utils.button("Close", enabled=1):
                imgui.close_current_popup()


            imgui.end_popup()

        # # Add non-square options before resolution settings
        # imgui.separator()
        # imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 1.0, 0.0)
        # imgui.text("Resolution Settings")
        # imgui.pop_style_color()
        
        # Add non-square option
        # _, non_square = imgui.checkbox("non-framing settings", not self.square)  # 注意这里用 not self.square
        # self.square = not non_square  # 反转逻辑

        # if not self.square:  # Only show these options in non-square mode
        #     imgui.indent(20)
        #     imgui.text("Crop Ratio Settings:")
        #     changed_width, new_width_ratio = imgui.input_int("Width Ratio", self.crop_width_ratio)


        #     changed_height, new_height_ratio = imgui.input_int("Height Ratio", self.crop_height_ratio)

            
        #     imgui.text(f"Current aspect ratio: {self.crop_width_ratio}:{self.crop_height_ratio}")
            
        #     changed_color, new_padding_color = imgui.combo("Padding Options", self.padding_color, ["Black", "White"])

        #     imgui.unindent(20)

        if imgui.button("Train", width=-1):
            imgui.open_popup("Training")
            print("training")
            
            target_data_path = self.data_path
            
            if not self.square:
                print(f"Starting non-square processing...")
                print(f"Crop ratio: {self.crop_width_ratio}:{self.crop_height_ratio}")
                print(f"Padding color: {'white' if self.padding_color == 1 else 'black'}")
                
                from training.dataset import process_non_square_dataset
                
                target_data_path = os.path.join(os.path.dirname(target_data_path), 'non_square_cache')
                try:
                    process_non_square_dataset(
                        input_path=self.data_path,
                        output_path=target_data_path,
                        crop_ratio=(int(self.crop_width_ratio), int(self.crop_height_ratio)),
                        padding_color=int(self.padding_color),
                        resize_mode=resize_mode[self.resize_mode].lower()
                    )
                    print(f"Non-square processing completed.")
                except Exception as e:
                    print(f"Error during non-square processing: {str(e)}")
                    return

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
                resolution=(int(self.img_size), int(self.height)),
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

        imgui.set_next_window_size( self.menu.app.content_width // 2, (self.menu.app.content_height // 2), imgui.ONCE)

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
                imgui.image(self.grid_texture.gl_id, self.menu.app.content_width // 1.7, (self.menu.app.content_height // 1.7))
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
                
