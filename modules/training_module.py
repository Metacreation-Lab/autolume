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

augs = ["ADA", "DiffAUG"]
ada_pipes = ['blit', 'geom', 'color', 'filter', 'noise', 'cutout', 'bg', 'bgc', 'bgcf', 'bgcfn', 'bgcfnc']
diffaug_pipes = ['color,translation,cutout', 'color,translation', 'color,cutout', 'color',
                 'translation', 'cutout,translation', 'cutout']
configs = ['auto', 'stylegan2', 'paper256', 'paper512', 'paper1024', 'cifar']

class TrainingModule:
    def __init__(self, menu):
        cwd = os.getcwd()
        self.save_path = os.path.join(cwd, "training-runs")
        self.data_path = os.path.join(cwd, "data")
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
        self.diffaug_pipe = 0
        self.img_factor = 1
        self.height_factor = 1
        self.img_size = self.start_res[0] * (2 ** self.img_factor)
        self.square = True
        self.height = self.start_res[1] * (2 ** self.height_factor)
        self.batch_size = 8
        self.save_path_dialog = BrowseWidget(self, "Save Path", os.path.abspath(os.getcwd()), [""], multiple=False,
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

        imgui.text("Train a model on your own data")

        _, self.save_path = imgui.input_text("Save Path", self.save_path, 1024)
        imgui.same_line()
        _clicked, save_path = self.save_path_dialog(self.menu.app.button_w)
        if _clicked:
            if len(save_path) > 0:
                self.save_path = save_path[0]
                print(self.save_path)
            else:
                self.save_path = ""
                print("No path selected")
        _, self.data_path = imgui.input_text("Data Path", self.data_path, 1024)
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
        _, self.resume_pkl = imgui.input_text("Resume Pkl", self.resume_pkl, 1024)
        imgui.same_line()
        if imgui_utils.button('Browse...', enabled=len(self.browse_cache) > 0, width=-1):
            imgui.open_popup('browse_pkls_popup_training')

        if imgui.begin_popup('browse_pkls_popup_training'):
            for pkl in self.browse_cache:
                clicked, _state = imgui.menu_item(pkl)
                if clicked:
                    self.resume_pkl = pkl
            imgui.end_popup()

        _, self.aug = imgui.combo("Augmentation", self.aug, augs)
        if self.aug == 0:
            _, self.ada_pipe = imgui.combo("Augmentation Pipeline", self.ada_pipe, ada_pipes)
        else:
            _, self.diffaug_pipe = imgui.combo("Augmentation Pipeline", self.diffaug_pipe, diffaug_pipes)

        _changed, start_res = imgui.input_int2("Start Resolution", *self.start_res)
        if _changed:
            if start_res[0] < 1:
                start_res[0] = 1
            if start_res[1] < 1:
                start_res[1] = 1
            self.start_res = start_res
            self.img_size = start_res[0] * (2 ** self.img_factor)
            self.height = start_res[1] * (2 ** self.height_factor)

        imgui.input_text("Width", str(self.img_size), 512,flags=imgui.INPUT_TEXT_READ_ONLY)
        imgui.same_line()
        if imgui.button("-##img_size", width=self.menu.app.font_size):
            self.img_factor = max(self.img_factor - 1, 1)   
            self.img_size = self.start_res[0] * (2 ** self.img_factor)
        imgui.same_line()
        if imgui.button("+##img_size", width=self.menu.app.font_size):
            self.img_factor = self.img_factor + 1
            self.img_size = self.start_res[0] * (2 ** self.img_factor)

        imgui.input_text("Height", str(self.height), 512, flags=imgui.INPUT_TEXT_READ_ONLY)
        imgui.same_line()
        if imgui.button("-##height", width=self.menu.app.font_size):
            self.height_factor = max(self.height_factor - 1, 1)
            self.height = self.start_res[1] * (2 ** self.height_factor)
        imgui.same_line()
        if imgui.button("+##height", width=self.menu.app.font_size):
            self.height_factor = self.height_factor + 1
            self.height = self.start_res[1] * (2 ** self.height_factor)

        if self.found_video:
            _, self.fps = imgui.input_int("FPS for frame extraction", self.fps)
            if self.fps < 1:
                self.fps = 1

        _, self.batch_size = imgui.input_int("Batch Size", self.batch_size)
        if self.batch_size < 1:
            self.batch_size = 1
        
        _, self.config = imgui.combo("Configuration", self.config, configs)

        imgui.set_next_window_size( self.menu.app.content_width // 4, (self.menu.app.content_height // 4), imgui.ONCE)

        if imgui.button("Advanced...", width=-1):
            imgui.open_popup("Advanced...")

        if imgui.begin_popup_modal("Advanced...")[0]:
            imgui.text("Advanced Training Options")
            _, self.glr = imgui.input_float("Generator Learning Rate", self.glr)
            _, self.dlr = imgui.input_float("Discriminator Learning Rate", self.dlr)
            _, self.gamma = imgui.input_int("Gamma", self.gamma)
            _, self.snap = imgui.input_int("Number of ticks between snapshots", self.snap)
            _, self.mirror = imgui.checkbox('Mirror', self.mirror)
            if imgui_utils.button("Done", enabled=1):
                imgui.close_current_popup()
            imgui.end_popup()


        if imgui.button("Train", width=-1):
            imgui.open_popup("Training")
            print("training")
            kwargs = dnnlib.EasyDict(
                outdir=self.save_path,
                data=self.data_path,
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
            if imgui_utils.button("Done", enabled=1):
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
                
