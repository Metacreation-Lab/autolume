import os
import zipfile

import imgui
import multiprocessing as mp

import dnnlib
from utils.gui_utils import imgui_utils
from train import main as train_main
from utils import dataset_tool
from widgets.browse_widget import BrowseWidget

augs = ["ADA", "DiffAUG"]
ada_pipes = ['blit', 'geom', 'color', 'filter', 'noise', 'cutout', 'bg', 'bgc', 'bgcf', 'bgcfn', 'bgcfnc']
diffaug_pipes = ['color,translation,cutout', 'color,translation', 'color,cutout', 'color',
                 'translation', 'cutout,translation', 'cutout']
sizes = ["64", "128", "256", "512", "1024"]
class TrainingModule:
    def __init__(self, menu):
        cwd = os.getcwd()
        self.save_path = os.path.join(cwd, "training-runs")
        self.data_path = os.path.join(cwd, "data")
        self.file_dialog = BrowseWidget(menu, "Dataset", os.path.abspath(os.path.join(os.getcwd(),"data")), ["*",""], multiple=False, traverse_folders=False)

        self.resume_pkl = ""
        self.browse_cache = []
        self.aug = 0
        self.ada_pipe = 7
        self.diffaug_pipe = 0
        self.img_size = 2
        self.square = True
        self.height = 2
        self.batch_size = 8
        for pkl in os.listdir("./models"):
            if pkl.endswith(".pkl"):
                print(pkl, os.path.join(os.getcwd(),"models",pkl))
                self.browse_cache.append(os.path.join(os.getcwd(),"models",pkl))

        self.menu = menu

        self.queue = mp.Queue()
        self.reply = mp.Queue()
        self.message = ""
        self.done = False
        self.training_process = mp.Process(target=train_main, args=(self.queue, self.reply))
        self.fps = 10
        self.found_video = False
        self._zipfile = None

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

        imgui.text("Training Module")

        _, self.save_path = imgui.input_text("Save Path", self.save_path, 1024)
        _, self.data_path = imgui.input_text("Data Path", self.data_path, 1024)
        imgui.same_line()
        _clicked, data_path = self.file_dialog()
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
            else:
                raise IOError('Path must point to a directory or zip')

            self.found_video = False
            # if any file in self__all_fnames is a video create a new subfolder where we save the frames based on fps using ffmpeg
            for fname in self._all_fnames:
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

        _, self.img_size = imgui.combo("Image Size" if self.square else "Width", self.img_size, sizes)
        if not(self.square):
            _, self.height = imgui.combo("Height", self.height, sizes)
        else:
            self.height = self.img_size

        imgui.same_line()
        _, self.square = imgui.checkbox("Square", self.square)

        if self.found_video:
            _, self.fps = imgui.input_int("FPS for frame extraction", self.fps)
            if self.fps < 1:
                self.fps = 1

        _, self.batch_size = imgui.input_int("Batch Size", self.batch_size)
        if self.batch_size < 1:
            self.batch_size = 1


        if imgui.button("Train"):
            imgui.open_popup("Training")
            print("training")
            kwargs = dnnlib.EasyDict(
                outdir=self.save_path,
                data=self.data_path,
                cfg="stylegan2",
                batch=self.batch_size,
                topk=None,
                gpus=1,
                gamma=10,
                z_dim=512,
                w_dim=512,
                cond=False,
                mirror=True,
                resolution=(int(sizes[self.img_size]), int(sizes[self.height])),
                aug="ada" if augs[self.aug] == "ADA" else "noaug",
                augpipe=ada_pipes[self.ada_pipe],
                resume=self.resume_pkl if self.resume_pkl != "" else None,
                freezed=0,
                p=0.2,
                target=0.6,
                batch_gpu=self.batch_size//1, #gpus param?
                cbase=32768,
                cmax=512,
                glr=None,
                dlr=0.002,
                map_depth=8,
                mbstd_group=4,
                initstrength=None,
                projected=False,
                diffaugment= diffaug_pipes[self.diffaug_pipe] if self.aug == 1 else None,
                desc="",
                metrics=[],
                kimg=25000,
                nkimg=0,
                tick=4,
                snap=1,
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

            self.queue.put(kwargs)
            self.training_process.start()

        if imgui.begin_popup_modal("Training")[0]:
            imgui.text("Training...")
            if self.message != "":
                imgui.text(self.message)
            if imgui_utils.button("Done", enabled=self.done):
                imgui.close_current_popup()
            imgui.end_popup()





