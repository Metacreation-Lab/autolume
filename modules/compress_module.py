import os

import imgui

import dnnlib
from utils.gui_utils import imgui_utils
from prune import main as prune_main
from train import main as train_main
from utils import dataset_tool

augs = ["ADA", "DiffAUG"]
ada_pipes = ['blit', 'geom', 'color', 'filter', 'noise', 'cutout', 'bg', 'bgc', 'bgcf', 'bgcfn', 'bgcfnc']
diffaug_pipes = ['color, translation, cutoff', 'color, translation', 'color, cutoff', 'color',
                 'translation', 'cutoff', 'translation', 'cutoff']
sizes = ["64", "128", "256", "512", "1024"]
class CompressModule:
    def __init__(self, menu):
        self.img_factor = 4
        self.height_factor = 4
        self.square = True
        cwd = os.getcwd()
        self.compress_pkl = ""
        self.save_path = os.path.join(cwd, "distillation-runs")
        self.save_path_distill = os.path.join(cwd, "training-runs")
        self.data_path = os.path.join(cwd, "data")
        self.resume_pkl = ""
        self.teacher_pkl = ""
        self.browse_cache = []
        self.compression_factor = 0.7
        self.n_samples = 400
        self.batch_size_compress = 10
        self.noise_prob = 0.05
        self.aug = 0
        self.ada_pipe = 7
        self.diffaug_pipe = 0
        self.img_size = 2 ** self.img_factor
        self.height = 2 ** self.height_factor
        self.batch_size = 8
        for pkl in os.listdir("./models"):
            if pkl.endswith(".pkl"):
                print(pkl, os.path.join(os.getcwd(),"models",pkl))
                self.browse_cache.append(os.path.join(os.getcwd(),"models",pkl))

        self.menu = menu

    @imgui_utils.scoped_by_object_id
    def __call__(self):
        imgui.text("Prune unnecessary layers from a network")
        _, self.save_path = imgui.input_text("Save Path", self.save_path, 1024)
        _, self.compress_pkl = imgui.input_text("Learner Pkl", self.compress_pkl, 1024)
        imgui.same_line()
        if imgui_utils.button('Browse...##compress', enabled=len(self.browse_cache) > 0, width=-1):
            imgui.open_popup('browse_pkls_popup_compress')
        if imgui.begin_popup('browse_pkls_popup_compress'):
            for pkl in self.browse_cache:
                clicked, _state = imgui.menu_item(pkl)
                if clicked:
                    self.compress_pkl = pkl
            imgui.end_popup()
        _, self.compression_factor = imgui.input_float("Compression Factor", self.compression_factor)
        _, self.n_samples = imgui.input_int("Number of Samples", self.n_samples)
        _, self.batch_size_compress = imgui.input_int("Batch Size", self.batch_size_compress)
        if self.batch_size_compress < 1:
            self.batch_size_compress = 1
        _, self.noise_prob = imgui.input_float("Noise Probability", self.noise_prob)

        if imgui_utils.button('Compress', enabled=len(self.compress_pkl) > 0, width=-1):
            print("Compressing")
            print(self.compress_pkl)
            prune_main(self.compress_pkl, self.save_path, n_samples=self.n_samples, batch_size=self.batch_size_compress,
                       noise_prob=self.noise_prob, remove_ratio=self.compression_factor)

        imgui.separator()
        imgui.text("Retrain a network with a teacher network")
        _, self.save_path_distill = imgui.input_text("Save Path", self.save_path, 1024)
        _, self.data_path = imgui.input_text("Data Path", self.data_path, 1024)
        _, self.resume_pkl = imgui.input_text("Learner Pkl", self.resume_pkl, 1024)
        imgui.same_line()
        if imgui_utils.button('Browse...##learner', enabled=len(self.browse_cache) > 0, width=-1):
            imgui.open_popup('browse_pkls_popup_learner')
        if imgui.begin_popup('browse_pkls_popup_learner'):
            for pkl in self.browse_cache:
                clicked, _state = imgui.menu_item(pkl)
                if clicked:
                    self.resume_pkl = pkl
            imgui.end_popup()
        _, self.teacher_pkl = imgui.input_text("Teacher Pkl", self.teacher_pkl, 1024)
        imgui.same_line()
        if imgui_utils.button('Browse...##teacher', enabled=len(self.browse_cache) > 0, width=-1):
            imgui.open_popup('browse_pkls_popup_teacher')

        if imgui.begin_popup('browse_pkls_popup_teacher'):
            for pkl in self.browse_cache:
                clicked, _state = imgui.menu_item(pkl)
                if clicked:
                    self.teacher_pkl = pkl
            imgui.end_popup()

        _, self.aug = imgui.combo("Augmentation", self.aug, augs)
        if self.aug == 0:
            _, self.ada_pipe = imgui.combo("Augmentation Pipeline", self.ada_pipe, ada_pipes)
        else:
            _, self.diffaug_pipe = imgui.combo("Augmentation Pipeline", self.diffaug_pipe, diffaug_pipes)
        # label = "Image Size" if self.square else "Width"
        # imgui.input_text(label, str(self.img_size), 512, flags=imgui.INPUT_TEXT_READ_ONLY)
        # imgui.same_line()
        # if imgui.button("-##img_size", width=self.menu.app.font_size):
        #     self.img_factor = max(self.img_factor - 1, 1)
        #     self.img_size = 2 ** self.img_factor
        # imgui.same_line()
        # if imgui.button("+##img_size", width=self.menu.app.font_size):
        #     self.img_factor = self.img_factor + 1
        #     self.img_size = 2 ** self.img_factor
        #
        # if not (self.square):
        #     imgui.input_text("Height", str(self.height), 512, flags=imgui.INPUT_TEXT_READ_ONLY)
        #     imgui.same_line()
        #     if imgui.button("-##height", width=self.menu.app.font_size):
        #         self.height_factor = max(self.height_factor - 1, 1)
        #         self.height = 2 ** self.height_factor
        #     imgui.same_line()
        #     if imgui.button("+##height", width=self.menu.app.font_size):
        #         self.height_factor = self.height_factor + 1
        #         self.height = 2 ** self.height_factor
        # else:
        #     self.height = self.img_size

        imgui.same_line()
        _, self.square = imgui.checkbox("Square", self.square)

        _, self.batch_size = imgui.input_int("Batch Size", self.batch_size)
        if self.batch_size < 1:
            self.batch_size = 1

        #add distillation params into gui


        if imgui.button("Train##compress", width=-1):
            kwargs = dnnlib.EasyDict(
                outdir=self.save_path_distill,
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
                aug="ada" if augs[self.aug] == "ADA" else "noaug",
                augpipe=ada_pipes[self.ada_pipe],
                resume=self.resume_pkl,
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
                kd_l1_lambda=3,
                kd_lpips_lambda=3,
                kd_mode="Output_Only",
                content_aware_kd=False,
                teacher = self.teacher_pkl,
                custom=True,
                lpips_image_size=256
            )
            train_main(**kwargs)





