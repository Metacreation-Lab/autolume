import imgui

import os

from utils.gui_utils import imgui_utils
import multiprocessing as mp

from modules.filedialog import FileDialog
from projection.bayle_projection import run_projection

class ProjectionModule:
    def __init__(self, menu):
        self.menu = menu
        self.app = menu.app
        self.file_dialog = FileDialog(self, "Target Image", os.path.abspath(os.getcwd()),
                                      ["*", ".jpg", ".png", ".jpeg", ".bmp"])

        self.network_path = ""
        self._pkl_data = dict()
        self._networks = dict()
        self.models = []
        self.target_fname = ""
        self.target_text = ""
        self.initial_latent = None
        self.outdir = ""
        self.save_video = False
        self.seed = 300
        self.lr = 0.1
        self.steps = 1000
        self.use_vgg = True
        self.use_clip = True
        self.use_pixel = True
        self.use_penalty = True
        self.use_center = True
        self.use_kmeans = True

        self.queue = mp.Queue()
        self.reply = mp.Queue()
        self.message = ""
        self.projected_img = None
        self.done_recording = False
        self.projection_process = mp.Process(target=run_projection, args=(self.queue, self.reply),
                                       daemon=True)

        for pkl in os.listdir("./models"):
            if pkl.endswith(".pkl"):
                self.models.append(os.path.join(os.getcwd(),"models",pkl))

    @imgui_utils.scoped_by_object_id
    def __call__(self):
        if self.reply.qsize() > 0:
            self.message, self.projected_img, self.done_recording = self.reply.get()
            while self.reply.qsize() > 0:
                self.message, self.projected_img, self.done_recording = self.reply.get()

        _, self.network_path = imgui_utils.input_text('##projection', self.network_path, 1024,
                                                        flags=(
                                                                    imgui.INPUT_TEXT_AUTO_SELECT_ALL | imgui.INPUT_TEXT_ENTER_RETURNS_TRUE),
                                                        width=(100),
                                                        help_text='<PATH> | <URL> | <RUN_DIR> | <RUN_ID> | <RUN_ID>/<KIMG>.pkl')
        imgui.same_line()
        if imgui_utils.button(f'Browse...##projection', enabled=len(self.models) > 0):
            imgui.open_popup(f'browse_pkls_popup##projection')
            self.browse_refocus = True

        if imgui.begin_popup(f'browse_pkls_popup##projection'):
            for pkl in self.models:
                clicked, _state = imgui.menu_item(pkl)
                if clicked:
                    self.network_path = pkl
            if self.browse_refocus:
                imgui.set_scroll_here()
                self.browse_refocus = False

            imgui.end_popup()
        imgui.same_line()


        joined = '\n'.join(self.target_fname)
        imgui_utils.input_text("##projection", joined, 1024, flags=imgui.INPUT_TEXT_READ_ONLY,
                               width=-self.app.button_w * 3, help_text="Input Files")
        imgui.same_line()
        _clicked, target_pth = self.file_dialog() # should have argument to only allow single file
        if _clicked:
            self.target_fname = target_pth

        _changed, self.target_text = imgui_utils.input_text('Target Text##target_text', self.target_text, 1024, flags=imgui.INPUT_TEXT_AUTO_SELECT_ALL, help_text='Text to be projected')
        _changed, self.outdir = imgui_utils.input_text('Output Directory##outdir', self.outdir, 1024, flags=imgui.INPUT_TEXT_AUTO_SELECT_ALL, help_text='Directory to save results')
        imgui.same_line()
        _changed, self.save_video = imgui.checkbox('Save Video##save_video', self.save_video)
        _changed, self.seed = imgui.input_int('Seed##seed', self.seed)
        _changed, self.lr = imgui.input_float('Learning Rate##lr', self.lr)
        _changed, self.steps = imgui.input_int('Steps##steps', self.steps)
        _changed, self.use_vgg = imgui.checkbox('Use VGG##use_vgg', self.use_vgg)
        _changed, self.use_clip = imgui.checkbox('Use CLIP##use_clip', self.use_clip)
        _changed, self.use_pixel = imgui.checkbox('Use Pixel##use_pixel', self.use_pixel)
        _changed, self.use_penalty = imgui.checkbox('Use Penalty##use_penalty', self.use_penalty)
        _changed, self.use_center = imgui.checkbox('Use Center##use_center', self.use_center)


        if imgui.button("Project"):
            imgui.open_popup('Project')
            self.projection_process.start()
            self.queue.put([self.network_path, self.target_fname[0], self.target_text if self.target_text != "" else None, self.initial_latent, self.outdir, self.save_video, self.seed, self.lr, self.steps, self.use_vgg, self.use_clip, self.use_pixel, self.use_penalty, self.use_center, self.use_kmeans])


        if imgui.begin_popup_modal('Project')[0]:
            imgui.text("Projecting...")
            if self.message != "":
                imgui.text(self.message)

            enabled = self.projected_img is not None
            if self.save_video:
                enabled = self.done_recording
            if imgui_utils.button("Done", enabled=enabled):
                imgui.close_current_popup()
            imgui.end_popup()
