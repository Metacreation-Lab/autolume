import PIL
import imgui

import os

import numpy as np
import torch
from PIL import ImageFilter
from PIL.Image import Image

from utils.gui_utils import imgui_utils, gl_utils
import multiprocessing as mp

from modules.filedialog import FileDialog
from projection.bayle_projection import run_projection
from widgets.browse_widget import BrowseWidget


class ProjectionModule:
    def __init__(self, menu):
        self.menu = menu
        self.app = menu.app
        self.file_dialog = BrowseWidget(self, "Target Image", os.path.abspath(os.getcwd()), ["*", ".jpg", ".png", ".jpeg", ".bmp"], multiple=False,
                                             traverse_folders=False, add_folder_button=False, width=self.app.button_w)

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
        self.done_projecting = False
        self.target_image = None
        self.target_texture = None
        self.projection_process = mp.Process(target=run_projection, args=(self.queue, self.reply),
                                       daemon=True)
        self.save_path_dialog = BrowseWidget(self, "Save Path", os.path.abspath(os.getcwd()), [""], multiple=False,
                                             traverse_folders=False, add_folder_button=True, width=self.app.button_w)
        self.projected_texture = None



        for pkl in os.listdir("./models"):
            if pkl.endswith(".pkl"):
                self.models.append(os.path.join(os.getcwd(),"models",pkl))

    @imgui_utils.scoped_by_object_id
    def __call__(self):
        if self.reply.qsize() > 0:
            self.message, projected_img, self.done_projecting, self.done_recording = self.reply.get()
            while self.reply.qsize() > 0:
                self.message, projected_img, self.done_projecting, self.done_recording = self.reply.get()

            if projected_img is not None:
                self.projected_img = projected_img

            if self.projected_img is not None:
                if self.projected_texture is None or not self.projected_texture.is_compatible(image=self.projected_img):
                    self.projected_texture = gl_utils.Texture(image=self.projected_img, width=self.projected_img.shape[1], height=self.projected_img.shape[0], channels=self.projected_img.shape[2])
                else:
                    self.projected_texture.update(self.projected_img)

        _, self.network_path = imgui_utils.input_text('##projection_network', self.network_path, 1024,
                                                        flags=(
                                                                    imgui.INPUT_TEXT_AUTO_SELECT_ALL | imgui.INPUT_TEXT_ENTER_RETURNS_TRUE),
                                                        width=-self.app.button_w - self.app.spacing,
                                                        help_text='<PATH> | <URL> | <RUN_DIR> | <RUN_ID> | <RUN_ID>/<KIMG>.pkl')
        imgui.same_line()
        if imgui_utils.button(f'Models ##projection', enabled=len(self.models) > 0, width = self.app.button_w):
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


        joined = '\n'.join(self.target_fname)
        imgui_utils.input_text("##projection_file", joined, 1024, flags=imgui.INPUT_TEXT_READ_ONLY,
                               width=- (self.app.button_w + self.app.spacing), help_text="Input Files")
        imgui.same_line()
        _clicked, target_pth = self.file_dialog(self.app.button_w) # should have argument to only allow single file
        if _clicked:
            self.target_fname = target_pth

        _changed, self.target_text = imgui_utils.input_text('Target Text##target_text', self.target_text, 1024, flags=imgui.INPUT_TEXT_AUTO_SELECT_ALL, help_text='Text to be projected', width=-self.app.button_w - self.app.spacing,)
        _changed, self.outdir = imgui_utils.input_text('##outdir', self.outdir, 1024, flags=imgui.INPUT_TEXT_AUTO_SELECT_ALL, help_text='Directory to save results', width=-self.app.button_w - self.app.spacing,)
        imgui.same_line()
        _clicked, save_path = self.save_path_dialog()
        if _clicked:
            if len(save_path) > 0:
                self.outdir = save_path[0]
                print(self.outdir)
            else:
                self.outdir = ""
                print("No path selected")

        _changed, self.save_video = imgui.checkbox('Save Video##save_video', self.save_video)
        with imgui_utils.item_width(-(self.app.button_w + self.app.spacing)):
            _changed, self.seed = imgui.input_int('Seed##seed', self.seed)
            _changed, self.lr = imgui.input_float('Learning Rate##lr', self.lr)
            _changed, self.steps = imgui.input_int('Steps##steps', self.steps)
        _changed, self.use_vgg = imgui.checkbox('Use VGG##use_vgg', self.use_vgg)
        if imgui.get_item_rect_max()[0] < imgui.get_window_content_region_max()[0] + self.app.content_width // 4:
            imgui.same_line()
        _changed, self.use_clip = imgui.checkbox('Use CLIP##use_clip', self.use_clip)
        if imgui.get_item_rect_max()[0] < imgui.get_window_content_region_max()[0] + self.app.content_width // 4:
            imgui.same_line()
        _changed, self.use_pixel = imgui.checkbox('Use Pixel##use_pixel', self.use_pixel)
        if imgui.get_item_rect_max()[0] < imgui.get_window_content_region_max()[0] + self.app.content_width // 4:
            imgui.same_line()
        _changed, self.use_penalty = imgui.checkbox('Use Penalty##use_penalty', self.use_penalty)
        if imgui.get_item_rect_max()[0] < imgui.get_window_content_region_max()[0] + self.app.content_width // 4:
            imgui.same_line()
        _changed, self.use_center = imgui.checkbox('Use Center##use_center', self.use_center)

        if imgui_utils.button("Project", width=imgui.get_content_region_available_width(), enabled=((self.network_path != "" or self.target_fname != "") and self.outdir != "")):
            imgui.open_popup('Project')
            self.done_projecting = False
            self.done_recording = False

            self.queue = mp.Queue()
            self.reply = mp.Queue()

            self.projection_process = mp.Process(target=run_projection, args=(self.queue, self.reply),
                                                 daemon=True)
            self.projection_process.start()
            self.queue.put([self.network_path, self.target_fname[0] if self.target_fname != "" else None, self.target_text if self.target_text != "" else None, self.initial_latent, self.outdir, self.save_video, self.seed, self.lr, self.steps, self.use_vgg, self.use_clip, self.use_pixel, self.use_penalty, self.use_center, self.use_kmeans])


        # set modal popup size to be 1/2 of the window size
        imgui.set_next_window_size( self.menu.app.content_width // 2, (self.menu.app.content_height // 4) * 3, imgui.ONCE)
        if imgui.begin_popup_modal('Project')[0]:
            imgui.text("Projecting...")
            if self.target_fname != "":
                self.target_image = PIL.Image.open(self.target_fname[0]).convert('RGB').filter(ImageFilter.SHARPEN)
                self.target_image = np.array(self.target_image, dtype=np.uint8)
                self.target_image = torch.tensor(self.target_image, device="cpu")
                self.target_texture = gl_utils.Texture(image=self.target_image, width=self.target_image.shape[1], height=self.target_image.shape[0], channels=self.target_image.shape[2])
            if self.message != "":
                imgui.text(self.message)
            if self.projected_img is not None and self.projected_texture is not None:
                if self.target_image is not None:
                    ratio = self.target_image.shape[1] / self.target_image.shape[0]
                else:
                    ratio = self.projected_img.shape[1] / self.projected_img.shape[0]
                shape = ratio*imgui.get_content_region_available_width()//2, imgui.get_content_region_available_width()//2
                imgui.image(self.projected_texture.gl_id, *shape)

                if self.target_texture is not None:
                    imgui.same_line()
                    imgui.image(self.target_texture.gl_id, *shape)

            enabled = self.projected_img is not None

            # set label to Done if self.done_projecting is True and not self.save_video or self.done_recording is True and self.save_video else Cancel
            label = "Done" if (self.done_projecting and not self.save_video) or (self.done_recording and self.save_video) else "Cancel"
            if imgui_utils.button(label, enabled=enabled):
                self.queue.put(True)

            if not self.save_video:
                if self.done_projecting:
                    imgui.close_current_popup()
                    self.done_projecting = False
                    self.projected_img = None
                    self.message = ""

                    self.projection_process.join()
                    self.projection_process.terminate()
                    self.projection_process.close()
            else:
                if self.done_recording:
                    imgui.close_current_popup()
                    self.done_recording = False
                    self.projected_img = None
                    self.message = ""

                    self.projection_process.join()
                    self.projection_process.terminate()
                    self.projection_process.close()
            imgui.end_popup()

