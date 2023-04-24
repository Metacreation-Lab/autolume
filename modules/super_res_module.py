import os
import imgui

from modules.filedialog import FileDialog
from utils.gui_utils import imgui_utils
from super_res import main as super_res_main
#from super_res import base_args
from dnnlib import EasyDict

args = EasyDict(result_path="", input_path=[""], model_type="Balance",
                outscale=3, width=4096, height=4096, sharpen_scale=1, scale_mode=0)
scale_factor = ['1', '2', '3', '4', '5', '6', '7', '8']


class SuperResModule:
    def __init__(self, menu):
        self.result_path=args.result_path
        self.input_path = args.input_path
        self.models = ['Quality','Balance','Fast']
        self.model_selected = 0
        self.model_type = self.models[self.model_selected]
        self.width = args.width
        self.height = args.height
        self.out_scale = args.outscale
        self.sharpen = args.sharpen_scale
        self.menu = menu
        self.app = menu.app
        self.file_dialog = FileDialog(self, "Videos and Images", os.path.abspath(os.getcwd()), ["*", ".mp4", ".avi", ".jpg", ".png", ".jpeg", ".bmp"])
        self.scale_mode = 0

    @imgui_utils.scoped_by_object_id
    def __call__(self):
        joined = '\n'.join(self.input_path)
        imgui_utils.input_text("##SRINPUT", joined, 1024, flags=imgui.INPUT_TEXT_READ_ONLY, width=-self.app.button_w * 3, help_text="Input Files")
        imgui.same_line()
        _clicked, input = self.file_dialog()
        if _clicked:
            self.input_path = input
            print(self.input_path)
        _, self.result_path = imgui.input_text("Destination Folder", self.result_path, 1024)
        self.models = ['Quality','Balance','Fast']
        if len(self.models) > 0:
            _, self.model_selected = imgui.combo("Model", self.model_selected, self.models)
            self.model_type = self.models[self.model_selected]
        clicked, self.scale_mode = imgui.combo("Scale Mode", self.scale_mode, ["Custom", "Scale"])
        if clicked:
            print(self.scale_mode)
        if self.scale_mode:
            _, self.out_scale = imgui.combo("Scale Factor", self.out_scale, scale_factor)
        else:
            _, self.height = imgui.input_int("Height", self.height)
            _, self.width = imgui.input_int("Width", self.width)
        _, self.sharpen = imgui.input_int("Sharpening Factor", self.sharpen)
        if self.sharpen<1:
            self.sharpen=1
        if imgui.is_item_hovered():
            imgui.set_tooltip("Additional sharpening performed after super resolution")
        try:
            if imgui.button("Super Resolution"):
                print("Super Resolution")
                args.result_path = self.result_path
                args.input_path = self.input_path
                args.model_type = self.model_type
                args.outscale = self.out_scale + 1
                args.out_height = self.height
                args.out_width = self.width
                args.sharpen_scale = self.sharpen
                args.scale_mode = self.scale_mode

                super_res_main(args)
        except Exception as e:
            print("SRR ERROR", e)
