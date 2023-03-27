import os
import imgui

from utils.gui_utils import imgui_utils
from super_res import main as super_res_main
from super_res import base_args
from dnnlib import EasyDict

args = EasyDict(result_path="", input_path="", model_type="Balance",
                outscale=4, width=4096, height=4096, sharpen_scale=4)
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

        self.scale_mode = 0

    @imgui_utils.scoped_by_object_id
    def __call__(self):
        _, self.input_path = imgui.input_text("Source Folder", self.input_path, 1024)
        _, self.result_path = imgui.input_text("Destination Folder", self.result_path, 1024)
        self.models = ['Quality','Balance','Fast']
        if len(self.models) > 0:
            _, self.model_selected = imgui.combo("Model", self.model_selected, self.models)

        clicked, self.scale_mode = imgui.combo("Scale Mode", self.scale_mode, ["Custom", "Scale"])
        if clicked:
            print(self.scale_mode)
        if self.scale_mode:
            _, self.out_scale = imgui.combo("Scale Factor", self.out_scale, scale_factor)
        else:
            _, self.height = imgui.input_int("Height", self.height)
            _, self.width = imgui.input_int("Width", self.width)
        _, self.sharpen = imgui.input_int("Sharpening Factor", self.sharpen)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Additional sharpening performed after super resolution")
        try:
            if imgui.button("Super Resolution"):
                print("Super Resolution")
                args.result_path = self.result_path
                args.input_path = self.input_path
                args.model_type = self.model_type
                args.outscale = self.out_scale
                args.out_height = self.height
                args.out_width = self.width
                args.sharpen_scale = self.sharpen

                super_res_main(args)
        except Exception as e:
            print(e)
