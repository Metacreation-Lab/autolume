import os
import imgui

from utils.gui_utils import imgui_utils
from super-res import main as super_res_main
from super-res import base_args

parser=base_args()
args = parser.parse_args()
scale_factor=[1,2,3,4,5,6,7,8]

class SuperResModule:
    def __init__(self, menu):
        self.menu = menu

    @imgui_utils.scoped_by_object_id
    def __call__(self):
        imgui.text("Super_Resolution Module")

        _, self.result_path = imgui.input_text("Save Path", self.save_path, 1024)
        _, self.input_path = imgui.input_text("File Path", self.file_path, 1024)
        _, self.model_path = imgui.input_text("Model Path", self.model_path, 1024)
        imgui.same_line()

        _, self.out_scale = imgui.combo("Out_scale", self.out_scale, scale_factor) 
        _, self.height = imgui.input_int("Height", self.height)
        _, self.width = imgui.input_int("Width", self.width)
        _, self.sharpen = imgui.input_int("Sharpen", self.sharpen)


        if imgui.button("Super Resolution"):
            print("Super Resolution")
            args.result_path=self.result_path
            args.input_path=self.input_path
            args.model_path=self.model_path
            args.out_scale=self.out_scale
            args.height=self.height
            args.width=self.width
            args.sharpen=self.sharpen
            
            super_res_main(args)