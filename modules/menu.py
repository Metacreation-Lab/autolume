# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import click
import cv2
import imgui
import numpy as np

from assets import RED, GREEN, OPAQUEGREEN
from utils.gui_utils import imgui_window, gl_utils
from utils.gui_utils import imgui_utils
from modules.pca_module import PCA_Module
from modules.training_module import TrainingModule
from modules.compress_module import CompressModule
from modules.network_surgery import SurgeryModule
from modules.projection_module import ProjectionModule
from modules.network_mixing import MixingModule

from modules.super_res_module import SuperResModule
#----------------------------------------------------------------------------
class Menu:
    def __init__(self, app):
        self.app = app
        self.show_help = False
        self.pca = PCA_Module(self)
        self.training = TrainingModule(self)
        self.compress = CompressModule(self)
        # self.network_surgery = SurgeryModule(self)
        self.projection = ProjectionModule(self)
        self.super_res = SuperResModule(self)
        self.mixing_module = MixingModule(self)
        self.logo = cv2.imread("assets/Autolume-logo.png", cv2.IMREAD_UNCHANGED)
        self.logo_texture = gl_utils.Texture(image=self.logo, width=self.logo.shape[1], height=self.logo.shape[0], channels=self.logo.shape[2])

        self.metacreation = cv2.imread("assets/metalogo.png", cv2.IMREAD_UNCHANGED)
        self.metacreation_texture = gl_utils.Texture(image=self.metacreation, width=self.metacreation.shape[1], height=self.metacreation.shape[0], channels=self.metacreation.shape[2])
        
        self.menu_height = 50

    def __call__(self):
        imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND,*OPAQUEGREEN)
        imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND_ACTIVE, *OPAQUEGREEN)
        imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND_COLLAPSED, *OPAQUEGREEN)

        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(self.app.content_width, self.menu_height)
        imgui.begin('##Menu', closable=False, flags=(
                imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS | imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_SCROLLBAR))

        imgui.get_window_draw_list().add_rect_filled(0, 0, self.app.content_width, self.menu_height, imgui.get_color_u32_rgba(*RED))

        logo_height = int(self.menu_height * 0.6)
        logo_width = int(logo_height * (self.logo.shape[1] / self.logo.shape[0]))
        imgui.set_cursor_pos_y((self.menu_height - logo_height) / 2)
        imgui.set_cursor_pos_x(20)
        imgui.image(self.logo_texture.gl_id, logo_width, logo_height)

        metacreation_height = logo_height
        metacreation_width = int(metacreation_height * (self.metacreation.shape[1] / self.metacreation.shape[0]))
        
        help_button_width = 150
        help_button_height = int(self.menu_height * 0.6)  
        help_button_y = (self.menu_height - help_button_height) / 2  
        
        imgui.same_line(self.app.content_width - (metacreation_width + help_button_width + 40))
        imgui.set_cursor_pos_y(help_button_y)
        
        imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, help_button_height * 0.2))
        if imgui_utils.button("Help On" if not self.show_help else "Help Off", 
                            width=help_button_width):
            self.show_help = not self.show_help
        imgui.pop_style_var()

        imgui.same_line(self.app.content_width - (metacreation_width + 20))
        imgui.set_cursor_pos_y((self.menu_height - metacreation_height) / 2)
        imgui.image(self.metacreation_texture.gl_id, metacreation_width, metacreation_height)
        imgui.end()

        main_window_y = self.menu_height
        main_window_height = int((self.app.content_height * 0.75) - self.menu_height)
        
        imgui.set_next_window_position(0, main_window_y)
        imgui.set_next_window_size(self.app.content_width//4, main_window_height)
        imgui.begin('Data Preparation & Training##Menu', closable=False, flags=(imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS))
        self.training()
        imgui.separator()
        imgui.end()

        imgui.set_next_window_position(self.app.content_width // 4, main_window_y)
        imgui.set_next_window_size(self.app.content_width // 4, main_window_height)
        imgui.begin('Projection##Menu', closable=False, flags=(imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS))
        self.projection()
        imgui.end()

        imgui.set_next_window_position((2 * self.app.content_width)// 4, main_window_y)
        imgui.set_next_window_size(self.app.content_width // 4, main_window_height)
        imgui.begin('Feature Extractor##Menu', closable=False, flags=(imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS))
        self.pca()
        imgui.end()

        imgui.set_next_window_position((3 * self.app.content_width)// 4, main_window_y)
        imgui.set_next_window_size(self.app.content_width // 4, main_window_height)
        imgui.begin('Super Resolution##Menu', closable=False, flags=(imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS))
        self.super_res()
        imgui.end()

        bottom_y = main_window_y + main_window_height
        bottom_height = self.app.content_height - bottom_y

        imgui.set_next_window_position(0, bottom_y)
        imgui.set_next_window_size(self.app.content_width // 2, bottom_height)
        imgui.begin('Model Mixing##Menu', closable=False, flags=(imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE| imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS))
        self.mixing_module()
        imgui.end()

        imgui.set_next_window_position(self.app.content_width//2, bottom_y)
        imgui.set_next_window_size(self.app.content_width//2, bottom_height)
        imgui.begin('Render##Menu', closable=False, flags=(imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE| imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS))
        imgui.text("Jump into Autolume-Live")
        if imgui_utils.button("START", width=self.app.button_w):
            self.app.start_renderer()
        imgui.end()

        imgui.pop_style_color(3)




