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

    def __call__(self):
        imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND,*OPAQUEGREEN)
        imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND_ACTIVE, *OPAQUEGREEN)
        imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND_COLLAPSED, *OPAQUEGREEN)

        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(self.app.content_width, 50)
        imgui.begin('##Menu', closable=False, flags=(
                imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS | imgui.WINDOW_NO_TITLE_BAR))
        # set red background
        imgui.get_window_draw_list().add_rect_filled(0, 0, self.app.content_width + (self.app.spacing*2), 50,
                                                     imgui.get_color_u32_rgba(*RED))

        #calculate logo shape ratio
        logo_ratio = self.logo.shape[1] / self.logo.shape[0]
        # logo with height of 30px centered in y axis
        imgui.set_cursor_pos_y(25 - (30/2))
        imgui.set_cursor_pos_x(20)
        imgui.image(self.logo_texture.gl_id, 30 * logo_ratio, 30)

        #calculate metacreation shape ratio
        metacreation_ratio = self.metacreation.shape[1] / self.metacreation.shape[0]
        # metacreation with height of 30px centered in y axis
        imgui.same_line(self.app.content_width - ((30 * metacreation_ratio) + 20))
        imgui.set_cursor_pos_y(25 - (30/2))
        imgui.image(self.metacreation_texture.gl_id, 30 * metacreation_ratio, 30, tint_color=(1, 1, 1, 1))
        imgui.end()

        # Begin control pane.
        imgui.set_next_window_position(0, 50)
        imgui.set_next_window_size(self.app.content_width//4, (self.app.content_height * 4) //5 - 50)
        imgui.begin('Train##Menu', closable=False, flags=(imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS))

        self.training()
        imgui.separator()
        #self.compress()
        imgui.end()

        imgui.set_next_window_position(self.app.content_width // 4, 50)
        imgui.set_next_window_size(self.app.content_width // 4, (self.app.content_height * 4) // 5 - 50)
        imgui.begin('Projection##Menu', closable=False, flags=(imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS))
        imgui.text("Find the closest match to your image")

        self.projection()
        imgui.end()

        imgui.set_next_window_position((2 * self.app.content_width)// 4, 50)
        imgui.set_next_window_size(self.app.content_width // 4, (self.app.content_height * 4) // 5 - 50)
        imgui.begin('Feature Extractor##Menu', closable=False, flags=(imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS))
        imgui.text("Extract Meaningful Directions from a Model")
        self.pca()
        imgui.end()

        imgui.set_next_window_position((3 * self.app.content_width)// 4, 50)
        imgui.set_next_window_size(self.app.content_width // 4, (self.app.content_height * 4) // 5 - 50)
        imgui.begin('Super Resolution##Menu', closable=False, flags=(imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS))
        imgui.text("Use AI to upscale your images and videos")
        self.super_res()
        imgui.end()

        imgui.set_next_window_position(0, (self.app.content_height * 3) // 4 + 50)
        imgui.set_next_window_size(self.app.content_width // 2, (self.app.content_height * 3) // 4 - 50)
        imgui.begin('Model Mixing##Menu', closable=False, flags=(imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE| imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS))
        imgui.text("Combine two models into one")
        self.mixing_module()
        imgui.end()

        imgui.set_next_window_position(self.app.content_width//2,(self.app.content_height * 3) // 4 + 50)
        imgui.set_next_window_size(self.app.content_width//2, (self.app.content_height * 3) // 4 - 50)
        imgui.begin('Render##Menu', closable=False, flags=(imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE| imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS))
        imgui.text("Jump into Autolume-Live")
        if imgui_utils.button("START", width=self.app.button_w):
            self.app.start_renderer()
        imgui.end()

        imgui.pop_style_color(3)




