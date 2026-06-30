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
import webbrowser

from assets import RED, GREEN, OPAQUEGREEN, HOVERGREEN
from utils.gui_utils import imgui_window, gl_utils
from utils.gui_utils import imgui_utils
from modules.pca_module import PCA_Module
from modules.training_module import TrainingModule
from modules.network_surgery import SurgeryModule
from modules.projection_module import ProjectionModule
from modules.network_mixing import MixingModule

from modules.super_res_module import SuperResModule

from utils.model_dir import models_dir
from utils.resource_paths import resource_path
from widgets.help_icon_widget import DOCS_BASE_URL
from widgets.model_download_widget import ModelDownloadWidget
#----------------------------------------------------------------------------
def draw_icon_text_button(app, icon_texture, label, width, height):
    """Self-drawn button: icon on the left + text label, with a hover
    background. Returns True when clicked. Mirrors the menu banner styling."""
    pos = imgui.get_cursor_screen_pos()
    clicked = imgui.invisible_button(f"##{label}_btn", width, height)
    hovered = imgui.is_item_hovered()
    draw_list = imgui.get_window_draw_list()
    if hovered:
        draw_list.add_rect_filled(
            pos[0], pos[1], pos[0] + width, pos[1] + height,
            imgui.get_color_u32_rgba(*HOVERGREEN))
    icon_size = int(app.font_size * 0.8)
    pad_x = app.font_size * 0.4
    gap = app.font_size * 0.35
    icon_y = pos[1] + (height - icon_size) / 2
    icon_x = pos[0] + pad_x
    draw_list.add_image(
        icon_texture.gl_id,
        (icon_x, icon_y),
        (icon_x + icon_size, icon_y + icon_size),
        col=imgui.get_color_u32_rgba(1, 1, 1, 1))
    text_x = icon_x + icon_size + gap
    text_y = pos[1] + (height - app.font_size) / 2
    draw_list.add_text(
        text_x, text_y, imgui.get_color_u32_rgba(1, 1, 1, 1), label)
    return clicked
#----------------------------------------------------------------------------
class Menu:
    def __init__(self, app):
        self.app = app
        self.model_downloader = ModelDownloadWidget(app, models_dir())
        self.pca = PCA_Module(self)
        self.training = TrainingModule(self)
        # self.network_surgery = SurgeryModule(self)
        self.projection = ProjectionModule(self)
        self.super_res = SuperResModule(self)
        self.mixing_module = MixingModule(self)
        self.logo = cv2.imread(str(resource_path("assets", "Autolume-logo.png")), cv2.IMREAD_UNCHANGED)
        self.logo_texture = gl_utils.Texture(image=self.logo, width=self.logo.shape[1], height=self.logo.shape[0], channels=self.logo.shape[2])

        self.metacreation = cv2.imread(str(resource_path("assets", "metalogo.png")), cv2.IMREAD_UNCHANGED)
        self.metacreation_texture = gl_utils.Texture(image=self.metacreation, width=self.metacreation.shape[1], height=self.metacreation.shape[0], channels=self.metacreation.shape[2])

        self.cog = cv2.imread(str(resource_path("assets", "cog.png")), cv2.IMREAD_UNCHANGED)
        self.cog_texture = gl_utils.Texture(image=self.cog, width=self.cog.shape[1], height=self.cog.shape[0], channels=self.cog.shape[2])

        self.web = cv2.imread(str(resource_path("assets", "web.png")), cv2.IMREAD_UNCHANGED)
        self.web_texture = gl_utils.Texture(image=self.web, width=self.web.shape[1], height=self.web.shape[0], channels=self.web.shape[2])

    def __call__(self):
        # Scale with the UI font, calibrated to 50px at font 23.
        self.menu_height = round(self.app.font_size * 2.2)
        imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND,*OPAQUEGREEN)
        imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND_ACTIVE, *OPAQUEGREEN)
        imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND_COLLAPSED, *OPAQUEGREEN)

        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(self.app.content_width, self.menu_height)
        # temporarily update the window style to remove annoying gaps around the banner
        imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (0, 0))
        imgui.push_style_var(imgui.STYLE_WINDOW_BORDERSIZE, 0)
        imgui.begin('##Menu', closable=False, flags=(
                imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS | imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_SCROLLBAR))
        imgui.pop_style_var(2)

        imgui.get_window_draw_list().add_rect_filled(0, 0, self.app.content_width, self.menu_height, imgui.get_color_u32_rgba(*RED))

        logo_height = int(self.menu_height * 0.6)
        logo_width = int(logo_height * (self.logo.shape[1] / self.logo.shape[0]))
        imgui.set_cursor_pos_y((self.menu_height - logo_height) / 2)
        imgui.set_cursor_pos_x(20)
        imgui.image(self.logo_texture.gl_id, logo_width, logo_height)

        metacreation_height = logo_height
        metacreation_width = int(metacreation_height * (self.metacreation.shape[1] / self.metacreation.shape[0]))

        button_height = int(self.menu_height * 0.6)
        button_y = (self.menu_height - button_height) / 2
        pad_x = self.app.font_size * 0.4

        def icon_text_width(label):
            icon_size = int(self.app.font_size * 0.8)
            gap = self.app.font_size * 0.35
            return pad_x + icon_size + gap + imgui.calc_text_size(label).x + pad_x

        doc_button_width = icon_text_width("Documentation")
        settings_button_width = icon_text_width("Settings")
        settings_button_x = self.app.content_width - (metacreation_width + doc_button_width + settings_button_width + 60)
        imgui.same_line(settings_button_x)
        imgui.set_cursor_pos_y(button_y)
        if draw_icon_text_button(self.app, self.cog_texture, "Settings", settings_button_width, button_height):
            self.app.open_settings()

        imgui.same_line(self.app.content_width - (metacreation_width + doc_button_width + 40))
        imgui.set_cursor_pos_y(button_y)
        if draw_icon_text_button(self.app, self.web_texture, "Documentation", doc_button_width, button_height):
            webbrowser.open(DOCS_BASE_URL)

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

        self.model_downloader()

        imgui.pop_style_color(3)




