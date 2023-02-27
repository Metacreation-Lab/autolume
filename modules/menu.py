# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import click
import imgui

from utils.gui_utils import imgui_window
from utils.gui_utils import imgui_utils
from modules.pca_module import PCA_Module
from modules.training_module import TrainingModule
from modules.compress_module import CompressModule
from modules.super_res_module import SuperResModule
from modules.autolume_live import States

#----------------------------------------------------------------------------
class Menu:
    def __init__(self, app):
        self.app = app
        self.pca = PCA_Module(self)
        self.training = TrainingModule(self)
        self.compress = CompressModule(self)
        self.super_res = SuperResModule(self)

    def __call__(self):

        # Make train and compress buttons call function as subprocess to avoid blocking
        # Begin control pane.
        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(self.app.content_width//4, (self.app.content_height * 4) //5)
        imgui.begin('Train##Menu', closable=False, flags=(imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE))
        imgui.text("Train")
        self.training()
        imgui.end()

        imgui.set_next_window_position(self.app.content_width // 4, 0)
        imgui.set_next_window_size(self.app.content_width // 4, (self.app.content_height * 4) // 5)
        imgui.begin('Compress##Menu', closable=False, flags=(imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE))
        imgui.text("Compress")
        self.compress()
        imgui.end()

        imgui.set_next_window_position((3 * self.app.content_width)// 4, 0)
        imgui.set_next_window_size(self.app.content_width // 4, (self.app.content_height * 4) // 5)
        imgui.begin('GANSPACE##Menu', closable=False, flags=(imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE))
        imgui.text("GANSPACE")
        self.pca()
        imgui.end()

        imgui.set_next_window_position((3 * self.app.content_width)// 4, 0)
        imgui.set_next_window_size(self.app.content_width // 3, (self.app.content_height * 4) // 5)
        imgui.begin('Super-res##Menu', closable=False, flags=(imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE))
        imgui.text("Super-res")
        self.super_res()
        imgui.end()

        imgui.set_next_window_position(0,(self.app.content_height * 4) // 5)
        imgui.set_next_window_size(self.app.content_width, (self.app.content_height * 4) // 5)
        imgui.begin('Render##Menu', closable=False, flags=(imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE))
        imgui.text("Renderer")
        if imgui_utils.button("START", width=self.app.button_w):
            self.app.start_renderer()
        imgui.end()


