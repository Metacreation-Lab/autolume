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
from modules.autolume_live import States

#----------------------------------------------------------------------------

class Welcome:
    def __init__(self, app):
        self.app = app

    @imgui_utils.scoped_by_object_id
    def __call__(self):
        # Begin control pane.
        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(self.app.content_width, self.app.content_height)
        imgui.begin('##welcome', closable=False, flags=(imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE))

        draw_list = imgui.get_window_draw_list()
        draw_list.add_text(self.app.content_width//2, self.app.content_height//2, imgui.get_color_u32_rgba(1, 1, 1, 1), "Autolume-Live!")
        if imgui_utils.button("START", self.app.button_w):
            self.app.state = States.MENU

        if self.app.state == States.MENU:
            self.app.open_menu()


        imgui.end()
