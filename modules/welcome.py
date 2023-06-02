# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import cv2
import imgui

from utils.gui_utils import imgui_utils, gl_utils
from modules.autolume_live import States

#----------------------------------------------------------------------------

class Welcome:
    def __init__(self, app):
        self.app = app
        self.splash = cv2.imread("assets/splashscreen.jpg")
        print(self.splash.shape)
        self.splash = cv2.cvtColor(self.splash, cv2.COLOR_BGR2RGB)
        self.splash_texture = gl_utils.Texture(image=self.splash, width=self.splash.shape[1], height=self.splash.shape[0], channels=self.splash.shape[2])

    @imgui_utils.scoped_by_object_id
    def __call__(self):
        # Begin control pane.

