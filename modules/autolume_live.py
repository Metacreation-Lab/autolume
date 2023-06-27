import gc
import os

import cv2
import imgui

from utils.gui_utils import imgui_window, gl_utils
from enum import IntEnum
class States(IntEnum):
    ERROR = -2
    CLOSE = -1
    WELCOME = 0
    MENU = 1
    RENDER = 2
    SPLASH = 3



class Autolume(imgui_window.ImguiWindow):
    # State handler that greets user, shows menu (allowing for training, compression, ganspace), and renders
    def __init__(self):
        super().__init__(title='Autolume-Live', window_width=3840, window_height=2160)

        self.state = 0
        self.running = True
        self.menu = None
        self.viz = None
        self.render_loop = None
        self.pkls = []
        self.splash_delay = 0

        self.splash = cv2.imread("assets/splashscreen.jpg", cv2.IMREAD_UNCHANGED)
        self.splash = cv2.cvtColor(self.splash, cv2.COLOR_BGRA2RGBA)
        self.splash_texture = gl_utils.Texture(image=self.splash, width=self.splash.shape[1],
                                               height=self.splash.shape[0], channels=self.splash.shape[2])

        # Initialize window.
        self.label_w = 0
        self.button_w = 0
        self.set_position(0, 0)
        self._adjust_font_size()
        self.skip_frame()  # Layout may change after first frame.

    def _adjust_font_size(self):
        old = self.font_size
        self.set_font_size(min(self.content_width / 120, self.content_height / 60))
        if self.font_size != old:
            self.skip_frame() # Layout changed.

    def close(self):
        super().close()


    def open_menu(self):
        from modules.menu import Menu
        print("opening Menu")
        # Initialize window.
        self.menu = Menu(self)

    def start_renderer(self):
        from modules.renderloop import AsyncRenderer
        from modules.visualizer import Visualizer

        self.render_loop = AsyncRenderer()
        self.viz = Visualizer(self, self.render_loop)

        # List pickles.
        if len(self.pkls) > 0:
            for pkl in self.pkls:
                self.viz.add_recent_pickle(pkl)
            self.viz.load_pickle(self.pkls[0])
        self.state = States.RENDER
        self.menu = None
        gc.collect()

    def set_visible_menu(self):
        from modules.menu import Menu
        print("setting visible menu ------------------------")
        self.state = States.MENU
        if self.viz is not None:
            self.viz.close()
            self.viz = None
        if self.render_loop is not None:
            self.render_loop.close()
            self.render_loop = None
        gc.collect()
        self.menu = Menu(self)

    def draw_frame(self):

        if self.state == States.SPLASH:
            self.set_window_size(self.splash_texture.width//2, self.splash_texture.height//2)
            self.hide_title_bar()

            # set size of app window/ frame to self.splash_texture.width //2 , self.splash_texture.height //2


        self.begin_frame()
        self.button_w = self.font_size * 5
        self.label_w = round(self.font_size * 4.5)

        if self.state == States.SPLASH:
            imgui.set_next_window_position(0, 0)
            imgui.set_next_window_size(self.content_width, self.content_height)
            imgui.begin('##welcome', closable=False,
                        flags=(imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_SCROLLBAR))
            imgui.image(self.splash_texture.gl_id, self.content_width, self.content_height)
            imgui.end()
            self.splash_delay -= 1
            if self.splash_delay <= 0:
                self.set_visible_menu()
                self.set_window_size(3840,2160)
                self.show_title_bar()


        if self.state == States.WELCOME:
            imgui.set_next_window_position(0, 0)
            imgui.set_next_window_size(self.content_width, self.content_height)
            imgui.begin('##welcome', closable=False,
                        flags=(imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE| imgui.WINDOW_NO_SCROLLBAR))
            imgui.image(self.splash_texture.gl_id, self.content_width, self.content_height)
            imgui.end()
            self.state = States.SPLASH
            self.splash_delay = 30



        if self.state == States.MENU:
            if self.menu is None:
                self.state = States.ERROR
            else:
                self.menu()

        if self.state == States.RENDER:
            if self.viz is None or self.render_loop is None:
                self.state = States.ERROR
            else:
                self.viz()

        if self.state == States.CLOSE or self.state == States.ERROR:
            self.stop()

        self._adjust_font_size()
        self.end_frame()

