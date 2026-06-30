import os
import sys

import cv2
import imgui
import time
import gc

from utils.gui_utils import imgui_window, gl_utils
from utils.resource_paths import get_version, resource_path
from enum import IntEnum
class States(IntEnum):
    ERROR = -2
    CLOSE = -1
    WELCOME = 0
    MENU = 1
    RENDER = 2
    PREPROCESSING = 4



class Autolume(imgui_window.ImguiWindow):
    # State handler that greets user, shows menu (allowing for training, compression, ganspace), and renders

    DEFAULT_FPS_LIMIT = 60

    # Number of frames to hold the startup splash screen before opening the menu.
    SPLASH_FRAMES = 30

    # Standard-DPI UI font size. Divided by the display's DPI scale so the UI is a
    # constant, DPI-appropriate size that does NOT change when the window resizes.
    # scale 1 (Windows/non-retina) -> 23 (clamps to the max key);
    # scale 2 (retina)             -> 11.5 -> snaps to the min key 14.
    BASE_FONT_SIZE = 23

    def __init__(self):
        super().__init__(title=f'Autolume-Live v{get_version()}', window_width=3840, window_height=2160)

        self.state = 0
        self.running = True
        self.menu = None
        self.viz = None
        self.render_loop = None
        self.pkls = []
        self.splash_delay = None
        self.data_preprocessing = None

        self.splash = cv2.imread(str(resource_path("assets", "splashscreen.jpg")), cv2.IMREAD_UNCHANGED)
        self.splash = cv2.cvtColor(self.splash, cv2.COLOR_BGRA2RGBA)
        self.splash_texture = gl_utils.Texture(image=self.splash, width=self.splash.shape[1],
                                               height=self.splash.shape[0], channels=self.splash.shape[2])

        # Initialize window.
        self.set_fps_limit(self.DEFAULT_FPS_LIMIT)
        self.label_w = 0
        self.button_w = 0
        self.set_position(0, 0)
        self._adjust_font_size()
        self.skip_frame()  # Layout may change after first frame.

    def _adjust_font_size(self):
        old = self.font_size
        self.set_font_size(self.BASE_FONT_SIZE / self._font_dpi_scale)
        if self.font_size != old:
            self.skip_frame() # Layout changed.

    def close(self):
        if self.data_preprocessing is not None:
            self.data_preprocessing.cleanup()
            self.data_preprocessing = None
        
        if self.menu is not None and hasattr(self.menu, 'training'):
            self.menu.training.cleanup_dataset_process()
        
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
        self.set_fps_limit(self.DEFAULT_FPS_LIMIT)
        if self.viz is not None:
            self.viz.close()
            self.viz = None
        if self.render_loop is not None:
            self.render_loop.close()
            self.render_loop = None
        if self.data_preprocessing is not None:
            self.data_preprocessing.cleanup() 
            self.data_preprocessing = None
        gc.collect()
        
        # Small delay to ensure OpenGL resources are properly released
        time.sleep(0.05)
        
        self.menu = Menu(self)

    # Preprocessing window
    def start_preprocessing(self):
        from modules.preprocessing_module import DataPreprocessing
        self.data_preprocessing = DataPreprocessing(self)
        self.data_preprocessing()
        self.state = States.PREPROCESSING

    def _draw_splash(self):
        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(self.content_width, self.content_height)
        # No padding or border so the splash image fills the window edge to edge.
        imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (0, 0))
        imgui.push_style_var(imgui.STYLE_WINDOW_BORDERSIZE, 0)
        imgui.begin('##welcome', closable=False,
                    flags=(imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_SCROLLBAR))
        imgui.image(self.splash_texture.gl_id, self.content_width, self.content_height)
        imgui.end()
        imgui.pop_style_var(2)

    def draw_frame(self):

        if self.state == States.WELCOME and self.splash_delay is None:
            # First splash frame: size the still-hidden window to the splash image and
            # center it. Must run before begin_frame() so the frame renders at the fitted
            # size. Skip borderless on Linux: Wayland can't restore decorations later.
            if sys.platform != 'linux':
                self.hide_title_bar()
            splash_w = self.splash_texture.width // 2
            splash_h = self.splash_texture.height // 2
            fit = min(0.8 * self.monitor_width / splash_w, 0.8 * self.monitor_height / splash_h, 1)
            self.set_window_size(int(splash_w * fit), int(splash_h * fit))
            self.center()
            self.splash_delay = self.SPLASH_FRAMES

        self.begin_frame()
        self.button_w = self.font_size * 5
        self.label_w = round(self.font_size * 4.5)

        if self.state == States.WELCOME:
            # Hold the splash image for SPLASH_FRAMES frames, then open the menu.
            self._draw_splash()
            self.splash_delay -= 1
            if self.splash_delay <= 0:
                self.set_visible_menu()
                if sys.platform != 'linux':
                    # Title bar height reads 0 while hidden; restore before sizing.
                    self.show_title_bar()
                self.maximize()

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

        # Preprocessing window
        if self.state == States.PREPROCESSING:
            if self.data_preprocessing is None:
                self.state = States.ERROR
            else:
                self.data_preprocessing()

        self._adjust_font_size()
        self.end_frame()

