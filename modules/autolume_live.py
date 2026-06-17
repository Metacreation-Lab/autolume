import os

import cv2
import imgui
import time
import gc

from utils.gui_utils import imgui_window, gl_utils
from utils.resource_paths import get_version
from enum import IntEnum
class States(IntEnum):
    ERROR = -2
    CLOSE = -1
    WELCOME = 0
    MENU = 1
    RENDER = 2
    SPLASH = 3
    PREPROCESSING = 4



class Autolume(imgui_window.ImguiWindow):
    # State handler that greets user, shows menu (allowing for training, compression, ganspace), and renders

    DEFAULT_FPS_LIMIT = 60

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
        self.splash_delay = 0
        self.data_preprocessing = None

        self.splash = cv2.imread("assets/splashscreen.jpg", cv2.IMREAD_UNCHANGED)
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

    def draw_frame(self):

        if self.state == States.WELCOME:
            # Configure the still-hidden window for the splash once: borderless,
            # at image scale but capped below the work area (reaching work-area
            # size would trigger the maximize path), and centered.
            self.hide_title_bar()
            splash_w = self.splash_texture.width // 2
            splash_h = self.splash_texture.height // 2
            fit = min(0.8 * self.monitor_width / splash_w, 0.8 * self.monitor_height / splash_h, 1)
            self.set_window_size(int(splash_w * fit), int(splash_h * fit))
            self.center()

        self.begin_frame()
        self.button_w = self.font_size * 5
        self.label_w = round(self.font_size * 4.5)

        if self.state == States.SPLASH:
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
            self.splash_delay -= 1
            if self.splash_delay <= 0:
                self.set_visible_menu()
                # Restore the decoration before sizing: with the title bar hidden its
                # height reads as 0, so the window ends up one title bar taller than
                # the work area and the bottom is cropped (visible on macOS, where
                # the later maximize does not re-fit the frame).
                self.show_title_bar()
                self.set_window_size(3840, 2160)


        if self.state == States.WELCOME:
            imgui.set_next_window_position(0, 0)
            imgui.set_next_window_size(self.content_width, self.content_height)
            # No padding or border so the splash image fills the window edge to edge.
            imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (0, 0))
            imgui.push_style_var(imgui.STYLE_WINDOW_BORDERSIZE, 0)
            imgui.begin('##welcome', closable=False,
                        flags=(imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE| imgui.WINDOW_NO_SCROLLBAR))
            imgui.image(self.splash_texture.gl_id, self.content_width, self.content_height)
            imgui.end()
            imgui.pop_style_var(2)
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

        # Preprocessing window
        if self.state == States.PREPROCESSING:
            if self.data_preprocessing is None:
                self.state = States.ERROR
            else:
                self.data_preprocessing()

        self._adjust_font_size()
        self.end_frame()

