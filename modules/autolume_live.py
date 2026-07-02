import logging
import sys
import webbrowser

import cv2
import imgui
import time
import gc

from assets import RED, OPAQUEGREEN, HOVERGREEN
from utils.gui_utils import imgui_window, gl_utils
from utils.model_dir import models_dir
from utils.resource_paths import get_version, resource_path
from widgets.help_icon_widget import DOCS_BASE_URL
from widgets.model_download_widget import ModelDownloadWidget
from enum import IntEnum

logger = logging.getLogger(__name__)

class States(IntEnum):
    ERROR = -2
    CLOSE = -1
    WELCOME = 0
    LIVE = 1
    PREPROCESSING = 4
    TRAINING = 5
    TOOLS = 6


def draw_icon_text_button(app, icon_texture, label, width, height):
    """Self-drawn button: icon on the left + text label, with a hover
    background. Returns True when clicked. Mirrors the navbar styling."""
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


class ModuleHost:
    """Minimal stand-in for the old Menu object: screen modules expect a
    parent exposing `app` and `model_downloader`."""
    def __init__(self, app):
        self.app = app
        self.model_downloader = app.model_downloader


class Autolume(imgui_window.ImguiWindow):
    # State handler that greets user, then hosts the navbar screens
    # (Prepare, Train, Perform, Tools).

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

        self.state = States.WELCOME
        self.running = True
        self.viz = None
        self.render_loop = None
        self.pkls = []
        self.splash_delay = None
        self.data_preprocessing = None
        self.settings = None
        self.settings_open = False

        self.model_downloader = ModelDownloadWidget(self, models_dir())

        self._training_module = None
        self._projection_module = None
        self._features_module = None
        self._super_res_module = None
        self._mixing_module = None

        self.splash = cv2.imread(str(resource_path("assets", "splashscreen.jpg")), cv2.IMREAD_UNCHANGED)
        self.splash = cv2.cvtColor(self.splash, cv2.COLOR_BGRA2RGBA)
        self.splash_texture = gl_utils.Texture(image=self.splash, width=self.splash.shape[1],
                                               height=self.splash.shape[0], channels=self.splash.shape[2])

        self.logo = cv2.imread(str(resource_path("assets", "Autolume-logo.png")), cv2.IMREAD_UNCHANGED)
        self.logo_texture = gl_utils.Texture(image=self.logo, width=self.logo.shape[1],
                                             height=self.logo.shape[0], channels=self.logo.shape[2])

        self.metacreation = cv2.imread(str(resource_path("assets", "metalogo.png")), cv2.IMREAD_UNCHANGED)
        self.metacreation_texture = gl_utils.Texture(image=self.metacreation, width=self.metacreation.shape[1],
                                                     height=self.metacreation.shape[0],
                                                     channels=self.metacreation.shape[2])

        self.cog = cv2.imread(str(resource_path("assets", "cog.png")), cv2.IMREAD_UNCHANGED)
        self.cog_texture = gl_utils.Texture(image=self.cog, width=self.cog.shape[1],
                                            height=self.cog.shape[0], channels=self.cog.shape[2])

        self.web = cv2.imread(str(resource_path("assets", "web.png")), cv2.IMREAD_UNCHANGED)
        self.web_texture = gl_utils.Texture(image=self.web, width=self.web.shape[1],
                                            height=self.web.shape[0], channels=self.web.shape[2])

        self.nav_icons = {}
        for icon_name, icon_state in [("prepare", States.PREPROCESSING), ("train", States.TRAINING),
                                      ("perform", States.LIVE), ("tools", States.TOOLS)]:
            icon_img = cv2.cvtColor(cv2.imread(str(resource_path("assets", f"{icon_name}.png")),
                                               cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
            self.nav_icons[icon_state] = gl_utils.Texture(image=icon_img, width=icon_img.shape[1],
                                                          height=icon_img.shape[0], channels=icon_img.shape[2])

        self.navbar_height = round(self.BASE_FONT_SIZE * 2.2)

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

        if self._training_module is not None and self._training_module.is_training:
            self._training_module._kill_training_process()

        super().close()

    def _cleanup_current_state(self):
        if self.viz is not None:
            self.viz.close()
            self.viz = None
        if self.render_loop is not None:
            self.render_loop.close()
            self.render_loop = None
        if self.data_preprocessing is not None:
            self.data_preprocessing.cleanup()
            self.data_preprocessing = None
        self.set_fps_limit(self.DEFAULT_FPS_LIMIT)
        gc.collect()

        # Small delay to ensure OpenGL resources are properly released
        time.sleep(0.05)

    def navigate_to(self, target_state):
        if target_state == self.state:
            return

        if self._is_training_active() and self.state == States.TRAINING:
            return

        self._cleanup_current_state()

        if target_state == States.LIVE:
            self._start_live()
        elif target_state == States.TRAINING:
            self._ensure_training_module()
        elif target_state == States.TOOLS:
            self._ensure_tool_modules()
        elif target_state == States.PREPROCESSING:
            self._start_preprocessing()

        self.state = target_state

    def _start_live(self):
        from modules.renderloop import AsyncRenderer
        from modules.visualizer import Visualizer

        self.render_loop = AsyncRenderer()
        self.viz = Visualizer(self, self.render_loop)

        # List pickles.
        if len(self.pkls) > 0:
            for pkl in self.pkls:
                self.viz.add_recent_pickle(pkl)
            self.viz.load_pickle(self.pkls[0])

    def _ensure_training_module(self):
        if self._training_module is None:
            from modules.training_module import TrainingModule
            self._training_module = TrainingModule(ModuleHost(self))

    def _ensure_tool_modules(self):
        if self._projection_module is None:
            from modules.projection_module import ProjectionModule
            self._projection_module = ProjectionModule(ModuleHost(self))
        if self._features_module is None:
            from modules.pca_module import PCA_Module
            self._features_module = PCA_Module(ModuleHost(self))
        if self._super_res_module is None:
            from modules.super_res_module import SuperResModule
            self._super_res_module = SuperResModule(ModuleHost(self))
        if self._mixing_module is None:
            from modules.network_mixing import MixingModule
            self._mixing_module = MixingModule(ModuleHost(self))

    def _is_training_active(self):
        return self._training_module is not None and self._training_module.is_training

    def _start_preprocessing(self):
        from modules.preprocessing_module import DataPreprocessing
        self.data_preprocessing = DataPreprocessing(self)

    def start_preprocessing(self):
        self.navigate_to(States.PREPROCESSING)

    # Settings modal (drawn as an overlay over the current screen, not its own state)
    def open_settings(self):
        from modules.settings import Settings
        if self.settings is None:
            self.settings = Settings(self)
        self.settings_open = True
        self.settings.open()

    def close_settings(self):
        self.settings_open = False

    def draw_navbar(self):
        # Scale with the UI font, calibrated to 50px at font 23.
        self.navbar_height = round(self.font_size * 2.2)
        training_active = self._is_training_active()

        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(self.content_width, self.navbar_height)
        # temporarily update the window style to remove annoying gaps around the banner
        imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (0, 0))
        imgui.push_style_var(imgui.STYLE_WINDOW_BORDERSIZE, 0)
        imgui.begin('##Navbar', closable=False, flags=(
            imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE |
            imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS |
            imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_SCROLLBAR))
        imgui.pop_style_var(2)

        imgui.get_window_draw_list().add_rect_filled(
            0, 0, self.content_width, self.navbar_height,
            imgui.get_color_u32_rgba(*RED))

        logo_height = int(self.navbar_height * 0.6)
        logo_width = int(logo_height * (self.logo.shape[1] / self.logo.shape[0]))
        imgui.set_cursor_pos_y((self.navbar_height - logo_height) / 2)
        imgui.set_cursor_pos_x(20)
        imgui.image(self.logo_texture.gl_id, logo_width, logo_height)

        nav_button_height = int(self.navbar_height * 0.6)
        nav_button_y = (self.navbar_height - nav_button_height) / 2

        nav_items = [
            ("Prepare", States.PREPROCESSING),
            ("Train", States.TRAINING),
            ("Perform", States.LIVE),
            ("Tools", States.TOOLS),
        ]

        nav_icon_size = int(nav_button_height * 0.85)
        nav_icon_text_gap = 8
        nav_item_spacing = 28
        draw_list = imgui.get_window_draw_list()

        imgui.same_line(20 + logo_width + 30)

        for i, (label, target_state) in enumerate(nav_items):
            if i > 0:
                imgui.same_line(0, nav_item_spacing)

            is_active = (self.state == target_state)
            nav_disabled = training_active and target_state != States.TRAINING

            text_size = imgui.calc_text_size(label)
            item_width = nav_icon_size + nav_icon_text_gap + text_size.x
            item_height = nav_button_height

            imgui.set_cursor_pos_y(nav_button_y)
            pos = imgui.get_cursor_screen_pos()
            clicked = imgui.invisible_button(f"##nav_{i}", item_width, item_height)
            hovered = imgui.is_item_hovered()

            if nav_disabled:
                alpha = 0.3
            elif is_active or hovered:
                alpha = 1.0
            else:
                alpha = 0.5
            color = imgui.get_color_u32_rgba(1.0, 1.0, 1.0, alpha)

            icon_x = pos.x
            icon_y = pos.y + (item_height - nav_icon_size) / 2
            draw_list.add_image(self.nav_icons[target_state].gl_id,
                                (icon_x, icon_y), (icon_x + nav_icon_size, icon_y + nav_icon_size),
                                (0, 0), (1, 1), color)

            text_x = icon_x + nav_icon_size + nav_icon_text_gap
            text_y = pos.y + (item_height - text_size.y) / 2
            draw_list.add_text(text_x, text_y, color, label)

            if clicked and not nav_disabled:
                self.navigate_to(target_state)

        metacreation_height = logo_height
        metacreation_width = int(metacreation_height * (self.metacreation.shape[1] / self.metacreation.shape[0]))

        button_height = int(self.navbar_height * 0.6)
        button_y = (self.navbar_height - button_height) / 2
        pad_x = self.font_size * 0.4

        def icon_text_width(label):
            icon_size = int(self.font_size * 0.8)
            gap = self.font_size * 0.35
            return pad_x + icon_size + gap + imgui.calc_text_size(label).x + pad_x

        doc_button_width = icon_text_width("Documentation")
        settings_button_width = icon_text_width("Settings")
        settings_button_x = self.content_width - (metacreation_width + doc_button_width + settings_button_width + 60)
        imgui.same_line(settings_button_x)
        imgui.set_cursor_pos_y(button_y)
        if draw_icon_text_button(self, self.cog_texture, "Settings", settings_button_width, button_height):
            self.open_settings()

        imgui.same_line(self.content_width - (metacreation_width + doc_button_width + 40))
        imgui.set_cursor_pos_y(button_y)
        if draw_icon_text_button(self, self.web_texture, "Documentation", doc_button_width, button_height):
            webbrowser.open(DOCS_BASE_URL)

        imgui.same_line(self.content_width - (metacreation_width + 20))
        imgui.set_cursor_pos_y((self.navbar_height - metacreation_height) / 2)
        imgui.image(self.metacreation_texture.gl_id, metacreation_width, metacreation_height)

        imgui.end()

    def _draw_module_fullscreen(self, title, module_callable):
        fullscreen_flags = (
            imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE |
            imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS | imgui.WINDOW_NO_TITLE_BAR)
        imgui.set_next_window_position(0, self.navbar_height)
        imgui.set_next_window_size(self.content_width, self.content_height - self.navbar_height)
        imgui.begin(title, closable=False, flags=fullscreen_flags)
        module_callable()
        imgui.end()

    def _draw_tools_grid(self):
        grid_flags = (
            imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE |
            imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS | imgui.WINDOW_NO_COLLAPSE)
        half_w = self.content_width / 2
        half_h = (self.content_height - self.navbar_height) / 2

        cells = [
            (0, 0, 'Projection', self._projection_module),
            (1, 0, 'Feature Extractor', self._features_module),
            (0, 1, 'Super Resolution', self._super_res_module),
            (1, 1, 'Model Mixing', self._mixing_module),
        ]

        for col, row, title, module in cells:
            x = col * half_w
            y = self.navbar_height + row * half_h
            imgui.set_next_window_position(x, y)
            imgui.set_next_window_size(half_w, half_h)
            imgui.begin(f'{title}##tools_grid', closable=False, flags=grid_flags)
            module()
            imgui.end()

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
            # Hold the splash image for SPLASH_FRAMES frames, then open the live screen.
            self._draw_splash()
            self.splash_delay -= 1
            if self.splash_delay <= 0:
                self.navigate_to(States.LIVE)
                if sys.platform != 'linux':
                    # Title bar height reads 0 while hidden; restore before sizing.
                    self.show_title_bar()
                self.maximize()
        else:
            imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND, *OPAQUEGREEN)
            imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND_ACTIVE, *OPAQUEGREEN)
            imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND_COLLAPSED, *OPAQUEGREEN)

            self.draw_navbar()

            if self.state == States.LIVE:
                if self.viz is None or self.render_loop is None:
                    self.state = States.ERROR
                else:
                    self.viz()

            elif self.state == States.TRAINING:
                if self._training_module is None:
                    self.state = States.ERROR
                else:
                    self._draw_module_fullscreen('Training##fullscreen', self._training_module)

            elif self.state == States.TOOLS:
                if (self._projection_module is None or self._features_module is None
                        or self._super_res_module is None or self._mixing_module is None):
                    self.state = States.ERROR
                else:
                    self._draw_tools_grid()

            elif self.state == States.PREPROCESSING:
                if self.data_preprocessing is None:
                    self.state = States.ERROR
                else:
                    self.data_preprocessing()

            if self.state in (States.TRAINING, States.TOOLS):
                self.model_downloader()

            if self.settings_open and self.settings is not None:
                self.settings()

            imgui.pop_style_color(3)

        self._adjust_font_size()
        self.end_frame()
