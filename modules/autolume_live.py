import logging
import sys
import webbrowser

import cv2
import imgui
import time
import gc

from assets import RED, OPAQUEGREEN, HOVERGREEN
from utils.gui_utils import imgui_window, gl_utils
from utils.resource_paths import get_version, resource_path
from widgets.help_icon_widget import DOCS_BASE_URL
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


def draw_icon_button(icon_texture, label, icon_size, pad, height):
    """Self-drawn icon button styled exactly like the navbar items (full-height
    hover background, same idle/hover alphas), with a tooltip naming the
    action. Returns True when clicked."""
    width = icon_size + 2 * pad
    pos = imgui.get_cursor_screen_pos()
    clicked = imgui.invisible_button(f"##{label}_btn", width, height)
    hovered = imgui.is_item_hovered()
    draw_list = imgui.get_window_draw_list()
    if hovered:
        draw_list.add_rect_filled(
            pos[0], pos[1], pos[0] + width, pos[1] + height,
            imgui.get_color_u32_rgba(*HOVERGREEN))
        imgui.set_tooltip(label)
    icon_x = pos[0] + pad
    icon_y = pos[1] + (height - icon_size) / 2
    alpha = 0.9 if hovered else 0.6
    draw_list.add_image(
        icon_texture.gl_id,
        (icon_x, icon_y),
        (icon_x + icon_size, icon_y + icon_size),
        col=imgui.get_color_u32_rgba(1, 1, 1, alpha))
    return clicked


class ModuleHost:
    """Minimal stand-in for the old Menu object: screen modules expect a
    parent exposing `app`."""
    def __init__(self, app):
        self.app = app


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

    # Navbar text, icons, and height are drawn this much larger than the base UI font.
    NAVBAR_SCALE = 1.25

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

        self._training_module = None
        self._projection_module = None
        self._features_module = None
        self._super_res_module = None
        self._mixing_module = None

        self.splash = cv2.imread(str(resource_path("assets", "splashscreen.jpg")), cv2.IMREAD_UNCHANGED)
        self.splash = cv2.cvtColor(self.splash, cv2.COLOR_BGRA2RGBA)
        self.splash_texture = gl_utils.Texture(image=self.splash, width=self.splash.shape[1],
                                               height=self.splash.shape[0], channels=self.splash.shape[2])

        self.logo = cv2.imread(str(resource_path("assets", "autolume-logo.png")), cv2.IMREAD_UNCHANGED)
        self.logo_texture = gl_utils.Texture(image=self.logo, width=self.logo.shape[1],
                                             height=self.logo.shape[0], channels=self.logo.shape[2])

        self.cog = cv2.imread(str(resource_path("assets", "cog.png")), cv2.IMREAD_UNCHANGED)
        self.cog_texture = gl_utils.Texture(image=self.cog, width=self.cog.shape[1],
                                            height=self.cog.shape[0], channels=self.cog.shape[2])

        self.book = cv2.imread(str(resource_path("assets", "book.png")), cv2.IMREAD_UNCHANGED)
        self.book_texture = gl_utils.Texture(image=self.book, width=self.book.shape[1],
                                             height=self.book.shape[0], channels=self.book.shape[2])

        self.navbar_height = round(self.BASE_FONT_SIZE * self.NAVBAR_SCALE * 2.2)

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
        # Scale with the UI font, calibrated to 50px at font 23 before NAVBAR_SCALE.
        nav_font = self.font_size * self.NAVBAR_SCALE
        self.navbar_height = round(nav_font * 2.2)
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
        # Enlarge text drawn in this window (and calc_text_size) to match the bar.
        imgui.set_window_font_scale(self.NAVBAR_SCALE)

        draw_list = imgui.get_window_draw_list()
        draw_list.add_rect_filled(
            0, 0, self.content_width, self.navbar_height,
            imgui.get_color_u32_rgba(*OPAQUEGREEN))
        # Hairline separator so the bar reads as chrome against the content below.
        draw_list.add_line(
            0, self.navbar_height - 1, self.content_width, self.navbar_height - 1,
            imgui.get_color_u32_rgba(1, 1, 1, 0.08))

        logo_x = round(nav_font * 0.9)
        logo_height = int(self.navbar_height * 0.6)
        logo_width = int(logo_height * (self.logo.shape[1] / self.logo.shape[0]))
        imgui.set_cursor_pos_y((self.navbar_height - logo_height) / 2)
        imgui.set_cursor_pos_x(logo_x)
        imgui.image(self.logo_texture.gl_id, logo_width, logo_height)

        nav_items = [
            ("Prepare", States.PREPROCESSING),
            ("Train", States.TRAINING),
            ("Perform", States.LIVE),
            ("Tools", States.TOOLS),
        ]

        item_pad = round(nav_font * 0.6)
        item_spacing = round(nav_font * 0.35)
        underline_height = max(2, round(nav_font * 0.15))

        imgui.same_line(logo_x + logo_width + round(nav_font * 1.3))

        for i, (label, target_state) in enumerate(nav_items):
            if i > 0:
                imgui.same_line(0, item_spacing)

            is_active = (self.state == target_state)
            nav_disabled = training_active and target_state != States.TRAINING

            text_size = imgui.calc_text_size(label)
            item_width = text_size.x + 2 * item_pad

            imgui.set_cursor_pos_y(0)
            pos = imgui.get_cursor_screen_pos()
            clicked = imgui.invisible_button(f"##nav_{i}", item_width, self.navbar_height)
            hovered = imgui.is_item_hovered() and not nav_disabled

            if hovered:
                draw_list.add_rect_filled(
                    pos.x, pos.y, pos.x + item_width, pos.y + self.navbar_height,
                    imgui.get_color_u32_rgba(*HOVERGREEN))

            if nav_disabled:
                alpha = 0.3
            elif is_active:
                alpha = 1.0
            elif hovered:
                alpha = 0.9
            else:
                alpha = 0.6
            color = imgui.get_color_u32_rgba(1.0, 1.0, 1.0, alpha)

            text_x = pos.x + item_pad
            text_y = pos.y + (self.navbar_height - text_size.y) / 2
            draw_list.add_text(text_x, text_y, color, label)

            if is_active:
                draw_list.add_rect_filled(
                    pos.x, pos.y + self.navbar_height - underline_height,
                    pos.x + item_width, pos.y + self.navbar_height,
                    imgui.get_color_u32_rgba(*RED))

            if clicked and not nav_disabled:
                self.navigate_to(target_state)

        icon_size = round(nav_font * 0.9)
        button_width = icon_size + 2 * item_pad
        edge_pad = round(nav_font * 0.9)

        imgui.same_line(self.content_width - (edge_pad + 2 * button_width + item_spacing))
        imgui.set_cursor_pos_y(0)
        if draw_icon_button(self.cog_texture, "Settings", icon_size, item_pad, self.navbar_height):
            self.open_settings()

        imgui.same_line(self.content_width - (edge_pad + button_width))
        imgui.set_cursor_pos_y(0)
        if draw_icon_button(self.book_texture, "Documentation", icon_size, item_pad, self.navbar_height):
            webbrowser.open(DOCS_BASE_URL)

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

            if self.settings_open and self.settings is not None:
                self.settings()

            imgui.pop_style_color(3)

        self._adjust_font_size()
        self.end_frame()
