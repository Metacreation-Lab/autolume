import os
import subprocess
import sys

import imgui

from assets import OPAQUEGREEN, RED
from utils.gui_utils import imgui_utils
from utils.resource_paths import get_version
from utils.user_data import config_file, data_root, default_data_root, set_data_root
from widgets.native_browser_widget import NativeBrowserWidget


def _open_in_file_manager(path):
    """Reveal a folder in the OS file manager, creating it first if needed."""
    os.makedirs(path, exist_ok=True)
    if sys.platform == "win32":
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.run(["open", path], check=False)
    else:
        subprocess.run(["xdg-open", path], check=False)


SETTINGS_POPUP = "Settings##Modal"


class Settings:
    """Application preferences modal. Currently exposes the user data folder.

    Drawn as an overlay over the menu: call :meth:`open` (via
    ``app.open_settings``) to show it while the menu keeps rendering
    underneath. Closing the modal — Close button or the title-bar X —
    lets ``app.close_settings`` drop the overlay flag.
    """

    def __init__(self, app):
        self.app = app
        self.browser = NativeBrowserWidget()
        # Edits stay in a working copy until the user applies them.
        self.pending_root = data_root()
        self.pending_font_size = app.ui_font_size
        self.status = ""
        self.status_error = False
        self._wants_open = False
        self._open = False

    def open(self):
        self._wants_open = True
        self._open = True

    def _apply(self, path):
        path = str(path)
        try:
            os.makedirs(path, exist_ok=True)
            if not os.access(path, os.W_OK):
                raise PermissionError(f"no write access to {path}")
        except OSError as e:
            self.status = f"This folder cannot be used. {e}"
            self.status_error = True
            return
        set_data_root(path)
        self.pending_root = data_root()
        self.status = f"Data folder set to {self.pending_root}"
        self.status_error = False

    @imgui_utils.scoped_by_object_id
    def __call__(self):
        if self._wants_open:
            imgui.open_popup(SETTINGS_POPUP)
            self._wants_open = False

        imgui.set_next_window_size(self.app.content_width * 0.5, 0)
        imgui.set_next_window_position(
            self.app.content_width * 0.5, self.app.content_height * 0.5,
            pivot_x=0.5, pivot_y=0.5)
        imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND, *OPAQUEGREEN)
        imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND_ACTIVE, *OPAQUEGREEN)
        imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND_COLLAPSED, *OPAQUEGREEN)

        opened, self._open = imgui.begin_popup_modal(
            SETTINGS_POPUP, visible=self._open,
            flags=(imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE))
        if opened:
            imgui.text("Data folder")
            imgui.text_colored(
                "Autolume stores your models, presets, captures, datasets, "
                "GANSpace features and training runs in this folder.",
                0.7, 0.7, 0.7)
            changed, self.pending_root = imgui_utils.input_text(
                "##data_root", self.pending_root, 1024, imgui.INPUT_TEXT_ENTER_RETURNS_TRUE,
                width=imgui.get_content_region_available_width(),
                help_text=default_data_root())
            if changed and self.pending_root:
                self._apply(self.pending_root)

            if imgui_utils.button("Browse...", width=self.app.font_size * 7):
                chosen = self.browser.select_directory("Select Autolume Data Folder", initial_dir=self.pending_root)
                if chosen:
                    self._apply(chosen)
            imgui.same_line()
            if imgui_utils.button("Apply", width=self.app.font_size * 7):
                self._apply(self.pending_root or default_data_root())
            imgui.same_line()
            if imgui_utils.button("Open Folder", width=self.app.font_size * 8):
                try:
                    _open_in_file_manager(data_root())
                except OSError as e:
                    self.status = f"Could not open the folder. {e}"
                    self.status_error = True
            imgui.same_line()
            if imgui_utils.button("Reset to Default", width=self.app.font_size * 10):
                self._apply(default_data_root())

            imgui.spacing()
            imgui.text_colored(
                "Model downloads and captures use the new folder right away; "
                "the save-path fields in the other tools update after a restart.",
                0.7, 0.7, 0.7)
            imgui.text_colored(f"Preferences file: {config_file()}", 0.5, 0.5, 0.5)
            if self.status:
                imgui.spacing()
                if self.status_error:
                    imgui.text_colored(self.status, *RED)
                else:
                    imgui.text_colored(self.status, 0.4, 0.8, 0.4)

            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            imgui.text("UI font size")
            imgui.text_colored(
                "Base size of the interface text. The whole UI scales with it. "
                "Applies on release and is remembered across launches.",
                0.7, 0.7, 0.7)
            _, self.pending_font_size = imgui.slider_int(
                "##ui_font_size", self.pending_font_size,
                self.app.MIN_UI_FONT_SIZE, self.app.MAX_UI_FONT_SIZE)
            # Rebuilding the font atlas is too heavy per drag tick; apply once
            # the slider is released (or after keyboard entry).
            if not imgui.is_mouse_down(0) and self.pending_font_size != self.app.ui_font_size:
                self.app.set_ui_font_size(self.pending_font_size)
                self.pending_font_size = self.app.ui_font_size
            imgui.same_line()
            if imgui_utils.button("Default", width=self.app.font_size * 7):
                self.app.set_ui_font_size(self.app.DEFAULT_UI_FONT_SIZE)
                self.pending_font_size = self.app.ui_font_size

            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            imgui.text_colored(f"Version {get_version()}", 0.7, 0.7, 0.7)

            imgui.spacing()
            if imgui.button("Close"):
                imgui.close_current_popup()

            imgui.end_popup()

        imgui.pop_style_color(3)

        if not self._open:
            self.app.close_settings()