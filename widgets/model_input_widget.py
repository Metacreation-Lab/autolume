import imgui

from utils.gui_utils import imgui_utils
from widgets.model_dropdown_widget import ModelDropdownButton
from widgets.native_browser_widget import NativeBrowserWidget

#----------------------------------------------------------------------------

class ModelInputWidget:
    """Input field + native 'Browse' + 'Models' dropdown for .pkl model paths."""

    PLACEHOLDER = 'Path to model (.pkl)'

    def __init__(self, app, dropdown=None, browser=None):
        self.app = app
        self.browser = browser or NativeBrowserWidget()
        self.dropdown = dropdown or ModelDropdownButton()

    @imgui_utils.scoped_by_object_id
    def __call__(self, value, width=-1, enabled=True,
                 flags=(imgui.INPUT_TEXT_AUTO_SELECT_ALL | imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)):
        """Draw the row and return (changed, value).

        changed is True on a typed commit (Enter), a Browse pick, or a Models
        pick (including the downloader's pending pick surfacing through the
        dropdown).
        """
        input_w = width - 2 * (self.app.button_w + self.app.spacing)
        if width > 0:
            input_w = max(input_w, self.app.button_w)
        if not enabled:
            flags |= imgui.INPUT_TEXT_READ_ONLY
        changed, value = imgui_utils.input_text('##model_path', value, 1024,
            flags=flags, width=input_w, help_text=self.PLACEHOLDER)
        if not enabled:
            changed = False
        if imgui.is_item_hovered() and not imgui.is_item_active() and value != '':
            imgui.set_tooltip(value)
        imgui.same_line()
        if imgui_utils.button('Browse##model', width=self.app.button_w, enabled=enabled):
            pkl = self.browser.select_model_file(initial_dir=value)
            if pkl:
                value = str(pkl)
                changed = True
        imgui.same_line()
        picked = self.dropdown(width=self.app.button_w, enabled=enabled)
        if picked is not None and enabled:
            value = picked
            changed = True
        return changed, value

#----------------------------------------------------------------------------
