import logging

import imgui

from assets import ACTIVE_RED, GREEN
from utils.gui_utils import imgui_utils
from utils.presets import PresetStore, load_preset, save_preset
from utils.user_data import data_path
from widgets.osc_menu import osc_address_picker

logger = logging.getLogger(__name__)

_DELETE_MODAL = "Delete Preset##preset_widget"


class PresetWidget:
    def __init__(self, viz):
        self.viz = viz
        self.store = PresetStore(str(data_path("presets")))
        self.new_name = ""
        self._new_name_active = False
        self.last_loaded = None
        self.delete_target = None
        self._open_delete_modal = False
        self.use_osc = False
        self.osc_addresses = "preset"

    def _load(self, name):
        if load_preset(self.viz, self.store.path(name)):
            self.last_loaded = name
            self.viz.app.skip_frame()

    def _save(self, name):
        save_preset(self.viz, self.store.path(name))
        self.store.invalidate()

    def _create_row(self):
        viz = self.viz
        spacing = imgui.get_style().item_spacing.x
        name_w = max(viz.app.button_w,
                     imgui.get_content_region_available_width()
                     - viz.app.button_w - spacing)
        with imgui_utils.item_width(name_w):
            _changed, self.new_name = imgui_utils.input_text(
                "##new_preset_name", self.new_name, 256,
                flags=imgui.INPUT_TEXT_AUTO_SELECT_ALL,
                help_text="Name for a new preset")
        enter = (self._new_name_active
                 and imgui.is_key_pressed(
                     imgui.get_key_index(imgui.KEY_ENTER), repeat=False))
        self._new_name_active = imgui.is_item_active()
        candidate = self.new_name.strip()
        valid = self.store.is_valid_name(candidate)
        imgui.same_line()
        clicked = imgui_utils.button("Save New##presets",
                                     width=viz.app.button_w, enabled=valid)
        if (clicked or enter) and valid:
            path = self.store.create(candidate)
            if path is not None and save_preset(viz, path):
                self.new_name = ""
            self.store.invalidate()

    def _preset_row(self, name):
        viz = self.viz
        # Scope the row by name instead of interpolating it into labels, since
        # "#" is legal in a preset name and would collide with imgui's ID syntax.
        imgui.push_id(name)
        spacing = imgui.get_style().item_spacing.x
        name_w = max(viz.app.button_w,
                     imgui.get_content_region_available_width()
                     - 3 * viz.app.button_w - 3 * spacing)
        loaded = name == self.last_loaded
        if loaded:
            imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, *GREEN)
        with imgui_utils.item_width(name_w):
            changed, edited = imgui_utils.input_text(
                "##preset_name", name, 256,
                flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE
                | imgui.INPUT_TEXT_AUTO_SELECT_ALL,
                help_text="Rename this preset")
        if loaded:
            imgui.pop_style_color(1)
        if changed:
            target = edited.strip()
            if target != name:
                if self.store.rename(name, target):
                    if self.last_loaded == name:
                        self.last_loaded = target
                else:
                    logger.warning("Could not rename preset %s to %s",
                                   name, target)
        imgui.same_line()
        if imgui_utils.button("Load", width=viz.app.button_w):
            self._load(name)
        imgui.same_line()
        if imgui_utils.button("Save", width=viz.app.button_w):
            self._save(name)
        imgui.same_line()
        if self._delete_button("Delete", width=viz.app.button_w):
            self.delete_target = name
            self._open_delete_modal = True
        imgui.pop_id()

    @staticmethod
    def _delete_button(label, width):
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, *ACTIVE_RED)
        imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, *ACTIVE_RED)
        clicked = imgui_utils.button(label, width=width)
        imgui.pop_style_color(2)
        return clicked

    def _delete_modal(self):
        viz = self.viz
        if self._open_delete_modal:
            imgui.open_popup(_DELETE_MODAL)
            self._open_delete_modal = False
        opened, _ = imgui_utils.begin_popup_modal(
            _DELETE_MODAL, dim_background=True,
            flags=imgui.WINDOW_NO_RESIZE | imgui.WINDOW_ALWAYS_AUTO_RESIZE)
        if not opened:
            return
        imgui.text(f"Delete preset '{self.delete_target}'?")
        imgui.text("This cannot be undone.")
        if imgui_utils.button("Cancel##preset_confirm", width=viz.app.button_w):
            self.delete_target = None
            imgui.close_current_popup()
        imgui.same_line()
        if self._delete_button("Delete##preset_confirm",
                               width=viz.app.button_w):
            self.store.delete(self.delete_target)
            if self.last_loaded == self.delete_target:
                self.last_loaded = None
            self.delete_target = None
            imgui.close_current_popup()
        imgui.end_popup()

    def _osc_row(self):
        viz = self.viz
        toggled, self.use_osc = imgui.checkbox("Use OSC##load", self.use_osc)
        if toggled:
            if self.use_osc:
                viz.osc_dispatcher.map(f"/{self.osc_addresses}",
                                       self.osc_handler)
            else:
                try:
                    viz.osc_dispatcher.unmap(f"/{self.osc_addresses}",
                                             self.osc_handler)
                except Exception:
                    logger.warning("OSC address %s is not mapped",
                                   self.osc_addresses)
        imgui.same_line()
        with imgui_utils.grayed_out(not self.use_osc):
            changed, osc_address = imgui_utils.input_text(
                "##OSC_load", self.osc_addresses, 256,
                imgui.INPUT_TEXT_CHARS_NO_BLANK |
                (imgui.INPUT_TEXT_READ_ONLY) * (not self.use_osc),
                width=viz.app.font_size * 5,
                help_text="Osc Address")
            imgui.same_line()
            picked, picked_address = osc_address_picker(
                viz, "OSC_load", self.osc_addresses, enabled=self.use_osc)
            if picked:
                changed, osc_address = True, picked_address
            if changed:
                try:
                    viz.osc_dispatcher.unmap(f"/{self.osc_addresses}",
                                             self.osc_handler)
                    self.osc_addresses = osc_address
                except Exception:
                    logger.warning("OSC address %s is not mapped",
                                   self.osc_addresses)
                viz.osc_dispatcher.map(f"/{self.osc_addresses}",
                                       self.osc_handler)

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if not show:
            self._new_name_active = False
            return
        self._create_row()
        names = self.store.names()
        imgui_utils.begin_child(self, "##preset_list", width=0,
                                height=viz.app.font_size * 14, border=True)
        if not names:
            hint = "No presets yet. Save one to get started."
            avail = imgui.get_content_region_available()
            text_w, text_h = imgui.calc_text_size(hint)
            imgui.set_cursor_pos((max(0, (avail.x - text_w) / 2),
                                  max(0, (avail.y - text_h) / 2)))
            imgui.text_disabled(hint)
        for name in names:
            self._preset_row(name)
        imgui_utils.end_child()
        self._delete_modal()
        self._osc_row()

    def osc_handler(self, address, *args):
        name = str(args[-1])
        if name in self.store.names():
            self._load(name)
        else:
            logger.warning("OSC preset load: no preset named %s", name)
