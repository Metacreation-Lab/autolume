"""Presets panel: save a look, recall a look, bring an old one across.

Saving reads the control snapshot and writes a file. Loading sends the whole
payload as one control event, so the control thread swaps the state in a
single step and a half applied preset is not a state the runtime can be in.

Importing an old preset folder is deliberately not a save followed by a load.
The importer attaches an error to every legacy expression the new evaluator
refuses, and a preset payload does not carry errors, so a round trip through
one would hand the performer a switched off mapping with no reason given.
Each imported value and binding is sent on its own instead.
"""

import logging
from pathlib import Path

from imgui_bundle import imgui, portable_file_dialogs as pfd

from autolume.live.core import presets
from autolume.live.core.events import ControlEvent
from autolume.live.core.params import BINDING_SET, REGISTRY
from autolume.live.core.presets_legacy import import_legacy_preset

logger = logging.getLogger(__name__)

_ERROR_COLOR = (1.0, 0.3, 0.3, 1.0)
# Sized in multiples of the font size so the panel holds its proportions on
# every display scale.
_NAME_EMS = 13.0
_NOTES_EMS = 9.0
_LIST_EMS = 10.0
_LIST_INTERVAL = 1.0

_NO_PRESETS = "No presets saved yet. Name one and press Save."
_NOTHING_IMPORTED = (
    "Nothing was imported from that folder. "
    "It may not be an old preset folder, or Autolume may not be allowed to read it."
)


def is_valid_name(name: str) -> bool:
    """A preset name has to be a plain file name, not a path into the disk."""
    stripped = name.strip()
    if not stripped or stripped in (".", ".."):
        return False
    return not any(character in stripped for character in "/\\")


class PresetsPanel:
    def __init__(self, runtime, directory: str | Path | None = None) -> None:
        self._runtime = runtime
        self._directory = Path(directory) if directory is not None else None
        self._name = ""
        self._message: str | None = None
        self._error: str | None = None
        self._notes: list[str] = []
        self._imported = False
        self._folder_dialog: pfd.select_folder | None = None
        self._names_cache: list[str] | None = None
        self._names_read = 0.0

    def gui(self) -> None:
        self._save_row()
        self._preset_list()
        self._import_row()
        self._report()

    def directory(self) -> Path:
        return self._directory if self._directory is not None else presets.preset_dir()

    def _save_row(self) -> None:
        imgui.separator_text("Save")
        imgui.set_next_item_width(imgui.get_font_size() * _NAME_EMS)
        _, self._name = imgui.input_text("##name", self._name)
        imgui.same_line()
        valid = is_valid_name(self._name)
        if not valid:
            imgui.begin_disabled()
        clicked = imgui.button("Save")
        if not valid:
            imgui.end_disabled()
        elif clicked:
            self._save(self._name.strip())

    def _save(self, name: str) -> None:
        self._reset_report()
        try:
            path = self.directory() / f"{name}.json"
            presets.save(self._runtime.control_store.snapshot(), path)
        except Exception as exc:
            logger.exception("Could not save preset %s", name)
            self._error = f"Could not save {name}. {_describe(exc)}"
            return
        self._names_cache = None
        self._message = f"Saved {name}."

    def _preset_list(self) -> None:
        imgui.separator_text("Presets")
        names = self._names()
        imgui.begin_child(
            "##presets", imgui.ImVec2(0.0, imgui.get_font_size() * _LIST_EMS)
        )
        if not names:
            imgui.text_disabled(_NO_PRESETS)
        for name in names:
            imgui.push_id(name)
            if imgui.button("Load"):
                self._load(name)
            imgui.same_line()
            imgui.text(name)
            imgui.pop_id()
        imgui.end_child()

    def _names(self) -> list[str]:
        """The preset names, rescanned on a timer rather than every frame.

        Listing hits the disk, and a panel drawn at frame rate has no reason to
        glob a directory sixty times a second. Saving invalidates it directly so
        a new preset shows up at once.
        """
        now = imgui.get_time()
        if self._names_cache is not None and now - self._names_read < _LIST_INTERVAL:
            return self._names_cache
        self._names_read = now
        try:
            self._names_cache = presets.list_presets(self.directory())
        except Exception as exc:
            logger.exception("Could not list presets")
            self._names_cache = []
            self._error = f"Could not read the presets folder. {_describe(exc)}"
        return self._names_cache

    def _load(self, name: str) -> None:
        self._reset_report()
        try:
            payload = presets.load(self.directory() / f"{name}.json")
        except Exception as exc:
            logger.exception("Could not load preset %s", name)
            self._error = f"Could not load {name}. {_describe(exc)}"
            return
        self._submit(presets.PRESET_APPLY, payload)
        self._message = f"Loaded {name}."

    def _import_row(self) -> None:
        imgui.separator_text("Old presets")
        if imgui.button("Import old preset"):
            self._folder_dialog = pfd.select_folder("Choose an old preset folder")
        if self._folder_dialog is None or not self._folder_dialog.ready():
            return
        folder = self._folder_dialog.result()
        self._folder_dialog = None
        if folder:
            self._import(folder)

    def _import(self, folder: str) -> None:
        self._reset_report()
        values, bindings, skipped = import_legacy_preset(folder)
        for name, value in values.items():
            spec = REGISTRY.get(name)
            if spec is None:
                continue
            self._submit(spec.address, value)
        # Sent one at a time rather than through a preset payload, so the error
        # the importer put on a rejected legacy expression survives.
        for binding in bindings:
            self._submit(BINDING_SET, binding)
        self._notes = list(skipped)
        self._imported = True
        if not values and not bindings:
            # Reported plainly. The importer cannot tell an unreadable folder
            # from an absent one, so a folder the user has no permission on
            # would otherwise make the Import button look broken.
            self._message = _NOTHING_IMPORTED
            return
        self._message = (
            f"Imported {len(values)} settings and {len(bindings)} mappings "
            f"from {Path(folder).name}."
        )

    def _report(self) -> None:
        if self._error:
            imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*_ERROR_COLOR))
            imgui.text_wrapped(self._error)
            imgui.pop_style_color()
        elif self._message:
            imgui.text_wrapped(self._message)
        if not self._imported:
            return
        imgui.begin_child(
            "##notes", imgui.ImVec2(0.0, imgui.get_font_size() * _NOTES_EMS)
        )
        if not self._notes:
            imgui.text_disabled("Everything in that folder came across.")
        for note in self._notes:
            imgui.bullet()
            imgui.text_wrapped(note)
        imgui.end_child()

    def _submit(self, address: str, value: object) -> None:
        self._runtime.submit(ControlEvent(address, value, source="ui"))

    def _reset_report(self) -> None:
        self._message = None
        self._error = None
        self._notes = []
        self._imported = False


def _describe(exc: Exception) -> str:
    return str(exc) or type(exc).__name__
