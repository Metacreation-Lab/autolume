"""Presets panel: save a look, recall a look.

Saving reads the control snapshot and writes a file. Loading sends the whole
payload as one control event, so the control thread swaps the state in a
single step and a half applied preset is not a state the runtime can be in.
"""

import logging
from pathlib import Path
from typing import Callable

from imgui_bundle import imgui

from autolume.live.core import presets
from autolume.live.core.events import ControlEvent
from autolume.live.core.mapping import preset_transforms
from autolume.live.errors import describe
from autolume.live.ui.theme import ERROR_COLOR

logger = logging.getLogger(__name__)

# Sized in multiples of the font size so the panel holds its proportions on
# every display scale.
_NAME_EMS = 13.0
_LIST_EMS = 10.0
_LIST_INTERVAL = 1.0

_NO_PRESETS = "No presets saved yet. Name one and press Save."
_NOTICE_POPUP = "Preset notice"


def missing_model_message(missing: str | None, missing2: str | None) -> str | None:
    """What the missing model modal says, or None when nothing is missing.

    Both models are reported, and in one modal rather than two: a preset saved
    from a mixing session names two files, a machine that has neither is the
    ordinary case when a preset travels, and two dialogs in a row for one Load
    is worse than one sentence naming both.

    The second model is the half that used to be dropped: `PresetData` has
    carried `missing_model2` since the mixing state was added to a payload, and
    nothing read it, so a mixing preset whose second model was absent loaded
    silently with `mixing_enabled` coming back on and no mix to show for it.
    """
    names = [name for name in (missing, missing2) if name]
    if not names:
        return None
    if len(names) == 1:
        return f"Model file {names[0]} is missing. The preset loaded without it."
    joined = " and ".join(names)
    return f"Model files {joined} are missing. The preset loaded without them."


def skipped_transforms_message(requested: int, applied: int) -> str | None:
    """What a load says about transform rows that did not survive it.

    A preset's rows are dropped in two places and neither can speak: the file
    reader drops a malformed row, and the control thread's own gate drops one
    whose values would take the render path down or stall it, which is what an
    erode kernel past the ceiling now does. Both used to be silent the same way
    a missing model was, so both are counted and reported together. The count
    is all this says, because which row it was is a number the performer has no
    use for and the chain in the bending panel already shows what loaded.
    """
    dropped = requested - applied
    if dropped <= 0:
        return None
    if dropped == 1:
        return "One transform in this preset could not be loaded. It was skipped."
    return (
        f"{dropped} transforms in this preset could not be loaded. "
        "They were skipped."
    )


def load_notice(*messages: str | None) -> str | None:
    """Everything a load has to say, as one paragraph, or None for nothing.

    One modal rather than one per finding, for the reason
    `missing_model_message` already gives about the two models: a preset that
    travelled to another machine can easily be missing a model and carry a row
    this build refuses, and two dialogs in a row for one Load is worse than two
    sentences in one.
    """
    said = [message for message in messages if message]
    return " ".join(said) if said else None


def is_valid_name(name: str) -> bool:
    """A preset name has to be a plain file name, not a path into the disk."""
    stripped = name.strip()
    if not stripped or stripped in (".", ".."):
        return False
    return not any(character in stripped for character in "/\\")


class PresetsPanel:
    def __init__(
        self,
        runtime,
        directory: str | Path | None = None,
        clock: Callable[[], float] = imgui.get_time,
    ) -> None:
        self._runtime = runtime
        self._directory = Path(directory) if directory is not None else None
        self._clock = clock
        self._name = ""
        self._message: str | None = None
        self._error: str | None = None
        # Kept apart from `_error`, which belongs to whatever the user last
        # asked for. Listing happens on a timer nobody asked for, so it clears
        # itself and may not speak over a save that failed.
        self._list_error: str | None = None
        self._names_cache: list[str] | None = None
        self._names_read = 0.0
        # What a just loaded preset could not find or could not keep, or None.
        # Cleared when the performer dismisses it. One paragraph, since a
        # preset that travelled names two models and a chain of rows.
        self._notice: str | None = None
        # Set by `_load`, consumed by `_notice_modal` later in the same frame.
        # `imgui.open_popup` may not be called from `_load`: that runs inside
        # the preset list's child window and inside `push_id(name)`, and imgui
        # seeds a popup id off the current window and id stack, so the id
        # opened there is not the id `begin_popup_modal` asks for at panel
        # scope. The modal then never draws, and the notice is only cleared
        # from inside it, so the panel stays stuck on it for the session.
        self._pending_notice = False

    def gui(self) -> None:
        self._save_row()
        self._preset_list()
        self._report()
        self._notice_modal()

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
            self._error = f"Could not save {name}. {describe(exc)}"
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
        now = self._clock()
        if self._names_cache is not None and now - self._names_read < _LIST_INTERVAL:
            return self._names_cache
        self._names_read = now
        try:
            self._names_cache = presets.list_presets(self.directory())
        except Exception as exc:
            logger.exception("Could not list presets")
            self._names_cache = []
            self._list_error = f"Could not read the presets folder. {describe(exc)}"
        else:
            # A folder that reads now is not a folder that failed to read, and
            # leaving the old failure up would keep the panel red for the rest
            # of the show and hide every message behind it.
            self._list_error = None
        return self._names_cache

    def _load(self, name: str) -> None:
        self._reset_report()
        try:
            payload = presets.load(self.directory() / f"{name}.json")
        except Exception as exc:
            logger.exception("Could not load preset %s", name)
            self._error = f"Could not load {name}. {describe(exc)}"
            return
        self._submit(presets.PRESET_APPLY, payload)
        self._message = f"Loaded {name}."
        notice = self._notice_for(payload)
        if notice is not None:
            self._notice = notice
            self._pending_notice = True

    def _notice_for(self, payload: dict) -> str | None:
        """What this load has to say beyond "Loaded", or None.

        Parsed a second time here, alongside the control thread's own parse,
        because reporting what a preset lost is a UI concern and the control
        thread has no channel back to this panel. The kept rows are counted
        through `preset_transforms`, the very function the control thread
        applies, so the count cannot disagree with what loads.
        """
        try:
            data = presets.from_payload(payload)
        except ValueError:
            return None
        raw = payload.get("transforms")
        requested = len(raw) if isinstance(raw, list) else 0
        applied = len(preset_transforms(data.transforms))
        return load_notice(
            missing_model_message(data.missing_model, data.missing_model2),
            skipped_transforms_message(requested, applied),
        )

    def _notice_modal(self) -> None:
        # Opened here rather than where the notice is set, at the same scope
        # `begin_popup_modal` is called from. See `_pending_notice`.
        if self._pending_notice:
            imgui.open_popup(_NOTICE_POPUP)
            self._pending_notice = False
        if self._notice is None:
            return
        visible, _ = imgui.begin_popup_modal(
            _NOTICE_POPUP, None, imgui.WindowFlags_.always_auto_resize
        )
        if not visible:
            return
        imgui.text_wrapped(self._notice)
        if imgui.button("OK"):
            self._notice = None
            imgui.close_current_popup()
        imgui.end_popup()

    def report_error(self) -> str | None:
        """The failure the panel shows, the last action's before the listing's.

        An error the performer's own click produced says more than a background
        rescan finding the folder missing, and only one of them can be shown.
        """
        return self._error or self._list_error

    def _report(self) -> None:
        error = self.report_error()
        if error:
            imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*ERROR_COLOR))
            imgui.text_wrapped(error)
            imgui.pop_style_color()
        elif self._message:
            imgui.text_wrapped(self._message)

    def _submit(self, address: str, value: object) -> None:
        self._runtime.submit(ControlEvent(address, value, source="ui"))

    def _reset_report(self) -> None:
        self._message = None
        self._error = None
