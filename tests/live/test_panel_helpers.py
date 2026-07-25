"""Pure helpers behind the panels.

imgui cannot be driven headless, so the drawing itself is verified by hand.
What can be pinned down is the logic the drawing reads from, which is why it
lives in module functions rather than inside the gui methods.
"""

import itertools

import numpy as np

from autolume.live.core import presets
from autolume.live.core.params import (
    BINDING_CLEAR,
    BINDING_SET,
    REGISTRY,
    Binding,
    ClearBinding,
    ControlState,
)
from autolume.live.errors import describe
from autolume.live.ui.panels.audio import (
    bar_value,
    device_index,
    device_labels,
    spectrum_ceiling,
    spectrum_values,
)
from autolume.live.ui.panels.mapping import (
    MappingPanel,
    bindable_specs,
    canonical_address,
    display_label,
    reference_note,
)
from autolume.live.ui.panels.presets import PresetsPanel, is_valid_name


def test_device_labels_never_empty_so_the_combo_stays_drawn():
    assert device_labels(()) == ["No input devices found"]
    assert device_labels(((3, "mic"), (7, "line in"))) == ["mic", "line in"]


def test_device_index_stays_inside_the_drawn_list():
    assert device_index((), 4) == 0
    assert device_index(((3, "mic"), (7, "line in")), 9) == 1
    assert device_index(((3, "mic"), (7, "line in")), -2) == 0
    assert device_index(((3, "mic"), (7, "line in")), 1) == 1


def test_spectrum_values_fall_back_to_idle_bars():
    idle = spectrum_values(None)
    assert idle.dtype == np.float32
    assert idle.size > 0
    assert spectrum_values(np.zeros(0, dtype=np.float32)).size == idle.size

    values = spectrum_values(np.array([1.0, 2.0], dtype=np.float64))
    assert values.dtype == np.float32
    assert list(values) == [1.0, 2.0]


def test_idle_bars_are_not_shared_between_frames():
    idle = spectrum_values(None)
    idle[0] = 5.0
    assert spectrum_values(None)[0] == 0.0


def test_spectrum_ceiling_keeps_silence_flat():
    quiet = spectrum_values(np.zeros(8, dtype=np.float32))
    loud = spectrum_values(np.array([0.0, 3.0], dtype=np.float32))
    broken = spectrum_values(np.array([0.0, np.nan], dtype=np.float32))
    assert spectrum_ceiling(quiet) > 0.0
    assert spectrum_ceiling(loud) == 3.0
    assert spectrum_ceiling(broken) == spectrum_ceiling(quiet)


def test_bar_value_shows_a_broken_feature_as_silence():
    assert bar_value(0.5) == 0.5
    assert bar_value(float("nan")) == 0.0
    assert bar_value(float("inf")) == 0.0
    assert bar_value("loud") == 0.0
    assert bar_value(None) == 0.0


def test_every_parameter_gets_a_row_including_the_model_path():
    """The model path is bindable, so it has a row like everything else.

    Under the opt in model a parameter is reachable from a controller only
    through its row, so leaving the text parameter out of this list was the
    whole of why a controller could not switch models. Pinned by name as well
    as by count, because that one parameter is the point.
    """
    names = [spec.name for spec in bindable_specs()]
    assert "pkl_path" in names
    assert "latent_x" in names
    assert names == [spec.name for spec in REGISTRY.values()]


def test_only_a_text_parameter_row_carries_a_note():
    """The expression field stays live on the model row, so the row explains it.

    A number is an index and the expression scales it, text is a name and no
    expression applies. Both halves have to be on the row or the field is the
    misleading kind: editable, and quietly doing nothing to half of what
    arrives. Every other row means one thing and needs no line under it.
    """
    note = reference_note(REGISTRY["pkl_path"])
    assert note is not None
    assert "position" in note and "expression above" in note
    assert "does not apply" in note
    assert reference_note(REGISTRY["latent_x"]) is None
    assert reference_note(REGISTRY["anim_playing"]) is None


def test_display_label_is_derived_from_the_registry_name():
    assert display_label("anim_speed_x") == "Anim speed x"
    assert display_label("fps_cap") == "Fps cap"


def test_canonical_address_matches_what_the_source_table_stores():
    assert canonical_address(" latent/x ") == "/latent/x"
    assert canonical_address("/audio/level") == "/audio/level"
    assert canonical_address("   ") == ""


def test_preset_names_may_not_walk_out_of_the_folder():
    assert is_valid_name("evening set")
    assert not is_valid_name("")
    assert not is_valid_name("   ")
    assert not is_valid_name("..")
    assert not is_valid_name("a/b")
    assert not is_valid_name("a\\b")


class RecordingStore:
    def snapshot(self):
        return ControlState()


class RecordingRuntime:
    """Captures what a panel submits, without a control loop behind it."""

    def __init__(self):
        self.events = []
        self.control_store = RecordingStore()

    def submit(self, event):
        self.events.append(event)


def commit_panel():
    return MappingPanel(RecordingRuntime())


def test_typed_source_becomes_a_binding_with_a_canonical_address():
    """And a live one. Naming a source is the ask, so the first one switches
    the row on rather than leaving a mapping the performer just picked doing
    nothing until they find the box beside it. Off by default protects them
    from traffic they did not configure, not from an address they typed.
    """
    panel = commit_panel()
    panel._commit("latent_x", None, source="audio/bass")
    (event,) = panel._runtime.events
    assert event.address == BINDING_SET
    assert event.value == Binding("latent_x", "/audio/bass", "x", True)
    assert event.source == "ui"


def test_typing_a_source_into_a_row_that_is_off_leaves_it_off():
    # The performer switched that one off on purpose, and retyping an address
    # is not them taking it back.
    panel = commit_panel()
    off = Binding("latent_x", "/audio/bass", "x", False)
    panel._commit("latent_x", off, source="/td/knob")
    (event,) = panel._runtime.events
    assert event.value == Binding("latent_x", "/td/knob", "x", False)


def test_emptying_the_source_clears_with_an_object_never_a_string():
    panel = commit_panel()
    bound = Binding("latent_x", "/audio/bass", "x", False)
    panel._commit("latent_x", bound, source="")
    (event,) = panel._runtime.events
    assert event.address == BINDING_CLEAR
    # A bare target string would be expressible over OSC, which would let any
    # peer on the port delete a performer's mapping mid show.
    assert not isinstance(event.value, str)
    assert event.value == ClearBinding("latent_x")


def test_emptying_an_unbound_source_submits_nothing():
    panel = commit_panel()
    panel._commit("latent_x", None, source="   ")
    assert panel._runtime.events == []


def test_switching_an_unmapped_row_on_records_it_as_a_sourceless_binding():
    """On has to be written down somewhere, and this is the somewhere.

    Absence of a row now means remote input is off, so switching one on cannot
    be expressed by leaving the state alone. As an ordinary binding it persists
    in a preset for free rather than needing a parallel set of enabled names.
    """
    panel = commit_panel()
    panel._commit("latent_x", None, enabled=True)
    (event,) = panel._runtime.events
    assert event.address == BINDING_SET
    assert event.value == Binding("latent_x", "", "x", True)


def test_switching_a_sourceless_row_back_off_returns_it_to_the_default():
    # And leaves nothing behind in the preset, so on and off again lands where
    # it started rather than storing a row that says what absence already says.
    panel = commit_panel()
    on = Binding("latent_x", "", "x", True)
    panel._commit("latent_x", on, enabled=False)
    (event,) = panel._runtime.events
    assert event.address == BINDING_CLEAR
    assert event.value == ClearBinding("latent_x")


def test_emptying_the_source_of_a_row_that_is_on_keeps_the_row():
    # Clearing it would switch remote input off behind the performer, and the
    # row still says something: listen on the parameter's own address.
    panel = commit_panel()
    bound = Binding("latent_x", "/audio/bass", "x", True)
    panel._commit("latent_x", bound, source="")
    (event,) = panel._runtime.events
    assert event.address == BINDING_SET
    assert event.value == Binding("latent_x", "", "x", True)


def test_emptying_the_source_of_a_row_that_is_off_clears_it():
    # Nothing is left to say: no address, off, and the value passed through is
    # exactly what a parameter with no row does.
    panel = commit_panel()
    bound = Binding("latent_x", "/audio/bass", "x", False)
    panel._commit("latent_x", bound, source="")
    (event,) = panel._runtime.events
    assert event.address == BINDING_CLEAR
    assert event.value == ClearBinding("latent_x")


def test_a_sourceless_row_with_an_expression_is_kept():
    # It shapes what will arrive on the parameter's own address, which is not
    # what an absent row does, so the typing survives even switched off.
    panel = commit_panel()
    panel._commit("latent_x", None, expression="x*2")
    (event,) = panel._runtime.events
    assert event.value == Binding("latent_x", "", "x*2", False)


def test_editing_only_the_expression_preserves_the_source_and_enabled_flag():
    panel = commit_panel()
    bound = Binding("truncation_psi", "/audio/level", "x", False)
    panel._commit("truncation_psi", bound, expression="x*2")
    (event,) = panel._runtime.events
    assert event.value == Binding("truncation_psi", "/audio/level", "x*2", False)


def test_a_blank_expression_falls_back_to_identity():
    panel = commit_panel()
    bound = Binding("latent_x", "/audio/bass", "x*2", True)
    panel._commit("latent_x", bound, expression="  ")
    (event,) = panel._runtime.events
    assert event.value.expression == "x"


def test_committing_forgets_the_drafts_for_that_parameter_only():
    panel = commit_panel()
    panel._drafts[("latent_x", "source")] = "typed"
    panel._drafts[("latent_y", "source")] = "other"
    panel._commit("latent_x", None, source="/a/b")
    assert ("latent_x", "source") not in panel._drafts
    assert panel._drafts[("latent_y", "source")] == "other"


def test_a_failure_carrying_no_message_is_still_described():
    # Shared with the audio transport, which reports its errors the same way.
    assert describe(OSError("folder is gone")) == "folder is gone"
    assert describe(KeyError()) == "KeyError"


def presets_at(directory):
    """A presets panel on a clock that is always past the rescan interval."""
    ticks = itertools.count(0.0, 10.0)
    return PresetsPanel(RecordingRuntime(), directory, clock=lambda: next(ticks))


def test_a_listing_failure_stops_being_reported_once_listing_works(
    tmp_path, monkeypatch
):
    panel = presets_at(tmp_path)

    def boom(directory):
        raise OSError("folder is gone")

    monkeypatch.setattr(presets, "list_presets", boom)
    assert panel._names() == []
    assert "presets folder" in (panel.report_error() or "")

    monkeypatch.undo()
    assert panel._names() == []
    assert panel.report_error() is None


def test_a_save_failure_stays_reported_through_a_later_rescan(tmp_path, monkeypatch):
    panel = presets_at(tmp_path)

    def boom(state, path):
        raise OSError("disk is full")

    monkeypatch.setattr(presets, "save", boom)
    panel._save("evening")
    assert "Could not save evening" in (panel.report_error() or "")

    monkeypatch.undo()
    panel._names()
    # The rescan says nothing about the failed save, so it may not clear it.
    assert "Could not save evening" in (panel.report_error() or "")


def test_a_saved_preset_is_reported_after_a_transient_listing_failure(
    tmp_path, monkeypatch
):
    panel = presets_at(tmp_path)

    def boom(directory):
        raise OSError("folder is busy")

    monkeypatch.setattr(presets, "list_presets", boom)
    panel._names()
    monkeypatch.undo()

    panel._save("evening")
    panel._names()

    # An error takes precedence over the message, so a stale one hides "Saved".
    assert panel.report_error() is None
    assert panel._message == "Saved evening."
