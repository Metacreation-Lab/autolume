"""Pure helpers behind the panels.

imgui cannot be driven headless, so the drawing itself is verified by hand.
What can be pinned down is the logic the drawing reads from, which is why it
lives in module functions rather than inside the gui methods.
"""

import numpy as np

from autolume.live.core.params import (
    BINDING_CLEAR,
    BINDING_SET,
    Binding,
    ClearBinding,
    ParamKind,
)
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
)
from autolume.live.ui.panels.presets import is_valid_name


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


def test_bindable_specs_leave_out_text_parameters():
    names = [spec.name for spec in bindable_specs()]
    assert "pkl_path" not in names
    assert "latent_x" in names
    assert all(spec.kind is not ParamKind.STR for spec in bindable_specs())


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


class RecordingRuntime:
    """Captures what a panel submits, without a control loop behind it."""

    def __init__(self):
        self.events = []

    def submit(self, event):
        self.events.append(event)


def commit_panel():
    return MappingPanel(RecordingRuntime())


def test_typed_source_becomes_a_binding_with_a_canonical_address():
    panel = commit_panel()
    panel._commit("latent_x", None, source="audio/bass")
    (event,) = panel._runtime.events
    assert event.address == BINDING_SET
    assert event.value == Binding("latent_x", "/audio/bass", "x", True)
    assert event.source == "ui"


def test_emptying_the_source_clears_with_an_object_never_a_string():
    panel = commit_panel()
    bound = Binding("latent_x", "/audio/bass", "x", True)
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
