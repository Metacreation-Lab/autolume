"""Pure helpers behind the panels.

imgui cannot be driven headless, so the drawing itself is verified by hand.
What can be pinned down is the logic the drawing reads from, which is why it
lives in module functions rather than inside the gui methods.
"""

import numpy as np

from autolume.live.core.params import ParamKind
from autolume.live.ui.panels.audio import (
    bar_value,
    device_index,
    device_labels,
    spectrum_ceiling,
    spectrum_values,
)
from autolume.live.ui.panels.mapping import (
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
