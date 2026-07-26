"""Pure helpers behind the panels.

imgui cannot be driven headless, so the drawing itself is verified by hand.
What can be pinned down is the logic the drawing reads from, which is why it
lives in module functions rather than inside the gui methods.
"""

import itertools

import numpy as np

from autolume.live.core import presets
from autolume.live.core.generator import ModelInfo
from autolume.live.core.params import (
    BINDING_CLEAR,
    BINDING_SET,
    REGISTRY,
    Binding,
    ClearBinding,
    ControlState,
    Keyframe,
)
from autolume.live.errors import describe
from autolume.live.ui.panels.audio import (
    bar_value,
    device_index,
    device_labels,
    spectrum_ceiling,
    spectrum_values,
)
from autolume.live.ui.panels.loop import (
    captured_keyframe,
    desired_noise_key,
    noise_table_pending,
)
from autolume.live.ui.panels.mapping import (
    MappingPanel,
    bindable_specs,
    canonical_address,
    display_label,
    reference_note,
)
from autolume.live.ui.panels.perform import load_vector_file, save_vector_file
from autolume.live.ui.panels.presets import (
    PresetsPanel,
    is_valid_name,
    missing_model_message,
)
from autolume.live.ui.panels.preview import (
    DisplayMode,
    PreviewPanel,
    PreviewStatus,
    centred_offset,
    dims_the_frame,
    displayed_size,
    frame_placement,
    magnifies,
    model_name,
    needs_refresh,
    preview_status,
)


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


WIKIART = "/home/vj/models/wikiart-1024.pkl"


def test_the_preview_names_the_model_it_is_loading():
    """A load takes seconds, and a still frame with no word on it is a hang.

    By filename, not by path: a status line that grew with the path would push
    the frame under it around every time a model in a deeper folder loaded.
    """
    status = preview_status(WIKIART, None, False, False)
    assert status.text == "Loading wikiart-1024.pkl."
    assert not status.error


def test_a_load_that_failed_is_reported_over_the_frames_still_arriving():
    """The state the preview used to lie about, and the one that matters most.

    A load failing partway through a set leaves the previous model rendering
    happily. Without this the performer sees a preview that simply ignored
    them, and the old wording said "Waiting for frames", which is the worst of
    the three because it implies something is still coming.
    """
    status = preview_status(None, "No such file", False, True)
    assert status.text == "No such file"
    assert status.error
    assert preview_status(None, "No such file", False, False).error


def test_the_model_being_loaded_is_named_ahead_of_the_last_one_that_failed():
    # The error belongs to the model before this one by then, and it would be
    # read as this one failing before it has had a chance to.
    status = preview_status(WIKIART, "No such file", False, False)
    assert status.text == "Loading wikiart-1024.pkl."
    assert not status.error


def test_a_preview_with_frames_in_it_says_nothing():
    # The line is still drawn, so nothing on the panel moves when it goes
    # quiet, but a performance is not the place to read a running commentary.
    status = preview_status(None, None, True, True)
    assert status.text == ""
    assert not status.error


def test_an_empty_preview_with_no_model_invites_one():
    status = preview_status(None, None, False, False)
    assert "No model loaded" in status.text
    assert "Browse" in status.text
    assert not status.error


def test_an_empty_preview_with_a_model_loaded_is_waiting_rather_than_empty():
    # The one state the old wording was right about, and now the only one it
    # is used for.
    assert preview_status(None, None, True, False).text == "Waiting for frames."


def test_every_preview_status_says_something_the_bundled_font_can_draw():
    texts = [
        preview_status(WIKIART, None, False, False).text,
        preview_status(None, None, False, False).text,
        preview_status(None, None, True, False).text,
    ]
    for text in texts:
        assert text.isascii()
        assert ";" not in text
        assert " - " not in text
        assert text.endswith(".")


def test_the_display_modes_are_named_for_what_they_actually_do():
    """The old app called these "Raw" and "Fit" and had both backwards.

    Its "Raw" kept the aspect ratio and the native size, which is what everyone
    means by fit, and its "Fit" grew the frame to meet the area, which is
    stretching. Carrying those names over would have carried the confusion with
    them.
    """
    assert [mode.value for mode in DisplayMode] == ["Fit", "Stretch"]
    assert "Raw" not in {mode.value for mode in DisplayMode}


def test_the_preview_opens_fitted_and_keeps_the_mode_to_itself():
    # Panel state, not a registry parameter: it describes this window and not
    # the performance, so nothing carries it into a preset or out over OSC.
    panel = PreviewPanel(RecordingRuntime())
    assert panel._mode is DisplayMode.FIT
    assert "display_mode" not in REGISTRY
    assert not any("display_mode" in name for name in REGISTRY)


SQUARE = (1024, 1024)
# Device pixels per point on a display that does not scale, which is what
# these cases are stated in. The scaled display has tests of its own.
ONE_TO_ONE = 1.0
FIT = DisplayMode.FIT
STRETCH = DisplayMode.STRETCH


def test_fit_scales_by_whichever_axis_runs_out_first():
    # A square frame in a wide panel is bounded by the height, and the width
    # left over is letterboxing.
    assert displayed_size(SQUARE, (1200.0, 600.0), FIT, ONE_TO_ONE) == (600, 600)
    assert displayed_size(SQUARE, (600.0, 1200.0), FIT, ONE_TO_ONE) == (600, 600)
    assert displayed_size((1024, 512), (600.0, 600.0), FIT, ONE_TO_ONE) == (600, 300)


def test_fit_never_magnifies_a_frame_smaller_than_the_panel():
    """Native size, and a model that renders small is meant to look small.

    Magnifying is not fitting, so a Fit that enlarged would be misnamed, and
    the size a model renders at is worth being able to see for what it is.
    This is the old app's "Raw" behaviour under a name that describes it.
    """
    assert displayed_size((256, 256), (900.0, 900.0), FIT, ONE_TO_ONE) == (256, 256)
    assert displayed_size((256, 128), (900.0, 900.0), FIT, ONE_TO_ONE) == (256, 128)


def test_fit_leaves_a_frame_that_fits_exactly_alone():
    # The boundary between native and shrunk, from both sides.
    assert displayed_size((600, 600), (600.0, 600.0), FIT, ONE_TO_ONE) == (600, 600)
    assert displayed_size((601, 601), (600.0, 600.0), FIT, ONE_TO_ONE) == (600, 600)
    assert displayed_size((599, 599), (600.0, 600.0), FIT, ONE_TO_ONE) == (599, 599)


def test_magnifying_is_the_only_thing_the_two_modes_disagree_about():
    # A square panel, so the shape cannot confuse the one real difference.
    assert displayed_size((256, 256), (900.0, 900.0), STRETCH, ONE_TO_ONE) == (900, 900)
    assert displayed_size((256, 256), (900.0, 900.0), FIT, ONE_TO_ONE) == (256, 256)


def test_the_modes_agree_exactly_on_a_frame_too_big_for_the_panel():
    # Above native size Fit's cap never binds, so there is nothing left to
    # tell them apart. A Stretch that cropped or distorted would fail here.
    for area in ((1200.0, 600.0), (599.5, 601.5), (300.0, 900.0)):
        assert displayed_size((1024, 1024), area, FIT, ONE_TO_ONE) == displayed_size(
            (1024, 1024), area, STRETCH, ONE_TO_ONE
        )


def test_a_frame_smaller_than_the_panel_is_still_centred_in_it():
    # Both modes letterbox, so both leave room, and both split it evenly.
    small = displayed_size((256, 256), (900.0, 700.0), FIT, ONE_TO_ONE)
    assert centred_offset(small, (900.0, 700.0)) == (322.0, 222.0)
    grown = displayed_size((256, 256), (900.0, 700.0), STRETCH, ONE_TO_ONE)
    assert grown == (700, 700)
    assert centred_offset(grown, (900.0, 700.0)) == (100.0, 0.0)


def test_stretch_keeps_the_aspect_ratio_whatever_shape_the_panel_is():
    """The bars land on the axis with room left, never on the tight one.

    A wide panel and a square frame leaves the width over. A tall panel and a
    wide frame leaves the height over. A mode that filled the panel would
    return the panel's own size in both cases and distort the picture.
    """
    assert displayed_size(SQUARE, (1200.0, 600.0), STRETCH, ONE_TO_ONE) == (600, 600)
    assert displayed_size((1024, 512), (300.0, 900.0), STRETCH, ONE_TO_ONE) == (
        300,
        150,
    )


def test_neither_mode_ever_exceeds_the_panel_it_is_drawn_in():
    """What keeps the preview from scrolling, at every shape of panel.

    Including the fraction that rounding leaves: the size is drawn in whole
    pixels, so rounding up a fitted frame by one would be a panel that scrolls
    by one, on a whole class of window sizes rather than on a memorable one.
    """
    frames = ((1024, 1024), (1024, 512), (3, 1000), (1, 1))
    areas = ((1200.0, 600.0), (599.5, 601.5), (37.0, 900.0), (1.0, 1.0))
    for frame in frames:
        for area in areas:
            for mode in DisplayMode:
                width, height = displayed_size(frame, area, mode, ONE_TO_ONE)
                left, top = centred_offset((width, height), area)
                assert width <= area[0] and height <= area[1]
                assert left + width <= area[0] and top + height <= area[1]


def test_a_panel_or_a_frame_with_no_area_leaves_nothing_to_draw():
    """Both are reachable and neither may divide by anything.

    A dock split dragged shut has no width, and there is no frame at all until
    the first one arrives.
    """
    for mode in DisplayMode:
        assert displayed_size(SQUARE, (0.0, 600.0), mode, ONE_TO_ONE) == (0, 0)
        assert displayed_size(SQUARE, (600.0, 0.0), mode, ONE_TO_ONE) == (0, 0)
        assert displayed_size(SQUARE, (-5.0, -5.0), mode, ONE_TO_ONE) == (0, 0)
        assert displayed_size((0, 0), (600.0, 600.0), mode, ONE_TO_ONE) == (0, 0)
        assert displayed_size((1024, 0), (600.0, 600.0), mode, ONE_TO_ONE) == (0, 0)


def test_only_a_viewport_with_a_frame_in_it_is_dimmed():
    """With nothing rendered there is nothing to dim.

    The words already stand out on an empty panel, so a dim there would be a
    grey rectangle explaining itself.
    """
    loading = PreviewStatus("Loading wikiart-1024.pkl.")
    assert dims_the_frame(loading, True)
    assert not dims_the_frame(loading, False)


def test_a_preview_that_is_running_is_never_dimmed():
    # A dim over a preview that is fine is the panel claiming to be busy when
    # it is not, and it would sit there for the whole of a set.
    assert not dims_the_frame(PreviewStatus(""), True)


def test_a_failed_load_dims_the_frame_it_could_not_replace():
    # The model that is still rendering is the one the performer is looking at
    # while they read why the next one did not arrive.
    assert dims_the_frame(PreviewStatus("No such file", True), True)


def test_centred_offset_puts_the_spare_room_on_both_sides():
    assert centred_offset((600.0, 600.0), (1200.0, 600.0)) == (300.0, 0.0)
    assert centred_offset((0.0, 0.0), (0.0, 0.0)) == (0.0, 0.0)


def test_centred_offset_never_starts_something_off_the_left_edge():
    # A status line longer than a narrow panel. Losing the end of it beats
    # losing the beginning as well.
    assert centred_offset((900.0, 20.0), (300.0, 300.0)) == (0.0, 140.0)


def test_centred_offset_lands_on_a_whole_pixel():
    """Half a pixel of offset resamples the image across the whole grid.

    However exactly the frame was sized to the panel, an odd amount of spare
    room either side would undo it, so the split is rounded down and the extra
    pixel goes to the far side.
    """
    assert centred_offset((100.0, 100.0), (301.0, 303.0)) == (100.0, 101.0)


def test_native_size_is_one_frame_pixel_per_point_on_every_display():
    """A model is the same physical size whatever the display scale.

    A 256 frame at native size asks for 256 points on a 1x display and on a 2x
    one. What changes is how many device pixels that covers, which is what the
    texture then has to hold: 256 on the first and 512 on the second.

    Sizing it in device pixels instead would halve a model on a 2x screen,
    against the app's own convention that nothing changes physical size with
    the display, and against the old app, which draws through a projection in
    points and so lands here too.
    """
    for scale in (1.0, 2.0):
        place = frame_placement((256, 256), (900.0, 900.0), FIT, scale)
        assert place.size == (256.0, 256.0)
        assert place.pixels == (int(256 * scale), int(256 * scale))


def test_a_frame_at_native_size_is_resampled_before_imgui_draws_it():
    """Which is what keeps it crisp rather than interpolated.

    On a 2x display native size covers four device pixels per frame pixel, and
    imgui's sampler would blend them, so this is a magnification even though
    the mode is called Fit and the frame is at its native size.
    """
    place = frame_placement((64, 64), (900.0, 900.0), FIT, 2.0)
    assert place.pixels == (128, 128)
    assert magnifies((64, 64), place.pixels)


def test_stretch_fills_the_panel_in_pixels_it_can_actually_draw():
    """A stretched frame is measured against the pixels, then asked for in points.

    The quad still meets the panel edge, and the texture behind it holds one
    pixel for every pixel the panel has rather than a quarter of them.
    """
    place = frame_placement((256, 256), (450.0, 450.0), STRETCH, 2.0)
    assert place.pixels == (900, 900)
    assert place.size == (450.0, 450.0)


def test_a_centred_frame_is_offset_in_points_and_stays_on_the_pixel_grid():
    # 1800 pixels of panel and 512 drawn leaves 644 pixels either side, which
    # is 322 points, and lands the frame on a whole pixel.
    place = frame_placement((256, 256), (900.0, 900.0), FIT, 2.0)
    assert place.offset == (322.0, 322.0)
    assert (place.offset[0] * 2.0).is_integer()


def test_an_unchanged_frame_is_never_uploaded_again():
    # The whole reason the mailbox carries a sequence number.
    assert not needs_refresh(7, 7, (256, 256), (256, 256))


def test_a_new_frame_is_uploaded():
    assert needs_refresh(8, 7, (256, 256), (256, 256))


def test_a_magnified_frame_is_uploaded_again_when_the_panel_resizes():
    """The enlarging happens before the upload, so the panel's size is in it.

    Without this the picture would keep whatever width the panel had when the
    frame arrived, and a dock split dragged wider would draw a small texture
    over a large quad, which is the interpolation the enlarging exists to
    avoid.
    """
    assert needs_refresh(7, 7, (900, 900), (700, 700))


def test_only_a_magnified_frame_is_resampled():
    """A frame being shrunk goes to imgui's sampler untouched.

    That is the case where interpolation is wanted, and the case where a blit
    would be work done to make the picture worse.
    """
    assert not magnifies((1024, 1024), (600, 600))
    assert not magnifies((1024, 1024), (1024, 1024))
    assert magnifies((256, 256), (900, 900))
    assert magnifies((256, 256), (256, 300))


def test_model_name_falls_back_to_the_whole_path_when_there_is_no_filename():
    assert model_name(WIKIART) == "wikiart-1024.pkl"
    assert model_name("wikiart.pkl") == "wikiart.pkl"
    assert model_name("/home/vj/models/") == "/home/vj/models/"


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


def test_a_preset_that_found_both_its_models_reports_nothing():
    assert missing_model_message(None, None) is None


def test_a_preset_missing_only_its_first_model_names_that_one():
    assert missing_model_message("a.pkl", None) == (
        "Model file a.pkl is missing. The preset loaded without it."
    )


def test_a_preset_missing_only_its_second_model_is_reported_at_all():
    # The half that used to be dropped. `PresetData.missing_model2` existed and
    # nothing read it, so a mixing preset whose second model was absent loaded
    # in silence with `mixing_enabled` coming back on.
    assert missing_model_message(None, "b.pkl") == (
        "Model file b.pkl is missing. The preset loaded without it."
    )


def test_a_preset_missing_both_models_names_both_in_one_sentence():
    assert missing_model_message("a.pkl", "b.pkl") == (
        "Model files a.pkl and b.pkl are missing. The preset loaded without them."
    )


def test_the_missing_model_sentence_is_drawable_by_the_bundled_font():
    for message in (
        missing_model_message("a.pkl", None),
        missing_model_message(None, "b.pkl"),
        missing_model_message("a.pkl", "b.pkl"),
    ):
        assert message.isascii(), message


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


# --- vector load and save (perform.py, task 9) --------------------------------


def test_load_vector_file_round_trips_an_npy_array(tmp_path):
    path = tmp_path / "vector.npy"
    np.save(path, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert load_vector_file(str(path)) == [1.0, 2.0, 3.0]


def test_save_vector_file_writes_an_npy_array_by_default(tmp_path):
    path = tmp_path / "vector.npy"
    save_vector_file(str(path), (1.0, -2.0, 3.5))
    assert np.load(path).tolist() == [1.0, -2.0, 3.5]


def test_a_saved_vector_loads_back_through_the_same_pair_of_helpers(tmp_path):
    path = tmp_path / "vector.npy"
    save_vector_file(str(path), (1.5, -2.5, 3.0))
    assert load_vector_file(str(path)) == [1.5, -2.5, 3.0]


def test_load_vector_file_reads_a_torch_tensor(tmp_path):
    import torch

    path = tmp_path / "vector.pt"
    torch.save(torch.tensor([4.0, 5.0, -6.0]), path)
    assert load_vector_file(str(path)) == [4.0, 5.0, -6.0]


def test_save_vector_file_writes_a_torch_tensor_for_pt(tmp_path):
    import torch

    path = tmp_path / "vector.pt"
    save_vector_file(str(path), (4.0, 5.0, -6.0))
    assert torch.load(path, weights_only=True).tolist() == [4.0, 5.0, -6.0]


def test_a_saved_torch_vector_loads_back_through_the_same_pair_of_helpers(tmp_path):
    path = tmp_path / "vector.pt"
    save_vector_file(str(path), (4.0, 5.0, -6.0))
    assert load_vector_file(str(path)) == [4.0, 5.0, -6.0]


# --- loop panel helpers (loop.py, task 9) --------------------------------------


def test_desired_noise_key_is_none_without_a_loaded_model():
    state = ControlState(noise_loop_seed=3, noise_radius=2.0)
    assert desired_noise_key(state, None) is None


def test_desired_noise_key_combines_the_state_and_the_models_z_dim():
    state = ControlState(noise_loop_seed=3, noise_radius=2.0)
    info = ModelInfo(pkl_path="model.pkl", z_dim=512, num_ws=16)
    assert desired_noise_key(state, info) == (3, 2.0, 512)


def test_noise_table_pending_is_false_with_no_desired_key():
    assert noise_table_pending(None, None) is False
    assert noise_table_pending(None, (1, 1.0, 4)) is False


def test_noise_table_pending_compares_the_built_key_to_the_desired_one():
    desired = (1, 2.0, 4)
    assert noise_table_pending(desired, None) is True
    assert noise_table_pending(desired, (1, 2.0, 4)) is False
    assert noise_table_pending(desired, (1, 3.0, 4)) is True


def test_captured_keyframe_in_seed_mode_onto_a_seed_keyframe_clears_vec():
    """Kind follows navigation mode, not the keyframe's own prior kind
    (`vector_mode` False here), and the field the new kind does not use is
    cleared rather than left stale.
    """
    keyframe = Keyframe("seed", seed_x=99.0, seed_y=99.0, vec=(9.0, 9.0), project=False)
    state = ControlState(vector_mode=False, latent_x=4.0, latent_y=-1.0)
    captured = captured_keyframe(keyframe, state)
    assert captured.kind == "seed"
    assert (captured.seed_x, captured.seed_y) == (4.0, -1.0)
    assert captured.vec == ()
    # Seed mode never reads `project` (`_keyframe_to_w`), so it is left
    # alone rather than copied from `latent_project`.
    assert captured.project is False


def test_captured_keyframe_in_seed_mode_onto_a_vector_keyframe_switches_kind():
    """The bug this fixes: navigating in seed mode and snapping onto a
    keyframe that happens to already be `"vec"` used to leave it `"vec"`,
    filled from `latent_x`/`latent_y` as though they were a vector, which
    they are not. Kind now always follows the navigation mode.
    """
    keyframe = Keyframe("vec", seed_x=1.0, seed_y=2.0, vec=(9.0, 9.0), project=False)
    state = ControlState(vector_mode=False, latent_x=4.0, latent_y=-1.0)
    captured = captured_keyframe(keyframe, state)
    assert captured.kind == "seed"
    assert (captured.seed_x, captured.seed_y) == (4.0, -1.0)
    assert captured.vec == ()
    assert captured.project is False


def test_captured_keyframe_in_vector_mode_onto_a_vector_keyframe_clears_seed_fields():
    keyframe = Keyframe("vec", seed_x=99.0, seed_y=99.0, vec=(0.0, 0.0), project=False)
    state = ControlState(
        vector_mode=True, latent_vec=(1.0, 2.0, 3.0), latent_project=True
    )
    captured = captured_keyframe(keyframe, state)
    assert captured.kind == "vec"
    assert captured.vec == (1.0, 2.0, 3.0)
    # Untouched by a later switch back to seed kind: a leftover seed
    # position here would reappear as one nobody chose.
    assert (captured.seed_x, captured.seed_y) == (0.0, 0.0)
    # Copied from `latent_project`, part of what produced the frame being
    # captured (it decides whether `vec` reads as a `z` or a `w`).
    assert captured.project is True


def test_captured_keyframe_in_vector_mode_onto_a_seed_keyframe_switches_kind():
    """The same bug as the seed-mode case, the other direction: navigating
    by vector and snapping onto a `"seed"` keyframe used to leave it
    `"seed"`, filled from stale `latent_x`/`latent_y` that render nothing
    like the frame actually on screen in vector mode.
    """
    keyframe = Keyframe("seed", seed_x=1.0, seed_y=2.0, project=False)
    state = ControlState(
        vector_mode=True, latent_vec=(1.0, 2.0, 3.0), latent_project=True
    )
    captured = captured_keyframe(keyframe, state)
    assert captured.kind == "vec"
    assert captured.vec == (1.0, 2.0, 3.0)
    assert (captured.seed_x, captured.seed_y) == (0.0, 0.0)
    assert captured.project is True
