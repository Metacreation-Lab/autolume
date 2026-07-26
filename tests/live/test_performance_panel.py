"""Pure helpers behind the Performance panel.

The panel's job is to say what the machine is actually doing, which is a
different fact from what was asked for, so every one of these is about a
disagreement between a parameter and a published status. imgui itself cannot
be driven headless; the sentences it draws can be.
"""

from autolume.live.io.ndi import NdiStatus
from autolume.live.io.recorder import RecorderStatus
from autolume.live.ui.panels.perform import combo_index
from autolume.live.ui.panels.performance import (
    DEVICE_LABELS,
    DEVICE_VALUES,
    device_note,
    elapsed_text,
    ndi_note,
    osc_note,
    recording_note,
    stats_text,
    superres_note,
    superres_state,
)


def test_every_device_value_has_a_label_to_read():
    assert len(DEVICE_VALUES) == len(DEVICE_LABELS)
    assert DEVICE_VALUES[0] == "auto"


def test_a_combo_finds_the_value_it_is_showing():
    assert combo_index("mps", DEVICE_VALUES) == 2


def test_a_value_the_combo_has_no_entry_for_selects_nothing():
    assert combo_index("rocm", DEVICE_VALUES) == -1


def test_no_device_the_combo_offers_is_ever_called_unavailable():
    # The note and the combo have to agree on which strings are unknown, or a
    # performer picks an entry and is told it does not exist.
    for value in DEVICE_VALUES:
        assert device_note(value, value, None) is None


def test_a_failed_device_switch_names_the_device_that_was_refused():
    note = device_note("cuda", "cpu", "No CUDA device is available.")
    assert note == "Could not switch to cuda. No CUDA device is available."


def test_an_unknown_device_string_is_named_rather_than_shown_as_blank():
    assert device_note("rocm", "cpu", None) == "rocm is not a device this build offers."


def test_an_auto_session_says_which_device_it_actually_picked():
    assert device_note("auto", "mps", None) == "Rendering on mps."


def test_a_device_running_exactly_what_was_asked_for_says_nothing():
    assert device_note("cpu", "cpu", None) is None


def test_a_device_with_nothing_loaded_yet_says_nothing():
    assert device_note("cuda", None, None) is None


def test_a_failed_rebind_names_both_the_refused_port_and_the_one_still_serving():
    note = osc_note(1338, 1400, "Address already in use")
    assert "1338" in note
    assert "Still listening on port 1400." in note


def test_a_failed_first_bind_has_no_port_left_to_name():
    note = osc_note(1338, None, "Address already in use")
    assert note == "Could not listen on port 1338. Address already in use"


def test_a_port_taken_and_scanned_past_is_reported_as_a_success():
    assert osc_note(1338, 1339, None) == "Port 1338 was taken. Listening on port 1339."


def test_a_bound_port_matching_the_request_still_says_where_it_is_listening():
    assert osc_note(1338, 1338, None) == "Listening on port 1338."


def test_osc_switched_off_says_so_rather_than_naming_a_port():
    assert osc_note(1338, None, None) == "OSC input is off."


def test_permanently_disabled_superres_reports_the_load_failure():
    note = superres_note("The weights are missing.", None)
    assert note == "Super-res is off. The weights are missing."


def test_a_load_failure_is_reported_ahead_of_a_transient_one():
    note = superres_note("The weights are missing.", "Out of memory")
    assert "The weights are missing." in note
    assert "Out of memory" not in note


def test_a_transient_failure_is_not_claimed_to_be_about_this_frame():
    # `last_error` survives the calls that short circuit before a forward
    # pass, so the wording must not promise the current frame is the one that
    # hit it (core/superres.py's recorded trade-off).
    note = superres_note(None, "Out of memory")
    assert note == "Super-res reported a problem. Out of memory"
    assert "frame" not in note


def test_healthy_superres_says_nothing():
    assert superres_note(None, None) is None


class Stage:
    def __init__(self, reason, last_error):
        self.disabled_reason = reason
        self.last_error = last_error


class Model:
    def __init__(self, stage):
        self._superres = stage


def test_superres_state_reads_the_stage_off_the_rendering_model():
    assert superres_state(Model(Stage("gone", "oom"))) == ("gone", "oom")


def test_a_model_without_a_superres_stage_reports_nothing_wrong():
    assert superres_state(object()) == (None, None)


def test_no_model_at_all_reports_nothing_wrong():
    assert superres_state(None) == (None, None)


def test_a_take_is_timed_the_way_a_take_is_talked_about():
    assert elapsed_text(0.0) == "0:00"
    assert elapsed_text(9.9) == "0:09"
    assert elapsed_text(61.0) == "1:01"
    assert elapsed_text(600.0) == "10:00"


def test_a_negative_elapsed_is_still_a_readable_zero():
    assert elapsed_text(-3.0) == "0:00"


def test_a_running_take_reports_how_long_it_has_been_going():
    note = recording_note(RecorderStatus(recording=True), 65.0)
    assert note == "Recording 1:05."


def test_a_take_behind_the_encoder_says_how_many_frames_it_lost():
    note = recording_note(
        RecorderStatus(recording=True, frames_dropped=12), 3.0
    )
    assert note == "Recording 0:03. The encoder is behind by 12 frames."


def test_a_take_keeping_up_does_not_mention_dropped_frames():
    note = recording_note(RecorderStatus(recording=True, frames_written=90), 1.0)
    assert "frames" not in note


def test_a_finished_take_names_the_file_it_wrote():
    note = recording_note(RecorderStatus(path="/tmp/take.mp4"), None)
    assert note == "Saved to /tmp/take.mp4."


def test_a_take_that_failed_shows_the_failure_ahead_of_the_path():
    note = recording_note(
        RecorderStatus(path="/tmp/take.mp4", error="Could not open it."), None
    )
    assert note == "Could not open it."


def test_nothing_recorded_yet_says_nothing():
    assert recording_note(RecorderStatus(), None) is None


def test_a_missing_ndi_runtime_is_the_only_thing_worth_saying():
    note = ndi_note(NdiStatus(sending=True, name="Autolume Live"), False)
    assert note == "NDI is not installed on this machine."


def test_a_sending_session_names_what_it_advertises():
    note = ndi_note(NdiStatus(sending=True, name="Autolume Live"), True)
    assert note == "Sending as Autolume Live."


def test_a_failed_session_shows_its_reason_rather_than_its_name():
    note = ndi_note(NdiStatus(name="Autolume Live", error="No senders left."), True)
    assert note == "No senders left."


def test_an_idle_installed_ndi_says_nothing():
    assert ndi_note(NdiStatus(), True) is None


def test_render_stats_report_the_rate_and_the_interval_it_works_out_to():
    assert stats_text(60.0) == "60.0 fps. Average interval 16.7 ms."


def test_nothing_rendering_yet_is_said_rather_than_divided_by():
    assert stats_text(0.0) == "Not rendering yet."


def test_every_panel_sentence_is_drawable_by_the_bundled_font():
    # The bundled font has no symbol range, so a Unicode glyph renders as a
    # question mark. Every sentence this panel can draw stays inside ASCII.
    sentences = [
        device_note("cuda", "cpu", "boom"),
        device_note("rocm", None, None),
        device_note("auto", "mps", None),
        osc_note(1338, 1400, "taken"),
        osc_note(1338, None, None),
        osc_note(1338, 1339, None),
        superres_note("gone", None),
        superres_note(None, "oom"),
        recording_note(RecorderStatus(recording=True, frames_dropped=2), 3.0),
        recording_note(RecorderStatus(path="/tmp/a.mp4"), None),
        ndi_note(NdiStatus(), False),
        ndi_note(NdiStatus(sending=True, name="Autolume Live"), True),
        stats_text(30.0),
        stats_text(0.0),
    ]
    for sentence in sentences:
        assert sentence is not None
        assert sentence.isascii(), sentence
