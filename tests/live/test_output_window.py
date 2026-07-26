"""Pure logic behind the fullscreen output window.

GL and GLFW cannot be driven headless, so window creation, drawing and
teardown are manual-only. What is pinned here is everything the driver
decides before it ever touches GL: the letterbox rectangle's arithmetic and
the lifecycle state machine that decides create, destroy, upload or nothing.
"""

from autolume.live.ui.output_window import (
    Action,
    Rect,
    decide_action,
    letterbox_rect,
    pending_status,
    suppressed_fullscreen,
)

# --- letterbox_rect -----------------------------------------------------


def test_letterbox_frame_wider_than_monitor():
    # 2:1 frame into a square monitor: bound by width, letterboxed top/bottom.
    rect = letterbox_rect((2000, 500), (1000, 1000))
    assert rect == Rect(0, 375, 1000, 250)


def test_letterbox_frame_taller_than_monitor():
    # 1:2 frame into a square monitor: bound by height, letterboxed left/right.
    rect = letterbox_rect((500, 2000), (1000, 1000))
    assert rect == Rect(375, 0, 250, 1000)


def test_letterbox_exact_match():
    rect = letterbox_rect((800, 600), (800, 600))
    assert rect == Rect(0, 0, 800, 600)


def test_letterbox_frame_larger_than_monitor_same_aspect():
    # Same 16:9 aspect on both sides: an even downscale, no letterboxing.
    rect = letterbox_rect((3840, 2160), (1920, 1080))
    assert rect == Rect(0, 0, 1920, 1080)


def test_letterbox_frame_smaller_than_monitor_same_aspect():
    # Same 16:9 aspect, frame smaller: magnified to fill, no letterboxing.
    rect = letterbox_rect((640, 360), (1920, 1080))
    assert rect == Rect(0, 0, 1920, 1080)


def test_letterbox_non_square_frame_on_non_square_monitor():
    # 4:3 frame into a 16:9 monitor: bound by height, pillarboxed left/right.
    rect = letterbox_rect((1024, 768), (1920, 1080))
    assert rect == Rect(240, 0, 1440, 1080)


def test_letterbox_result_is_centred():
    rect = letterbox_rect((1024, 768), (1920, 1080))
    assert rect.x == (1920 - rect.width) // 2
    assert rect.y == (1080 - rect.height) // 2


def test_letterbox_no_frame_yet_draws_nothing():
    rect = letterbox_rect((0, 0), (1920, 1080))
    assert rect == Rect(0, 0, 0, 0)


def test_letterbox_no_monitor_area_draws_nothing():
    rect = letterbox_rect((1024, 768), (0, 0))
    assert rect == Rect(0, 0, 0, 0)


def test_letterbox_rounds_without_distorting_aspect():
    # 1000x999 does not divide evenly into 1920x1080 on either axis, so this
    # exercises int(round(...)) landing off the exact value and the //2
    # centring absorbing an odd pixel of leftover, neither of which the other
    # cases (chosen to divide evenly) touch at all.
    rect = letterbox_rect((1000, 999), (1920, 1080))
    assert rect == Rect(419, 0, 1081, 1080)
    frame_aspect = 1000 / 999
    rect_aspect = rect.width / rect.height
    assert abs(frame_aspect - rect_aspect) < 0.001
    assert rect.x == (1920 - rect.width) // 2
    assert rect.y == (1080 - rect.height) // 2


# --- decide_action --------------------------------------------------------


def test_param_turns_on_with_no_window_creates():
    action = decide_action(
        fullscreen=True, exists=False, close_requested=False, latest_seq=0, last_seq=-1
    )
    assert action is Action.CREATE


def test_param_off_with_no_window_does_nothing():
    action = decide_action(
        fullscreen=False, exists=False, close_requested=False, latest_seq=0, last_seq=-1
    )
    assert action is Action.NONE


def test_new_frame_while_open_uploads():
    action = decide_action(
        fullscreen=True, exists=True, close_requested=False, latest_seq=5, last_seq=4
    )
    assert action is Action.UPLOAD


def test_same_seq_does_not_reupload():
    action = decide_action(
        fullscreen=True, exists=True, close_requested=False, latest_seq=5, last_seq=5
    )
    assert action is Action.NONE


def test_same_seq_arriving_twice_stays_none():
    # Calling decide_action again with the seq the driver already recorded
    # from the first UPLOAD must not ask for a second one.
    first = decide_action(
        fullscreen=True, exists=True, close_requested=False, latest_seq=5, last_seq=4
    )
    assert first is Action.UPLOAD
    second = decide_action(
        fullscreen=True, exists=True, close_requested=False, latest_seq=5, last_seq=5
    )
    assert second is Action.NONE


def test_param_goes_false_while_window_open_destroys():
    action = decide_action(
        fullscreen=False, exists=True, close_requested=False, latest_seq=5, last_seq=5
    )
    assert action is Action.DESTROY


def test_window_reports_close_while_param_still_true_destroys():
    action = decide_action(
        fullscreen=True, exists=True, close_requested=True, latest_seq=5, last_seq=5
    )
    assert action is Action.DESTROY


def test_close_wins_over_a_pending_frame():
    # A frame arrived (latest_seq != last_seq) in the same tick the window
    # reports it should close. Closing wins: the frame is never uploaded.
    action = decide_action(
        fullscreen=True, exists=True, close_requested=True, latest_seq=6, last_seq=5
    )
    assert action is Action.DESTROY


def test_close_wins_when_param_also_already_false():
    action = decide_action(
        fullscreen=False, exists=True, close_requested=True, latest_seq=5, last_seq=5
    )
    assert action is Action.DESTROY


def test_no_window_and_close_requested_is_moot():
    # close_requested cannot be true with no window in practice, but the
    # function must not misread it as a reason to create one.
    action = decide_action(
        fullscreen=False, exists=False, close_requested=True, latest_seq=0, last_seq=-1
    )
    assert action is Action.NONE


# --- suppressed_fullscreen -------------------------------------------------


def test_suppression_masks_a_stale_true():
    # This window just submitted fullscreen=False, setting the deadline to
    # 10.0, and the control loop has not published it back yet, so this poll
    # still reads the stale True the submit is trying to undo.
    fullscreen, deadline = suppressed_fullscreen(True, suppress_until=10.0, now=9.9)
    assert fullscreen is False
    assert deadline == 10.0


def test_suppression_releases_a_genuine_reenable_after_the_deadline():
    # A submit-off followed later by a real re-enable: once the deadline has
    # passed, a True is no longer assumed stale and is handed straight
    # through, with the deadline cleared so it does not linger.
    fullscreen, deadline = suppressed_fullscreen(True, suppress_until=10.0, now=10.25)
    assert fullscreen is True
    assert deadline is None


def test_suppression_releases_exactly_at_the_deadline():
    fullscreen, deadline = suppressed_fullscreen(True, suppress_until=10.0, now=10.0000001)
    assert fullscreen is True
    assert deadline is None


def test_no_active_suppression_passes_true_through():
    fullscreen, deadline = suppressed_fullscreen(True, suppress_until=None, now=5.0)
    assert fullscreen is True
    assert deadline is None


def test_no_active_suppression_passes_false_through():
    fullscreen, deadline = suppressed_fullscreen(False, suppress_until=None, now=5.0)
    assert fullscreen is False
    assert deadline is None


# --- pending_status ---------------------------------------------------------


def test_status_survives_a_poll_before_its_dwell_ends():
    # Written at t=0 with a 5 second dwell (deadline 5.0); a poll partway
    # through must still show it, unrelated to whatever fullscreen is doing.
    status, deadline = pending_status(
        "Fullscreen output is unavailable.", status_until=5.0, now=2.0
    )
    assert status == "Fullscreen output is unavailable."
    assert deadline == 5.0


def test_status_disappears_after_its_dwell():
    status, deadline = pending_status(
        "Fullscreen output is unavailable.", status_until=5.0, now=5.0
    )
    assert status is None
    assert deadline is None


def test_no_status_stays_empty():
    status, deadline = pending_status(None, status_until=None, now=1.0)
    assert status is None
    assert deadline is None


def test_status_with_no_deadline_clears_defensively():
    # Should not happen in practice (every set carries a deadline), but a
    # status with nothing to expire it must not read as permanent.
    status, deadline = pending_status("stray", status_until=None, now=1.0)
    assert status is None
    assert deadline is None
