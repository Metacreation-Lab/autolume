"""Pure logic behind the fullscreen output window.

GL and GLFW cannot be driven headless, so window creation, drawing and
teardown are manual-only. What is pinned here is everything the driver
decides before it ever touches GL: the letterbox rectangle's arithmetic and
the lifecycle state machine that decides create, destroy, upload or nothing.
"""

from autolume.live.ui.output_window import Action, Rect, decide_action, letterbox_rect

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
