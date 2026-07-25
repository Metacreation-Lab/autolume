from autolume.live.core.touch import TOUCH_GRACE, TouchTracker


def test_untouched_name_is_not_held():
    assert not TouchTracker().is_held("latent_x", 0.0)


def test_name_is_held_from_the_moment_it_is_begun():
    tracker = TouchTracker()
    tracker.begin("latent_x", 10.0)
    assert tracker.is_held("latent_x", 10.0)


def test_a_begun_name_stays_held_however_long_the_drag_lasts():
    tracker = TouchTracker()
    tracker.begin("latent_x", 10.0)
    assert tracker.is_held("latent_x", 10.0 + 3600.0)


def test_name_stays_held_inside_the_grace_window():
    tracker = TouchTracker()
    tracker.begin("latent_x", 10.0)
    tracker.end("latent_x", 11.0)
    assert tracker.is_held("latent_x", 11.0 + TOUCH_GRACE / 2)


def test_name_is_released_once_the_grace_window_has_passed():
    tracker = TouchTracker()
    tracker.begin("latent_x", 10.0)
    tracker.end("latent_x", 11.0)
    assert not tracker.is_held("latent_x", 11.0 + TOUCH_GRACE)


def test_only_the_touched_name_is_held():
    tracker = TouchTracker()
    tracker.begin("latent_x", 10.0)
    assert not tracker.is_held("latent_y", 10.0)


def test_end_without_begin_is_a_no_op():
    tracker = TouchTracker()
    tracker.end("latent_x", 10.0)
    assert not tracker.is_held("latent_x", 10.0)
    assert not tracker.is_held("latent_x", 10.0 + TOUCH_GRACE / 2)


def test_repeated_begin_is_idempotent():
    tracker = TouchTracker()
    tracker.begin("latent_x", 10.0)
    tracker.begin("latent_x", 12.0)
    tracker.end("latent_x", 13.0)
    assert not tracker.is_held("latent_x", 13.0 + TOUCH_GRACE)


def test_begin_after_a_release_holds_the_name_again():
    tracker = TouchTracker()
    tracker.begin("latent_x", 10.0)
    tracker.end("latent_x", 11.0)
    tracker.begin("latent_x", 20.0)
    assert tracker.is_held("latent_x", 20.0 + 3600.0)
