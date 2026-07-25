import logging

from autolume.live.core import touch as touch_module
from autolume.live.core.touch import TOUCH_GRACE, TOUCH_HOLD_LIMIT, TouchTracker


def test_untouched_name_is_not_held():
    assert not TouchTracker().is_held("latent_x", 0.0)


def test_name_is_held_from_the_moment_it_is_begun():
    tracker = TouchTracker()
    tracker.begin("latent_x", 10.0)
    assert tracker.is_held("latent_x", 10.0)


def test_a_begun_name_stays_held_for_the_whole_drag():
    tracker = TouchTracker()
    tracker.begin("latent_x", 10.0)
    assert tracker.is_held("latent_x", 10.0 + TOUCH_HOLD_LIMIT / 2)


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
    assert tracker.is_held("latent_x", 20.0 + TOUCH_HOLD_LIMIT / 2)


def test_a_hold_that_is_never_ended_lapses_at_the_ceiling():
    tracker = TouchTracker()
    tracker.begin("latent_x", 10.0)
    assert not tracker.is_held("latent_x", 10.0 + TOUCH_HOLD_LIMIT)


def test_a_lapsed_hold_is_reported_once_and_names_the_parameter(caplog):
    tracker = TouchTracker()
    tracker.begin("latent_x", 10.0)
    with caplog.at_level(logging.WARNING, logger=touch_module.__name__):
        for _ in range(5):
            assert not tracker.is_held("latent_x", 10.0 + TOUCH_HOLD_LIMIT)
    logged = [r for r in caplog.records if r.name == touch_module.__name__]
    assert len(logged) == 1
    assert "latent_x" in logged[0].getMessage()


def test_a_name_can_be_held_again_after_a_lapse():
    tracker = TouchTracker()
    tracker.begin("latent_x", 10.0)
    assert not tracker.is_held("latent_x", 10.0 + TOUCH_HOLD_LIMIT)
    tracker.begin("latent_x", 100.0)
    assert tracker.is_held("latent_x", 100.0)


def test_a_long_drag_that_ends_normally_still_gets_its_grace():
    tracker = TouchTracker()
    tracker.begin("latent_x", 10.0)
    tracker.end("latent_x", 10.0 + TOUCH_HOLD_LIMIT * 2)
    assert tracker.is_held("latent_x", 10.0 + TOUCH_HOLD_LIMIT * 2)
    assert not tracker.is_held("latent_x", 10.0 + TOUCH_HOLD_LIMIT * 2 + TOUCH_GRACE)
