import numpy as np

from autolume.live.core.generator import ModelInfo
from autolume.live.core.motion import MOTION_PARAMS, WalkState, drives, integrate
from autolume.live.core.params import Binding, ControlState
from autolume.live.core.touch import TOUCH_GRACE, TouchTracker

_INFO = ModelInfo(pkl_path="model.pkl", z_dim=2, num_ws=8)


def test_not_playing_is_identity():
    state = ControlState(latent_x=1.0)
    assert integrate(state, 0.5) == state


def test_playing_advances_by_speed_times_dt():
    state = ControlState(anim_playing=True, anim_speed_x=2.0, anim_speed_y=-1.0)
    out = integrate(state, 0.5)
    assert out.latent_x == 1.0
    assert out.latent_y == -0.5


def test_zero_or_negative_dt_is_identity():
    state = ControlState(anim_playing=True, anim_speed_x=2.0)
    assert integrate(state, 0.0) == state
    assert integrate(state, -1.0) == state


def test_deterministic_over_split_intervals():
    state = ControlState(anim_playing=True, anim_speed_x=1.0)
    whole = integrate(state, 1.0)
    halves = integrate(integrate(state, 0.5), 0.5)
    assert abs(whole.latent_x - halves.latent_x) < 1e-9


def test_a_held_axis_does_not_move_and_resumes_after_the_grace():
    state = ControlState(anim_playing=True, anim_speed_x=2.0, anim_speed_y=2.0)
    touch = TouchTracker()
    touch.begin("latent_x", 10.0)

    held = integrate(state, 0.5, touch, 10.0)
    assert held.latent_x == 0.0
    assert held.latent_y == 1.0

    touch.end("latent_x", 10.0)
    in_grace = integrate(state, 0.5, touch, 10.0 + TOUCH_GRACE / 2.0)
    assert in_grace.latent_x == 0.0

    resumed = integrate(state, 0.5, touch, 10.0 + TOUCH_GRACE)
    assert resumed.latent_x == 1.0


def test_an_enabled_binding_takes_the_axis_away_from_motion():
    state = ControlState(
        anim_playing=True,
        anim_speed_x=2.0,
        anim_speed_y=2.0,
        bindings=(Binding("latent_x", "/audio/level"),),
    )
    out = integrate(state, 0.5)
    assert out.latent_x == 0.0
    assert out.latent_y == 1.0


def test_a_disabled_binding_leaves_the_axis_to_motion():
    state = ControlState(
        anim_playing=True,
        anim_speed_x=2.0,
        bindings=(Binding("latent_x", "/audio/level", enabled=False),),
    )
    assert integrate(state, 0.5).latent_x == 1.0


def test_a_row_with_no_source_leaves_the_axis_to_motion():
    # It writes when a message happens to arrive on the parameter's own
    # address, exactly like the unmapped default, so it takes nothing away.
    state = ControlState(
        anim_playing=True,
        anim_speed_x=2.0,
        bindings=(Binding("latent_x", "", "x*2"),),
    )
    assert integrate(state, 0.5).latent_x == 1.0
    assert drives(state, "latent_x")


def test_clearing_a_binding_hands_the_axis_back_to_motion():
    bound = ControlState(
        anim_playing=True,
        anim_speed_x=2.0,
        bindings=(Binding("latent_x", "/audio/level"),),
    )
    assert integrate(bound, 0.5).latent_x == 0.0
    cleared = ControlState(anim_playing=True, anim_speed_x=2.0)
    assert integrate(cleared, 0.5).latent_x == 1.0


def test_a_binding_on_one_axis_does_not_stop_the_other():
    state = ControlState(
        anim_playing=True,
        anim_speed_x=2.0,
        anim_speed_y=2.0,
        bindings=(Binding("latent_y", "/audio/level"),),
    )
    out = integrate(state, 0.5)
    assert out.latent_x == 1.0
    assert out.latent_y == 0.0


def test_a_non_finite_step_is_refused_like_any_other_write():
    state = ControlState(anim_playing=True, latent_x=1e308, anim_speed_x=1e308)
    assert integrate(state, 1.0).latent_x == 1e308


def test_motion_drives_an_axis_only_while_the_animation_plays():
    assert drives(ControlState(anim_playing=True), "latent_x")
    assert not drives(ControlState(anim_playing=False), "latent_x")


def test_motion_drives_nothing_it_does_not_write():
    playing = ControlState(anim_playing=True, vector_mode=True)
    assert MOTION_PARAMS == ("latent_x", "latent_y", "latent_vec")
    assert not drives(playing, "truncation_psi")
    assert not drives(playing, "anim_speed_x")


def test_an_enabled_binding_takes_the_axis_off_motion_but_a_disabled_one_does_not():
    bound = ControlState(
        anim_playing=True, bindings=(Binding("latent_x", "/audio/level"),)
    )
    assert not drives(bound, "latent_x")
    assert drives(bound, "latent_y")

    parked = ControlState(
        anim_playing=True,
        bindings=(Binding("latent_x", "/audio/level", enabled=False),),
    )
    assert drives(parked, "latent_x")


def test_a_held_axis_is_not_motion_driven_until_the_grace_is_over():
    state = ControlState(anim_playing=True)
    touch = TouchTracker()
    touch.begin("latent_x", 10.0)
    assert not drives(state, "latent_x", touch, 10.0)
    touch.end("latent_x", 10.0)
    assert not drives(state, "latent_x", touch, 10.0 + TOUCH_GRACE / 2.0)
    assert drives(state, "latent_x", touch, 10.0 + TOUCH_GRACE)


def test_the_predicate_answers_exactly_what_the_integrator_does():
    """The marker the UI draws from this must never disagree with the engine.

    Every combination of the conditions either function looks at, checked as
    one question: did the integrator move this axis, and did the predicate say
    it would.
    """
    touch = TouchTracker()
    touch.begin("latent_x", 10.0)
    cases = []
    for playing in (True, False):
        for binding in (
            (),
            (Binding("latent_x", "/a"),),
            (Binding("latent_x", "/a", enabled=False),),
            (Binding("latent_y", "/a"),),
        ):
            for tracker in (None, touch):
                cases.append(
                    (
                        ControlState(
                            anim_playing=playing,
                            anim_speed_x=2.0,
                            anim_speed_y=2.0,
                            bindings=binding,
                        ),
                        tracker,
                    )
                )

    for state, tracker in cases:
        out = integrate(state, 0.5, tracker, 10.0)
        for name in MOTION_PARAMS:
            moved = getattr(out, name) != getattr(state, name)
            assert drives(state, name, tracker, 10.0) is moved, (state, tracker, name)


# --- vector walk ---------------------------------------------------------


def test_vector_walk_drifts_toward_the_target_by_gain_times_speed_times_dt():
    walk = WalkState(np.random.RandomState(0))
    walk.target = np.array([10.0, 0.0])
    state = ControlState(
        anim_playing=True, vector_mode=True, anim_speed_x=0.5, latent_vec=(0.0, 0.0)
    )
    out = integrate(state, 0.1, model_info=_INFO, walk=walk)
    assert out.latent_vec == (0.5, 0.0)


def test_negative_speed_drifts_the_same_direction_toward_the_target():
    walk = WalkState(np.random.RandomState(0))
    walk.target = np.array([10.0, 0.0])
    state = ControlState(
        anim_playing=True, vector_mode=True, anim_speed_x=-0.5, latent_vec=(0.0, 0.0)
    )
    out = integrate(state, 0.1, model_info=_INFO, walk=walk)
    assert out.latent_vec == (0.5, 0.0)


def test_retarget_fires_under_the_threshold_and_draws_from_the_injected_rng():
    expected_target = np.random.RandomState(123).randn(2)
    walk = WalkState(np.random.RandomState(123))
    walk.target = np.array([0.5, 0.0])  # distance 0.5 from (0, 0), under 1.0
    state = ControlState(
        anim_playing=True, vector_mode=True, anim_speed_x=1.0, latent_vec=(0.0, 0.0)
    )
    integrate(state, 0.1, model_info=_INFO, walk=walk)
    assert np.array_equal(walk.target, expected_target)


def test_a_held_vector_does_not_walk():
    walk = WalkState(np.random.RandomState(0))
    state = ControlState(
        anim_playing=True, vector_mode=True, anim_speed_x=1.0, latent_vec=(5.0, 5.0)
    )
    touch = TouchTracker()
    touch.begin("latent_vec", 10.0)
    out = integrate(state, 0.5, touch, 10.0, model_info=_INFO, walk=walk)
    assert out.latent_vec == state.latent_vec


def test_a_binding_on_latent_x_does_not_suppress_the_vector_walk():
    walk = WalkState(np.random.RandomState(0))
    state = ControlState(
        anim_playing=True,
        vector_mode=True,
        anim_speed_x=1.0,
        latent_vec=(5.0, 5.0),
        bindings=(Binding("latent_x", "/audio/level"),),
    )
    out = integrate(state, 0.1, model_info=_INFO, walk=walk)
    assert out.latent_x == state.latent_x
    assert out.latent_vec != state.latent_vec


def test_loop_active_freezes_both_the_seed_walk_and_the_vector_walk():
    walk = WalkState(np.random.RandomState(0))
    state = ControlState(
        anim_playing=True,
        vector_mode=True,
        loop_active=True,
        anim_speed_x=1.0,
        anim_speed_y=1.0,
        latent_vec=(5.0, 5.0),
    )
    out = integrate(state, 0.5, model_info=_INFO, walk=walk)
    assert out == state


def test_an_empty_vector_with_the_walk_on_is_a_no_op():
    walk = WalkState(np.random.RandomState(0))
    state = ControlState(anim_playing=True, vector_mode=True, anim_speed_x=1.0)
    out = integrate(state, 0.5, model_info=_INFO, walk=walk)
    assert out.latent_vec == ()
    assert walk.target is None


def test_a_vector_that_no_longer_matches_z_dim_is_left_alone():
    walk = WalkState(np.random.RandomState(0))
    state = ControlState(
        anim_playing=True,
        vector_mode=True,
        anim_speed_x=1.0,
        latent_vec=(1.0, 2.0, 3.0),
    )
    out = integrate(state, 0.5, model_info=_INFO, walk=walk)
    assert out.latent_vec == state.latent_vec


def test_the_walk_runs_without_model_info_as_long_as_the_vector_is_not_empty():
    walk = WalkState(np.random.RandomState(0))
    state = ControlState(
        anim_playing=True, vector_mode=True, anim_speed_x=1.0, latent_vec=(5.0, 5.0)
    )
    out = integrate(state, 0.5, walk=walk)
    assert out.latent_vec != state.latent_vec


def test_vector_mode_off_leaves_the_vector_alone_even_with_a_walk_state():
    walk = WalkState(np.random.RandomState(0))
    state = ControlState(
        anim_playing=True, anim_speed_x=1.0, latent_vec=(5.0, 5.0)
    )
    out = integrate(state, 0.5, model_info=_INFO, walk=walk)
    assert out.latent_vec == state.latent_vec


def test_no_walk_state_means_the_vector_never_moves():
    state = ControlState(
        anim_playing=True, vector_mode=True, anim_speed_x=1.0, latent_vec=(5.0, 5.0)
    )
    out = integrate(state, 0.5, model_info=_INFO)
    assert out.latent_vec == state.latent_vec


def test_drives_answers_for_the_vector_too():
    assert drives(ControlState(anim_playing=True, vector_mode=True), "latent_vec")
    assert not drives(ControlState(anim_playing=True, vector_mode=False), "latent_vec")
    loop_state = ControlState(
        anim_playing=True, vector_mode=True, loop_active=True
    )
    assert not drives(loop_state, "latent_vec")
    assert not drives(ControlState(anim_playing=True, loop_active=True), "latent_x")
    assert not drives(ControlState(anim_playing=True, loop_active=True), "latent_y")
