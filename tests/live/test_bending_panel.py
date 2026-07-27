"""Pure helpers behind the Bending panel.

The load bearing one is `resolve_indices`. It is what turns "half the channels
at random" into a fixed tuple at edit time, which is the contract that kills
the legacy per-frame resample: the old app re-rolled every frame, so a saved
look was never the look that was saved. Every mode's boundaries are pinned here
because a selection that reaches past a tensor is a device side assert on CUDA,
not an exception.
"""

import numpy as np
import pytest

from autolume.live.core.generator import LayerInfo
from autolume.live.core.mapping import (
    _MAX_KERNEL_SIZE,
    _OPERATOR_ARITY,
    _validate_transform,
)
from autolume.live.core.params import Transform
from autolume.live.ui.panels.bending import (
    MAX_KERNEL_SIZE,
    MODE_ALL,
    MODE_CLUSTER,
    MODE_RANDOM,
    MODE_RANGE,
    OPERATOR_NAMES,
    OPERATORS,
    SELECTION_LABELS,
    SELECTION_MODES,
    IndexSelection,
    capture_choices,
    clamp_param,
    cluster_ids,
    cluster_indices,
    default_params,
    is_bendable,
    is_torgb,
    layer_marks,
    layer_noise_value,
    layer_ratio_value,
    load_cluster_file,
    load_direction_file,
    merged_directions,
    param_kinds,
    param_label,
    params_for_op,
    randomized_directions,
    resolve_indices,
    selected_layer,
    visible_layers,
)

CATALOG = (
    LayerInfo("b4.conv1", 512, 4, 4),
    LayerInfo("b4.torgb", 3, 4, 4),
    LayerInfo("b8.conv0", 512, 8, 8),
    LayerInfo("b8.conv1", 512, 8, 8),
    LayerInfo("b8.torgb", 3, 8, 8),
    LayerInfo("b16.conv0", 256, 16, 16),
    LayerInfo("b16.conv1", 256, 16, 16),
    LayerInfo("b16.torgb", 3, 16, 16),
    LayerInfo("output", 3, 16, 16),
)


def selection(**fields):
    made = IndexSelection()
    for name, value in fields.items():
        setattr(made, name, value)
    return made


# --- the operator table -----------------------------------------------------


def test_the_panel_offers_exactly_the_operators_the_validator_accepts():
    # Restated in the panel rather than imported, so this is what stops the two
    # from drifting: an operator the panel offers and the validator rejects
    # would draw a row that silently never applies.
    assert set(OPERATOR_NAMES) == set(_OPERATOR_ARITY)


def test_every_operator_declares_as_many_fields_as_the_validator_demands():
    for name, kinds in OPERATORS:
        assert len(kinds) == _OPERATOR_ARITY[name], name


def test_every_operator_starts_at_a_value_the_validator_accepts():
    # A freshly added transform has to be applicable straight away. The
    # validator drops an invalid one silently, so a bad default would look like
    # Add doing nothing at all.
    for name in OPERATOR_NAMES:
        transform = Transform(name, "b8.conv1", default_params(name), (0,))
        assert _validate_transform(transform) is not None, name


def test_a_switch_operator_never_carries_a_bool():
    # `bool` is an `int` subclass and the validator rejects one outright in any
    # numeric parameter, so a checkbox has to submit 1.0 and 0.0.
    for name, kinds in OPERATORS:
        if "switch" not in kinds:
            continue
        for value in default_params(name):
            assert not isinstance(value, bool)
            assert isinstance(value, float)


def test_only_the_two_kernel_operators_take_a_whole_number():
    kernels = {name for name, kinds in OPERATORS if "kernel" in kinds}
    assert kernels == {"erode", "dilate"}


def test_a_parameter_with_no_name_worth_reading_gets_an_empty_label():
    assert param_label("translate", 0) == "x"
    assert param_label("translate", 1) == "y"
    assert param_label("invert", 0) == ""
    assert param_label("translate", 5) == ""


def test_switching_operators_of_the_same_arity_keeps_the_dialled_in_number():
    assert params_for_op("dilate", (7.0,)) == (7.0,)


def test_growing_the_arity_fills_the_new_slot_from_the_defaults():
    assert params_for_op("translate", (3.0,)) == (3.0, 0.0)


def test_shrinking_the_arity_drops_the_slot_that_no_longer_exists():
    assert params_for_op("rotate", (3.0, 9.0)) == (3.0,)


def test_a_switch_never_inherits_a_number_from_the_previous_operator():
    # Any nonzero number reads as on, so inheriting 0.4 from a threshold would
    # silently turn the switch on rather than carry a value.
    assert params_for_op("invert", (0.4,)) == default_params("invert")


def test_a_kernel_is_rounded_to_a_whole_number_of_at_least_one():
    assert clamp_param("erode", 0, 0.0) == 1.0
    assert clamp_param("erode", 0, 2.4) == 2.0
    assert clamp_param("erode", 0, -5.0) == 1.0


def test_a_kernel_is_capped_rather_than_left_to_stall_the_render():
    # Typing 500 where 50 was meant is one keystroke. The cost of a kernel is
    # k squared taps per pixel, so an uncapped one does not raise, it blocks
    # the render thread on the frame in flight for minutes.
    assert clamp_param("erode", 0, 501.0) == float(MAX_KERNEL_SIZE)
    assert clamp_param("dilate", 0, 1e9) == float(MAX_KERNEL_SIZE)
    assert clamp_param("erode", 0, float(MAX_KERNEL_SIZE)) == float(MAX_KERNEL_SIZE)


def test_the_panel_and_the_validator_agree_on_the_kernel_ceiling():
    # Restated in the panel rather than imported, same as the operator table,
    # so this is what stops the two from drifting: a field that lets a value
    # through which the validator then drops would silently cost the edit.
    assert MAX_KERNEL_SIZE == _MAX_KERNEL_SIZE
    capped = clamp_param("erode", 0, 5000.0)
    assert _validate_transform(Transform("erode", "b8.conv1", (capped,), (0,)))


def test_a_scale_of_zero_is_corrected_rather_than_dropped():
    # The validator refuses a scale under 1e-6 outright, and its refusal drops
    # the whole transform, so typing zero on the way to something else would
    # cost the edit.
    corrected = clamp_param("scale", 0, 0.0)
    assert _validate_transform(Transform("scale", "b8.conv1", (corrected,), (0,)))
    assert clamp_param("scale", 0, -0.0000001) < 0.0


def test_an_ordinary_number_is_left_exactly_as_typed():
    assert clamp_param("rotate", 0, -37.5) == -37.5


def test_every_field_kind_the_panel_knows_how_to_draw_is_one_of_three():
    for _, kinds in OPERATORS:
        for kind in kinds:
            assert kind in ("float", "kernel", "switch")


def test_an_unknown_operator_has_no_fields_rather_than_raising():
    assert param_kinds("sobel") == ()
    assert default_params("sobel") == ()


# --- the layer list ---------------------------------------------------------


def test_the_list_keeps_the_convolutions_the_torgbs_and_the_output():
    assert is_bendable("b8.conv1")
    assert is_bendable("b8.torgb")
    assert is_bendable("output")


def test_the_list_drops_everything_that_is_not_a_tensor_to_bend():
    assert not is_bendable("b8")
    assert not is_bendable("b8.resample_filter")
    assert not is_bendable("b8.conv1.affine")


def test_a_torgb_layer_is_recognised_wherever_it_sits():
    assert is_torgb("b8.torgb")
    assert not is_torgb("b8.conv1")
    assert not is_torgb("output")


def test_advanced_shows_every_layer_the_filter_keeps_in_network_order():
    names = [layer.name for layer in visible_layers(CATALOG, simple=False)]
    assert names == [layer.name for layer in CATALOG]


def test_simple_shows_one_row_per_resolution_plus_the_output():
    names = [layer.name for layer in visible_layers(CATALOG, simple=True)]
    assert names == ["b4.conv1", "b8.conv0", "b16.conv0", "output"]


def test_simple_survives_a_gap_in_the_resolution_ladder():
    # The legacy widget counted 4, 8, 16 upward and stopped at the first
    # resolution missing from that walk, dropping every deeper layer silently.
    gapped = (
        LayerInfo("b4.conv1", 512, 4, 4),
        LayerInfo("b32.conv0", 256, 32, 32),
        LayerInfo("b64.conv0", 128, 64, 64),
    )
    names = [layer.name for layer in visible_layers(gapped, simple=True)]
    assert names == ["b4.conv1", "b32.conv0", "b64.conv0"]


def test_a_model_with_no_catalog_has_no_rows():
    assert visible_layers((), simple=True) == ()
    assert visible_layers((), simple=False) == ()


def test_a_row_is_marked_for_the_transforms_and_the_noise_it_carries():
    transforms = (Transform("ablate", "b8.conv1", (1.0,), (0,)),)
    noise = (("b16.conv0", 0.5),)
    ratios = (("b4.conv1", 2.0, 1.0),)
    assert layer_marks("b8.conv1", transforms, noise, ratios) == (True, False)
    assert layer_marks("b16.conv0", transforms, noise, ratios) == (False, True)
    assert layer_marks("b4.conv1", transforms, noise, ratios) == (False, True)
    assert layer_marks("b8.torgb", transforms, noise, ratios) == (False, False)


def test_a_noise_entry_of_zero_is_not_a_mark():
    # The sparse mapping stores absence for a neutral value, but a preset or a
    # dragged slider can leave an explicit zero behind, and a mark on it would
    # send a performer looking for an edit that is not there.
    assert layer_marks("b8.conv1", (), (("b8.conv1", 0.0),), ()) == (False, False)


def test_a_neutral_ratio_is_not_a_mark():
    assert layer_marks("b8.conv1", (), (), (("b8.conv1", 1.0, 1.0),)) == (False, False)


def test_a_layer_with_no_entry_reads_as_neutral():
    assert layer_noise_value((), "b8.conv1") == 0.0
    assert layer_ratio_value((), "b8.conv1") == (1.0, 1.0)


def test_a_layer_with_an_entry_reads_it_back():
    assert layer_noise_value((("b8.conv1", 0.75),), "b8.conv1") == 0.75
    assert layer_ratio_value((("b8.conv1", 2.0, 0.5),), "b8.conv1") == (2.0, 0.5)


def test_a_selection_that_outlived_its_model_falls_back_to_the_first_row():
    layers = visible_layers(CATALOG, simple=True)
    assert selected_layer(layers, "b8.conv0").name == "b8.conv0"
    assert selected_layer(layers, "b256.conv0").name == layers[0].name
    assert selected_layer((), "anything") is None


# --- the capture layer combo ------------------------------------------------


def test_the_capture_combo_offers_the_final_output_first_and_as_an_empty_value():
    values, labels = capture_choices(CATALOG)
    assert values[0] == ""
    assert labels[0] == "Final output"


def test_the_capture_combo_never_offers_the_output_layer_twice():
    values, _ = capture_choices(CATALOG)
    assert "output" not in values


def test_the_capture_combo_names_the_shape_of_each_layer():
    values, labels = capture_choices(CATALOG)
    assert values[1] == "b4.conv1"
    assert labels[1] == "b4.conv1 (512 x 4)"


def test_every_capture_label_is_drawable_by_the_bundled_font():
    _, labels = capture_choices(CATALOG)
    for label in labels:
        assert label.isascii(), label


# --- index resolution -------------------------------------------------------


def test_all_resolves_to_every_channel():
    assert resolve_indices(selection(mode=MODE_ALL), "b8.conv1", 4) == (0, 1, 2, 3)


def test_all_on_a_layer_with_no_channels_resolves_to_nothing():
    assert resolve_indices(selection(mode=MODE_ALL), "b8.conv1", 0) == ()


def test_a_percent_of_zero_selects_nothing():
    picked = resolve_indices(selection(mode=MODE_RANDOM, percent=0.0), "l", 512)
    assert picked == ()


def test_a_percent_of_one_hundred_selects_every_channel():
    picked = resolve_indices(selection(mode=MODE_RANDOM, percent=100.0), "l", 8)
    assert picked == tuple(range(8))


def test_a_percent_is_clamped_rather_than_believed():
    assert resolve_indices(selection(mode=MODE_RANDOM, percent=400.0), "l", 8) == tuple(
        range(8)
    )
    assert resolve_indices(selection(mode=MODE_RANDOM, percent=-50.0), "l", 8) == ()


def test_a_percent_rounds_down_so_it_never_asks_for_more_than_there_is():
    picked = resolve_indices(selection(mode=MODE_RANDOM, percent=99.0), "l", 3)
    assert len(picked) == 2


def test_a_seed_reproduces_the_same_channels():
    # The whole point of resolving at edit time: the same look, saved and
    # reloaded, has to be the same channels. The legacy app re-rolled every
    # frame and could not promise this.
    first = resolve_indices(selection(mode=MODE_RANDOM, percent=50.0, seed=7), "l", 512)
    second = resolve_indices(
        selection(mode=MODE_RANDOM, percent=50.0, seed=7), "l", 512
    )
    assert first == second
    assert len(first) == 256


def test_a_different_seed_gives_different_channels():
    first = resolve_indices(selection(mode=MODE_RANDOM, percent=50.0, seed=1), "l", 512)
    second = resolve_indices(
        selection(mode=MODE_RANDOM, percent=50.0, seed=2), "l", 512
    )
    assert first != second


def test_a_random_selection_comes_back_sorted_and_without_repeats():
    picked = resolve_indices(selection(mode=MODE_RANDOM, percent=30.0, seed=3), "l", 64)
    assert list(picked) == sorted(set(picked))


def test_a_range_is_half_open_the_way_a_python_range_is():
    picked = resolve_indices(selection(mode=MODE_RANGE, low=2, high=5), "l", 512)
    assert picked == (2, 3, 4)


def test_a_range_wider_than_the_layer_is_clamped_to_the_layer():
    picked = resolve_indices(selection(mode=MODE_RANGE, low=0, high=9999), "l", 4)
    assert picked == (0, 1, 2, 3)


def test_a_range_starting_past_the_layer_selects_nothing():
    assert resolve_indices(selection(mode=MODE_RANGE, low=900, high=999), "l", 4) == ()


def test_a_negative_range_bound_is_clamped_to_zero():
    picked = resolve_indices(selection(mode=MODE_RANGE, low=-10, high=2), "l", 4)
    assert picked == (0, 1)


def test_a_backwards_range_selects_nothing_rather_than_reversing():
    assert resolve_indices(selection(mode=MODE_RANGE, low=5, high=2), "l", 512) == ()


def test_every_mode_resolves_to_indices_the_validator_accepts():
    # The validator refuses a non-int, a bool or a negative index and drops the
    # whole transform when it does, silently. `random.sample` and `range` both
    # hand back plain ints, and this is what keeps that true.
    for mode, fields in (
        (MODE_ALL, {}),
        (MODE_RANDOM, {"percent": 50.0, "seed": 11}),
        (MODE_RANGE, {"low": 1, "high": 5}),
    ):
        picked = resolve_indices(selection(mode=mode, **fields), "l", 8)
        transform = Transform("ablate", "b8.conv1", (1.0,), picked)
        assert _validate_transform(transform) is not None, mode
        for index in picked:
            assert isinstance(index, int) and not isinstance(index, bool)


def test_no_mode_can_resolve_to_an_index_the_layer_does_not_have():
    channels = 6
    for mode, fields in (
        (MODE_ALL, {}),
        (MODE_RANDOM, {"percent": 100.0, "seed": 4}),
        (MODE_RANGE, {"low": 0, "high": 9999}),
    ):
        picked = resolve_indices(selection(mode=mode, **fields), "l", channels)
        assert all(0 <= index < channels for index in picked), mode


# --- cluster files ----------------------------------------------------------


CLUSTERS = {
    "b8.conv1": [
        {"cluster_index": 0, "feature_index": 3},
        {"cluster_index": 1, "feature_index": 5},
        {"cluster_index": 0, "feature_index": 1},
        {"cluster_index": 0, "feature_index": 1},
    ],
    "b16.conv0": [{"cluster_index": 2, "feature_index": 7}],
}


def test_a_cluster_selects_the_features_it_names_on_that_layer():
    assert cluster_indices(CLUSTERS, "b8.conv1", 0, 512) == (1, 3)
    assert cluster_indices(CLUSTERS, "b8.conv1", 1, 512) == (5,)


def test_cluster_indices_are_plain_ints_the_validator_accepts():
    picked = cluster_indices(CLUSTERS, "b8.conv1", 0, 512)
    assert _validate_transform(Transform("ablate", "l", (1.0,), picked)) is not None


def test_a_cluster_file_is_read_per_layer():
    assert cluster_indices(CLUSTERS, "b16.conv0", 2, 512) == (7,)
    assert cluster_indices(CLUSTERS, "b16.conv0", 0, 512) == ()


def test_a_cluster_feature_past_the_layer_is_dropped():
    assert cluster_indices(CLUSTERS, "b8.conv1", 0, 2) == (1,)


def test_a_layer_the_file_says_nothing_about_selects_nothing():
    assert cluster_indices(CLUSTERS, "b32.conv0", 0, 512) == ()


def test_a_missing_or_malformed_cluster_file_selects_nothing():
    assert cluster_indices(None, "b8.conv1", 0, 512) == ()
    assert cluster_indices({"b8.conv1": "not a list"}, "b8.conv1", 0, 512) == ()
    assert cluster_indices({"b8.conv1": [1, 2, 3]}, "b8.conv1", 0, 512) == ()
    assert cluster_indices({"b8.conv1": [{"x": 1}]}, "b8.conv1", 0, 512) == ()


def test_a_cluster_file_lists_the_clusters_it_names_for_a_layer():
    assert cluster_ids(CLUSTERS, "b8.conv1") == (0, 1)
    assert cluster_ids(CLUSTERS, "b16.conv0") == (2,)
    assert cluster_ids(CLUSTERS, "b32.conv0") == ()
    assert cluster_ids(None, "b8.conv1") == ()


def test_a_cluster_mode_with_nothing_loaded_selects_nothing():
    assert resolve_indices(selection(mode=MODE_CLUSTER), "b8.conv1", 512) == ()


def test_a_cluster_file_is_read_with_the_safe_loader(tmp_path):
    path = tmp_path / "clusters.yaml"
    path.write_text(
        "b8.conv1:\n"
        "  - cluster_index: 0\n"
        "    feature_index: 4\n",
        encoding="utf-8",
    )
    config = load_cluster_file(str(path))
    assert cluster_indices(config, "b8.conv1", 0, 512) == (4,)


def test_a_cluster_file_that_would_construct_a_python_object_is_refused(tmp_path):
    # `yaml.load` builds arbitrary objects from a tag like this. The safe
    # loader raises instead, which is what keeps a file a performer picked from
    # being a code path.
    path = tmp_path / "evil.yaml"
    path.write_text("!!python/object/apply:os.system ['echo hi']\n", encoding="utf-8")
    with pytest.raises(Exception):
        load_cluster_file(str(path))


def test_a_cluster_file_that_is_not_a_mapping_is_refused(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_cluster_file(str(path))


# --- the adjuster -----------------------------------------------------------


def test_a_direction_file_holding_one_vector_reads_as_one_direction(tmp_path):
    path = tmp_path / "one.npy"
    np.save(path, np.arange(4, dtype=np.float32))
    assert load_direction_file(str(path)) == ((0.0, 1.0, 2.0, 3.0),)


def test_a_direction_file_holding_a_batch_reads_as_several(tmp_path):
    path = tmp_path / "batch.npy"
    np.save(path, np.zeros((3, 4), dtype=np.float32))
    loaded = load_direction_file(str(path))
    assert len(loaded) == 3
    assert all(len(row) == 4 for row in loaded)


def test_a_direction_file_with_a_leading_singleton_axis_is_squeezed(tmp_path):
    # The old app squeezed too: a vector saved from a `[1, 1, 512]` tensor is
    # still one direction.
    path = tmp_path / "squeeze.npy"
    np.save(path, np.zeros((1, 1, 6), dtype=np.float32))
    assert load_direction_file(str(path)) == ((0.0,) * 6,)


def test_a_direction_file_of_the_wrong_rank_is_refused_by_name(tmp_path):
    path = tmp_path / "deep.npy"
    np.save(path, np.zeros((2, 3, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        load_direction_file(str(path))


def test_a_batch_load_replaces_the_whole_set():
    merged = merged_directions(((1.0, 1.0),), ((2.0, 2.0), (3.0, 3.0)), None)
    assert merged == ((2.0, 2.0), (3.0, 3.0))


def test_a_batch_load_of_more_than_eight_keeps_the_first_eight():
    merged = merged_directions((), tuple((float(i),) for i in range(12)), None)
    assert len(merged) == 8
    assert merged[0] == (0.0,)


def test_a_slot_load_replaces_only_that_slot():
    merged = merged_directions(((1.0,), (2.0,), (3.0,)), ((9.0,),), 1)
    assert merged == ((1.0,), (9.0,), (3.0,))


def test_a_slot_load_into_the_next_free_slot_appends():
    merged = merged_directions(((1.0,), (2.0,)), ((3.0,),), 2)
    assert merged == ((1.0,), (2.0,), (3.0,))


def test_the_first_slot_load_starts_the_set():
    assert merged_directions((), ((1.0, 2.0),), 0) == ((1.0, 2.0),)


def test_a_slot_load_past_the_next_free_slot_is_refused():
    assert merged_directions(((1.0,),), ((2.0,),), 4) is None


def test_a_slot_load_of_a_different_length_is_refused():
    # Every direction is summed against one W row, so a set of mixed lengths is
    # a set where some slots silently do nothing.
    assert merged_directions(((1.0, 2.0),), ((3.0,),), 0) is None


def test_a_slot_load_of_a_batch_is_refused():
    assert merged_directions(((1.0,),), ((2.0,), (3.0,)), 0) is None


def test_an_empty_file_is_refused_rather_than_emptying_the_set():
    assert merged_directions(((1.0,),), (), None) is None


def test_loading_the_same_file_twice_is_not_reported_as_a_failure():
    same = ((1.0, 2.0),)
    assert merged_directions(same, same, 0) == same


def test_randomize_keeps_the_number_of_directions_there_already_are():
    current = ((0.0,) * 4, (0.0,) * 4, (0.0,) * 4)
    randomized = randomized_directions(current, 4, seed=1)
    assert len(randomized) == 3
    assert all(len(row) == 4 for row in randomized)


def test_randomize_with_nothing_loaded_makes_one_direction():
    assert len(randomized_directions((), 512, seed=1)) == 1


def test_randomize_uses_the_models_own_width():
    randomized = randomized_directions((), 1024, seed=1)
    assert len(randomized[0]) == 1024


def test_randomize_is_reproducible_from_its_seed():
    assert randomized_directions((), 8, seed=5) == randomized_directions((), 8, seed=5)
    assert randomized_directions((), 8, seed=5) != randomized_directions((), 8, seed=6)


# --- the selection mode list ------------------------------------------------


def test_every_selection_mode_has_a_label_to_read():
    assert len(SELECTION_MODES) == len(SELECTION_LABELS)


def test_every_selection_mode_has_a_branch_of_its_own_behind_it():
    # A mode with no branch would fall through to the cluster case and select
    # nothing at all, which looks the same as a working mode nobody has
    # configured yet. Told apart by giving each mode state only its own branch
    # reads and checking the answers differ.
    picked = {
        mode: resolve_indices(
            selection(mode=mode, percent=50.0, seed=1, low=0, high=2), "l", 8
        )
        for mode in SELECTION_MODES
    }
    assert picked[MODE_ALL] == tuple(range(8))
    assert len(picked[MODE_RANDOM]) == 4
    assert picked[MODE_RANGE] == (0, 1)
    assert picked[MODE_CLUSTER] == ()
