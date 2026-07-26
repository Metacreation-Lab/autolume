"""Self-check for `tools/make_demo_presets.py`'s preset-construction logic.

The script itself needs a real `.pkl` model, and for the mixing presets a
loaded pair, neither of which is available here. This exercises every pure
construction helper against a fake layer catalog, and the mixing helpers
against tiny but genuine `custom_stylegan2.Generator`s (the same convention
`tests/live/test_mixing.py` uses), so the payload shape the script hands to
`presets.save` is proven even though the script's own CLI path is not run.

The script lives in `tools/`, which is not an installed package, so it is
loaded here by file path rather than by import statement.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

from autolume.live.core.generator import LayerInfo
from autolume.live.core.params import ControlState

pytestmark = [
    pytest.mark.filterwarnings(
        r"ignore:.*torch\.cuda\.amp\.custom_fwd.*:FutureWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:pkg_resources is deprecated.*:DeprecationWarning"
    ),
]

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "tools" / "make_demo_presets.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("make_demo_presets", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mdp = _load_script()

FAKE_LAYERS = (
    LayerInfo("b4.conv0", channels=8, width=4, height=4),
    LayerInfo("b4.conv1", channels=8, width=4, height=4),
    LayerInfo("b8.conv0", channels=8, width=8, height=8),
    LayerInfo("b8.conv1", channels=8, width=8, height=8),
    LayerInfo("b16.conv0", channels=6, width=16, height=16),
    LayerInfo("b16.conv1", channels=6, width=16, height=16),
    LayerInfo("output", channels=3, width=16, height=16),
)


def test_conv_layers_excludes_the_output_alias():
    assert all(layer.name != "output" for layer in mdp.conv_layers(FAKE_LAYERS))


def test_pick_position_pair_is_earliest_and_latest_conv_layer():
    early, late = mdp.pick_position_pair(FAKE_LAYERS)
    assert early.name == "b4.conv0"
    assert late.name == "b16.conv1"


def test_pick_position_pair_fails_clearly_with_fewer_than_two_layers():
    with pytest.raises(SystemExit):
        mdp.pick_position_pair((FAKE_LAYERS[0],))


def test_pick_survey_layer_is_a_mid_resolution_conv_layer():
    layer = mdp.pick_survey_layer(FAKE_LAYERS)
    assert layer.name != "output"
    assert layer.width == 8


def test_all_indices_covers_every_channel():
    layer = FAKE_LAYERS[0]
    assert mdp.all_indices(layer) == tuple(range(layer.channels))


def test_subset_indices_is_smaller_than_all_channels():
    layer = FAKE_LAYERS[0]
    subset = mdp.subset_indices(layer, count=3)
    assert len(subset) == 3
    assert len(subset) < layer.channels


def test_translate_offset_is_never_zero():
    assert mdp.translate_offset(1) > 0


@pytest.mark.parametrize("op", mdp.OPERATORS)
def test_operator_transform_respects_its_arity(op):
    layer = FAKE_LAYERS[0]
    transform = mdp.operator_transform(op, layer, mdp.all_indices(layer))
    expected_arity = 2 if op == "translate" else 1
    assert len(transform.params) == expected_arity
    assert transform.layer == layer.name


def test_operator_transform_scale_respects_the_minimum_magnitude_guard():
    transform = mdp.operator_transform("scale", FAKE_LAYERS[0], (0,))
    assert abs(transform.params[0]) >= 1e-6


@pytest.mark.parametrize("op", ["erode", "dilate"])
def test_operator_transform_kernel_is_a_whole_number_of_one_or_more(op):
    transform = mdp.operator_transform(op, FAKE_LAYERS[0], (0,))
    assert transform.params[0].is_integer()
    assert transform.params[0] >= 1


def test_build_bending_presets_covers_every_documented_demo():
    built = mdp.build_bending_presets("/models/a.pkl", FAKE_LAYERS)
    names = [name for name, _ in built]
    assert len(names) == len(set(names))
    assert all(name.startswith(mdp.PREFIX) for name in names)
    # 2 position + 11 operators + 2 channel-selection + 2 capture-layer.
    assert len(built) == 2 + len(mdp.OPERATORS) + 2 + 2
    for _, state in built:
        assert isinstance(state, ControlState)
        assert state.pkl_path == "/models/a.pkl"
        for transform in state.transforms:
            layer = next(l for l in FAKE_LAYERS if l.name == transform.layer)
            assert all(0 <= i < layer.channels for i in transform.indices)


def test_build_bending_presets_layer_position_pair_shares_operator_and_params():
    built = mdp.build_bending_presets("/models/a.pkl", FAKE_LAYERS)
    by_name = dict(built)
    early = by_name["demo-bend-position-1-early-b4.conv0"]
    late = by_name["demo-bend-position-2-late-b16.conv1"]
    assert early.transforms[0].op == late.transforms[0].op
    assert early.transforms[0].params == late.transforms[0].params
    assert early.transforms[0].layer != late.transforms[0].layer


def test_build_bending_presets_channel_selection_pair_isolates_indices():
    built = mdp.build_bending_presets("/models/a.pkl", FAKE_LAYERS)
    by_name = dict(built)
    all_channels = by_name["demo-bend-channels-1-all"]
    subset = by_name["demo-bend-channels-2-subset"]
    assert all_channels.transforms[0].op == subset.transforms[0].op
    assert all_channels.transforms[0].layer == subset.transforms[0].layer
    assert all_channels.transforms[0].params == subset.transforms[0].params
    assert len(subset.transforms[0].indices) < len(all_channels.transforms[0].indices)


def test_build_bending_presets_capture_layer_pair_isolates_grayscale_and_channel():
    built = mdp.build_bending_presets("/models/a.pkl", FAKE_LAYERS)
    by_name = dict(built)
    color = by_name["demo-capture-layer-1-color"]
    gray = by_name["demo-capture-layer-2-grayscale"]
    assert color.capture_layer and color.capture_layer == gray.capture_layer
    assert color.grayscale is False
    assert gray.grayscale is True
    assert gray.base_channel != 0


def test_all_selection_never_exceeds_the_named_models_own_length():
    entries = mdp.all_selection("A", ("n0", "n1"), total=5)
    assert entries == ("A", "A", "X", "X", "X")


def test_split_selection_ends_in_a_trailing_x_run_only():
    names_a = tuple(f"synthesis.b{res}.conv0.weight" for res in (4, 8, 16, 32))
    names_b = tuple(f"synthesis.b{res}.conv0.weight" for res in (4, 8, 16))
    entries = mdp.split_selection(names_a, names_b, total=4)
    assert "A" in entries
    assert "B" in entries
    if "X" in entries:
        first_x = entries.index("X")
        assert all(entry == "X" for entry in entries[first_x:])


def test_split_selection_puts_a_before_b():
    names_a = tuple(f"synthesis.b{res}.conv0.weight" for res in (4, 8, 16))
    names_b = tuple(f"synthesis.b{res}.conv0.weight" for res in (4, 8, 16))
    entries = mdp.split_selection(names_a, names_b, total=3)
    assert entries.index("A") < entries.index("B")


def test_truncated_selection_drops_exactly_the_top_resolution_stage():
    names_a = tuple(f"synthesis.b{res}.conv0.weight" for res in (4, 8, 16, 16))
    entries = mdp.truncated_selection(names_a, total=4)
    assert entries == ("A", "A", "X", "X")


def test_truncated_selection_fails_clearly_with_a_single_resolution_stage():
    names_a = ("synthesis.b4.conv0.weight", "synthesis.b4.conv1.weight")
    with pytest.raises(SystemExit):
        mdp.truncated_selection(names_a, total=2)


class _FakeLoadedModel:
    """The one attribute `build_mixing_presets` reads off a loaded model."""

    def __init__(self, G):
        self.G = G


def _tiny_generator(seed, z_dim=8, img_resolution=16, channel_max=8):
    """A tiny but genuine custom stylegan2 generator, perturbed off its
    fresh initialisation so two of them are never accidentally identical.
    """
    from architectures import custom_stylegan2

    torch.manual_seed(seed)
    gen = custom_stylegan2.Generator(
        z_dim=z_dim,
        c_dim=0,
        w_dim=8,
        img_channels=3,
        img_resolution=img_resolution,
        mapping_kwargs={},
        synthesis_kwargs={"channel_base": 64, "channel_max": channel_max},
    )
    with torch.no_grad():
        for parameter in gen.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.1 + seed * 0.02)
    return gen


def test_build_mixing_presets_writes_the_four_documented_variants():
    model_a = _FakeLoadedModel(_tiny_generator(1))
    model_b = _FakeLoadedModel(_tiny_generator(2))
    built = mdp.build_mixing_presets("/models/a.pkl", "/models/b.pkl", model_a, model_b)
    assert [name for name, _ in built] == [
        "demo-mix-1-all-a",
        "demo-mix-2-all-b",
        "demo-mix-3-split-middle",
        "demo-mix-4-truncate-x",
    ]
    expected_length = len(mdp.mixing.conv_names(model_a.G))
    for name, state in built:
        assert state.pkl_path == "/models/a.pkl"
        assert state.pkl2 == "/models/b.pkl"
        assert state.mixing_enabled is True
        assert len(state.combined_layers) == expected_length


def test_build_mixing_presets_all_a_is_an_a_and_x_only_control():
    model_a = _FakeLoadedModel(_tiny_generator(1))
    model_b = _FakeLoadedModel(_tiny_generator(2))
    built = mdp.build_mixing_presets("/models/a.pkl", "/models/b.pkl", model_a, model_b)
    entries = dict(built)["demo-mix-1-all-a"].combined_layers
    assert set(entries) <= {"A", "X"}
    assert entries[0] == "A"


def test_build_mixing_presets_fails_clearly_on_an_incompatible_pair():
    # A mismatched z_dim makes the all-B variant's mapping network the wrong
    # shape for a generator built against model A's z_dim, so combine()
    # raises the documented incompatibility error.
    model_a = _FakeLoadedModel(_tiny_generator(1, z_dim=8))
    model_b = _FakeLoadedModel(_tiny_generator(2, z_dim=4))
    with pytest.raises(SystemExit):
        mdp.build_mixing_presets("/models/a.pkl", "/models/b.pkl", model_a, model_b)
