import warnings

import pytest
import torch

# kornia's lightglue submodule triggers a torch FutureWarning on import
# (unrelated to the operators under test) as a side effect of merely
# importing kornia at all, which autolume.bending.transform_layers does.
warnings.filterwarnings(
    "ignore", message=r".*torch\.cuda\.amp\.custom_fwd.*", category=FutureWarning
)

from autolume.bending.transform_layers import (
    Ablate,
    BinaryThreshold,
    Dilate,
    Erode,
    FlipHorizontal,
    FlipVertical,
    Invert,
    ManipulationLayer,
    Rotate,
    Scale,
    ScalarMultiply,
    Translate,
)

INDICES = [2, 5]
UNTOUCHED = [i for i in range(8) if i not in INDICES]


@pytest.fixture
def base_tensor():
    torch.manual_seed(0)
    return torch.rand(1, 8, 16, 16)


def assert_untouched(original, mutated):
    for i in UNTOUCHED:
        assert torch.equal(original[:, i], mutated[:, i])


OPERATOR_CASES = [
    pytest.param(Translate(), (2.0, 3.0), id="translate"),
    pytest.param(Rotate(), [30.0], id="rotate"),
    pytest.param(Scale(), [0.5], id="scale"),
    pytest.param(Erode(), [3], id="erode"),
    pytest.param(Dilate(), [3], id="dilate"),
    pytest.param(Invert(), [True], id="invert"),
    pytest.param(FlipHorizontal(), [True], id="flip-h"),
    pytest.param(FlipVertical(), [True], id="flip-v"),
    pytest.param(BinaryThreshold(), [0.5], id="binary-thresh"),
    pytest.param(ScalarMultiply(), [2.0], id="scalar-multiply"),
    pytest.param(Ablate(), [True], id="ablate"),
]


@pytest.mark.parametrize("op, params", OPERATOR_CASES)
def test_operator_only_touches_selected_indices(base_tensor, op, params):
    x = base_tensor.clone()
    out = op.forward(x, params, INDICES)

    assert_untouched(base_tensor, out)
    assert not torch.equal(base_tensor[:, INDICES], out[:, INDICES])


def test_translate_zero_is_identity(base_tensor):
    x = base_tensor.clone()
    out = Translate().forward(x, (0.0, 0.0), INDICES)
    assert torch.allclose(out, base_tensor, atol=1e-6)


def test_rotate_zero_is_identity(base_tensor):
    x = base_tensor.clone()
    out = Rotate().forward(x, [0.0], INDICES)
    assert torch.allclose(out, base_tensor, atol=1e-6)


def test_scale_one_is_identity(base_tensor):
    x = base_tensor.clone()
    out = Scale().forward(x, [1.0], INDICES)
    assert torch.allclose(out, base_tensor, atol=1e-6)


def test_erode_kernel_one_is_identity(base_tensor):
    x = base_tensor.clone()
    out = Erode().forward(x, [1], INDICES)
    assert torch.allclose(out, base_tensor, atol=1e-6)


def test_dilate_kernel_one_is_identity(base_tensor):
    x = base_tensor.clone()
    out = Dilate().forward(x, [1], INDICES)
    assert torch.allclose(out, base_tensor, atol=1e-6)


def test_invert_twice_is_identity(base_tensor):
    x = base_tensor.clone()
    once = Invert().forward(x, [True], INDICES)
    twice = Invert().forward(once.clone(), [True], INDICES)
    assert torch.allclose(twice, base_tensor, atol=1e-6)


def test_flip_h_flips_width_and_flip_v_flips_height():
    """The axes the names promise, which both operators used to have backwards.

    Asserted against an explicitly asymmetric pattern rather than a random
    tensor: a flip on the wrong axis is still a flip, so it changes the tensor
    and passes every test that only checks that something happened. The pattern
    is different along each axis and is its own mirror on neither.
    """
    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])

    flipped_h = FlipHorizontal().forward(x.clone(), [True], [0])
    assert flipped_h.tolist() == [[[[2.0, 1.0], [4.0, 3.0]]]]

    flipped_v = FlipVertical().forward(x.clone(), [True], [0])
    assert flipped_v.tolist() == [[[[3.0, 4.0], [1.0, 2.0]]]]


@pytest.mark.parametrize(
    "op", [FlipHorizontal(), FlipVertical()], ids=["flip-h", "flip-v"]
)
def test_a_flip_is_its_own_inverse(base_tensor, op):
    x = base_tensor.clone()
    once = op.forward(x, [True], INDICES)
    twice = op.forward(once.clone(), [True], INDICES)
    assert torch.equal(twice, base_tensor)


def test_a_flip_with_a_falsy_switch_is_identity(base_tensor):
    for op in (FlipHorizontal(), FlipVertical()):
        assert torch.equal(op.forward(base_tensor.clone(), [False], INDICES), base_tensor)


def test_manipulation_layer_empty_indices_returns_input_untouched(base_tensor):
    x = base_tensor.clone()
    out = ManipulationLayer().forward(
        x, {"transformID": "invert", "params": [True], "indices": []}
    )
    assert torch.equal(out, base_tensor)


def test_manipulation_layer_unknown_transform_id_raises_key_error(base_tensor):
    with pytest.raises(KeyError):
        ManipulationLayer().forward(
            base_tensor.clone(),
            {"transformID": "not-a-real-operator", "params": [1], "indices": [0]},
        )
