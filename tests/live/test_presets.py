import base64
import dataclasses
import json
import logging
from pathlib import Path, PureWindowsPath

import numpy as np
import pytest

from autolume.live.core import params, presets
from autolume.live.core.events import ControlEvent
from autolume.live.core.mapping import apply_event
from autolume.live.core.params import Binding, ControlState, Keyframe

PRESETS_LOGGER = "autolume.live.core.presets"

# Exact multiples of 0.5 and 0.25 so the float32 round trip loses nothing: the
# round trip test below compares whole `ControlState` equality, and a value
# that is not exactly representable in float32 would make that assertion flaky.
SAMPLE_LATENT_VEC = tuple(i * 0.5 for i in range(512))
SAMPLE_KEYFRAME_VEC = tuple(i * -0.25 for i in range(512))

# Every preset parameter differs from its default, so a round trip that keeps a
# value by accident cannot pass. `test_sample_state_is_non_default_everywhere`
# holds this honest as the registry grows. `keyframe_count` and `keyframes`
# agree on four stops on purpose: the two used to be able to disagree after a
# preset load (see mapping.py's `_apply_preset`), and a fixture that itself
# held them out of sync would hide a regression instead of catching one.
SAMPLE = ControlState(
    pkl_path=None,
    latent_x=1.5,
    latent_y=-2.25,
    anim_playing=True,
    anim_speed_x=-1.5,
    anim_speed_y=3.0,
    truncation_psi=1.25,
    global_noise=0.5,
    noise_enabled=False,
    noise_seed=99,
    noise_anim=True,
    vector_mode=True,
    latent_project=False,
    loop_active=True,
    loop_uses_time=False,
    loop_time=12.5,
    loop_speed=2.5,
    loop_alpha=0.75,
    loop_index=3,
    keyframe_count=4,
    perfect_loop=True,
    noise_loop=True,
    noise_radius=5.0,
    noise_loop_seed=42,
    # pulse_ip / pulse_port are left at their defaults, like fps_cap above:
    # they are preset=False (see
    # test_the_pulse_destination_is_a_property_of_the_machine_not_of_the_look
    # below), so a non-default value here would break the round trip test.
    pulse_address="/loop/pulse",
    latent_vec=SAMPLE_LATENT_VEC,
    keyframes=(
        Keyframe("seed", 0.0, 0.0, (), True),
        Keyframe("vec", 1.0, -1.0, SAMPLE_KEYFRAME_VEC, False),
        Keyframe("seed", 2.0, 1.5, (), True),
        Keyframe("seed", 3.0, 0.0, (), True),
    ),
    bindings=(
        Binding("truncation_psi", "/audio/bass", "0.4+x"),
        Binding("latent_x", "/ctl/1", "x*2", enabled=False),
    ),
)


def _use_data_root(monkeypatch, root: Path) -> None:
    """Point `data_path` at `root` so model resolution is hermetic in tests."""
    from utils import user_data

    monkeypatch.setattr(user_data, "_prefs", {"version": 1, "data_root": str(root)})
    monkeypatch.setattr(user_data, "_data_root", str(root))


def _array_payload(values, **overrides) -> dict:
    payload = {
        "dtype": "float32",
        "shape": [len(values)],
        "b64": base64.b64encode(
            np.asarray(values, dtype="<f4").tobytes()
        ).decode("ascii"),
    }
    payload.update(overrides)
    return payload


def warnings_from(caplog):
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == PRESETS_LOGGER and r.levelname == "WARNING"
    ]


def preset_names():
    return {name for name, spec in params.REGISTRY.items() if spec.preset}


def apply_payload(state, payload):
    return apply_event(state, ControlEvent(presets.PRESET_APPLY, payload))


def test_sample_state_is_non_default_everywhere():
    default = ControlState()
    for name in preset_names():
        assert getattr(SAMPLE, name) != getattr(default, name), name
    assert SAMPLE.bindings != default.bindings
    assert SAMPLE.latent_vec != default.latent_vec
    assert SAMPLE.keyframes != default.keyframes


def test_apply_address_is_reserved():
    assert presets.PRESET_APPLY.startswith("/preset/")
    assert presets.PRESET_APPLY not in params.BY_ADDRESS


def test_round_trip_restores_every_preset_param_and_bindings(tmp_path):
    """Also the vector and the mixed seed/vec keyframe loop, exactly.

    `SAMPLE` carries a latent vector and a loop mixing both keyframe kinds, so
    this one round trip covers the format's whole payload, not just scalars.
    """
    path = tmp_path / "look.json"
    presets.save(SAMPLE, path)
    assert apply_payload(ControlState(), presets.load(path)) == SAMPLE


def test_saved_file_is_json_with_the_expected_envelope(tmp_path):
    path = tmp_path / "look.json"
    presets.save(SAMPLE, path)
    with open(path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)
    assert payload["format"] == presets.FORMAT
    assert payload["version"] == presets.VERSION
    assert payload["model"] is None
    assert set(payload["params"]) == preset_names()
    assert "pkl_path" not in payload["params"]
    assert payload["params"]["truncation_psi"] == 1.25
    assert payload["params"]["noise_enabled"] is False
    assert payload["latent_vec"]["dtype"] == "float32"
    assert payload["latent_vec"]["shape"] == [512]
    assert len(payload["keyframes"]) == 4
    assert payload["keyframes"][0] == {
        "kind": "seed", "seed_x": 0.0, "seed_y": 0.0, "project": True, "vec": None
    }
    assert payload["keyframes"][1]["kind"] == "vec"
    assert payload["keyframes"][1]["vec"]["dtype"] == "float32"
    assert payload["bindings"] == [
        {
            "target": "truncation_psi",
            "source": "/audio/bass",
            "expression": "0.4+x",
            "enabled": True,
        },
        {
            "target": "latent_x",
            "source": "/ctl/1",
            "expression": "x*2",
            "enabled": False,
        },
    ]


def test_a_parameter_put_on_the_network_stays_on_it_through_a_reload(tmp_path):
    """A row with a switch and no source is how remote input is turned on.

    It is the one record the new default cannot infer, so it has to survive the
    round trip. Without it a performer recalls the look they set up before the
    show and every controller they wired to it is deaf.
    """
    path = tmp_path / "look.json"
    on = ControlState(bindings=(Binding("anim_playing", "", "x", enabled=True),))
    presets.save(on, path)
    assert apply_payload(ControlState(), presets.load(path)).bindings == on.bindings


def test_a_preset_with_no_record_for_a_parameter_leaves_it_off_the_network(tmp_path):
    """Absence means off, on recall as much as at startup.

    A preset written before remote input became opt in carries no row for a
    parameter that was listening by default, so recalling it now leaves that
    parameter deaf. That is deliberate and unmigrated: the absent row recorded
    a default nobody chose, and synthesizing one for every parameter would
    reinstate the very state this change exists to abolish.
    """
    path = tmp_path / "look.json"
    presets.save(ControlState(truncation_psi=1.25), path)
    state = apply_payload(ControlState(), presets.load(path))
    assert state.truncation_psi == 1.25
    assert state.bindings == ()


def test_the_frame_limit_is_a_property_of_the_machine_not_of_the_look(
    tmp_path, monkeypatch
):
    """A look saved on a laptop capped at 30 must not cap the stage machine.

    The model path is the opposite case and stays persisted: it is what the
    look looks like, while the frame limit is what the hardware can do.
    """
    _use_data_root(monkeypatch, tmp_path)
    model_file = tmp_path / "models" / "look.pkl"
    model_file.parent.mkdir(parents=True)
    model_file.write_bytes(b"")
    state = dataclasses.replace(SAMPLE, fps_cap=30, pkl_path=str(model_file))
    path = tmp_path / "look.json"
    presets.save(state, path)
    payload = presets.load(path)
    assert "fps_cap" not in payload["params"]
    assert payload["model"] == {"name": "look.pkl"}
    assert apply_payload(ControlState(fps_cap=144), payload).fps_cap == 144
    assert presets.from_payload(payload).params["pkl_path"] == str(model_file)


def test_the_pulse_destination_is_a_property_of_the_machine_not_of_the_look(
    tmp_path,
):
    """A look saved on one LAN must not misdirect pulses on another.

    The pulse address is the opposite case and stays persisted: it names the
    message within the patch, which is what a receiving rig matches on and is
    meaningfully part of the look, the same distinction `fps_cap` draws
    between the machine and the look above.
    """
    state = dataclasses.replace(
        SAMPLE, pulse_address="/loop/pulse", pulse_ip="10.0.0.5", pulse_port=9000
    )
    path = tmp_path / "look.json"
    presets.save(state, path)
    payload = presets.load(path)
    assert "pulse_ip" not in payload["params"]
    assert "pulse_port" not in payload["params"]
    assert payload["params"]["pulse_address"] == "/loop/pulse"
    current = ControlState(pulse_ip="192.168.1.1", pulse_port=6000)
    loaded = apply_payload(current, payload)
    assert loaded.pulse_ip == "192.168.1.1"
    assert loaded.pulse_port == 6000
    assert loaded.pulse_address == "/loop/pulse"


def test_params_written_follow_the_registry_preset_flag(monkeypatch):
    spec = params.REGISTRY["latent_y"]
    monkeypatch.setitem(
        params.REGISTRY, "latent_y", dataclasses.replace(spec, preset=False)
    )
    payload = presets.to_payload(SAMPLE)
    assert "latent_y" not in payload["params"]
    assert "latent_x" in payload["params"]


def test_non_preset_param_in_file_is_skipped_on_read(monkeypatch, caplog):
    spec = params.REGISTRY["latent_y"]
    payload = presets.to_payload(SAMPLE)
    monkeypatch.setitem(
        params.REGISTRY, "latent_y", dataclasses.replace(spec, preset=False)
    )
    with caplog.at_level(logging.WARNING):
        data = presets.from_payload(payload)
    assert "latent_y" not in data.params
    assert data.params["latent_x"] == 1.5
    assert any("latent_y" in message for message in warnings_from(caplog))


def test_binding_error_is_never_written_and_loads_as_none(tmp_path):
    state = dataclasses.replace(
        SAMPLE,
        bindings=(Binding("latent_x", "/ctl/1", "x*2", error="boom"),),
    )
    path = tmp_path / "look.json"
    presets.save(state, path)
    with open(path, "r", encoding="utf-8") as fp:
        raw = fp.read()
    assert "boom" not in raw
    assert "error" not in json.loads(raw)["bindings"][0]
    bindings = presets.from_payload(json.loads(raw)).bindings
    assert bindings[0].error is None


def test_path_values_are_str_wrapped(tmp_path, monkeypatch):
    _use_data_root(monkeypatch, tmp_path)
    state = dataclasses.replace(SAMPLE, pkl_path=PureWindowsPath(r"C:\models\m.pkl"))
    path = tmp_path / "look.json"
    presets.save(state, path)
    with open(path, "r", encoding="utf-8") as fp:
        stored = json.load(fp)["model"]
    assert stored == {"path": r"C:\models\m.pkl"}


def test_unknown_param_key_is_skipped_and_the_rest_applied(caplog):
    payload = presets.to_payload(SAMPLE)
    payload["params"]["gone_in_v9"] = 3.0
    with caplog.at_level(logging.WARNING):
        state = apply_payload(ControlState(), payload)
    assert state.truncation_psi == 1.25
    assert not hasattr(state, "gone_in_v9")
    assert any("gone_in_v9" in message for message in warnings_from(caplog))


def test_missing_param_key_keeps_the_current_value():
    payload = presets.to_payload(SAMPLE)
    del payload["params"]["truncation_psi"]
    current = ControlState(truncation_psi=0.33)
    state = apply_payload(current, payload)
    assert state.truncation_psi == 0.33
    assert state.latent_x == 1.5


def test_null_param_value_keeps_the_current_value():
    payload = presets.to_payload(SAMPLE)
    payload["params"]["truncation_psi"] = None
    state = apply_payload(ControlState(truncation_psi=0.42), payload)
    assert state.truncation_psi == 0.42
    assert state.latent_x == 1.5


def test_newer_version_loads_with_a_warning(tmp_path, caplog):
    payload = presets.to_payload(SAMPLE)
    payload["version"] = presets.VERSION + 1
    path = tmp_path / "future.json"
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp)
    with caplog.at_level(logging.WARNING):
        loaded = presets.load(path)
    assert loaded["params"]["truncation_psi"] == 1.25
    assert any("version" in message for message in warnings_from(caplog))
    assert apply_payload(ControlState(), loaded).truncation_psi == 1.25


@pytest.mark.parametrize("payload", [{"format": "other-app", "version": 1}, {}, []])
def test_wrong_format_raises(tmp_path, payload):
    path = tmp_path / "wrong.json"
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp)
    with pytest.raises(ValueError):
        presets.load(path)
    with pytest.raises(ValueError):
        presets.from_payload(payload)


def test_corrupt_file_raises(tmp_path):
    path = tmp_path / "truncated.json"
    path.write_text('{"format": "autolume-live-pre', encoding="utf-8")
    with pytest.raises(ValueError):
        presets.load(path)


def test_out_of_range_value_is_clamped_on_apply():
    payload = presets.to_payload(SAMPLE)
    payload["params"]["truncation_psi"] = 9.0
    payload["params"]["noise_seed"] = -20
    state = apply_payload(ControlState(), payload)
    assert state.truncation_psi == 2.0
    assert state.noise_seed == 0


def test_uncoercible_value_leaves_the_current_value():
    payload = presets.to_payload(SAMPLE)
    payload["params"]["latent_x"] = "not a number"
    state = apply_payload(ControlState(latent_x=7.0), payload)
    assert state.latent_x == 7.0
    assert state.latent_y == -2.25


def test_non_finite_value_is_skipped_and_the_rest_applied(caplog):
    payload = presets.to_payload(SAMPLE)
    payload["params"]["global_noise"] = float("nan")
    payload["params"]["noise_seed"] = float("inf")
    current = ControlState(global_noise=0.25, noise_seed=3)
    with caplog.at_level(logging.WARNING):
        state = apply_payload(current, payload)
    assert state.global_noise == 0.25
    assert state.noise_seed == 3
    assert state.truncation_psi == 1.25
    assert state.latent_x == 1.5
    messages = warnings_from(caplog)
    assert any("global_noise" in message for message in messages)
    assert any("noise_seed" in message for message in messages)


def test_non_finite_json_literals_from_a_hand_edited_file_are_skipped(tmp_path):
    # json.load accepts bare NaN and Infinity, so a hand edited file carries
    # them straight in.
    path = tmp_path / "hand_edited.json"
    raw = json.dumps(presets.to_payload(SAMPLE))
    raw = raw.replace('"global_noise": 0.5', '"global_noise": NaN')
    raw = raw.replace('"latent_y": -2.25', '"latent_y": -Infinity')
    path.write_text(raw, encoding="utf-8")
    current = ControlState(global_noise=0.25, latent_y=1.0)
    state = apply_payload(current, presets.load(path))
    assert state.global_noise == 0.25
    assert state.latent_y == 1.0
    assert state.truncation_psi == 1.25


def test_absent_bindings_key_clears_mappings_and_says_so(caplog):
    payload = presets.to_payload(SAMPLE)
    del payload["bindings"]
    current = ControlState(bindings=(Binding("latent_x", "/ctl/1", "x"),))
    with caplog.at_level(logging.WARNING):
        state = apply_payload(current, payload)
    assert state.bindings == ()
    messages = warnings_from(caplog)
    assert any("clear" in message.lower() for message in messages)
    assert not any("NoneType" in message for message in messages)


def test_absent_params_key_keeps_current_values_without_a_parse_complaint(caplog):
    payload = presets.to_payload(SAMPLE)
    del payload["params"]
    current = ControlState(truncation_psi=0.33)
    with caplog.at_level(logging.WARNING):
        state = apply_payload(current, payload)
    assert state.truncation_psi == 0.33
    assert not any("NoneType" in message for message in warnings_from(caplog))


def test_duplicate_binding_targets_collapse_to_one(caplog):
    payload = presets.to_payload(SAMPLE)
    payload["bindings"].append(
        {"target": "truncation_psi", "source": "/ctl/9", "expression": "x"}
    )
    with caplog.at_level(logging.WARNING):
        bindings = presets.from_payload(payload).bindings
    assert [b.target for b in bindings] == ["truncation_psi", "latent_x"]
    assert bindings[0].source == "/audio/bass"
    assert any("truncation_psi" in message for message in warnings_from(caplog))


@pytest.mark.parametrize(
    "entry",
    [
        "not a mapping",
        {"source": "/ctl/1"},
        {"target": "nope", "source": "/ctl/1"},
        {"target": "latent_y", "source": None},
        {"target": "latent_y", "source": "/ctl/1", "expression": 3.0},
        {"target": ["latent_y"], "source": "/ctl/1"},
    ],
)
def test_malformed_binding_entry_is_skipped(entry, caplog):
    payload = presets.to_payload(SAMPLE)
    payload["bindings"].append(entry)
    with caplog.at_level(logging.WARNING):
        bindings = presets.from_payload(payload).bindings
    assert [b.target for b in bindings] == ["truncation_psi", "latent_x"]
    assert warnings_from(caplog)


def test_binding_fields_default_when_absent():
    payload = presets.to_payload(SAMPLE)
    payload["bindings"] = [{"target": "latent_y", "source": "/ctl/2"}]
    bindings = presets.from_payload(payload).bindings
    assert bindings == (Binding("latent_y", "/ctl/2", "x", True, None),)


def test_bad_expression_is_kept_so_it_can_be_fixed():
    payload = presets.to_payload(SAMPLE)
    payload["bindings"] = [
        {"target": "latent_y", "source": "/ctl/2", "expression": "x +* 2"}
    ]
    bindings = presets.from_payload(payload).bindings
    assert bindings[0].expression == "x +* 2"


def test_non_mapping_params_and_non_list_bindings_are_ignored(caplog):
    payload = {"format": presets.FORMAT, "version": 1, "params": 3, "bindings": "x"}
    with caplog.at_level(logging.WARNING):
        data = presets.from_payload(payload)
    assert data.params == {}
    assert data.bindings == ()
    assert data.latent_vec == ControlState().latent_vec
    assert data.keyframes == ControlState().keyframes
    assert data.missing_model is None
    assert len(warnings_from(caplog)) == 2


# --- keyframe_count / keyframes divergence (bug fixed in mapping.py) -------


def test_preset_apply_keeps_keyframe_count_and_keyframes_in_sync():
    """Reproduces the bug: a preset used to be able to set one without the other.

    `_apply_preset` (mapping.py) now derives `keyframe_count` from the loaded
    `keyframes` tuple instead of applying the payload's scalar directly, so the
    two can never disagree after a preset load.
    """
    payload = presets.to_payload(SAMPLE)
    state = apply_payload(ControlState(), payload)
    assert state.keyframe_count == len(state.keyframes) == 4


def test_preset_apply_ignores_a_keyframe_count_that_disagrees_with_keyframes():
    payload = presets.to_payload(SAMPLE)
    # A hand edited or stale file: keyframe_count claims 3 but the keyframes
    # list still holds 4 entries. The list wins.
    payload["params"]["keyframe_count"] = 3
    state = apply_payload(ControlState(), payload)
    assert len(state.keyframes) == 4
    assert state.keyframe_count == 4


# --- latent_vec / keyframes defaults and array descriptor rejection -------


def test_missing_latent_vec_and_keyframes_load_at_their_defaults():
    payload = presets.to_payload(SAMPLE)
    del payload["latent_vec"]
    del payload["keyframes"]
    data = presets.from_payload(payload)
    assert data.latent_vec == ControlState().latent_vec
    assert data.keyframes == ControlState().keyframes
    state = apply_payload(ControlState(), payload)
    assert state.latent_vec == ()
    assert state.keyframes == ControlState().keyframes
    assert state.keyframe_count == len(ControlState().keyframes)


def test_array_wrong_dtype_is_rejected(caplog):
    payload = presets.to_payload(SAMPLE)
    payload["latent_vec"]["dtype"] = "float64"
    with caplog.at_level(logging.WARNING):
        data = presets.from_payload(payload)
    assert data.latent_vec == ControlState().latent_vec
    assert data.params["truncation_psi"] == 1.25
    assert any("latent_vec" in m for m in warnings_from(caplog))


def test_array_shape_and_byte_length_disagreement_is_rejected(caplog):
    payload = presets.to_payload(SAMPLE)
    payload["latent_vec"]["shape"] = [len(SAMPLE.latent_vec) + 1]
    with caplog.at_level(logging.WARNING):
        data = presets.from_payload(payload)
    assert data.latent_vec == ControlState().latent_vec
    assert data.params["truncation_psi"] == 1.25
    assert any("latent_vec" in m for m in warnings_from(caplog))


def test_array_undecodable_base64_is_rejected(caplog):
    payload = presets.to_payload(SAMPLE)
    payload["latent_vec"]["b64"] = "not base64 at all!!"
    with caplog.at_level(logging.WARNING):
        data = presets.from_payload(payload)
    assert data.latent_vec == ControlState().latent_vec
    assert data.params["truncation_psi"] == 1.25
    assert any("latent_vec" in m for m in warnings_from(caplog))


def test_array_non_finite_value_after_decode_is_rejected(caplog):
    payload = presets.to_payload(SAMPLE)
    corrupt = [float("nan")] + [0.0] * (len(SAMPLE.latent_vec) - 1)
    payload["latent_vec"]["b64"] = base64.b64encode(
        np.asarray(corrupt, dtype="<f4").tobytes()
    ).decode("ascii")
    with caplog.at_level(logging.WARNING):
        data = presets.from_payload(payload)
    assert data.latent_vec == ControlState().latent_vec
    assert data.params["truncation_psi"] == 1.25
    assert any("latent_vec" in m for m in warnings_from(caplog))


def test_array_endianness_is_explicit_not_native():
    """The wire format is always little-endian, regardless of the encoding
    host's own native order.

    There is no hardware here with a different native order, so this proves it
    without one: build the same values from an array explicitly typed
    big-endian and convert it the way the wire format requires. If the code
    depended on the host's native order rather than forcing `<f4`, this
    conversion could differ on a big-endian host; forcing it makes the bytes,
    and therefore the decoded values, identical no matter where they came from.
    """
    values = (1.5, -2.25, 100.0, 0.0, -8.0, 0.25)
    little_native = presets._encode_array(values)
    big_source = np.asarray(values, dtype=">f4").astype("<f4")
    big_twin = _array_payload(values, b64=base64.b64encode(big_source.tobytes()).decode("ascii"))
    assert big_twin["b64"] == little_native["b64"]
    assert presets._decode_array(big_twin, "x") == values
    assert presets._decode_array(little_native, "x") == values


# --- model reference --------------------------------------------------


def test_model_under_models_folder_saves_as_a_name(tmp_path, monkeypatch):
    _use_data_root(monkeypatch, tmp_path)
    model_file = tmp_path / "models" / "rivers-1024.pkl"
    model_file.parent.mkdir(parents=True)
    model_file.write_bytes(b"")
    payload = presets.to_payload(dataclasses.replace(SAMPLE, pkl_path=str(model_file)))
    assert payload["model"] == {"name": "rivers-1024.pkl"}


def test_model_outside_models_folder_saves_as_a_path(tmp_path, monkeypatch):
    _use_data_root(monkeypatch, tmp_path)
    (tmp_path / "models").mkdir()
    outside = tmp_path / "elsewhere" / "rivers-1024.pkl"
    outside.parent.mkdir()
    outside.write_bytes(b"")
    payload = presets.to_payload(dataclasses.replace(SAMPLE, pkl_path=str(outside)))
    assert payload["model"] == {"path": str(outside)}


def test_model_name_resolves_against_the_local_models_folder_on_load(
    tmp_path, monkeypatch
):
    _use_data_root(monkeypatch, tmp_path)
    model_file = tmp_path / "models" / "rivers-1024.pkl"
    model_file.parent.mkdir(parents=True)
    model_file.write_bytes(b"")
    payload = presets.to_payload(SAMPLE)
    payload["model"] = {"name": "rivers-1024.pkl"}
    data = presets.from_payload(payload)
    assert data.params["pkl_path"] == str(model_file)
    assert data.missing_model is None


def test_missing_model_name_reports_rather_than_raising(tmp_path, monkeypatch, caplog):
    _use_data_root(monkeypatch, tmp_path)
    (tmp_path / "models").mkdir()
    payload = presets.to_payload(SAMPLE)
    payload["model"] = {"name": "ghost.pkl"}
    with caplog.at_level(logging.WARNING):
        data = presets.from_payload(payload)
    assert data.missing_model == "ghost.pkl"
    assert "pkl_path" not in data.params
    assert data.params["truncation_psi"] == 1.25
    state = apply_payload(ControlState(pkl_path="/current/model.pkl"), payload)
    assert state.pkl_path == "/current/model.pkl"


def test_missing_model_path_reports_rather_than_raising(tmp_path, monkeypatch):
    _use_data_root(monkeypatch, tmp_path)
    payload = presets.to_payload(SAMPLE)
    payload["model"] = {"path": str(tmp_path / "nowhere" / "ghost.pkl")}
    data = presets.from_payload(payload)
    assert data.missing_model == "ghost.pkl"
    assert "pkl_path" not in data.params


def test_null_or_missing_model_key_leaves_the_current_model_alone():
    payload = presets.to_payload(SAMPLE)
    payload["model"] = None
    data = presets.from_payload(payload)
    assert "pkl_path" not in data.params
    assert data.missing_model is None
    del payload["model"]
    data = presets.from_payload(payload)
    assert "pkl_path" not in data.params
    assert data.missing_model is None
    state = apply_payload(ControlState(pkl_path="/current/model.pkl"), payload)
    assert state.pkl_path == "/current/model.pkl"


def test_legacy_params_pkl_path_with_no_model_key_loads_without_raising(caplog):
    """The one existing test preset, written in the older shape, still opens.

    Its `params.pkl_path` is a non-preset key now and is ignored; its other
    params and bindings load normally, and it arrives with no model.
    """
    payload = {
        "format": presets.FORMAT,
        "version": 1,
        "params": {
            "pkl_path": "/Users/ucodia/autolume/models/rivers-1024.pkl",
            "truncation_psi": 1.1,
        },
        "bindings": [{"target": "latent_x", "source": "/ctl/1", "expression": "x"}],
    }
    with caplog.at_level(logging.WARNING):
        data = presets.from_payload(payload)
    assert "pkl_path" not in data.params
    assert data.params["truncation_psi"] == 1.1
    assert data.missing_model is None
    assert data.bindings == (Binding("latent_x", "/ctl/1", "x"),)
    state = apply_payload(ControlState(), payload)
    assert state.pkl_path is None
    assert state.truncation_psi == 1.1


# --- hostile input never raises -----------------------------------------


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.__setitem__("model", "not a dict"),
        lambda p: p.__setitem__("model", {"name": 12345}),
        lambda p: p.__setitem__("model", {"path": 12345}),
        lambda p: p.__setitem__("model", {}),
        lambda p: p.__setitem__("latent_vec", "not a dict"),
        lambda p: p.__setitem__(
            "latent_vec", {"dtype": "float32", "shape": "bad", "b64": "AA=="}
        ),
        lambda p: p.__setitem__(
            "latent_vec", {"dtype": "float32", "shape": [0], "b64": ""}
        ),
        lambda p: p.__setitem__(
            "latent_vec", {"dtype": "float32", "shape": [-1], "b64": "AA=="}
        ),
        lambda p: p.__setitem__("keyframes", "not a list"),
        lambda p: p.__setitem__("keyframes", []),
        lambda p: p.__setitem__("keyframes", [None, 1, "x"]),
        lambda p: p.__setitem__("keyframes", [{"kind": "vec"}]),
        lambda p: p.__setitem__("keyframes", [{"kind": "nope"}]),
        lambda p: p.__setitem__("keyframes", [{"kind": "seed", "seed_x": float("nan")}]),
    ],
)
def test_nothing_in_the_read_path_raises_on_hostile_input(mutate):
    payload = presets.to_payload(SAMPLE)
    mutate(payload)
    data = presets.from_payload(payload)
    assert isinstance(data, presets.PresetData)
    apply_payload(ControlState(), payload)


def test_list_presets_returns_sorted_names_without_suffix(tmp_path):
    for name in ("sunset.json", "acid.json", "notes.txt"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    (tmp_path / "nested.json").mkdir()
    assert presets.list_presets(tmp_path) == ["acid", "sunset"]


def test_list_presets_on_a_missing_directory_is_empty(tmp_path):
    assert presets.list_presets(tmp_path / "nothing here") == []


def test_preset_dir_lives_under_the_data_root_and_is_created(tmp_path, monkeypatch):
    from utils import user_data

    monkeypatch.setattr(user_data, "_prefs", {"version": 1, "data_root": str(tmp_path)})
    monkeypatch.setattr(user_data, "_data_root", str(tmp_path))
    directory = presets.preset_dir()
    assert directory == tmp_path / "live" / "presets"
    assert directory.is_dir()
    presets.save(SAMPLE, directory / "look.json")
    assert presets.list_presets() == ["look"]


def test_save_creates_missing_parent_directories(tmp_path):
    path = tmp_path / "deep" / "nested" / "look.json"
    presets.save(SAMPLE, path)
    assert presets.load(path)["params"]["truncation_psi"] == 1.25


def test_save_accepts_a_str_path(tmp_path):
    presets.save(SAMPLE, str(tmp_path / "look.json"))
    assert presets.list_presets(str(tmp_path)) == ["look"]


def test_a_failed_save_leaves_the_previous_file_intact(tmp_path):
    path = tmp_path / "look.json"
    presets.save(SAMPLE, path)
    # Not serializable, so the write dies partway through, standing in for the
    # interrupted or full disk write a performer would otherwise lose a look to.
    doomed = dataclasses.replace(SAMPLE, pkl_path=object())
    with pytest.raises(TypeError):
        presets.save(doomed, path)
    assert presets.load(path)["params"]["truncation_psi"] == 1.25
    assert [entry.name for entry in tmp_path.iterdir()] == ["look.json"]
