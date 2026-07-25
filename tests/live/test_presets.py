import dataclasses
import json
import logging
from pathlib import PureWindowsPath

import pytest

from autolume.live.core import params, presets
from autolume.live.core.events import ControlEvent
from autolume.live.core.mapping import apply_event
from autolume.live.core.params import Binding, ControlState

PRESETS_LOGGER = "autolume.live.core.presets"

# Every preset parameter differs from its default, so a round trip that keeps a
# value by accident cannot pass. `test_sample_state_is_non_default_everywhere`
# holds this honest as the registry grows.
SAMPLE = ControlState(
    pkl_path="/models/look.pkl",
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
    pulse_address="/loop/pulse",
    pulse_ip="10.0.0.5",
    pulse_port=9000,
    bindings=(
        Binding("truncation_psi", "/audio/bass", "0.4+x"),
        Binding("latent_x", "/ctl/1", "x*2", enabled=False),
    ),
)


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


def test_apply_address_is_reserved():
    assert presets.PRESET_APPLY.startswith("/preset/")
    assert presets.PRESET_APPLY not in params.BY_ADDRESS


def test_round_trip_restores_every_preset_param_and_bindings(tmp_path):
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
    assert set(payload["params"]) == preset_names()
    assert payload["params"]["truncation_psi"] == 1.25
    assert payload["params"]["noise_enabled"] is False
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


def test_the_frame_limit_is_a_property_of_the_machine_not_of_the_look(tmp_path):
    """A look saved on a laptop capped at 30 must not cap the stage machine.

    The model path is the opposite case and stays persisted: it is what the
    look looks like, while the frame limit is what the hardware can do.
    """
    path = tmp_path / "look.json"
    presets.save(dataclasses.replace(SAMPLE, fps_cap=30), path)
    payload = presets.load(path)
    assert "fps_cap" not in payload["params"]
    assert "pkl_path" in payload["params"]
    assert apply_payload(ControlState(fps_cap=144), payload).fps_cap == 144


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
        values, _ = presets.from_payload(payload)
    assert "latent_y" not in values
    assert values["latent_x"] == 1.5
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
    _, bindings = presets.from_payload(json.loads(raw))
    assert bindings[0].error is None


def test_path_values_are_str_wrapped(tmp_path):
    state = dataclasses.replace(SAMPLE, pkl_path=PureWindowsPath(r"C:\models\m.pkl"))
    path = tmp_path / "look.json"
    presets.save(state, path)
    with open(path, "r", encoding="utf-8") as fp:
        stored = json.load(fp)["params"]["pkl_path"]
    assert stored == r"C:\models\m.pkl"


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
    payload = presets.to_payload(dataclasses.replace(SAMPLE, pkl_path=None))
    assert payload["params"]["pkl_path"] is None
    state = apply_payload(ControlState(pkl_path="/models/current.pkl"), payload)
    assert state.pkl_path == "/models/current.pkl"


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
        _, bindings = presets.from_payload(payload)
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
        _, bindings = presets.from_payload(payload)
    assert [b.target for b in bindings] == ["truncation_psi", "latent_x"]
    assert warnings_from(caplog)


def test_binding_fields_default_when_absent():
    payload = presets.to_payload(SAMPLE)
    payload["bindings"] = [{"target": "latent_y", "source": "/ctl/2"}]
    _, bindings = presets.from_payload(payload)
    assert bindings == (Binding("latent_y", "/ctl/2", "x", True, None),)


def test_bad_expression_is_kept_so_it_can_be_fixed():
    payload = presets.to_payload(SAMPLE)
    payload["bindings"] = [
        {"target": "latent_y", "source": "/ctl/2", "expression": "x +* 2"}
    ]
    _, bindings = presets.from_payload(payload)
    assert bindings[0].expression == "x +* 2"


def test_non_mapping_params_and_non_list_bindings_are_ignored(caplog):
    payload = {"format": presets.FORMAT, "version": 1, "params": 3, "bindings": "x"}
    with caplog.at_level(logging.WARNING):
        values, bindings = presets.from_payload(payload)
    assert values == {}
    assert bindings == ()
    assert len(warnings_from(caplog)) == 2


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
