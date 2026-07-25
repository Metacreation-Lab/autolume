import logging
import os
import pickle
import sys

import pytest
import torch

import dnnlib
from autolume.live.core import presets_legacy
from autolume.live.core.params import REGISTRY, Binding
from autolume.live.core.presets_legacy import import_legacy_preset, is_legacy_preset

LEGACY_LOGGER = "autolume.live.core.presets_legacy"

# A folder with no permissions is still readable by root, so the test would
# assert nothing there. Windows ignores the mode bits entirely.
needs_posix_permissions = pytest.mark.skipif(
    sys.platform.startswith("win") or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="needs POSIX permission bits and a non root user",
)

# Key order matches the legacy widgets, so binding order in the assertions below
# is the order the old app built its OSC menus in.
SEED_KEYS = ("project", "seed", "anim", "speed", "model")
VEC_KEYS = ("project", "anim", "speed", "model", "vector", "randomize")
TRUNC_KEYS = (
    "Diversity",
    "Global Noise",
    "Noise On/Off",
    "Noise Seed",
    "Animate Noise",
    "Reset",
)


def make_latent(**overrides):
    latent = dnnlib.EasyDict(
        vec=torch.randn(1, 512),
        next=torch.randn(1, 512),
        x=0,
        y=0,
        frac_x=0.0,
        frac_y=0.0,
        update_mode=0,
        speed=0.25,
        mode=True,
        project=True,
    )
    latent.update(overrides)
    return latent


def make_menu(keys, addresses=None, use_osc=None, mappings=None):
    addresses = addresses or {}
    use_osc = use_osc or {}
    mappings = mappings or {}
    return (
        dict.fromkeys(keys, True),
        dnnlib.EasyDict((key, bool(use_osc.get(key, True))) for key in keys),
        dnnlib.EasyDict((key, addresses.get(key, "...")) for key in keys),
        dnnlib.EasyDict((key, addresses.get(key, "...")) for key in keys),
        dnnlib.EasyDict((key, mappings.get(key, "x")) for key in keys),
    )


def make_trunc_params(**overrides):
    params = dnnlib.EasyDict(
        trunc_psi=0.8,
        global_noise=1.0,
        noise_enable=True,
        noise_seed=0,
        noise_anim=False,
        reset=False,
    )
    params.update(overrides)
    return params


def write_pickle(directory, name, payload):
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / name, "wb") as fp:
        pickle.dump(payload, fp)


def write_latent(directory, latent=None, seed_menu=None, vec_menu=None):
    write_pickle(
        directory,
        "latent.pkl",
        (
            latent if latent is not None else make_latent(),
            seed_menu if seed_menu is not None else make_menu(SEED_KEYS),
            vec_menu if vec_menu is not None else make_menu(VEC_KEYS),
        ),
    )


def write_trunc(directory, params=None, menu=None):
    write_pickle(
        directory,
        "trunc.pkl",
        (
            params if params is not None else make_trunc_params(),
            menu if menu is not None else make_menu(TRUNC_KEYS),
        ),
    )


def full_preset(directory):
    write_latent(
        directory,
        latent=make_latent(x=3.5, y=-2.0, speed=1.25, update_mode=1),
        seed_menu=make_menu(
            SEED_KEYS,
            addresses={
                "project": "ctl/project",
                "seed": "ctl/seed",
                "anim": "ctl/anim",
                "speed": "ctl/speed",
                "model": "ctl/model",
            },
            use_osc={"speed": False},
            mappings={"seed": "x*2"},
        ),
    )
    write_trunc(
        directory,
        params=make_trunc_params(
            trunc_psi=1.2,
            global_noise=0.4,
            noise_enable=False,
            noise_seed=7,
            noise_anim=True,
        ),
        menu=make_menu(
            TRUNC_KEYS,
            addresses={
                "Diversity": "ctl/psi",
                "Global Noise": "/ctl/gn",
                "Noise On/Off": "ctl/non",
                "Noise Seed": "ctl/seed2",
                "Reset": "ctl/reset",
            },
            mappings={"Diversity": "x/2", "Noise On/Off": "x > 0.5"},
        ),
    )
    return directory


def mentions(skipped, needle):
    return [note for note in skipped if needle in note]


def test_full_preset_imports_every_mapped_parameter(tmp_path):
    params, _, _ = import_legacy_preset(full_preset(tmp_path / "old"))
    assert params == {
        "latent_x": 3.5,
        "latent_y": -2.0,
        "anim_speed_x": 1.25,
        "anim_playing": True,
        "truncation_psi": 1.2,
        "global_noise": 0.4,
        "noise_enabled": False,
        "noise_seed": 7,
        "noise_anim": True,
    }


def test_full_preset_imports_every_mapped_binding(tmp_path):
    _, bindings, _ = import_legacy_preset(full_preset(tmp_path / "old"))
    assert bindings == (
        Binding("latent_x", "/ctl/seed", "x*2", True, None),
        Binding("anim_playing", "/ctl/anim", "x", True, None),
        Binding("anim_speed_x", "/ctl/speed", "x", False, None),
        Binding("truncation_psi", "/ctl/psi", "x/2", True, None),
        Binding("global_noise", "/ctl/gn", "x", True, None),
        Binding("noise_enabled", "/ctl/non", "x > 0.5", True, None),
        Binding("noise_seed", "/ctl/seed2", "x", True, None),
    )


def test_full_preset_reports_the_controls_it_could_not_carry_over(tmp_path):
    _, _, skipped = import_legacy_preset(full_preset(tmp_path / "old"))
    assert mentions(skipped, "mapping for projection")
    assert mentions(skipped, "mapping for model switching")
    assert mentions(skipped, "mapping for reset")
    assert not mentions(skipped, "Vector mode")
    # The preset never left the defaults for these, so they are not worth a note.
    assert not mentions(skipped, "projection setting")
    assert not mentions(skipped, "latent vector")


def test_import_accepts_a_str_path(tmp_path):
    params, bindings, _ = import_legacy_preset(str(full_preset(tmp_path / "old")))
    assert params["latent_x"] == 3.5
    assert bindings


def test_every_imported_binding_targets_a_real_parameter(tmp_path):
    params, bindings, _ = import_legacy_preset(full_preset(tmp_path / "old"))
    assert params and bindings
    assert set(params) <= set(REGISTRY)
    assert {binding.target for binding in bindings} <= set(REGISTRY)


def test_seed_speed_maps_to_the_x_axis_speed_only(tmp_path):
    params, _, _ = import_legacy_preset(full_preset(tmp_path / "old"))
    assert params["anim_speed_x"] == 1.25
    assert "anim_speed_y" not in params


def test_stopped_update_mode_imports_as_not_playing(tmp_path):
    directory = tmp_path / "old"
    write_latent(directory, latent=make_latent(update_mode=0))
    params, _, _ = import_legacy_preset(directory)
    assert params["anim_playing"] is False


def test_vector_mode_preset_is_reported_as_unavailable(tmp_path):
    directory = tmp_path / "old"
    write_latent(directory, latent=make_latent(mode=False))
    _, _, skipped = import_legacy_preset(directory)
    assert mentions(skipped, "Vector mode")


def test_vector_mode_osc_mappings_are_reported(tmp_path):
    directory = tmp_path / "old"
    write_latent(
        directory,
        vec_menu=make_menu(VEC_KEYS, addresses={"vector": "ctl/vec", "anim": "ctl/va"}),
    )
    _, bindings, skipped = import_legacy_preset(directory)
    assert bindings == ()
    assert mentions(skipped, "vector mode")


def test_unconfigured_vector_mode_menu_is_not_reported(tmp_path):
    directory = tmp_path / "old"
    write_latent(directory)
    _, _, skipped = import_legacy_preset(directory)
    assert not mentions(skipped, "vector mode")


def test_missing_latent_file_still_imports_truncation(tmp_path):
    directory = tmp_path / "old"
    write_trunc(directory, params=make_trunc_params(trunc_psi=1.4, noise_seed=12))
    params, _, skipped = import_legacy_preset(directory)
    assert params["truncation_psi"] == 1.4
    assert params["noise_seed"] == 12
    assert "latent_x" not in params
    assert mentions(skipped, "latent.pkl")


def test_missing_truncation_file_still_imports_latent(tmp_path):
    directory = tmp_path / "old"
    write_latent(directory, latent=make_latent(x=1.5))
    params, _, skipped = import_legacy_preset(directory)
    assert params["latent_x"] == 1.5
    assert "truncation_psi" not in params
    assert mentions(skipped, "trunc.pkl")


def test_corrupt_pickle_is_reported_and_the_other_file_still_imports(tmp_path, caplog):
    directory = tmp_path / "old"
    write_trunc(directory, params=make_trunc_params(trunc_psi=1.1))
    (directory / "latent.pkl").write_bytes(b"not a pickle at all")
    with caplog.at_level(logging.WARNING, logger=LEGACY_LOGGER):
        params, _, skipped = import_legacy_preset(directory)
    assert params["truncation_psi"] == 1.1
    assert mentions(skipped, "latent.pkl")
    assert caplog.records


def test_short_tuple_imports_what_it_can_and_reports_the_rest(tmp_path):
    directory = tmp_path / "old"
    write_pickle(directory, "latent.pkl", (make_latent(x=2.5),))
    params, bindings, skipped = import_legacy_preset(directory)
    assert params["latent_x"] == 2.5
    assert bindings == ()
    assert mentions(skipped, "latent.pkl")


def test_extra_tuple_element_imports_what_it_can_and_reports_the_rest(tmp_path):
    directory = tmp_path / "old"
    write_pickle(
        directory,
        "trunc.pkl",
        (make_trunc_params(trunc_psi=1.3), make_menu(TRUNC_KEYS), "from the future"),
    )
    params, _, skipped = import_legacy_preset(directory)
    assert params["truncation_psi"] == 1.3
    assert mentions(skipped, "trunc.pkl")


def test_pickle_that_is_not_a_preset_is_reported(tmp_path):
    directory = tmp_path / "old"
    write_pickle(directory, "latent.pkl", {"nothing": "useful"})
    params, bindings, skipped = import_legacy_preset(directory)
    assert params == {}
    assert bindings == ()
    assert mentions(skipped, "latent.pkl")


def test_malformed_osc_section_is_reported_without_losing_the_parameters(tmp_path):
    directory = tmp_path / "old"
    write_pickle(directory, "trunc.pkl", (make_trunc_params(noise_seed=5), "broken"))
    params, bindings, skipped = import_legacy_preset(directory)
    assert params["noise_seed"] == 5
    assert bindings == ()
    assert mentions(skipped, "OSC")


@pytest.mark.parametrize("placeholder", ["", "...", "osc address", "   "])
def test_placeholder_addresses_produce_no_bindings(tmp_path, placeholder):
    directory = tmp_path / "old"
    addresses = dict.fromkeys(TRUNC_KEYS, placeholder)
    # One real address, so a no-op importer cannot pass this by returning nothing.
    addresses["Diversity"] = "ctl/psi"
    write_trunc(directory, menu=make_menu(TRUNC_KEYS, addresses=addresses))
    _, bindings, skipped = import_legacy_preset(directory)
    assert bindings == (Binding("truncation_psi", "/ctl/psi", "x", True, None),)
    assert not mentions(skipped, "mapping for reset")


def test_disabled_legacy_control_imports_as_a_disabled_binding(tmp_path):
    directory = tmp_path / "old"
    write_trunc(
        directory,
        menu=make_menu(
            TRUNC_KEYS,
            addresses={"Diversity": "ctl/psi"},
            use_osc=dict.fromkeys(TRUNC_KEYS, False),
        ),
    )
    _, bindings, _ = import_legacy_preset(directory)
    assert bindings == (Binding("truncation_psi", "/ctl/psi", "x", False, None),)


HOSTILE = [
    "__import__('os').system('ls')",
    "().__class__.__bases__[0].__subclasses__()",
    "open('SENTINEL', 'w').write('pwned')",
]


@pytest.mark.parametrize("mapping", HOSTILE)
def test_hostile_mapping_is_imported_disabled_and_never_evaluated(
    tmp_path, monkeypatch, mapping
):
    directory = tmp_path / "old"
    monkeypatch.chdir(tmp_path)
    write_trunc(
        directory,
        menu=make_menu(
            TRUNC_KEYS,
            addresses={"Diversity": "ctl/psi"},
            mappings={"Diversity": mapping},
        ),
    )
    _, bindings, skipped = import_legacy_preset(directory)
    assert len(bindings) == 1
    binding = bindings[0]
    assert binding.target == "truncation_psi"
    assert binding.expression == mapping
    assert binding.enabled is False
    assert binding.error
    assert mentions(skipped, "Diversity")
    # Evaluating any of these would leave a trace on disk or in the process.
    assert not (tmp_path / "SENTINEL").exists()
    assert list(tmp_path.iterdir()) == [directory]


def test_invalid_mapping_syntax_is_imported_disabled_with_the_error(tmp_path):
    directory = tmp_path / "old"
    write_trunc(
        directory,
        menu=make_menu(
            TRUNC_KEYS,
            addresses={"Global Noise": "ctl/gn"},
            mappings={"Global Noise": "x +* 2"},
        ),
    )
    _, bindings, skipped = import_legacy_preset(directory)
    assert len(bindings) == 1
    binding = bindings[0]
    assert binding.target == "global_noise"
    assert binding.source == "/ctl/gn"
    assert binding.expression == "x +* 2"
    assert binding.enabled is False
    assert binding.error is not None
    assert "invalid syntax" in binding.error
    assert mentions(skipped, "Global Noise")


def test_unknown_legacy_control_name_is_reported(tmp_path):
    directory = tmp_path / "old"
    keys = TRUNC_KEYS + ("Warp Factor",)
    write_trunc(
        directory,
        menu=make_menu(
            keys,
            addresses={"Diversity": "ctl/psi", "Warp Factor": "ctl/warp"},
        ),
    )
    _, bindings, skipped = import_legacy_preset(directory)
    assert bindings == (Binding("truncation_psi", "/ctl/psi", "x", True, None),)
    assert mentions(skipped, "Warp Factor")


def test_unsupported_legacy_files_are_each_reported(tmp_path):
    directory = tmp_path / "old"
    write_latent(directory)
    for name in ("layer.pkl", "adjuster.pkl", "looper.pkl", "pickle.pkl", "collap.pkl"):
        write_pickle(directory, name, ("whatever",))
    _, _, skipped = import_legacy_preset(directory)
    for needle in ("Layer", "direction", "Looping", "model list", "bending"):
        assert mentions(skipped, needle), needle
    assert not mentions(skipped, "mixing")


def test_unsupported_files_are_reported_without_being_unpickled(tmp_path):
    directory = tmp_path / "old"
    write_latent(directory)
    (directory / "mix.pkl").write_bytes(b"not a pickle at all")
    _, _, skipped = import_legacy_preset(directory)
    assert mentions(skipped, "mixing")


def test_a_directory_that_is_not_a_preset_imports_nothing(tmp_path):
    params, bindings, skipped = import_legacy_preset(tmp_path / "nowhere")
    assert params == {}
    assert bindings == ()
    assert len(skipped) == 1


def test_is_legacy_preset_recognizes_an_old_preset_folder(tmp_path):
    directory = tmp_path / "old"
    write_latent(directory)
    assert is_legacy_preset(directory) is True
    assert is_legacy_preset(str(directory)) is True


def test_is_legacy_preset_rejects_a_new_format_preset(tmp_path):
    from autolume.live.core import presets
    from autolume.live.core.params import ControlState

    presets.save(ControlState(), tmp_path / "look.json")
    write_latent(tmp_path / "old")
    assert is_legacy_preset(tmp_path / "look.json") is False
    assert is_legacy_preset(tmp_path) is False
    assert is_legacy_preset(tmp_path / "old") is True


def test_is_legacy_preset_rejects_an_unrelated_folder(tmp_path):
    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
    write_latent(tmp_path / "old")
    assert is_legacy_preset(tmp_path) is False
    assert is_legacy_preset(tmp_path / "missing") is False
    assert is_legacy_preset(tmp_path / "old") is True


@pytest.fixture
def unreadable_preset(tmp_path):
    """A real legacy folder the current user is not allowed to look inside."""
    directory = tmp_path / "old"
    write_latent(directory)
    directory.chmod(0o000)
    try:
        yield directory
    finally:
        directory.chmod(0o700)


@needs_posix_permissions
def test_unreadable_preset_folder_is_not_a_crash(unreadable_preset):
    params, bindings, skipped = import_legacy_preset(unreadable_preset)
    assert params == {}
    assert bindings == ()
    assert skipped


@needs_posix_permissions
def test_is_legacy_preset_survives_an_unreadable_folder(unreadable_preset):
    assert is_legacy_preset(unreadable_preset) is False


def test_a_failing_probe_still_returns_what_was_gathered(tmp_path, monkeypatch, caplog):
    directory = tmp_path / "old"
    write_trunc(directory, params=make_trunc_params(trunc_psi=1.7))
    real_is_file = presets_legacy._is_file

    def explode(path):
        if path.name in presets_legacy.UNSUPPORTED_FILES:
            raise PermissionError(13, "Permission denied", str(path))
        return real_is_file(path)

    monkeypatch.setattr(presets_legacy, "_is_file", explode)
    with caplog.at_level(logging.WARNING, logger=LEGACY_LOGGER):
        params, _, skipped = import_legacy_preset(directory)
    assert params["truncation_psi"] == 1.7
    assert mentions(skipped, "could not be read")
    assert caplog.records


def test_a_tensor_where_a_number_was_expected_is_reported(tmp_path, caplog):
    directory = tmp_path / "old"
    # bool() on a multi element tensor raises RuntimeError, not ValueError.
    write_latent(directory, latent=make_latent(x=2.0, update_mode=torch.randn(4)))
    with caplog.at_level(logging.WARNING, logger=LEGACY_LOGGER):
        params, _, skipped = import_legacy_preset(directory)
    assert params["latent_x"] == 2.0
    assert "anim_playing" not in params
    assert mentions(skipped, "update_mode")
    assert caplog.records


def test_a_mapping_that_fails_to_compile_at_all_is_reported(tmp_path, monkeypatch):
    directory = tmp_path / "old"

    def explode(source):
        raise RuntimeError("compiler fell over")

    monkeypatch.setattr(presets_legacy, "compile_expression", explode)
    write_trunc(
        directory,
        menu=make_menu(TRUNC_KEYS, addresses={"Diversity": "ctl/psi"}),
    )
    _, bindings, skipped = import_legacy_preset(directory)
    assert len(bindings) == 1
    assert bindings[0].enabled is False
    assert bindings[0].error
    assert mentions(skipped, "Diversity")


def test_a_mapping_that_is_not_text_is_reported(tmp_path):
    directory = tmp_path / "old"
    write_trunc(
        directory,
        menu=make_menu(
            TRUNC_KEYS,
            addresses={"Diversity": "ctl/psi"},
            mappings={"Diversity": torch.randn(2)},
        ),
    )
    _, bindings, skipped = import_legacy_preset(directory)
    assert bindings == (Binding("truncation_psi", "/ctl/psi", "x", True, None),)
    assert mentions(skipped, "Diversity")


def test_a_blank_mapping_is_not_reported(tmp_path):
    directory = tmp_path / "old"
    write_trunc(
        directory,
        menu=make_menu(
            TRUNC_KEYS, addresses={"Diversity": "ctl/psi"}, mappings={"Diversity": ""}
        ),
    )
    _, bindings, skipped = import_legacy_preset(directory)
    assert bindings == (Binding("truncation_psi", "/ctl/psi", "x", True, None),)
    assert not mentions(skipped, "Diversity")


def test_an_untouched_preset_reports_nothing_the_performer_did_not_set(tmp_path):
    directory = tmp_path / "old"
    write_latent(directory)
    write_trunc(directory)
    _, _, skipped = import_legacy_preset(directory)
    assert skipped == []


def test_projection_is_reported_only_when_it_was_turned_off(tmp_path):
    on = tmp_path / "on"
    write_latent(on, latent=make_latent(project=True))
    off = tmp_path / "off"
    write_latent(off, latent=make_latent(project=False))
    assert not mentions(import_legacy_preset(on)[2], "projection setting")
    assert mentions(import_legacy_preset(off)[2], "projection setting")


def test_saved_vectors_are_reported_only_in_vector_mode(tmp_path):
    seeds = tmp_path / "seeds"
    write_latent(seeds, latent=make_latent(mode=True))
    vectors = tmp_path / "vectors"
    write_latent(vectors, latent=make_latent(mode=False))
    assert not mentions(import_legacy_preset(seeds)[2], "latent vector")
    vector_notes = import_legacy_preset(vectors)[2]
    assert mentions(vector_notes, "saved latent vector")
    assert mentions(vector_notes, "queued latent vector")


def test_the_model_note_names_a_control_that_exists(tmp_path):
    directory = tmp_path / "old"
    write_latent(directory)
    write_pickle(directory, "pickle.pkl", ("whatever",))
    _, _, skipped = import_legacy_preset(directory)
    assert mentions(skipped, "Open model")
    assert not mentions(skipped, "model panel")


def test_skipped_notes_read_as_plain_sentences(tmp_path, monkeypatch):
    directory = full_preset(tmp_path / "old")
    write_pickle(directory, "mix.pkl", ("whatever",))
    _, _, skipped = import_legacy_preset(directory)
    # The notes only a damaged or vector mode preset reaches, gathered here so
    # every note the module can produce is held to the same style.
    awkward = tmp_path / "awkward"
    write_latent(
        awkward,
        latent=make_latent(mode=False, project=False, speed="fast"),
        seed_menu=make_menu(
            SEED_KEYS,
            addresses={"seed": "ctl/seed", "project": "ctl/project"},
            mappings={"seed": torch.randn(2)},
        ),
        vec_menu=make_menu(VEC_KEYS, addresses={"vector": "ctl/vec"}),
    )
    skipped += import_legacy_preset(awkward)[2]
    monkeypatch.setattr(presets_legacy, "_import_trunc", lambda *args: 1 / 0)
    skipped += import_legacy_preset(awkward)[2]
    assert mentions(skipped, "Part of this preset folder")
    assert len(skipped) > 10
    for note in skipped:
        assert note == note.strip()
        assert note.endswith(".")
        assert ";" not in note
        assert " - " not in note
        assert note.isascii(), note
