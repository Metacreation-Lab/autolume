import json

import pytest

from utils import session_state


@pytest.fixture(autouse=True)
def isolated_state(monkeypatch, tmp_path):
    monkeypatch.setattr(session_state, "cache_path",
                        lambda *parts: tmp_path.joinpath(*parts))
    monkeypatch.setattr(session_state, "_state", None)
    yield tmp_path


def reload_from_disk(monkeypatch):
    monkeypatch.setattr(session_state, "_state", None)


def test_missing_file_reads_as_empty():
    assert session_state.load() == {}
    assert session_state.get("diffusion", "prompt") is None
    assert session_state.get("diffusion", "prompt", "fallback") == "fallback"


def test_set_persists_across_a_reload(monkeypatch, isolated_state):
    session_state.set("diffusion", "prompt", "a zombie")
    assert (isolated_state / "session.json").is_file()
    reload_from_disk(monkeypatch)
    assert session_state.get("diffusion", "prompt") == "a zombie"


def test_sections_do_not_collide():
    session_state.set("diffusion", "prompt", "one")
    session_state.set("pickle", "prompt", "two")
    assert session_state.get("diffusion", "prompt") == "one"
    assert session_state.get("pickle", "prompt") == "two"


def test_corrupt_file_falls_back_to_empty(monkeypatch, isolated_state):
    (isolated_state / "session.json").write_text("{not json", encoding="utf-8")
    reload_from_disk(monkeypatch)
    assert session_state.load() == {}
    session_state.set("diffusion", "prompt", "recovered")
    assert session_state.get("diffusion", "prompt") == "recovered"


def test_unwritable_location_does_not_raise(monkeypatch):
    monkeypatch.setattr(session_state, "cache_path",
                        lambda *parts: pytest.importorskip("pathlib").Path("\0bad") / "session.json")
    session_state.set("diffusion", "prompt", "still fine")
    assert session_state.get("diffusion", "prompt") == "still fine"  # in memory


def test_push_recent_is_most_recent_first_and_deduped():
    session_state.push_recent("diffusion", "prompts", "first")
    session_state.push_recent("diffusion", "prompts", "second")
    session_state.push_recent("diffusion", "prompts", "first")
    assert session_state.get_recent("diffusion", "prompts") == ["first", "second"]


def test_push_recent_caps_the_list():
    for i in range(15):
        session_state.push_recent("diffusion", "prompts", f"p{i}", limit=10)
    recent = session_state.get_recent("diffusion", "prompts")
    assert len(recent) == 10 and recent[0] == "p14" and recent[-1] == "p5"


def test_push_recent_ignores_empty_values():
    session_state.push_recent("diffusion", "prompts", "kept")
    session_state.push_recent("diffusion", "prompts", "")
    assert session_state.get_recent("diffusion", "prompts") == ["kept"]


def test_get_recent_tolerates_wrong_types(isolated_state, monkeypatch):
    (isolated_state / "session.json").write_text(
        json.dumps({"diffusion": {"prompts": ["ok", 5, None]}}), encoding="utf-8")
    reload_from_disk(monkeypatch)
    assert session_state.get_recent("diffusion", "prompts") == ["ok"]
    session_state.set("diffusion", "prompts", "not a list")
    assert session_state.get_recent("diffusion", "prompts") == []
