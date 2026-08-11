import os

import pytest

from utils.presets import WIDGET_FILES, PresetStore, load_preset, save_preset


@pytest.fixture
def root(tmp_path):
    return tmp_path / "presets"


def make_preset(root, name):
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "latent.pkl").write_bytes(b"x")
    return folder


def test_names_empty_when_root_missing(root):
    assert PresetStore(root).names() == []


def test_names_lists_only_folders_with_marker(root):
    make_preset(root, "b")
    (root / "empty").mkdir()  # leftover from the old 12-slot behavior
    (root / "stray.txt").write_text("not a preset")
    assert PresetStore(root).names() == ["b"]


def test_names_sorted_case_insensitive(root):
    for name in ("banana", "Apple", "cherry"):
        make_preset(root, name)
    assert PresetStore(root).names() == ["Apple", "banana", "cherry"]


def test_external_changes_are_picked_up(root):
    make_preset(root, "a")
    store = PresetStore(root)
    assert store.names() == ["a"]
    make_preset(root, "b")
    assert store.names() == ["a", "b"]


def test_path_is_a_str_under_root(root):
    p = PresetStore(root).path("sunset")
    assert isinstance(p, str)
    assert p == str(root / "sunset")


@pytest.mark.parametrize("name", [
    "", " ", "  ", "a/b", "a\\b", "a:b", 'a"b', "a?b", "a*b", "a<b", "a>b",
    "a|b", " padded", "padded ", ".hidden", "trailing.", "CON", "con",
    "Nul.txt", "com1",
])
def test_is_valid_name_rejects_bad_names(root, name):
    assert not PresetStore(root).is_valid_name(name)


def test_is_valid_name_rejects_duplicates_case_insensitive(root):
    make_preset(root, "Sunset")
    store = PresetStore(root)
    assert not store.is_valid_name("Sunset")
    assert not store.is_valid_name("sunset")
    assert store.is_valid_name("sunset 2")


def test_module_is_ui_free():
    import inspect

    import utils.presets

    source = inspect.getsource(utils.presets)
    assert "imgui" not in source
    assert "widgets" not in source
    assert "modules" not in source


def test_create_returns_path_and_reuses_empty_leftover_folders(root):
    store = PresetStore(root)
    (root / "3").mkdir(parents=True)  # empty leftover slot folder
    assert store.create("3") == str(root / "3")
    assert store.create("fresh") == str(root / "fresh")
    assert (root / "fresh").is_dir()


def test_create_rejects_invalid_and_taken_names(root):
    make_preset(root, "taken")
    store = PresetStore(root)
    assert store.create("taken") is None
    assert store.create("a/b") is None
    assert store.create("") is None


def test_created_folder_appears_after_marker_and_invalidate(root):
    store = PresetStore(root)
    path = store.create("new")
    assert store.names() == []  # no marker file yet
    (root / "new" / "latent.pkl").write_bytes(b"x")
    store.invalidate()
    assert store.names() == ["new"]
    assert path == str(root / "new")


def test_rename_moves_folder(root):
    make_preset(root, "old")
    store = PresetStore(root)
    assert store.rename("old", "new") is True
    assert store.names() == ["new"]
    assert (root / "new" / "latent.pkl").is_file()
    assert not (root / "old").exists()


def test_rename_rejects_unknown_taken_or_malformed(root):
    make_preset(root, "a")
    make_preset(root, "b")
    store = PresetStore(root)
    assert store.rename("missing", "c") is False
    assert store.rename("a", "B") is False   # taken, case-insensitive
    assert store.rename("a", "x/y") is False
    assert store.names() == ["a", "b"]


def test_rename_allows_case_only_change(root):
    make_preset(root, "sunset")
    store = PresetStore(root)
    assert store.rename("sunset", "Sunset") is True
    assert store.names() == ["Sunset"]


def test_rename_refuses_to_consume_an_empty_leftover_folder(root):
    make_preset(root, "old")
    (root / "3").mkdir()  # leftover from the old 12-slot behavior
    store = PresetStore(root)
    assert store.rename("old", "3") is False
    assert store.names() == ["old"]
    assert (root / "3").is_dir()


def test_rename_to_same_name_is_a_noop(root):
    make_preset(root, "a")
    assert PresetStore(root).rename("a", "a") is True


def test_delete_removes_folder(root):
    make_preset(root, "gone")
    make_preset(root, "kept")
    store = PresetStore(root)
    assert store.delete("gone") is True
    assert store.names() == ["kept"]
    assert not (root / "gone").exists()


def test_delete_unknown_returns_false(root):
    assert PresetStore(root).delete("missing") is False


class StubWidget:
    def __init__(self, fail=False):
        self.fail = fail
        self.saved = []
        self.loaded = []

    def save(self, path):
        if self.fail:
            raise RuntimeError("boom")
        self.saved.append(path)

    def load(self, path):
        if self.fail:
            raise RuntimeError("boom")
        self.loaded.append(path)


class FakeViz:
    def __init__(self):
        for attr, _ in WIDGET_FILES:
            setattr(self, attr, StubWidget())


def test_widget_files_covers_the_eight_widgets():
    assert [a for a, _ in WIDGET_FILES] == [
        "latent_widget", "trunc_noise_widget", "layer_widget",
        "adjuster_widget", "looping_widget", "pickle_widget",
        "collapsed_widget", "mixing_widget"]
    assert [f for _, f in WIDGET_FILES] == [
        "latent.pkl", "trunc.pkl", "layer.pkl", "adjuster.pkl",
        "looper.pkl", "pickle.pkl", "collap.pkl", "mix.pkl"]


def test_save_preset_creates_folder_and_saves_every_widget(tmp_path):
    viz = FakeViz()
    path = str(tmp_path / "presets" / "new")
    assert save_preset(viz, path) is True
    assert os.path.isdir(path)
    for attr, filename in WIDGET_FILES:
        assert getattr(viz, attr).saved == [os.path.join(path, filename)]


def test_save_preset_failure_returns_false(tmp_path):
    viz = FakeViz()
    viz.layer_widget = StubWidget(fail=True)
    assert save_preset(viz, str(tmp_path / "p")) is False


def test_load_preset_loads_every_widget(tmp_path):
    viz = FakeViz()
    path = str(tmp_path / "p")
    assert load_preset(viz, path) is True
    for attr, filename in WIDGET_FILES:
        assert getattr(viz, attr).loaded == [os.path.join(path, filename)]


def test_load_preset_tolerates_missing_pickle_pkl(tmp_path):
    viz = FakeViz()
    viz.pickle_widget = StubWidget(fail=True)
    assert load_preset(viz, str(tmp_path / "p")) is True
    assert viz.mixing_widget.loaded  # widgets after pickle still load


def test_load_preset_aborts_on_other_failures(tmp_path):
    viz = FakeViz()
    viz.trunc_noise_widget = StubWidget(fail=True)
    assert load_preset(viz, str(tmp_path / "p")) is False
    assert viz.latent_widget.loaded    # before the failure
    assert not viz.layer_widget.loaded  # after the failure
