import numpy as np
import pytest

from features.extraction import ExtractionConfig
from features import store


@pytest.fixture
def model_file(tmp_path):
    path = tmp_path / "network-snapshot-000100.pkl"
    path.write_bytes(b"model-a-bytes")
    return path


def dirs(n=4, dim=16):
    rng = np.random.default_rng(0)
    return rng.normal(size=(n, dim)).astype(np.float32)


def test_roundtrip(tmp_path, model_file):
    root = tmp_path / "features"
    saved = store.save(model_file, dirs(), ExtractionConfig(), root=root)
    assert saved.exists()
    fs = store.lookup(model_file, root=root)
    assert np.array_equal(fs.directions, dirs())
    assert fs.metadata["estimator"] == "fbpca"
    assert fs.metadata["model_sha256"] == store.model_hash(model_file)
    assert fs.metadata["model_path"] == str(model_file)


def test_lookup_missing_returns_none(tmp_path, model_file):
    assert store.lookup(model_file, root=tmp_path / "features") is None


def test_same_stem_different_content_do_not_collide(tmp_path, model_file):
    root = tmp_path / "features"
    other_dir = tmp_path / "other-run"
    other_dir.mkdir()
    other = other_dir / "network-snapshot-000100.pkl"  # same stem
    other.write_bytes(b"model-b-bytes")
    store.save(model_file, dirs(), ExtractionConfig(), root=root)
    assert store.lookup(other, root=root) is None
    store.save(other, dirs(n=3), ExtractionConfig(), root=root)
    assert store.lookup(model_file, root=root).directions.shape == (4, 16)
    assert store.lookup(other, root=root).directions.shape == (3, 16)


def test_survives_model_file_move(tmp_path, model_file):
    root = tmp_path / "features"
    store.save(model_file, dirs(), ExtractionConfig(), root=root)
    moved = tmp_path / "renamed.pkl"
    model_file.rename(moved)
    # Same content, new name: hash matches but the stem differs, so the file
    # is keyed under the old name. lookup() must fall back to a hash match.
    assert store.lookup(moved, root=root) is not None


def test_corrupt_file_returns_none(tmp_path, model_file):
    root = tmp_path / "features"
    path = store.save(model_file, dirs(), ExtractionConfig(), root=root)
    path.write_bytes(b"garbage")
    assert store.lookup(model_file, root=root) is None


def test_no_tmp_file_left_behind(tmp_path, model_file):
    root = tmp_path / "features"
    store.save(model_file, dirs(), ExtractionConfig(), root=root)
    assert [p.name for p in root.iterdir() if p.suffix != ".npz"] == []
