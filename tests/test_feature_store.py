import json
from dataclasses import dataclass

import numpy as np
import pytest

from features import store


@dataclass(frozen=True)
class Cfg:
    n_components: int = 4
    seed: int = 0
    n_samples: int = 2048


@pytest.fixture
def model_file(tmp_path):
    path = tmp_path / "network-snapshot-000100.pkl"
    path.write_bytes(b"model-a-bytes")
    return path


def dirs(n=4, dim=16):
    rng = np.random.default_rng(0)
    return rng.normal(size=(n, dim)).astype(np.float32)


def sigmas(n=4):
    return np.linspace(3.0, 1.0, n).astype(np.float32)


def test_roundtrip(tmp_path, model_file):
    root = tmp_path / "features"
    saved = store.save(model_file, dirs(), sigmas(), Cfg(), root=root)
    assert saved.exists()
    fs = store.lookup(model_file, root=root)
    assert np.array_equal(fs.directions, dirs())
    assert np.array_equal(fs.sigmas, sigmas())
    assert fs.metadata["version"] == 2
    assert fs.metadata["n_components"] == 4
    assert "estimator" not in fs.metadata
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
    store.save(model_file, dirs(), sigmas(), Cfg(), root=root)
    assert store.lookup(other, root=root) is None
    store.save(other, dirs(n=3), sigmas(n=3), Cfg(), root=root)
    assert store.lookup(model_file, root=root).directions.shape == (4, 16)
    assert store.lookup(other, root=root).directions.shape == (3, 16)


def test_survives_model_file_move(tmp_path, model_file):
    root = tmp_path / "features"
    store.save(model_file, dirs(), sigmas(), Cfg(), root=root)
    moved = tmp_path / "renamed.pkl"
    model_file.rename(moved)
    # Same content, new name: hash matches but the stem differs, so the file
    # is keyed under the old name. lookup() must fall back to a hash match.
    assert store.lookup(moved, root=root) is not None


def test_corrupt_file_returns_none(tmp_path, model_file):
    root = tmp_path / "features"
    path = store.save(model_file, dirs(), sigmas(), Cfg(), root=root)
    path.write_bytes(b"garbage")
    assert store.lookup(model_file, root=root) is None


def test_no_tmp_file_left_behind(tmp_path, model_file):
    root = tmp_path / "features"
    store.save(model_file, dirs(), sigmas(), Cfg(), root=root)
    assert [p.name for p in root.iterdir() if p.suffix != ".npz"] == []


def test_old_format_version_reads_as_absent(tmp_path, model_file):
    root = tmp_path / "features"
    path = store.save(model_file, dirs(), sigmas(), Cfg(), root=root)
    with np.load(path) as data:
        metadata = json.loads(data["metadata"].item())
    metadata["version"] = 1
    with open(path, "wb") as f:
        np.savez(f, directions=dirs(), sigmas=sigmas(), metadata=json.dumps(metadata))
    assert store.lookup(model_file, root=root) is None
