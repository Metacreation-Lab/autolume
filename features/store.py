"""Feature set storage keyed by model file content hash."""
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from utils.user_data import data_path

FORMAT_VERSION = 1

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureSet:
    directions: np.ndarray
    metadata: dict


def model_hash(model_path):
    h = hashlib.sha256()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _root(root):
    return Path(root) if root is not None else data_path("features")


def feature_path(model_path, root=None, digest=None):
    digest = digest or model_hash(model_path)
    return _root(root) / f"{Path(model_path).stem}-{digest[:8]}.npz"


def save(model_path, directions, config, root=None):
    digest = model_hash(model_path)
    path = feature_path(model_path, root=root, digest=digest)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "version": FORMAT_VERSION,
        "estimator": config.estimator,
        "n_features": int(directions.shape[0]),
        "seed": config.seed,
        "n_samples": config.n_samples,
        "sparsity": config.sparsity,
        "model_path": str(model_path),
        "model_sha256": digest,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, directions=directions.astype(np.float32),
                 metadata=json.dumps(metadata))
    os.replace(tmp, path)
    return path


def _read(path):
    with np.load(path) as data:
        metadata = json.loads(data["metadata"].item())
        directions = np.array(data["directions"], dtype=np.float32)
    if metadata.get("version") != FORMAT_VERSION:
        return None
    return FeatureSet(directions=directions, metadata=metadata)


def lookup(model_path, root=None):
    """Feature set for this model file, or None if absent or unreadable."""
    try:
        digest = model_hash(model_path)
        path = feature_path(model_path, root=root, digest=digest)
        if path.exists():
            return _read(path)
        # Renamed model: same content hashed under a different stem.
        root_dir = _root(root)
        if root_dir.is_dir():
            for candidate in root_dir.glob(f"*-{digest[:8]}.npz"):
                fs = _read(candidate)
                if fs is not None and fs.metadata.get("model_sha256") == digest:
                    return fs
        return None
    except Exception:
        logger.exception("Failed to read features for %s", model_path)
        return None
