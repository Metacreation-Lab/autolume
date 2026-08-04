"""Curated checkpoint catalog for the diffusion stage.

Separate from the StyleGAN catalog in ``utils/downloads.py``: that one lists
single .pkl files, this one mostly lists HuggingFace repos, which are
directories. Both kinds resolve to the same thing here -- a list of files to
fetch -- so the downloader has one code path.

Every entry is a checkpoint that has been run through this stage at one
denoising step and confirmed to produce usable output. That verification is the
whole point of a curated list, and no external catalog can supply it.
"""
import csv
import io
import logging
import os

import requests

from utils.resource_paths import is_frozen, resource_path

logger = logging.getLogger(__name__)

CATALOG_FILE = "diffusion_models.csv"
CATALOG_URL = ("https://raw.githubusercontent.com/Metacreation-Lab/autolume/main/"
               + CATALOG_FILE)

REQUIRED_COLUMNS = ("name", "style", "base_model", "source", "ref", "dest",
                    "size_mb", "author", "license")
OPTIONAL_COLUMNS = ("variant", "trigger_words")

# Pickles execute arbitrary code on load, so no format that can carry one is
# ever fetched. A repo offering nothing else is not shippable.
WEIGHT_SUFFIX = ".safetensors"
KEEP_SUFFIXES = (WEIGHT_SUFFIX, ".json", ".txt", ".model")
DROP_MARKERS = (".bin", ".ckpt", ".pth", ".msgpack", ".onnx", ".pt", ".non_ema")

HF_API = "https://huggingface.co/api/models/{ref}"
HF_FILE = "https://huggingface.co/{ref}/resolve/main/{path}"


def _parse(handle):
    rows = []
    for row in csv.DictReader(handle):
        if not all((row.get(col) or "").strip() for col in REQUIRED_COLUMNS):
            logger.warning("Skipping malformed %s row: %s", CATALOG_FILE, row)
            continue
        entry = {col: row[col].strip() for col in REQUIRED_COLUMNS}
        for col in OPTIONAL_COLUMNS:
            entry[col] = (row.get(col) or "").strip()
        if entry["source"] not in ("hf", "file"):
            logger.warning("Skipping %s row with unknown source %r", CATALOG_FILE, entry["source"])
            continue
        rows.append(entry)
    return rows


def load_catalog():
    """Catalog entries. Frozen builds try GitHub first so the curated list can
    be refreshed without a release, exactly as the StyleGAN catalog does."""
    if is_frozen():
        try:
            response = requests.get(CATALOG_URL, timeout=(10, 30))
            response.raise_for_status()
            rows = _parse(io.StringIO(response.text))
            if rows:
                return rows
            raise ValueError("remote catalog contained no usable rows")
        except Exception as e:
            logger.warning("Could not fetch %s from GitHub (%s); using bundled copy",
                           CATALOG_FILE, e)
    try:
        with open(resource_path(CATALOG_FILE), newline="", encoding="utf-8") as f:
            return _parse(f)
    except OSError as e:
        logger.error("Could not load %s: %s", CATALOG_FILE, e)
        return []


def _wanted(path, variant, diffusers_repo):
    lowered = path.lower()
    if any(marker in lowered for marker in DROP_MARKERS):
        return False
    if not lowered.endswith(KEEP_SUFFIXES):
        return False
    if lowered.endswith(WEIGHT_SUFFIX):
        # Many diffusers repos also ship the whole model as one root-level
        # checkpoint. It is redundant next to the folder layout and doubles the
        # download: sd-turbo carries 2.5 GB of it, PaperCut 4 GB.
        if diffusers_repo and "/" not in path:
            return False
        # a repo shipping both precisions would otherwise be downloaded twice
        return (".fp16." in lowered) == (variant == "fp16")
    return True


def local_name(path, variant):
    """Destination name for a fetched file.

    An fp16 repo file is stored under its plain name: diffusers only looks for
    the unsuffixed weights unless it is passed variant="fp16", and nothing in
    the wrapper lets us pass that through. safetensors carries its own dtype, so
    the renamed file still loads as half precision.
    """
    if variant == "fp16" and path.lower().endswith(WEIGHT_SUFFIX):
        return path.replace(".fp16.", ".")
    return path


def resolve_files(entry, session=None):
    """[(url, path relative to the model root)] for an entry, pickle-free.

    Relative to the model root, not to the checkpoints folder: the caller owns
    where the model lands, and it downloads into a staging folder already named
    for the entry. Prefixing dest here would nest the model inside itself.
    """
    if entry["source"] == "file":
        return [(entry["ref"], os.path.basename(entry["dest"]))]

    get = (session or requests).get
    response = get(HF_API.format(ref=entry["ref"]), timeout=(10, 30))
    response.raise_for_status()
    variant = entry.get("variant", "")
    files = [s["rfilename"] for s in response.json().get("siblings", [])]
    diffusers_repo = "model_index.json" in files
    wanted = [f for f in files if _wanted(f, variant, diffusers_repo)]
    if not any(f.lower().endswith(WEIGHT_SUFFIX) for f in wanted):
        raise ValueError(f"{entry['ref']} has no safetensors weights for variant "
                         f"{variant or 'default'}; it must not be in the catalog")
    return [(HF_FILE.format(ref=entry["ref"], path=f),
             os.path.join(*local_name(f, variant).split("/")))
            for f in wanted]


def is_installed(entry, checkpoints_dir):
    """True when the entry is present and loadable, not merely started.

    A diffusers folder needs its index and its unet: a half-finished download
    that has model_index.json but no weights would otherwise read as installed.
    """
    target = os.path.join(checkpoints_dir, entry["dest"])
    if entry["source"] == "file":
        return os.path.isfile(target)
    return (os.path.isfile(os.path.join(target, "model_index.json"))
            and os.path.isdir(os.path.join(target, "unet")))
