"""Curated checkpoint catalog for the diffusion stage.

Single-file .safetensors checkpoints, the same artifact ComfyUI, A1111 and
Civitai all use. Diffusers' multi-folder layout is deliberately not supported:
it is a library convention rather than an ecosystem one, and a folder per model
does not belong in a checkpoints directory next to the LoRAs.

Every entry has been run through this stage at one denoising step and confirmed
to produce usable output. That verification is the point of a curated list and
no external catalog can supply it.
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

REQUIRED_COLUMNS = ("name", "style", "base_model", "filename", "url",
                    "size_mb", "author", "license")
OPTIONAL_COLUMNS = ("trigger_words",)

# Pickles execute arbitrary code on load, so nothing else is ever listed.
WEIGHT_SUFFIX = ".safetensors"


def _parse(handle):
    rows = []
    for row in csv.DictReader(handle):
        if not all((row.get(col) or "").strip() for col in REQUIRED_COLUMNS):
            logger.warning("Skipping malformed %s row: %s", CATALOG_FILE, row)
            continue
        entry = {col: row[col].strip() for col in REQUIRED_COLUMNS}
        for col in OPTIONAL_COLUMNS:
            entry[col] = (row.get(col) or "").strip()
        if not entry["filename"].endswith(WEIGHT_SUFFIX):
            logger.warning("Skipping %s row that is not safetensors: %s",
                           CATALOG_FILE, entry["filename"])
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


def destination(entry, checkpoints_dir):
    """Where this entry's checkpoint lives once downloaded."""
    return os.path.join(checkpoints_dir, entry["filename"])


def is_installed(entry, checkpoints_dir):
    return os.path.isfile(destination(entry, checkpoints_dir))
