"""Best effort import of a preset folder saved by the previous Autolume.

The old format is a directory of pickled positional tuples with no version
field and no key names. Only the two files whose contents still have a home in
the new runtime are read, `latent.pkl` and `trunc.pkl`. Everything the new
runtime cannot express yet, vectors, keyframes, layer bending and model mixing,
is reported in `skipped` rather than half imported, and nothing here raises: a
performer opening an old folder gets a list of what did not come across, not a
crash.

Legacy mapping strings were stored to be handed to `eval`, so they are treated
as hostile text. Every one is compiled by `expr.compile_expression` and an
expression the new evaluator rejects is imported disabled with its error
attached, never executed. The pickles themselves are still read with `pickle`,
which is the format's own trust boundary and cannot be tightened without losing
the ability to read real presets.
"""

import logging
import pickle
from pathlib import Path

from autolume.live.core.expr import ExpressionError, compile_expression
from autolume.live.core.params import Binding

logger = logging.getLogger(__name__)

LATENT_FILE = "latent.pkl"
TRUNC_FILE = "trunc.pkl"

# Named so the panel can tell an old preset folder from a new `.json` file.
# Presence of any of them is what makes a directory a legacy preset.
UNSUPPORTED_FILES = {
    "layer.pkl": "Layer view settings were not imported.",
    "adjuster.pkl": "Latent direction sliders were not imported. "
    "Directions are not available yet.",
    "looper.pkl": "Looping keyframes were not imported. Looping is not available yet.",
    "pickle.pkl": "The model list was not imported. "
    "Pick your model in the model panel.",
    "collap.pkl": "Layer bending was not imported. "
    "Network bending is not available yet.",
    "mix.pkl": "Model mixing was not imported. Model mixing is not available yet.",
}

LEGACY_FILES = (LATENT_FILE, TRUNC_FILE, *UNSUPPORTED_FILES)

# What the old address field holds when the performer never filled it in.
_PLACEHOLDER_ADDRESSES = frozenset({"", "...", "osc address"})

_LATENT_PARAMS = (
    ("x", "latent_x", float),
    ("y", "latent_y", float),
    # The old seed walk only ever moved along x, so its single speed is the x speed.
    ("speed", "anim_speed_x", float),
    ("update_mode", "anim_playing", lambda value: bool(value != 0)),
)

_LATENT_UNSUPPORTED = {
    "project": "The projection setting was not imported. "
    "Projection is not available yet.",
    "vec": "The saved latent vector was not imported.",
    "next": "The queued latent vector was not imported.",
}

_VECTOR_MENU_NOTE = (
    "The OSC mappings for vector mode were not imported. "
    "Vector mode is not available yet."
)

_TRUNC_PARAMS = (
    ("trunc_psi", "truncation_psi", float),
    ("global_noise", "global_noise", float),
    ("noise_enable", "noise_enabled", bool),
    ("noise_seed", "noise_seed", int),
    ("noise_anim", "noise_anim", bool),
)

_SEED_MENU_TARGETS = {
    "seed": "latent_x",
    "speed": "anim_speed_x",
    "anim": "anim_playing",
}

_SEED_MENU_UNSUPPORTED = {
    "project": "The OSC mapping for projection was not imported. "
    "Projection is not available yet.",
    "model": "The OSC mapping for model switching was not imported. "
    "Model switching is not available yet.",
}

_TRUNC_MENU_TARGETS = {
    "Diversity": "truncation_psi",
    "Global Noise": "global_noise",
    "Noise On/Off": "noise_enabled",
    "Noise Seed": "noise_seed",
    "Animate Noise": "noise_anim",
}

_TRUNC_MENU_UNSUPPORTED = {
    "Reset": "The OSC mapping for reset was not imported. "
    "There is no reset control yet."
}


def is_legacy_preset(directory: str | Path) -> bool:
    """True if `directory` is a folder saved by the previous Autolume."""
    root = Path(directory)
    return root.is_dir() and any((root / name).is_file() for name in LEGACY_FILES)


def _read_pickle(path: Path, skipped: list[str]) -> object | None:
    if not path.is_file():
        skipped.append(f"No {path.name} was found in this preset.")
        return None
    try:
        with open(path, "rb") as fp:
            return pickle.load(fp)
    except Exception:
        logger.exception("Could not read legacy preset file %s", path)
        skipped.append(
            f"{path.name} could not be read. It is damaged or from another version."
        )
        return None


def _sections(data: object, label: str, expected: int, skipped: list[str]) -> tuple:
    """Return `data` as a tuple of sections, reporting an unexpected shape.

    A short or long tuple still yields the sections it does have, so a file
    written by a slightly different build imports what it can.
    """
    if not isinstance(data, (list, tuple)):
        skipped.append(f"{label} does not hold a preset. Nothing was imported from it.")
        return ()
    if len(data) != expected:
        skipped.append(
            f"{label} does not have the expected contents. "
            "Only the parts that could be read were imported."
        )
    return tuple(data)


def _read_params(
    section: object,
    label: str,
    table: tuple,
    unsupported: dict[str, str],
    params: dict,
    skipped: list[str],
) -> None:
    if not isinstance(section, dict):
        skipped.append(f"The settings in {label} could not be read.")
        return
    for legacy_name, target, convert in table:
        if legacy_name not in section:
            continue
        try:
            params[target] = convert(section[legacy_name])
        except (TypeError, ValueError):
            logger.warning(
                "Ignoring legacy value %r for %s", section[legacy_name], legacy_name
            )
            skipped.append(f"The saved value for {legacy_name} could not be read.")
    for legacy_name, note in unsupported.items():
        if legacy_name in section:
            skipped.append(note)


def _address(raw: object) -> str | None:
    """Normalize a stored address, or None if it was never configured."""
    if not isinstance(raw, str):
        return None
    address = raw.strip()
    if address.lower() in _PLACEHOLDER_ADDRESSES:
        return None
    return address if address.startswith("/") else f"/{address}"


def _binding(target: str, address: str, key: str, raw_mapping: object) -> Binding:
    """Build a binding, disabling it if the new evaluator rejects its expression."""
    expression = raw_mapping.strip() if isinstance(raw_mapping, str) else ""
    expression = expression or "x"
    try:
        compile_expression(expression)
    except ExpressionError as exc:
        logger.warning(
            "Legacy mapping %r for %s is not valid: %s", expression, key, exc
        )
        return Binding(target, address, expression, enabled=False, error=str(exc))
    return Binding(target, address, expression, enabled=True, error=None)


def _read_menu(
    section: object,
    label: str,
    targets: dict[str, str],
    unsupported: dict[str, str],
    bindings: list[Binding],
    skipped: list[str],
) -> None:
    parts = _sections(section, f"The OSC section of {label}", 5, skipped)
    if len(parts) < 5:
        return
    _, use_osc, addresses, _, mappings = parts[:5]
    if not isinstance(addresses, dict):
        skipped.append(f"The OSC addresses in {label} could not be read.")
        return
    if not isinstance(use_osc, dict):
        use_osc = {}
    if not isinstance(mappings, dict):
        mappings = {}
    for key, raw_address in addresses.items():
        address = _address(raw_address)
        # An unconfigured control lost nothing, so it is not worth reporting.
        if address is None:
            continue
        target = targets.get(key)
        if target is None:
            skipped.append(
                unsupported.get(key)
                or f"The OSC mapping for {key} is not recognized. It was not imported."
            )
            continue
        binding = _binding(target, address, key, mappings.get(key))
        if binding.error is not None:
            skipped.append(
                f"The OSC mapping for {key} is not valid in this version. "
                "It was imported switched off so you can fix it."
            )
        elif not use_osc.get(key, False):
            binding = Binding(target, address, binding.expression, enabled=False)
        bindings.append(binding)


def _has_configured_address(section: object) -> bool:
    """True if an OSC menu section holds at least one address worth reporting."""
    if not isinstance(section, (list, tuple)) or len(section) < 5:
        return False
    addresses = section[2]
    if not isinstance(addresses, dict):
        return False
    return any(_address(raw) is not None for raw in addresses.values())


def _import_latent(root: Path, params: dict, bindings: list, skipped: list) -> None:
    data = _read_pickle(root / LATENT_FILE, skipped)
    if data is None:
        return
    parts = _sections(data, LATENT_FILE, 3, skipped)
    if not parts:
        return
    latent = parts[0]
    _read_params(
        latent, LATENT_FILE, _LATENT_PARAMS, _LATENT_UNSUPPORTED, params, skipped
    )
    # `mode` false means the old preset was performing on vectors, not seeds.
    if isinstance(latent, dict) and not latent.get("mode", True):
        skipped.append(
            "Vector mode is not available yet. This preset was imported in seed mode."
        )
    if len(parts) > 1:
        _read_menu(
            parts[1],
            LATENT_FILE,
            _SEED_MENU_TARGETS,
            _SEED_MENU_UNSUPPORTED,
            bindings,
            skipped,
        )
    # The third section is the vector mode OSC menu. Nothing in it has a home in
    # the new runtime, so it is only reported, and only when it was configured.
    if len(parts) > 2 and _has_configured_address(parts[2]):
        skipped.append(_VECTOR_MENU_NOTE)


def _import_trunc(root: Path, params: dict, bindings: list, skipped: list) -> None:
    data = _read_pickle(root / TRUNC_FILE, skipped)
    if data is None:
        return
    parts = _sections(data, TRUNC_FILE, 2, skipped)
    if not parts:
        return
    _read_params(parts[0], TRUNC_FILE, _TRUNC_PARAMS, {}, params, skipped)
    if len(parts) > 1:
        _read_menu(
            parts[1],
            TRUNC_FILE,
            _TRUNC_MENU_TARGETS,
            _TRUNC_MENU_UNSUPPORTED,
            bindings,
            skipped,
        )


def import_legacy_preset(
    directory: str | Path,
) -> tuple[dict, tuple[Binding, ...], list[str]]:
    """Read an old preset folder into parameter values, bindings and notes.

    `skipped` is the user facing list of everything that did not come across,
    ready to be shown in the presets panel. Nothing raises: whatever could be
    gathered is returned and the rest is reported.
    """
    root = Path(directory)
    if not root.is_dir():
        return {}, (), [f"{root.name} is not a preset folder. Nothing was imported."]
    params: dict = {}
    bindings: list[Binding] = []
    skipped: list[str] = []
    _import_latent(root, params, bindings, skipped)
    _import_trunc(root, params, bindings, skipped)
    # Reported from the file name alone. These are never unpickled, since
    # nothing in them can be imported yet.
    for name, note in UNSUPPORTED_FILES.items():
        if (root / name).is_file():
            skipped.append(note)
    return params, tuple(bindings), skipped
