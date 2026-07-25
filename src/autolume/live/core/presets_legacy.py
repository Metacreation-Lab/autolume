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

from autolume.live.core.expr import compile_expression
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
    "Use the Open model button to load your model.",
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

# What the old app started with, so a preset that still holds it configured
# nothing and does not deserve a note.
_PROJECT_DEFAULT = True

_PROJECT_NOTE = (
    "The projection setting was not imported. Projection is not available yet."
)
_VEC_NOTE = "The saved latent vector was not imported."
_NEXT_NOTE = "The queued latent vector was not imported."

_VECTOR_MODE_NOTE = (
    "Vector mode is not available yet. This preset was imported in seed mode."
)

_VECTOR_MENU_NOTE = (
    "The OSC mappings for vector mode were not imported. "
    "Vector mode is not available yet."
)

_UNREADABLE_NOTE = (
    "Part of this preset folder could not be read. "
    "Some of its settings were not imported."
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


def _is_dir(path: Path) -> bool:
    """`Path.is_dir` that treats an unreachable path as absent.

    `pathlib` only swallows the not found errors. A folder copied off a network
    share or out of another user's home answers with `EACCES` instead, which
    would otherwise escape from a probe that runs before any of the guarded
    reading below.
    """
    try:
        return path.is_dir()
    except OSError:
        logger.warning("Could not look at %s", path, exc_info=True)
        return False


def _is_file(path: Path) -> bool:
    """`Path.is_file` that treats an unreachable path as absent. See `_is_dir`."""
    try:
        return path.is_file()
    except OSError:
        logger.warning("Could not look at %s", path, exc_info=True)
        return False


def is_legacy_preset(directory: str | Path) -> bool:
    """True if `directory` is a folder saved by the previous Autolume."""
    root = Path(directory)
    return _is_dir(root) and any(_is_file(root / name) for name in LEGACY_FILES)


def _read_pickle(path: Path, skipped: list[str]) -> object | None:
    if not _is_file(path):
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
        # Every value here came out of a pickle written by another program, so
        # no assumption about its type holds. A tensor in a number's place, for
        # one, makes the update mode converter raise `RuntimeError`.
        except Exception:
            logger.warning(
                "Ignoring legacy value %r for %s",
                section[legacy_name],
                legacy_name,
                exc_info=True,
            )
            skipped.append(f"The saved value for {legacy_name} could not be read.")


def _address(raw: object) -> str | None:
    """Normalize a stored address, or None if it was never configured."""
    if not isinstance(raw, str):
        return None
    address = raw.strip()
    if address.lower() in _PLACEHOLDER_ADDRESSES:
        return None
    return address if address.startswith("/") else f"/{address}"


def _binding(
    target: str, address: str, key: str, raw_mapping: object, skipped: list[str]
) -> Binding:
    """Build a binding, disabling it if the new evaluator rejects its expression."""
    if isinstance(raw_mapping, str):
        expression = raw_mapping.strip()
    else:
        # A field the performer left blank means pass the value through, but a
        # mapping that pickled as something other than text is a corrupt value
        # being normalized, which the performer should hear about.
        if raw_mapping is not None:
            logger.warning("Legacy mapping for %s is not text: %r", key, raw_mapping)
            skipped.append(
                f"The saved OSC mapping for {key} was not readable text. "
                "It was imported passing the value through unchanged."
            )
        expression = ""
    expression = expression or "x"
    try:
        compile_expression(expression)
    # `compile_expression` promises `ExpressionError`, but the source reaching it
    # came out of a pickle, so a wider failure is caught rather than raised at a
    # performer who only clicked Import.
    except Exception as exc:
        logger.warning(
            "Legacy mapping %r for %s is not valid", expression, key, exc_info=True
        )
        error = str(exc) or type(exc).__name__
        return Binding(target, address, expression, enabled=False, error=error)
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
        binding = _binding(target, address, key, mappings.get(key), skipped)
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


def _in_vector_mode(latent: dict) -> bool:
    """True if the old preset was performing on vectors rather than on seeds."""
    try:
        return not latent.get("mode", True)
    except Exception:
        logger.warning("Unreadable legacy latent mode", exc_info=True)
        return False


def _report_latent_unsupported(latent: dict, skipped: list[str]) -> None:
    """Report the latent fields with no home yet, but only when they were used.

    Every real latent structure carries `project`, `vec` and `next`, so a note on
    mere presence would put three lines about things the performer never touched
    in front of every single import. `project` counts as used once it differs
    from what the old app started with, and the two vectors only ever mattered in
    vector mode, which is the only mode that read them.
    """
    project = latent.get("project", _PROJECT_DEFAULT)
    try:
        changed = bool(project != _PROJECT_DEFAULT)
    except Exception:
        # Something unexpected sits in the field, which is itself worth saying.
        logger.warning("Unreadable legacy projection setting", exc_info=True)
        changed = True
    if changed:
        skipped.append(_PROJECT_NOTE)
    if not _in_vector_mode(latent):
        return
    skipped.append(_VECTOR_MODE_NOTE)
    if "vec" in latent:
        skipped.append(_VEC_NOTE)
    if "next" in latent:
        skipped.append(_NEXT_NOTE)


def _import_latent(root: Path, params: dict, bindings: list, skipped: list) -> None:
    data = _read_pickle(root / LATENT_FILE, skipped)
    if data is None:
        return
    parts = _sections(data, LATENT_FILE, 3, skipped)
    if not parts:
        return
    latent = parts[0]
    _read_params(latent, LATENT_FILE, _LATENT_PARAMS, params, skipped)
    if isinstance(latent, dict):
        _report_latent_unsupported(latent, skipped)
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
    _read_params(parts[0], TRUNC_FILE, _TRUNC_PARAMS, params, skipped)
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
    if not _is_dir(root):
        return {}, (), [f"{root.name} is not a preset folder. Nothing was imported."]
    params: dict = {}
    bindings: list[Binding] = []
    skipped: list[str] = []
    # Everything below reports its own failures, so this catches only what no
    # handler saw coming. A performer opening an old folder gets what was
    # gathered plus a note either way, and a probe added later cannot turn into
    # a crash on the Import button.
    try:
        _import_latent(root, params, bindings, skipped)
        _import_trunc(root, params, bindings, skipped)
        # Reported from the file name alone. These are never unpickled, since
        # nothing in them can be imported yet.
        for name, note in UNSUPPORTED_FILES.items():
            if _is_file(root / name):
                skipped.append(note)
    except Exception:
        logger.exception("Could not finish reading legacy preset folder %s", root)
        skipped.append(_UNREADABLE_NOTE)
    return params, tuple(bindings), skipped
