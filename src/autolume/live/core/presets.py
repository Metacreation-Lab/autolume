"""Save and restore a performance look as one versioned JSON file.

Which parameters persist is driven by `ParamSpec.preset`, so the format cannot
drift from the parameters that exist. Reading is forgiving in one direction
only: unknown or malformed entries are logged and skipped, missing ones keep
their current value, but a file that is not a preset raises. Values are applied
through `apply_value`, so a hand edited file cannot push a parameter out of
range, and expressions are never evaluated here.
"""

import json
import logging
import os
from pathlib import Path

from autolume.live.core.params import REGISTRY, Binding, ControlState

logger = logging.getLogger(__name__)

FORMAT = "autolume-live-preset"
VERSION = 1

# Structured control address. It carries the whole payload so the control loop
# applies a preset in a single state replacement.
PRESET_APPLY = "/preset/apply"

_BINDING_FIELDS = ("target", "source", "expression")


def _jsonable(value: object) -> object:
    """Return `value` as something `json.dump` accepts.

    Paths reach the state from file dialogs and are not serializable, so they
    are written as text. This is the recurring `WindowsPath` bug class.
    """
    return os.fspath(value) if isinstance(value, os.PathLike) else value


def to_payload(state: ControlState) -> dict:
    return {
        "format": FORMAT,
        "version": VERSION,
        "params": {
            name: _jsonable(getattr(state, name))
            for name, spec in REGISTRY.items()
            if spec.preset
        },
        "bindings": [
            {
                "target": binding.target,
                "source": binding.source,
                "expression": binding.expression,
                "enabled": binding.enabled,
            }
            for binding in state.bindings
        ],
    }


def _check_envelope(payload: object) -> None:
    """Reject anything that is not a preset, and warn about a newer format.

    A wrong `format` is a wrong file, not an old file, so it raises rather than
    loading a surprising subset of it.
    """
    if not isinstance(payload, dict):
        raise ValueError(f"preset payload is {type(payload).__name__}, not an object")
    if payload.get("format") != FORMAT:
        raise ValueError(f"not an {FORMAT} file: format is {payload.get('format')!r}")
    version = payload.get("version")
    if not isinstance(version, int):
        logger.warning(
            "Preset version %r is not a number, loading best effort", version
        )
    elif version > VERSION:
        logger.warning(
            "Preset version %d is newer than %d, loading what this build knows",
            version,
            VERSION,
        )


def _read_params(raw: object) -> dict:
    if not isinstance(raw, dict):
        logger.warning("Ignoring preset params of type %s", type(raw).__name__)
        return {}
    values = {}
    for name, value in raw.items():
        spec = REGISTRY.get(name)
        if spec is None or not spec.preset:
            logger.warning(
                "Skipping preset parameter %s, this build does not persist it", name
            )
            continue
        # A null is how an unset parameter is written. Keeping the current value
        # matches how a missing key behaves and stops a loaded model from being
        # replaced by the string "None".
        if value is not None:
            values[name] = value
    return values


def _read_bindings(raw: object) -> tuple[Binding, ...]:
    if not isinstance(raw, list):
        logger.warning("Ignoring preset bindings of type %s", type(raw).__name__)
        return ()
    bindings: list[Binding] = []
    seen: set[str] = set()
    for entry in raw:
        binding = _read_binding(entry)
        if binding is None:
            continue
        # At most one binding per target, enforced here because this is where a
        # hand edited file could introduce a second one. Downstream code that
        # updates a binding only replaces the first match.
        if binding.target in seen:
            logger.warning("Skipping extra preset binding for %s", binding.target)
            continue
        seen.add(binding.target)
        bindings.append(binding)
    return tuple(bindings)


def _read_binding(entry: object) -> Binding | None:
    if not isinstance(entry, dict):
        logger.warning("Skipping preset binding of type %s", type(entry).__name__)
        return None
    fields = {name: entry.get(name) for name in _BINDING_FIELDS}
    fields["expression"] = entry.get("expression", "x")
    if not all(isinstance(fields[name], str) for name in _BINDING_FIELDS):
        logger.warning("Skipping malformed preset binding %r", entry)
        return None
    if fields["target"] not in REGISTRY:
        logger.warning(
            "Skipping preset binding for %s, this build has no such parameter",
            fields["target"],
        )
        return None
    # A bad expression is kept so the performer can see and fix the typo. It is
    # compiled by the control loop, which records the error on the binding.
    return Binding(enabled=bool(entry.get("enabled", True)), error=None, **fields)


def from_payload(payload: dict) -> tuple[dict, tuple[Binding, ...]]:
    """Split a payload into parameter values and bindings, both validated.

    Raises `ValueError` if `payload` is not a preset. Values are returned
    uncoerced; clamping happens when they are applied to a state.
    """
    _check_envelope(payload)
    return _read_params(payload.get("params")), _read_bindings(payload.get("bindings"))


def save(state: ControlState, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(to_payload(state), fp, indent=2)


def load(path: str | Path) -> dict:
    """Read and validate a preset file, returning its payload."""
    with open(path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)
    _check_envelope(payload)
    return payload


def preset_dir() -> Path:
    """Directory holding the user's presets, created on demand."""
    # Imported here because the user data root is a legacy flat root module,
    # and only this function needs it. The control path stays independent of it.
    from utils.user_data import data_path

    directory = data_path("live") / "presets"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def list_presets(directory: str | Path | None = None) -> list[str]:
    """Preset names, without the `.json` suffix, sorted."""
    root = Path(directory) if directory is not None else preset_dir()
    return sorted(path.stem for path in root.glob("*.json") if path.is_file())
