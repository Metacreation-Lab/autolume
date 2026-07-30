"""Process environment both Autolume apps set before their heavy imports.

The release ships two executables that share one folder of dependencies:
``Autolume`` (``main.py``) and ``Autolume Live`` (``main_live.py``). They are
separate processes, so each one declares here what it wants from the shared
folder rather than inheriting whatever the other app or the surrounding shell
left behind.

Everything in this module has to run before ``torch``, ``glfw``, ``imgui`` or
``imgui_bundle`` are imported: the variables it writes are read once, at the
moment those libraries load.

The GLFW choice is the reason this module is shared rather than a block in
each launcher. The bundle carries two GLFW native libraries and each app must
load exactly one of them:

``pyglfw``
    The library ``release.py`` collects from the ``glfw`` wheel into
    ``glfw/``, next to pyglfw's ``library.py``, where pyglfw's own frozen
    loader finds it. This is the legacy app's GLFW. Leaving ``PYGLFW_LIBRARY``
    unset is what keeps pyglfw's Linux ``x11``/``wayland`` variant selection
    (and with it ``PYGLFW_LIBRARY_VARIANT``) working, so the choice is
    expressed by clearing an inherited value, not by setting one.

``imgui_bundle``
    The library the ``imgui_bundle`` wheel ships next to its compiled
    extension, which that extension links through ``@loader_path``. The live
    app must load this one and only this one: hello_imgui creates the main
    window through it, and ``output_window.py`` drives a second window through
    pyglfw, whose window handles are only valid inside the same native GLFW.
    Two GLFW images in one process is what produces the macOS ``Class
    GLFWHelper is implemented in both ...`` warning.

    Importing ``imgui_bundle`` sets ``PYGLFW_LIBRARY`` to this library as a
    side effect, so the app would mostly get it anyway. We set it here instead
    because "mostly" depends on nothing else importing ``glfw`` first, and
    because an inherited ``PYGLFW_LIBRARY`` would otherwise silently win (its
    own search honours a pre-existing value).

    Note for Linux: ``PYGLFW_LIBRARY_VARIANT`` has no effect on the live app.
    It selects between the two variants inside the ``glfw`` wheel, and the
    live app uses neither; the XWayland behaviour of its windows is a property
    of the GLFW that ``imgui_bundle`` ships.
"""
import importlib.util
import os
import sys
from pathlib import Path

from utils import resource_paths

PYGLFW = "pyglfw"
IMGUI_BUNDLE = "imgui_bundle"

# Filenames of the GLFW library shipped inside the imgui_bundle package, in
# the order imgui_bundle's own search uses. macOS is the name its extension
# links through @loader_path (verified with otool), so it names the same file
# rather than one of the identical copies alongside it.
_IMGUI_BUNDLE_GLFW = {
    "darwin": ("libglfw.3.dylib",),
    "win32": ("glfw3.dll",),
    "linux": ("libglfw.so.3", "libglfw.3.so", "libglfw.so.3.4", "libglfw.so.3.3"),
}


def _runtime_root() -> Path:
    """Bundle root when frozen, repository root from source."""
    return resource_paths.resource_root()


def _torch_extensions_dir() -> str:
    """Build cache for the custom CUDA ops.

    Precompiled entries ship inside the bundle; in development the cache lives
    at the repository root. Inherited by spawned worker processes.
    """
    return str(_runtime_root() / "torch_extensions")


def _bin_dir() -> str:
    """Directory holding the bundled ffmpeg/ffprobe/ninja executables.

    They live in a ``bin/`` subdirectory rather than the bundle root so the
    extensionless ``ffmpeg`` binary cannot collide with the ``ffmpeg`` python
    package. In development the same tree may be nested one level deeper,
    which is what the walk covers.
    """
    bin_root = _runtime_root() / "bin"
    if resource_paths.is_frozen():
        return str(bin_root)
    for root, _dirs, files in os.walk(bin_root):
        if "ffmpeg.exe" in files:
            return root
    return str(bin_root)


def _imgui_bundle_glfw() -> str | None:
    """Path to the GLFW library inside the installed imgui_bundle, if present."""
    names = _IMGUI_BUNDLE_GLFW.get(
        sys.platform if sys.platform in _IMGUI_BUNDLE_GLFW else "linux"
    )
    candidates = []
    spec = importlib.util.find_spec("imgui_bundle")  # does not execute the package
    if spec is not None and spec.submodule_search_locations:
        candidates.append(Path(list(spec.submodule_search_locations)[0]))
    # A frozen bundle may collect the library next to the extension (the layout
    # find_spec reports) or flatten it to the bundle root; check both.
    candidates.append(_runtime_root() / "imgui_bundle")
    candidates.append(_runtime_root())
    for directory in candidates:
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return str(candidate)
    return None


def _configure_glfw(provider: str) -> None:
    if provider == PYGLFW:
        # Hand the choice back to pyglfw's own resolution, including its Linux
        # variant selection, and drop anything inherited.
        os.environ.pop("PYGLFW_LIBRARY", None)
        if sys.platform == "linux" and os.environ.get("WAYLAND_DISPLAY"):
            # Use X11 GLFW (XWayland) so GNOME decorates the window; native
            # Wayland requires libdecor-gtk which is unavailable or crashes on
            # some systems.
            os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")
        return
    if provider != IMGUI_BUNDLE:
        raise ValueError(f"unknown GLFW provider: {provider!r}")
    library = _imgui_bundle_glfw()
    if library is None:
        # Nothing found: leave the variable unset so imgui_bundle's own search
        # runs on import and reports its own diagnostics.
        os.environ.pop("PYGLFW_LIBRARY", None)
        return
    os.environ["PYGLFW_LIBRARY"] = library


def configure(glfw_provider: str) -> None:
    """Set the environment for this process. Call before any heavy import."""
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", _torch_extensions_dir())

    if sys.platform == "darwin":
        # Must be set before torch is imported.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    _configure_glfw(glfw_provider)

    bin_dir = _bin_dir()
    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    if resource_paths.is_frozen():
        # imageio-ffmpeg's own binary is pruned from the bundle (release.py);
        # route it to the ffmpeg already shipped in bin/.
        os.environ.setdefault(
            "IMAGEIO_FFMPEG_EXE",
            os.path.join(bin_dir, "ffmpeg.exe" if os.name == "nt" else "ffmpeg"),
        )
