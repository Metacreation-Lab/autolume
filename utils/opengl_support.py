"""Startup check that the machine provides the OpenGL context the UI needs.

Machines without a GPU or its driver (virtual machines, remote desktop
sessions, fresh installs) either fail to create the GLFW window outright or
fall back to a software renderer (e.g. Windows' "GDI Generic", OpenGL 1.1)
that cannot run the imgui backend's GLSL 3.30 shaders. Both used to surface
as an access violation deep inside the GUI stack and trigger the crash-report
dialog. GlfwWindow routes window creation through the checked helpers below,
which raise OpenGLUnsupportedError with a user-readable message instead;
main.py catches it, shows a native dialog via show_unsupported_dialog(), and
exits cleanly without offering a crash report.

Import-light on purpose (no glfw/OpenGL at module level): main.py imports it
before the heavy dependency stack loads.
"""

import logging
import re
import subprocess
import sys

logger = logging.getLogger(__name__)

DIALOG_TITLE = "Autolume"

# The imgui backend compiles GLSL 3.30 shaders on Windows/Linux; macOS runs
# the fixed-pipeline backend on the 2.1 compatibility context it offers.
MIN_VERSION = (2, 1) if sys.platform == "darwin" else (3, 3)

_REQUIREMENT = ("Autolume requires an OpenGL compatible GPU and up-to-date "
                "graphics drivers to run.")


class OpenGLUnsupportedError(Exception):
    """The machine cannot provide the OpenGL context the UI needs."""


def checked_init():
    """glfw.init() that raises OpenGLUnsupportedError on failure."""
    import glfw
    try:
        ok = glfw.init()
    except glfw.GLFWError as err:
        raise OpenGLUnsupportedError(_REQUIREMENT) from err
    if not ok:
        raise OpenGLUnsupportedError(_REQUIREMENT)


def checked_create_window(width, height, title):
    """glfw.create_window() that raises instead of returning a NULL window.

    Without a capable driver GLFW returns a NULL window (pyGLFW only warns),
    and the first call that dereferences it dies with an access violation.
    pyGLFW hands back the raw ctypes pointer, so failure is a falsy NULL
    pointer object, not None.
    """
    import glfw
    try:
        window = glfw.create_window(width=width, height=height, title=title,
                                    monitor=None, share=None)
    except glfw.GLFWError as err:
        raise OpenGLUnsupportedError(_REQUIREMENT) from err
    if not window:
        raise OpenGLUnsupportedError(_REQUIREMENT)
    return window


def check_context_version():
    """Verify the current context meets MIN_VERSION; requires a current context.

    Catches software fallbacks where window creation succeeds but the context
    is far below what the UI needs (e.g. GDI Generic OpenGL 1.1).
    """
    import OpenGL.GL as gl
    version = gl.glGetString(gl.GL_VERSION)
    renderer = gl.glGetString(gl.GL_RENDERER)
    version = version.decode("utf-8", "replace") if version else ""
    renderer = renderer.decode("utf-8", "replace") if renderer else "unknown"
    if parse_gl_version(version) >= MIN_VERSION:
        return
    logger.error("OpenGL context below the %d.%d minimum: version=%r renderer=%r",
                 *MIN_VERSION, version, renderer)
    raise OpenGLUnsupportedError(_REQUIREMENT)


def parse_gl_version(version_string):
    """Leading "major.minor" of a GL_VERSION string; (0, 0) if unparseable."""
    match = re.match(r"(\d+)\.(\d+)", version_string or "")
    if not match:
        return (0, 0)
    return (int(match.group(1)), int(match.group(2)))


def show_unsupported_dialog(message):
    """Best-effort native error dialog; never raises."""
    print(message, file=sys.stderr)
    try:
        if sys.platform == "win32":
            import ctypes
            MB_OK, MB_ICONERROR = 0x0, 0x10
            ctypes.windll.user32.MessageBoxW(
                None, message, DIALOG_TITLE, MB_OK | MB_ICONERROR)
        elif sys.platform == "darwin":
            script = ('display dialog "%s" with title "%s" buttons {"OK"} '
                      'default button "OK" with icon stop'
                      % (message.replace('"', "'"), DIALOG_TITLE))
            subprocess.run(["osascript", "-e", script],
                           capture_output=True, timeout=300)
        else:
            subprocess.run(
                ["zenity", "--error", "--title", DIALOG_TITLE,
                 "--text", message],
                capture_output=True, timeout=300)
    except Exception:
        pass
