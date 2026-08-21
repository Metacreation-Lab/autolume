import ctypes
import importlib
import sys
import types

import pytest

from utils import opengl_support


def test_parse_plain_version():
    assert opengl_support.parse_gl_version("1.1.0") == (1, 1)


def test_parse_vendor_suffix():
    assert opengl_support.parse_gl_version("4.6.0 NVIDIA 551.86") == (4, 6)


def test_parse_unparseable():
    assert opengl_support.parse_gl_version("") == (0, 0)
    assert opengl_support.parse_gl_version(None) == (0, 0)
    assert opengl_support.parse_gl_version("Mesa") == (0, 0)


@pytest.fixture
def gl_stub(monkeypatch):
    """Install a fake OpenGL.GL so no real context is needed."""
    def install(version, renderer):
        stub = types.SimpleNamespace(
            GL_VERSION="version", GL_RENDERER="renderer",
            glGetString=lambda name: {"version": version,
                                      "renderer": renderer}[name])
        monkeypatch.setitem(sys.modules, "OpenGL.GL", stub)
        # ``import OpenGL.GL as gl`` binds the package attribute when the real
        # module has already been imported, bypassing the sys.modules entry.
        monkeypatch.setattr(importlib.import_module("OpenGL"), "GL", stub,
                            raising=False)
    return install


def test_software_renderer_raises(gl_stub, caplog):
    gl_stub(b"1.1.0", b"GDI Generic")
    with pytest.raises(opengl_support.OpenGLUnsupportedError) as exc:
        opengl_support.check_context_version()
    assert "OpenGL compatible GPU" in str(exc.value)
    assert "GDI Generic" in caplog.text


def test_missing_version_string_raises(gl_stub):
    gl_stub(None, None)
    with pytest.raises(opengl_support.OpenGLUnsupportedError):
        opengl_support.check_context_version()


def test_capable_context_passes(gl_stub):
    gl_stub(b"4.6.0 NVIDIA 551.86", b"NVIDIA GeForce RTX 3060")
    opengl_support.check_context_version()


def test_null_pointer_window_raises(monkeypatch):
    # pyGLFW returns the raw ctypes pointer from glfwCreateWindow: on failure
    # that is a falsy NULL pointer object, not None (the no-GPU crash on the
    # winget validation VM).
    null_window = ctypes.POINTER(ctypes.c_void_p)()
    assert null_window is not None and not null_window
    stub = types.SimpleNamespace(
        GLFWError=RuntimeError,
        create_window=lambda **kwargs: null_window)
    monkeypatch.setitem(sys.modules, "glfw", stub)
    with pytest.raises(opengl_support.OpenGLUnsupportedError):
        opengl_support.checked_create_window(640, 480, "test")
