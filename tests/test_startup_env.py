"""The two-app GLFW policy in utils.startup_env.

Only the environment decisions are covered. What each library then loads is a
property of the built bundle and is verified by launching it.
"""
import os
import sys

import pytest

from utils import startup_env


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("PYGLFW_LIBRARY", "PYGLFW_LIBRARY_VARIANT", "WAYLAND_DISPLAY"):
        monkeypatch.delenv(name, raising=False)


def test_pyglfw_drops_an_inherited_library(monkeypatch):
    """Legacy resolves GLFW through pyglfw itself, whatever the parent set."""
    monkeypatch.setenv("PYGLFW_LIBRARY", "/somewhere/else/libglfw.3.dylib")

    startup_env.configure(startup_env.PYGLFW)

    assert "PYGLFW_LIBRARY" not in os.environ


def test_pyglfw_keeps_the_xwayland_workaround(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")

    startup_env.configure(startup_env.PYGLFW)

    assert os.environ["PYGLFW_LIBRARY_VARIANT"] == "x11"


def test_pyglfw_leaves_a_chosen_variant_alone(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setenv("PYGLFW_LIBRARY_VARIANT", "wayland")

    startup_env.configure(startup_env.PYGLFW)

    assert os.environ["PYGLFW_LIBRARY_VARIANT"] == "wayland"


def test_live_pins_the_imgui_bundle_library(monkeypatch):
    """The live app names its GLFW rather than depending on import order."""
    monkeypatch.setenv("PYGLFW_LIBRARY", "/somewhere/else/libglfw.3.dylib")

    startup_env.configure(startup_env.IMGUI_BUNDLE)

    library = os.environ["PYGLFW_LIBRARY"]
    assert os.path.isfile(library)
    assert os.path.basename(os.path.dirname(library)) == "imgui_bundle"


def test_live_falls_back_to_the_imgui_bundle_search(monkeypatch):
    """No library found means no claim: imgui_bundle searches on import."""
    monkeypatch.setattr(startup_env, "_imgui_bundle_glfw", lambda: None)
    monkeypatch.setenv("PYGLFW_LIBRARY", "/somewhere/else/libglfw.3.dylib")

    startup_env.configure(startup_env.IMGUI_BUNDLE)

    assert "PYGLFW_LIBRARY" not in os.environ


def test_unknown_provider_is_refused():
    with pytest.raises(ValueError):
        startup_env.configure("system")


def test_torch_extensions_dir_is_not_overridden(monkeypatch):
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", "/custom/cache")

    startup_env.configure(startup_env.PYGLFW)

    assert os.environ["TORCH_EXTENSIONS_DIR"] == "/custom/cache"


def test_bin_dir_is_prepended_to_path(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")

    startup_env.configure(startup_env.PYGLFW)

    first = os.environ["PATH"].split(os.pathsep)[0]
    assert os.path.basename(first) == "bin"
