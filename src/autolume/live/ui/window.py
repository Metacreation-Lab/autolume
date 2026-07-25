"""Window bootstrap: hello_imgui runner pumped by the asyncio event loop."""

import asyncio

from imgui_bundle import hello_imgui, imgui, immvision

from autolume.live.ui.panels import (
    AudioPanel,
    MappingPanel,
    PerformPanel,
    PresetsPanel,
    PreviewPanel,
)

_CONTROLS_SPACE = "ControlsSpace"
_PATCH_SPACE = "PatchSpace"
_MAIN_SPACE = "MainDockSpace"


def _dockable(label: str, space: str, gui) -> hello_imgui.DockableWindow:
    window = hello_imgui.DockableWindow()
    window.label = label
    window.dock_space_name = space
    window.gui_function = gui
    return window


def _split(initial: str, new: str, direction, ratio: float):
    split = hello_imgui.DockingSplit()
    split.initial_dock = initial
    split.new_dock = new
    split.direction = direction
    split.ratio = ratio
    return split


def _build_runner_params(runtime) -> hello_imgui.RunnerParams:
    immvision.use_rgb_color_order()
    # The mapping panel owns the editor that every control's right click menu
    # opens, so it is built first and handed to the panel drawing the controls.
    mapping = MappingPanel(runtime)
    perform = PerformPanel(runtime, mapping.popup)
    preview = PreviewPanel(runtime)
    audio = AudioPanel(runtime)
    presets = PresetsPanel(runtime)

    params = hello_imgui.RunnerParams()
    params.app_window_params.window_title = "Autolume Live"
    params.app_window_params.window_geometry.size = (1280, 800)
    params.app_window_params.restore_previous_geometry = True
    params.platform_backend_type = hello_imgui.PlatformBackendType.glfw
    params.renderer_backend_type = hello_imgui.RendererBackendType.open_gl3
    params.ini_folder_type = hello_imgui.IniFolderType.app_user_config_folder
    params.fps_idling.enable_idling = False

    params.imgui_window_params.default_imgui_window_type = (
        hello_imgui.DefaultImGuiWindowType.provide_full_screen_dock_space
    )
    params.docking_params.docking_splits = [
        _split(_MAIN_SPACE, _CONTROLS_SPACE, imgui.Dir.left, 0.35),
        _split(_CONTROLS_SPACE, _PATCH_SPACE, imgui.Dir.down, 0.5),
    ]
    params.docking_params.dockable_windows = [
        _dockable("Controls", _CONTROLS_SPACE, perform.gui),
        _dockable("Audio", _PATCH_SPACE, audio.gui),
        _dockable("Mapping", _PATCH_SPACE, mapping.gui),
        _dockable("Presets", _PATCH_SPACE, presets.gui),
        _dockable("Preview", _MAIN_SPACE, preview.gui),
    ]
    return params


async def _ui_main(runtime) -> None:
    params = _build_runner_params(runtime)
    await hello_imgui.run_async(params)


def run_ui(runtime) -> None:
    asyncio.run(_ui_main(runtime))
