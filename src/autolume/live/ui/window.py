"""Window bootstrap: hello_imgui runner pumped by the asyncio event loop."""

import asyncio

from imgui_bundle import hello_imgui, imgui, immvision

from autolume.live.ui.panels import PerformPanel, PreviewPanel


def _build_runner_params(runtime) -> hello_imgui.RunnerParams:
    immvision.use_rgb_color_order()
    perform = PerformPanel(runtime)
    preview = PreviewPanel(runtime)

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
    controls = hello_imgui.DockableWindow()
    controls.label = "Controls"
    controls.dock_space_name = "MainDockSpace"
    controls.gui_function = perform.gui
    view = hello_imgui.DockableWindow()
    view.label = "Preview"
    view.dock_space_name = "MainDockSpace"
    view.gui_function = preview.gui
    split = hello_imgui.DockingSplit()
    split.initial_dock = "MainDockSpace"
    split.new_dock = "ControlsSpace"
    split.direction = imgui.Dir.left
    split.ratio = 0.35
    controls.dock_space_name = "ControlsSpace"
    params.docking_params.docking_splits = [split]
    params.docking_params.dockable_windows = [controls, view]
    return params


async def _ui_main(runtime) -> None:
    params = _build_runner_params(runtime)
    await hello_imgui.run_async(params)


def run_ui(runtime) -> None:
    asyncio.run(_ui_main(runtime))
