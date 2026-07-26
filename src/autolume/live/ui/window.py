"""Window bootstrap: hello_imgui runner pumped by the asyncio event loop."""

import asyncio

from imgui_bundle import hello_imgui, imgui, immvision

from autolume.live.ui import theme
from autolume.live.ui.panels import (
    AudioPanel,
    LoopPanel,
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


def _viewport_dockable(label: str, space: str, gui) -> hello_imgui.DockableWindow:
    """A dockable window that is a viewport rather than a form.

    Every other panel is a form. Its content is rows of controls and it wants
    the window padding the theme gives it, because text against a window edge
    is cramped. A viewport's content is an image, it is meant to reach the
    edges, and the same padding reads as a mount around the picture.

    Padding is read once, by `Begin`, so there is nowhere inside the panel to
    drop it from. That is the whole reason this window opens itself:
    `call_begin_end` hands the `Begin` to us so the style can be pushed in
    front of it. Docking is unaffected, because a window is docked by name
    through the dock builder rather than by whoever calls `Begin`.

    The preview is the only viewport today. The fullscreen output in the parity
    plan is the same surface with even less chrome, so it goes through here as
    well rather than repeating the reasoning.
    """
    window = _dockable(label, space, lambda: _viewport_body(label, gui))
    window.call_begin_end = False
    return window


def _viewport_body(label: str, gui) -> None:
    """Open `label` with no padding, draw `gui` in it, and always close it.

    `End` is unconditional because imgui pairs it with `Begin` and not with
    what `Begin` returned.
    """
    imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(*theme.NO_PADDING))
    opened, _ = imgui.begin(label)
    imgui.pop_style_var()
    if opened:
        gui()
    imgui.end()


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
    loop = LoopPanel(runtime, mapping.popup)
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

    params.callbacks.setup_imgui_style = theme.apply_theme
    params.callbacks.load_additional_fonts = theme.load_fonts

    params.imgui_window_params.default_imgui_window_type = (
        hello_imgui.DefaultImGuiWindowType.provide_full_screen_dock_space
    )
    # A saved layout wins over the code default, and the condition to apply a
    # layout is first use ever. Naming this layout makes it fire again for
    # installs that already have an ini from an earlier pane arrangement,
    # which is what keeps a new dockable window from hiding behind a saved
    # one that predates it.
    params.docking_params.layout_name = "PatchLoop"
    params.docking_params.docking_splits = [
        _split(_MAIN_SPACE, _CONTROLS_SPACE, imgui.Dir.left, 0.35),
        _split(_CONTROLS_SPACE, _PATCH_SPACE, imgui.Dir.down, 0.5),
    ]
    # No per-window minimum size is set here for "Loop", even though its
    # keyframe row has one it cares about (`loop.py`,
    # `keyframe_row_required_width`): a docked window's size is the dock
    # node's, and neither
    # `hello_imgui.DockableWindow` (its own `window_size` field is documented
    # unused once docked) nor plain `imgui.set_next_window_size_constraints`
    # binds to a dock node splitter drag, only to an undocked window's own
    # resize. The only mechanism that does reach a dock node's size is the
    # internal `imgui.internal.DockNode.size`, forced back every frame,
    # which fights the performer's own drag and the neighbouring split; not
    # used, for that reason. The keyframe row instead reflows to a narrower
    # two line layout below its floor (`loop.py`, `_keyframe_row_two_line`),
    # so a narrow dock is cramped rather than silently overflowing.
    params.docking_params.dockable_windows = [
        _dockable("Controls", _CONTROLS_SPACE, perform.gui),
        _dockable("Loop", _PATCH_SPACE, loop.gui),
        _dockable("Audio", _PATCH_SPACE, audio.gui),
        _dockable("Mapping", _PATCH_SPACE, mapping.gui),
        _dockable("Presets", _PATCH_SPACE, presets.gui),
        _viewport_dockable("Preview", _MAIN_SPACE, preview.gui),
    ]
    return params


async def _ui_main(runtime) -> None:
    params = _build_runner_params(runtime)
    await hello_imgui.run_async(params)


def run_ui(runtime) -> None:
    asyncio.run(_ui_main(runtime))
