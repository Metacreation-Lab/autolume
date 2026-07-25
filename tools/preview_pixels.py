"""Check what the preview panel actually puts on the screen.

    uv run tools/preview_pixels.py

Opens a window, draws a known frame through the real `PreviewPanel` in both
display modes, reads the framebuffer back, and reports the geometry and whether
anything was interpolated. It exits on its own.

This exists because the preview took four rounds of screenshots to get right,
and every round was a hypothesis shipped without a way to check it. Reading the
framebuffer back answers in one number what a person squinting at a screenshot
cannot: how many device pixels a frame pixel covers, and whether the edges
between them are hard.

**How to read the output.** A magnified frame drawn without interpolation is
made of blocks, so a scanline across it holds long runs of one colour. The run
lengths printed are those blocks. A run of 1 in the middle of a magnified image
means a blended edge, which means something is interpolating that should not
be. Fit on a 2x display should be blocks of two, because native size is one
frame pixel per point.

It needs a display. It opens a real window for a few frames, so it will not run
over ssh or in a headless shell, and it is a tool rather than a test for that
reason.
"""

import numpy as np
from imgui_bundle import hello_imgui, imgui, immvision

from autolume.live.ui.panels.preview import DisplayMode, PreviewPanel

WINDOW = (500, 400)


def checkerboard(size: int = 64) -> np.ndarray:
    """A frame whose every pixel differs from its neighbours.

    Which is what makes a blended edge visible at all: over a smooth photograph
    an interpolated magnification and a blocky one look nearly the same, and
    the whole point here is to tell them apart.
    """
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[::2, ::2] = 255
    frame[1::2, 1::2] = (255, 0, 0)
    frame.flags.writeable = False
    return frame


class Runtime:
    """The little the panel asks of a runtime, and nothing else."""

    def __init__(self, frame):
        self._frame = frame
        self.preview = self
        self.model_host = self

    def latest(self):
        return 1, self._frame

    def pending(self):
        return None

    def error(self):
        return None

    def current(self):
        return self


def capture(mode: DisplayMode, frame: np.ndarray) -> np.ndarray:
    """One window, three frames drawn, and the last framebuffer read back."""
    panel = PreviewPanel(Runtime(frame))
    panel._mode = mode
    drawn = []

    def gui():
        imgui.set_next_window_pos(imgui.ImVec2(0, 0))
        imgui.set_next_window_size(imgui.ImVec2(*WINDOW))
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(0, 0))
        imgui.begin("Preview", None, imgui.WindowFlags_.no_decoration)
        imgui.pop_style_var()
        panel.gui()
        imgui.end()
        drawn.append(1)
        if len(drawn) == 3:
            hello_imgui.get_runner_params().app_shall_exit = True

    immvision.use_rgb_color_order()
    params = hello_imgui.RunnerParams()
    params.app_window_params.window_title = f"preview pixels: {mode.value}"
    params.app_window_params.window_geometry.size = WINDOW
    params.platform_backend_type = hello_imgui.PlatformBackendType.glfw
    params.renderer_backend_type = hello_imgui.RendererBackendType.open_gl3
    params.imgui_window_params.default_imgui_window_type = (
        hello_imgui.DefaultImGuiWindowType.no_default_window
    )
    params.callbacks.show_gui = gui
    # Otherwise every run drops a settings file next to whatever it was run
    # from, and a tool that litters the repo is a tool nobody runs twice.
    params.ini_filename = ""
    params.ini_folder_type = hello_imgui.IniFolderType.temp_folder
    hello_imgui.run(params)
    return np.asarray(hello_imgui.final_app_window_screenshot())


def drawn_bounds(image: np.ndarray) -> tuple[int, int, int]:
    """Where the frame landed in the captured window.

    Returned as a left, top and side, because the frame is square here and the
    panel letterboxes rather than crops.

    Found by exclusion rather than by correlation: everything in the window is
    one flat colour except the frame, so the frame is the bounding box of what
    is not that colour. Correlating a resampling of the frame against the image
    also works, but its score is nearly flat within a block of the true size,
    and a box a block too large takes in letterboxing and reports it as a
    blended edge. This is exact.
    """
    # The window's own border is drawn at the very edge, so it is cropped off
    # before anything is measured or it would be found instead of the frame.
    margin = 4
    inner = image[margin:-margin, margin:-margin]
    colours, counts = np.unique(inner.reshape(-1, 3), axis=0, return_counts=True)
    background = colours[counts.argmax()]
    content = np.any(inner != background, axis=2)
    rows = np.flatnonzero(content.any(axis=1))
    columns = np.flatnonzero(content.any(axis=0))
    side = min(rows[-1] - rows[0] + 1, columns[-1] - columns[0] + 1)
    return int(columns[0]) + margin, int(rows[0]) + margin, int(side)


def report(mode: DisplayMode, frame: np.ndarray) -> None:
    image = capture(mode, frame)
    left, top, side = drawn_bounds(image)
    # A block in from each edge. The frame's own outermost pixels sit against
    # the letterboxing, so a scanline taken right to the edge ends on a partial
    # block and reports it as a short run. Interpolation shows across the whole
    # image rather than only at its border, so the interior is where to count.
    block = max(1, side // frame.shape[0])
    scanline = image[top + side // 2, left + block : left + side - block, 0]
    changes = np.flatnonzero(np.diff(scanline)) + 1
    runs = np.diff(np.concatenate([[0], changes, [len(scanline)]]))
    print(f"{mode.value}:")
    print(f"  framebuffer      {image.shape[1]}x{image.shape[0]} device pixels")
    print(
        f"  frame drawn at   {side}x{side}, {side / frame.shape[0]:.2f} per frame pixel"
    )
    print(f"  run lengths      {sorted(set(runs.tolist()))}")
    print(
        f"  runs of one      {int((runs == 1).sum())} (any at all means a blended edge)"
    )


def main() -> None:
    frame = checkerboard()
    for mode in DisplayMode:
        report(mode, frame)


if __name__ == "__main__":
    main()
