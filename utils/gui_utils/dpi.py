"""Display-scale math for the imgui UI, in one place.

Three coordinate spaces are involved; every function here converts between them:

- **UI units**   — DPI-independent design units. The app expresses sizes it wants
                   to look physically constant (e.g. the base font) in these.
- **Logical units** — the space imgui lays out and draws in. Equal to GLFW window
                   units on every platform *except* XWayland HiDPI, where GLFW
                   reports window size in physical pixels and the renderer scales
                   back to logical units itself.
- **Physical pixels** — framebuffer pixels actually lit on the monitor.

The two scale factors that relate them:

- ``monitor_scale``  — physical pixels per UI unit (the OS "display scale":
                       1.0 at 100%, 1.5 at 150%, 2.0 on retina / 200%).
- ``pixels_per_logical`` — physical pixels per logical unit, i.e. the resolution
                       the font atlas is rasterized at (2 on retina and XWayland
                       HiDPI, 1 otherwise).

All functions take the raw GLFW window handle and are pure queries.
"""

import sys

import glfw
import numpy as np

# Contrast curve applied to the font atlas on 1x displays, where unhinted small
# glyphs rasterize with soft coverage: cut the faint AA halo below the offset and
# push stems toward solid. A stronger cousin of imgui's RasterizerMultiply, which
# pyimgui 1.4.1 does not expose; values chosen by visual comparison on a 1x monitor.
FONT_CONTRAST_OFFSET = 30
FONT_CONTRAST_GAIN = 1.55


def monitor_scale(window):
    """OS display scale of the monitor the window is on: physical px per UI unit.

    1.0 = 100%, 1.5 = 150%, 2.0 = retina / 200%. Per-window and updated by the OS
    when the window moves to another monitor.
    """
    xscale, _ = glfw.get_window_content_scale(window)
    return max(1.0, xscale)


def window_unit_scale(window):
    """GLFW window units per logical unit.

    1.0 everywhere except XWayland HiDPI. macOS and native Wayland report logical
    window units directly. Windows reports physical pixels but the app uses them
    as its logical drawing space, so no correction there either. Only XWayland
    reports physical pixels *and* expects logical layout, so its window size is
    divided by the OS content scale.
    """
    if sys.platform != 'linux':
        return 1.0
    fb_w, _ = glfw.get_framebuffer_size(window)
    win_w, _ = glfw.get_window_size(window)
    if fb_w == win_w:
        xscale, _ = glfw.get_window_content_scale(window)
        return max(1.0, xscale)
    return 1.0


def pixels_per_logical(window):
    """Physical pixels per logical unit — the font atlas rasterization scale.

    2 on retina / XWayland HiDPI, 1 on standard-DPI monitors.
    """
    fb_w, _ = glfw.get_framebuffer_size(window)
    logical_w = max(glfw.get_window_size(window)[0] / window_unit_scale(window), 1)
    return max(1, round(fb_w / logical_w))


def is_native_1x(window):
    """True when 1 logical unit maps to exactly 1 physical pixel (a 100% monitor).

    This is the gate for atlas sharpening: only there do the font's fractional
    glyph advances land on fractional pixels and smear under bilinear sampling.
    ``window`` may be None (renderer not yet bound to a window) -> False.
    """
    if window is None:
        return False
    return monitor_scale(window) <= 1.0 and pixels_per_logical(window) == 1


def scale_ui_size(size, window):
    """Convert a UI-unit size to logical units for the window's current monitor.

    ``monitor_scale`` grows the size to the display's physical density;
    ``pixels_per_logical`` divides back out where window units are already scaled
    (retina points, XWayland-corrected logical units), leaving the UI physically
    constant across monitors and platforms.
    """
    return round(size * monitor_scale(window) / pixels_per_logical(window))


def sharpen_font_alpha(alpha):
    """Apply the 1x contrast curve to a font-atlas alpha array (float or int)."""
    return np.clip((alpha.astype(np.float32) - FONT_CONTRAST_OFFSET) * FONT_CONTRAST_GAIN, 0, 255)
