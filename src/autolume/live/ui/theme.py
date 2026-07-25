"""The one place the app's appearance is defined.

Every colour and every metric the UI uses lives here, as data. Panels import
names from this module; nothing under ``ui/`` may hold a colour of its own or
push a numeric literal into the style, and ``tests/live/test_theme.py``
enforces both.

That rule exists because the old app broke it in a way that is easy to miss.
Its metrics were centralised too, in one ``set_default_style`` function, but
the function took the values as arguments and two callers passed different
ones, so the applied style depended on which module had run last. The tables
below take no arguments for that reason, and neither does :func:`apply_theme`.

The surface palette is the old app's, renamed. The old names are not
descriptive (``GREEN`` is teal, ``DARKGRAY`` is white at five percent) and they
still appear throughout the legacy widgets, so the mapping is recorded in
``plan-theme.md`` rather than carried into the names here.

The chip colours below are deliberately not drawn from it. A surface colour is
chosen to sit quietly behind text; a chip colour has to be told apart from
three others at a glance, on eleven rows at once, by an eye that may not
separate hues. Autolume's teal lands at 1.75 to 1 against the idle grey and its
red at 2.99 to 1 over a bright frame, where the gutter needs 2.0 and 4.0, so
using the brand values here would trade a real distinction for a nominal one.
The tests in ``tests/live/test_chip_layout.py`` hold those ratios, and they are
the reason this file carries two palettes instead of one.
"""

from __future__ import annotations

import logging
from pathlib import Path

from imgui_bundle import hello_imgui, imgui

log = logging.getLogger(__name__)

Color = tuple[float, float, float, float]

RED: Color = (235 / 255, 28 / 255, 59 / 255, 1.0)
RED_DIM: Color = (235 / 255, 28 / 255, 59 / 255, 0.40)
TEAL: Color = (39 / 255, 147 / 255, 150 / 255, 1.0)
TEAL_HALF: Color = (39 / 255, 148 / 255, 151 / 255, 0.5)
TEAL_HOVER: Color = (39 / 255, 148 / 255, 151 / 255, 0.3)
TEAL_FAINT: Color = (39 / 255, 148 / 255, 151 / 255, 0.15)
TEAL_SOLID: Color = (23 / 255, 39 / 255, 40 / 255, 1.0)
NEAR_BLACK: Color = (20 / 255, 20 / 255, 20 / 255, 1.0)
WHITE_05: Color = (1.0, 1.0, 1.0, 0.05)
WHITE_10: Color = (1.0, 1.0, 1.0, 0.10)
MID_GRAY: Color = (77 / 255, 77 / 255, 77 / 255, 1.0)
CHARCOAL: Color = (26 / 255, 26 / 255, 26 / 255, 1.0)

BINDING_COLOR: Color = (0.35, 0.75, 1.0, 1.0)
MOTION_COLOR: Color = (0.45, 0.92, 0.45, 1.0)
ERROR_COLOR: Color = (1.0, 0.3, 0.3, 1.0)

_SPACING = 9.0

STYLE_VARS: dict[str, float | tuple[float, float]] = {
    # One less than the rest, where the old app used its single spacing value
    # for this too. At the narrowest panel the app supports, the extra pixel a
    # side is what pushes the longest row past the content edge.
    "window_padding": (8.0, 8.0),
    "item_spacing": (_SPACING, _SPACING),
    # The gap between a widget and its label, which the old app also set from
    # its one spacing knob. Nine there costs every labelled row five pixels of
    # width and buys nothing, so this is imgui's value rather than the legacy's.
    "item_inner_spacing": (4.0, 4.0),
    "columns_min_spacing": _SPACING,
    "indent_spacing": 23.0,
    "scrollbar_size": 15.0,
    "frame_padding": (4.0, 3.0),
    "window_border_size": 1.0,
    "child_border_size": 1.0,
    "popup_border_size": 1.0,
    "frame_border_size": 1.0,
    "window_rounding": 0.0,
    "child_rounding": 0.0,
    "popup_rounding": 0.0,
    "frame_rounding": 0.0,
    "scrollbar_rounding": 0.0,
    "grab_rounding": 0.0,
}

COLOR_OVERRIDES: dict[imgui.Col_, Color] = {
    imgui.Col_.window_bg: NEAR_BLACK,
    imgui.Col_.child_bg: WHITE_05,
    imgui.Col_.border: TEAL_HALF,
    imgui.Col_.frame_bg: WHITE_05,
    imgui.Col_.frame_bg_hovered: WHITE_10,
    imgui.Col_.frame_bg_active: TEAL_HALF,
    imgui.Col_.title_bg: TEAL_FAINT,
    imgui.Col_.title_bg_active: TEAL_HOVER,
    imgui.Col_.title_bg_collapsed: TEAL_FAINT,
    imgui.Col_.menu_bar_bg: WHITE_05,
    imgui.Col_.scrollbar_bg: NEAR_BLACK,
    imgui.Col_.check_mark: TEAL,
    imgui.Col_.slider_grab: TEAL_HALF,
    imgui.Col_.slider_grab_active: TEAL,
    imgui.Col_.button: TEAL_FAINT,
    imgui.Col_.button_hovered: TEAL_HOVER,
    imgui.Col_.button_active: TEAL_HALF,
    imgui.Col_.header: TEAL_FAINT,
    imgui.Col_.header_hovered: TEAL_HOVER,
    imgui.Col_.header_active: TEAL_HALF,
    # The old app had no docking, so the tab and dock colours have no legacy
    # value to match. They take the header family so a tab reads as the same
    # material as the panel header it replaces, and the dimmed variants drop a
    # step so an unfocused dock node recedes.
    imgui.Col_.tab: TEAL_FAINT,
    imgui.Col_.tab_hovered: TEAL_HOVER,
    imgui.Col_.tab_selected: TEAL_HALF,
    imgui.Col_.tab_selected_overline: TEAL,
    imgui.Col_.tab_dimmed: WHITE_05,
    imgui.Col_.tab_dimmed_selected: TEAL_FAINT,
    imgui.Col_.tab_dimmed_selected_overline: TEAL_HOVER,
    imgui.Col_.docking_preview: TEAL_HOVER,
    imgui.Col_.docking_empty_bg: NEAR_BLACK,
}

NO_PADDING: tuple[float, float] = (0.0, 0.0)

_VEC2_VARS = frozenset(
    name for name, value in STYLE_VARS.items() if isinstance(value, tuple)
)


def configure_style(style: imgui.Style) -> None:
    """Write the whole theme into ``style``.

    Takes the destination rather than reading the current context so it can be
    checked against a bare :class:`imgui.Style`, and so no caller is ever
    offered a value to pass.
    """
    imgui.style_colors_dark(style)
    for name, value in STYLE_VARS.items():
        setattr(style, name, imgui.ImVec2(*value) if name in _VEC2_VARS else value)
    for index, color in COLOR_OVERRIDES.items():
        style.set_color_(index, color)


def apply_theme() -> None:
    """The ``setup_imgui_style`` callback."""
    configure_style(imgui.get_style())


FONT_SIZE = 16.0

_FONT_FILE = "OpenSans-Regular.ttf"


def font_path() -> Path | None:
    """The bundled text face, or ``None`` when it is not on disk."""
    # Imported here so the tables above can be checked without the flat root
    # package on the path. Plan 5 moves `utils` under `src/` and dissolves it.
    from utils import resource_paths

    path = resource_paths.resource_path("assets", _FONT_FILE)
    return path if path.is_file() else None


def load_fonts() -> None:
    """The ``load_additional_fonts`` callback.

    Replacing the hello_imgui default drops its icon font with it, which costs
    nothing while no panel draws an icon glyph.
    """
    path = font_path()
    if path is None:
        log.warning("No %s found. Falling back to the built in font.", _FONT_FILE)
        return
    hello_imgui.load_font(
        str(path), FONT_SIZE, hello_imgui.FontLoadingParams(inside_assets=False)
    )
