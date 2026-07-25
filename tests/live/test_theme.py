"""The theme as data, plus the one thing only a real style can answer.

Most of this checks the tables against the imgui build we ship: a metric named
in `STYLE_VARS` that imgui does not have, or a colour index that moved between
versions, is silently a no-op at runtime and shows up as one wrong looking
widget weeks later. `imgui.Style()` constructs without a context, so the whole
apply path can be exercised here rather than by eye.
"""

import ast
import inspect
import logging
from pathlib import Path

import pytest
from imgui_bundle import imgui

from autolume.live.ui import theme

PALETTE = {
    "RED": theme.RED,
    "RED_DIM": theme.RED_DIM,
    "TEAL": theme.TEAL,
    "TEAL_HALF": theme.TEAL_HALF,
    "TEAL_HOVER": theme.TEAL_HOVER,
    "TEAL_FAINT": theme.TEAL_FAINT,
    "TEAL_SOLID": theme.TEAL_SOLID,
    "NEAR_BLACK": theme.NEAR_BLACK,
    "WHITE_05": theme.WHITE_05,
    "WHITE_10": theme.WHITE_10,
    "MID_GRAY": theme.MID_GRAY,
    "CHARCOAL": theme.CHARCOAL,
}


@pytest.mark.parametrize("name", sorted(PALETTE))
def test_every_palette_entry_is_a_normalised_rgba(name):
    color = PALETTE[name]
    assert len(color) == 4
    assert all(isinstance(part, float) for part in color)
    assert all(0.0 <= part <= 1.0 for part in color)


def test_the_driver_marker_colours_are_not_surface_colours():
    """The two palettes stay apart, and this is the reminder of why.

    Reaching for `TEAL` and `RED` here is the obvious tidying and it is wrong:
    both fail the gutter's contrast guards, which is what the module docstring
    records and `test_driver_marker_layout.py` measures. This fails first and points
    there, rather than leaving the tidier to find out from a contrast ratio.
    """
    surfaces = set(PALETTE.values())
    assert theme.MOTION_COLOR not in surfaces
    assert theme.ERROR_COLOR not in surfaces
    assert theme.BINDING_COLOR not in surfaces


def test_every_style_var_is_a_real_style_attribute():
    style = imgui.Style()
    missing = [name for name in theme.STYLE_VARS if not hasattr(style, name)]
    assert missing == []


def test_every_colour_override_is_a_colour_index():
    wrong = [key for key in theme.COLOR_OVERRIDES if not isinstance(key, imgui.Col_)]
    assert wrong == []


def test_no_colour_override_invents_a_shade():
    known = set(PALETTE.values())
    strays = {
        imgui.Col_(key).name: value
        for key, value in theme.COLOR_OVERRIDES.items()
        if value not in known
    }
    assert strays == {}


def test_the_metrics_are_the_old_apps():
    assert theme.STYLE_VARS["indent_spacing"] == 23.0
    assert theme.STYLE_VARS["scrollbar_size"] == 15.0
    assert theme.STYLE_VARS["frame_padding"] == (4.0, 3.0)
    roundings = [key for key in theme.STYLE_VARS if key.endswith("_rounding")]
    assert len(roundings) == 6
    assert all(theme.STYLE_VARS[key] == 0.0 for key in roundings)
    borders = [key for key in theme.STYLE_VARS if key.endswith("_border_size")]
    assert len(borders) == 4
    assert all(theme.STYLE_VARS[key] == 1.0 for key in borders)


def test_applying_the_theme_takes_no_values():
    assert inspect.signature(theme.apply_theme).parameters == {}
    assert list(inspect.signature(theme.configure_style).parameters) == ["style"]


def test_configure_style_writes_every_metric():
    style = imgui.Style()
    theme.configure_style(style)
    assert style.indent_spacing == 23.0
    assert style.scrollbar_size == 15.0
    assert style.scrollbar_rounding == 0.0
    assert style.frame_border_size == 1.0
    assert (style.window_padding.x, style.window_padding.y) == (8.0, 8.0)
    assert (style.item_spacing.x, style.item_spacing.y) == (9.0, 9.0)
    assert (style.frame_padding.x, style.frame_padding.y) == (4.0, 3.0)


def test_configure_style_writes_every_colour():
    style = imgui.Style()
    theme.configure_style(style)
    for index, expected in theme.COLOR_OVERRIDES.items():
        written = style.color_(index)
        assert (written.x, written.y, written.z, written.w) == pytest.approx(expected)


def test_configure_style_starts_from_the_dark_base():
    # popup_bg is deliberately not overridden, so it proves the base ran.
    style = imgui.Style()
    theme.configure_style(style)
    popup = style.color_(imgui.Col_.popup_bg)
    assert (popup.x, popup.y, popup.z) == pytest.approx((0.08, 0.08, 0.08), abs=1e-3)


def test_the_font_ships_with_the_app():
    path = theme.font_path()
    assert path is not None
    assert path.suffix == ".ttf"


def test_a_missing_font_file_is_not_a_font(monkeypatch, tmp_path):
    from utils import resource_paths

    monkeypatch.setattr(
        resource_paths, "resource_path", lambda *parts: tmp_path.joinpath(*parts)
    )
    assert theme.font_path() is None


def test_loading_a_missing_font_warns_instead_of_raising(monkeypatch, caplog):
    monkeypatch.setattr(theme, "font_path", lambda: None)
    with caplog.at_level(logging.WARNING):
        theme.load_fonts()
    assert "built in font" in caplog.text


# Ownership: theme.py is the only place appearance is decided. A colour or a
# metric written anywhere else is invisible to the theme, so restyling the app
# means hunting through every panel for numbers that look like they matter.
# The source is parsed rather than matched, because a regex cannot tell a
# colour tuple from any other three floats, nor a literal argument from the
# name of a constant. It is parsed rather than imported so that a module which
# needs a live imgui context still gets checked.

UI_DIR = Path(theme.__file__).parent


def _ui_trees():
    """Every ui module except theme.py, as (relative path, parsed tree)."""
    for path in sorted(UI_DIR.rglob("*.py")):
        if path.name == "theme.py":
            continue
        yield path.relative_to(UI_DIR.parent), ast.parse(path.read_text())


def _is_number(node):
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float))


def _all_numbers(nodes):
    return bool(nodes) and all(_is_number(node) for node in nodes)


def _called_name(call):
    """The bare name of what a call calls, dotted or not."""
    func = call.func
    return func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)


def test_no_module_defines_a_colour_outside_the_theme():
    offenders = []
    for relative, tree in _ui_trees():
        for node in tree.body:
            if isinstance(node, ast.Assign):
                value = node.value
            elif isinstance(node, ast.AnnAssign):
                value = node.value
            else:
                continue
            if not isinstance(value, (ast.Tuple, ast.List)):
                continue
            # Two numbers are a size, not a colour.
            if len(value.elts) in (3, 4) and _all_numbers(value.elts):
                offenders.append(f"{relative}:{node.lineno}")
    assert offenders == []


def test_no_module_pushes_a_style_literal_outside_the_theme():
    pushes = {"push_style_var", "push_style_var_x", "push_style_var_y"}
    offenders = []
    for relative, tree in _ui_trees():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _called_name(node) not in pushes or len(node.args) < 2:
                continue
            value = node.args[1]
            literal = (
                _is_number(value)
                or (
                    isinstance(value, (ast.Tuple, ast.List))
                    and _all_numbers(value.elts)
                )
                or (
                    isinstance(value, ast.Call)
                    and _called_name(value) == "ImVec2"
                    and _all_numbers(value.args)
                )
            )
            if literal:
                offenders.append(f"{relative}:{node.lineno}")
    assert offenders == []
