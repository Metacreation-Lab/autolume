# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import contextlib
import imgui
from assets.colors import *


# ----------------------------------------------------------------------------

def set_default_style(color_scheme='dark', spacing=9, indent=23, scrollbar=15):
    s = imgui.get_style()
    s.window_padding = [spacing, spacing]
    s.item_spacing = [spacing, spacing]
    s.item_inner_spacing = [spacing, spacing]
    s.columns_min_spacing = spacing
    s.indent_spacing = indent
    s.scrollbar_size = scrollbar
    s.frame_padding = [4, 3]
    s.window_border_size = 1
    s.child_border_size = 1
    s.popup_border_size = 1
    s.frame_border_size = 1
    s.window_rounding = 0
    s.child_rounding = 0
    s.popup_rounding = 0
    s.frame_rounding = 0
    s.scrollbar_rounding = 0
    s.grab_rounding = 0

    getattr(imgui, f'style_colors_{color_scheme}')(s)
    # s.colors[imgui.COLOR_TEXT] = black
    # s.colors[imgui.COLOR_TEXT_DISABLED] = gray
    # s.colors[imgui.COLOR_WINDOW_BACKGROUND] = GREEN
    s.colors[imgui.COLOR_CHILD_BACKGROUND] = DARKGRAY
    # s.colors[imgui.COLOR_POPUP_BACKGROUND] = [0., 1., 0., 1.00]
    s.colors[imgui.COLOR_BORDER] = GREEN
    # s.colors[imgui.COLOR_BORDER_SHADOW] = [0., 1., 0., 0.00]
    s.colors[imgui.COLOR_FRAME_BACKGROUND] = DARKGRAY
    s.colors[imgui.COLOR_FRAME_BACKGROUND_HOVERED] = GRAY
    s.colors[imgui.COLOR_FRAME_BACKGROUND_ACTIVE] = GREEN
    s.colors[imgui.COLOR_TITLE_BACKGROUND] = DARKGREEN
    s.colors[imgui.COLOR_TITLE_BACKGROUND_ACTIVE] = HOVERGREEN
    s.colors[imgui.COLOR_TITLE_BACKGROUND_COLLAPSED] = DARKGREEN
    s.colors[imgui.COLOR_MENUBAR_BACKGROUND] = DARKGRAY
    s.colors[imgui.COLOR_SCROLLBAR_BACKGROUND] = BLACK
    # s.colors[imgui.COLOR_SCROLLBAR_GRAB] = [0., 1., 0., 1.00]
    # s.colors[imgui.COLOR_SCROLLBAR_GRAB_HOVERED] = [0., 1., 0., 1.00]
    # s.colors[imgui.COLOR_SCROLLBAR_GRAB_ACTIVE] = [0., 1., 0., 1.00]
    s.colors[imgui.COLOR_CHECK_MARK] = LIGHTGREEN
    s.colors[imgui.COLOR_SLIDER_GRAB] = GREEN
    s.colors[imgui.COLOR_SLIDER_GRAB_ACTIVE] = LIGHTGREEN
    s.colors[imgui.COLOR_BUTTON] = DARKGREEN
    s.colors[imgui.COLOR_BUTTON_HOVERED] = HOVERGREEN
    s.colors[imgui.COLOR_BUTTON_ACTIVE] = GREEN
    s.colors[imgui.COLOR_HEADER] = DARKGREEN
    s.colors[imgui.COLOR_HEADER_HOVERED] = HOVERGREEN
    s.colors[imgui.COLOR_HEADER_ACTIVE] = GREEN
    # s.colors[imgui.COLOR_SEPARATOR] = [0., 1., 0., 1.00]
    # s.colors[imgui.COLOR_SEPARATOR_HOVERED] = red_hovered
    # s.colors[imgui.COLOR_SEPARATOR_ACTIVE] = red_active
    # s.colors[imgui.COLOR_RESIZE_GRIP] = red
    # s.colors[imgui.COLOR_RESIZE_GRIP_HOVERED] = red_hovered
    # s.colors[imgui.COLOR_RESIZE_GRIP_ACTIVE] = red_active
    # s.colors[imgui.COLOR_PLOT_LINES] = white
    # s.colors[imgui.COLOR_PLOT_LINES_HOVERED] = white
    # s.colors[imgui.COLOR_PLOT_HISTOGRAM] = [0., 1., 0., 1.00]
    # s.colors[imgui.COLOR_PLOT_HISTOGRAM_HOVERED] = [0., 1., 0., 1.00]
    # s.colors[imgui.COLOR_TEXT_SELECTED_BACKGROUND] = red
    # s.colors[imgui.COLOR_DRAG_DROP_TARGET] = [0.8, 0.8, 0.8, 1.00]
    # s.colors[imgui.COLOR_NAV_HIGHLIGHT] = red
    # s.colors[imgui.COLOR_NAV_WINDOWING_HIGHLIGHT] = red
    # s.colors[imgui.COLOR_NAV_WINDOWING_DIM_BACKGROUND] = red
    # s.colors[imgui.COLOR_MODAL_WINDOW_DIM_BACKGROUND] = red



    c0 = s.colors[imgui.COLOR_MENUBAR_BACKGROUND]
    c1 = s.colors[imgui.COLOR_FRAME_BACKGROUND]
    # s.colors[imgui.COLOR_POPUP_BACKGROUND] = [x * 0.7 + y * 0.3 for x, y in zip(c0, c1)][:3] + [1]


# ----------------------------------------------------------------------------

@contextlib.contextmanager
def grayed_out(cond=True):
    if cond:
        s = imgui.get_style()
        text = s.colors[imgui.COLOR_TEXT_DISABLED]
        grab = s.colors[imgui.COLOR_SCROLLBAR_GRAB]
        back = s.colors[imgui.COLOR_MENUBAR_BACKGROUND]
        imgui.push_style_color(imgui.COLOR_TEXT, *text)
        imgui.push_style_color(imgui.COLOR_CHECK_MARK, *grab)
        imgui.push_style_color(imgui.COLOR_SLIDER_GRAB, *grab)
        imgui.push_style_color(imgui.COLOR_SLIDER_GRAB_ACTIVE, *grab)
        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, *GRAY)
        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_HOVERED, *GRAY)
        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_ACTIVE, *GRAY)
        imgui.push_style_color(imgui.COLOR_BUTTON, *GRAY)
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, *GRAY)
        imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, *GRAY)
        imgui.push_style_color(imgui.COLOR_HEADER, *back)
        imgui.push_style_color(imgui.COLOR_HEADER_HOVERED, *back)
        imgui.push_style_color(imgui.COLOR_HEADER_ACTIVE, *back)
        imgui.push_style_color(imgui.COLOR_POPUP_BACKGROUND, *back)
        imgui.push_style_color(imgui.COLOR_BORDER, *LIGHTGRAY)
        yield
        imgui.pop_style_color(15)
    else:
        yield


# ----------------------------------------------------------------------------

@contextlib.contextmanager
def item_width(width=None):
    if width is not None:
        imgui.push_item_width(width)
        yield
        imgui.pop_item_width()
    else:
        yield


# ----------------------------------------------------------------------------

def scoped_by_object_id(method):
    def decorator(self, *args, **kwargs):
        imgui.push_id(str(id(self)))
        res = method(self, *args, **kwargs)
        imgui.pop_id()
        return res

    return decorator


# ----------------------------------------------------------------------------

def color_button(label, color, width=0, height=0):
    s = imgui.get_style()
    imgui.push_style_color(imgui.COLOR_BUTTON, *color)
    imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, *color)
    imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, *color)
    clicked = imgui.button(label, width=width, height=height)
    imgui.pop_style_color(3)
    return clicked

def button(label, width=0, enabled=True):
    with grayed_out(not enabled):
        clicked = imgui.button(label, width=width)
    clicked = clicked and enabled
    return clicked




# ----------------------------------------------------------------------------

def collapsing_header(text, visible=None, flags=0, default=False, enabled=True, show=True):
    expanded = False
    if show:
        if default:
            flags |= imgui.TREE_NODE_DEFAULT_OPEN
        if not enabled:
            flags |= imgui.TREE_NODE_LEAF
        with grayed_out(not enabled):
            expanded, visible = imgui.collapsing_header(text, visible=visible, flags=flags)
        expanded = expanded and enabled
    return expanded, visible


# ----------------------------------------------------------------------------

def popup_button(label, width=0, enabled=True):
    if button(label, width, enabled):
        imgui.open_popup(label)
    opened = imgui.begin_popup(label)
    return opened


# ----------------------------------------------------------------------------

def input_text(label, value, buffer_length, flags, width=None, help_text=''):
    old_value = value
    color = list(imgui.get_style().colors[imgui.COLOR_TEXT])
    if value == '':
        color[-1] *= 0.5
    with item_width(width):
        imgui.push_style_color(imgui.COLOR_TEXT, *color)
        value = value if value != '' else help_text
        changed, value = imgui.input_text(label, value, buffer_length, flags)
        value = value if value != help_text else ''
        imgui.pop_style_color(1)
    if not flags & imgui.INPUT_TEXT_ENTER_RETURNS_TRUE:
        changed = (value != old_value)
    return changed, value


# ----------------------------------------------------------------------------

def drag_previous_control(enabled=True):
    dragging = False
    dx = 0
    dy = 0
    try:
        if imgui.begin_drag_drop_source(imgui.DRAG_DROP_SOURCE_NO_PREVIEW_TOOLTIP):
            if enabled:
                dragging = True
                dx, dy = imgui.get_mouse_drag_delta()
                imgui.reset_mouse_drag_delta()
            imgui.end_drag_drop_source()
    except:
        pass
    return dragging, dx, dy


# ----------------------------------------------------------------------------

def drag_button(label, width=0, enabled=True):
    clicked = button(label, width=width, enabled=enabled)
    dragging, dx, dy = drag_previous_control(enabled=enabled)
    return clicked, dragging, dx, dy


# ----------------------------------------------------------------------------

def drag_hidden_window(label, x, y, width, height, enabled=True):
    imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, 0, 0, 0, 0)
    imgui.push_style_color(imgui.COLOR_BORDER, 0, 0, 0, 0)
    imgui.set_next_window_position(x, y)
    imgui.set_next_window_size(width, height)
    imgui.begin(label, closable=False,
                flags=(imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE))
    dragging, dx, dy = drag_previous_control(enabled=enabled)
    imgui.end()
    imgui.pop_style_color(2)
    return dragging, dx, dy


# ----------------------------------------------------------------------------

def drag_float_slider(label, value, min_value, max_value, format):
    changed, value = imgui.slider_float(label, value, min_value, max_value, format)

    # if using slider and moving mouse outside the slider area, the value will continue to increase/decrease

    if imgui.is_item_active() and imgui.is_mouse_dragging():
        changed = True
        # calculate the delta between the mouse position and the slider position if mouse on the right side of the slider calculate in regards to the max value
        if imgui.get_mouse_pos()[0] > imgui.get_item_rect_max()[0]:
            value += (imgui.get_mouse_drag_delta()[0] / (imgui.get_item_rect_size()[0]))
    return changed, value


def img_checkbox(img, value, width, label=""):
    clicked = False
    # draw a square of size width x width
    draw_list = imgui.get_window_draw_list()
    posx, posy = imgui.get_cursor_screen_pos()
    draw_list.add_rect_filled(posx, posy, posx + width, posy + width, imgui.get_color_u32_rgba(0,0,0,1), 2)
    #draw outline of square
    draw_list.add_rect(posx, posy, posx + width, posy + width, imgui.get_color_u32_rgba(*LIGHTGRAY), rounding=0)
    # if clicked in area of square, change value
    if imgui.is_mouse_clicked(0) and imgui.is_mouse_hovering_rect(posx, posy, posx + width, posy + width):
        clicked = True
        value = not value

    # draw img in center of box if value is true
    if value:
        draw_list.add_image(img, (posx + (width * 0.2), posy + (width * 0.2)), (posx + (width * 0.8), posy + (width * 0.8)), col=imgui.get_color_u32_rgba(1,1,1,1))

    dim_color = list(imgui.get_style().colors[imgui.COLOR_TEXT])
    draw_list.add_text(posx + width + 5, posy, imgui.get_color_u32_rgba(*dim_color), label)

    return clicked, value
