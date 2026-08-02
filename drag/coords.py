"""Mapping between screen coordinates and generator pixel coordinates.

image_area is the rectangle the rendered image occupies on screen:
[x0, y0, disp_w, disp_h]. image_size is (iw, ih) in generator pixels.
Points use (y, x) order to match the drag engine; screen positions (x, y).
"""


def screen_to_image(sx, sy, image_area, image_size):
    x0, y0, disp_w, disp_h = image_area
    iw, ih = image_size
    px = (sx - x0) / disp_w * iw - 0.5
    py = (sy - y0) / disp_h * ih - 0.5
    inside = (0 <= sx - x0 < disp_w) and (0 <= sy - y0 < disp_h)
    return py, px, inside


def image_to_screen(py, px, image_area, image_size):
    x0, y0, disp_w, disp_h = image_area
    iw, ih = image_size
    sx = x0 + (px + 0.5) / iw * disp_w
    sy = y0 + (py + 0.5) / ih * disp_h
    return sx, sy
