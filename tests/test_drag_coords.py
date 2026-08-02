from drag.coords import image_to_screen, screen_to_image


AREA = [100.0, 50.0, 512.0, 512.0]   # image displayed at 2x zoom
SIZE = (256, 256)


def test_round_trip():
    sx, sy = image_to_screen(10, 20, AREA, SIZE)
    py, px, inside = screen_to_image(sx, sy, AREA, SIZE)
    assert inside
    assert abs(py - 10) < 1e-6 and abs(px - 20) < 1e-6


def test_zoom_scaling():
    sx, sy = image_to_screen(0, 0, AREA, SIZE)
    assert abs(sx - 101.0) < 1e-6   # x0 + half a 2x pixel
    assert abs(sy - 51.0) < 1e-6


def test_outside_flag():
    _, _, inside = screen_to_image(99.0, 50.0, AREA, SIZE)
    assert not inside
    _, _, inside = screen_to_image(100.0 + 512.0, 50.0, AREA, SIZE)
    assert not inside
    _, _, inside = screen_to_image(300.0, 300.0, AREA, SIZE)
    assert inside
