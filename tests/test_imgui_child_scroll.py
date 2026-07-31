import imgui
import pytest

from utils.gui_utils import imgui_utils


class FakeImgui:
    """Records begin_child flags and replays a scripted ScrollMax per label."""

    def __init__(self, scroll_max):
        self.scroll_max = scroll_max
        self.calls = []
        self._stack = []

    def begin_child(self, label, width, height, border, flags):
        self.calls.append((label, flags))
        self._stack.append(label)

    def get_scroll_max_y(self):
        return self.scroll_max.get(self._stack[-1], 0.0)

    def end_child(self):
        self._stack.pop()


@pytest.fixture
def fake(monkeypatch):
    def install(scroll_max=None):
        stub = FakeImgui(scroll_max or {})
        for name in ('begin_child', 'get_scroll_max_y', 'end_child'):
            monkeypatch.setattr(imgui_utils.imgui, name, getattr(stub, name))
        return stub
    imgui_utils._open_children.clear()
    return install


class Owner:
    pass


def frame(owner, label, flags=0):
    imgui_utils.begin_child(owner, label, flags=flags)
    imgui_utils.end_child()


def test_unscrollable_child_lets_the_wheel_through(fake):
    stub = fake()
    frame(Owner(), '##list')
    _, flags = stub.calls[0]
    assert flags & imgui.WINDOW_NO_SCROLL_WITH_MOUSE


def test_no_scrollbar_is_stripped_so_imgui_165_forwards_to_the_parent(fake):
    # imgui 1.65 refuses to climb to the parent when the child sets NoScrollbar,
    # so the flag has to go for the forwarding to actually happen.
    stub = fake()
    frame(Owner(), '##list2', flags=imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_SCROLL_WITH_MOUSE)
    _, flags = stub.calls[0]
    assert not flags & imgui.WINDOW_NO_SCROLLBAR


def test_scrollable_child_keeps_the_wheel_on_the_next_frame(fake):
    stub = fake({'##list': 120.0})
    owner = Owner()
    frame(owner, '##list')          # first frame: nothing known yet
    frame(owner, '##list')          # second frame: last frame said it scrolls
    assert stub.calls[0][1] & imgui.WINDOW_NO_SCROLL_WITH_MOUSE
    assert not stub.calls[1][1] & imgui.WINDOW_NO_SCROLL_WITH_MOUSE


def test_author_flags_are_restored_once_the_child_can_scroll(fake):
    stub = fake({'##list2': 40.0})
    owner = Owner()
    author = imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_SCROLL_WITH_MOUSE
    frame(owner, '##list2', flags=author)
    frame(owner, '##list2', flags=author)
    assert stub.calls[1][1] == author


def test_child_becoming_empty_reopens_the_wheel(fake):
    stub = fake({'##list': 120.0})
    owner = Owner()
    frame(owner, '##list')
    frame(owner, '##list')
    stub.scroll_max['##list'] = 0.0  # layers unloaded
    frame(owner, '##list')
    frame(owner, '##list')
    assert stub.calls[3][1] & imgui.WINDOW_NO_SCROLL_WITH_MOUSE


def test_same_label_in_two_widgets_does_not_share_state(fake):
    # layer_widget and collapsable_layer both name their children ##list.
    stub = fake({'##list': 0.0})
    scrolls, empty = Owner(), Owner()
    stub.scroll_max['##list'] = 120.0
    frame(scrolls, '##list')
    frame(scrolls, '##list')
    stub.scroll_max['##list'] = 0.0
    frame(empty, '##list')
    frame(empty, '##list')
    assert not stub.calls[1][1] & imgui.WINDOW_NO_SCROLL_WITH_MOUSE
    assert stub.calls[3][1] & imgui.WINDOW_NO_SCROLL_WITH_MOUSE


def test_nested_children_unwind_in_order(fake):
    stub = fake({'##outer': 200.0, '##inner': 0.0})
    owner = Owner()
    for _ in range(2):
        imgui_utils.begin_child(owner, '##outer')
        imgui_utils.begin_child(owner, '##inner')
        imgui_utils.end_child()
        imgui_utils.end_child()
    assert not imgui_utils._open_children
    assert not stub.calls[2][1] & imgui.WINDOW_NO_SCROLL_WITH_MOUSE   # outer scrolls
    assert stub.calls[3][1] & imgui.WINDOW_NO_SCROLL_WITH_MOUSE       # inner does not
