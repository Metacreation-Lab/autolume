import imgui
from utils.gui_utils import imgui_window
from enum import IntEnum
class States(IntEnum):
    ERROR = -2
    CLOSE = -1
    WELCOME = 0
    MENU = 1
    RENDER = 2

from modules.visualizer import Visualizer
from modules.menu import Menu
from modules.welcome import Welcome


class Autolume(imgui_window.ImguiWindow):
    # State handler that greets user, shows menu (allowing for training, compression, ganspace), and renders
    def __init__(self):
        super().__init__(title='Autolume-Live', window_width=3840, window_height=2160)
        self.state = 0
        self.running = True
        self.welcome = Welcome(self)
        self.menu = None
        self.viz = None
        self.pkls = []

        # Initialize window.
        self.label_w = 0
        self.button_w = 0
        self.set_position(0, 0)
        self._adjust_font_size()
        self.skip_frame()  # Layout may change after first frame.

    def _adjust_font_size(self):
        old = self.font_size
        self.set_font_size(min(self.content_width / 120, self.content_height / 60))
        if self.font_size != old:
            self.skip_frame() # Layout changed.

    def close(self):
        super().close()


    def open_menu(self):
        print("opening Menu")
        # Initialize window.
        self.menu = Menu(self)

    def start_renderer(self):
        self.viz = Visualizer(self)

        # List pickles.
        if len(self.pkls) > 0:
            for pkl in self.pkls:
                self.viz.add_recent_pickle(pkl)
            self.viz.load_pickle(self.pkls[0])
        self.state = States.RENDER

    def draw_frame(self):
        self.begin_frame()
        self.button_w = self.font_size * 5
        self.label_w = round(self.font_size * 4.5)

        # print("running", States(self.state).name)
        if States.WELCOME == self.state:
            self.welcome()

        if self.state == States.MENU:
            if self.menu is None:
                self.state = States.ERROR
            else:
                self.menu()

        if self.state == States.RENDER:
            if self.viz is None:
                self.state = States.ERROR
            else:
                self.viz()

        if self.state == States.CLOSE or self.state == States.ERROR:
            self.stop()

        self._adjust_font_size()
        self.end_frame()

