import os

import imgui
from utils.gui_utils import imgui_utils
from utils.gui_utils import imgui_window

class BrowseWidget():

    def __init__(self, parent, title, directory, extensions, width=0, enabled=True):
        self.parent = parent
        self.title = title
        self.directory = directory
        self.extensions = extensions
        self.files = []
        self.selected = None
        self.width = width
        self.enabled = enabled
        self.open = False


    def open_window(self):
        # Opens new window with file browser shows only files with given extensions in current directory and displays other directories by double clicking on a directory we enter said directory, single click selects files and if a directory is selected we recursively select all files in it

        imgui.set_next_window_size((self.parent.app.content_width *3)//4, self.parent.app.content_height//2)
        window_out = imgui.begin(self.title, True)
        self.open = window_out[0]

        # draw the top bar with the current directory and a button that goes up one directory
        imgui.begin_child("top_bar", 0, self.parent.app.font_size * 2, border=True, flags=imgui.WINDOW_NO_SCROLLBAR)

        if imgui.button("^", self.parent.app.font_size * 1.5):
            self.directory = os.path.dirname(self.directory)
        
        imgui.same_line()
        _changed, self.directory = imgui.input_text("##directory", self.directory, 1000, imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
        if _changed:
            directory = os.path.normpath(directory)
            if os.path.isdir(directory):
                self.directory = directory
            else:
                print("Invalid directory")
        imgui.end_child()

        # draw the file list
        imgui.begin_child("file_list", 0, -self.parent.app.font_size * 2, border=True, flags=0)
        # get all files and folders in the current directory
        self.files = os.listdir(self.directory)
        # draw the files
        for f in self.files:
            # if the file is a directory draw it as a selectable or if it is a file and has the correct extension draw it as a selectable
            if os.path.isdir(os.path.join(self.directory, f)):
                # single click selects double clicks enters directory_popup
                if imgui.selectable(f, self.selected == f, imgui.SELECTABLE_ALLOW_DOUBLE_CLICK)[0]:
                    self.selected = f
                    if imgui.is_mouse_double_clicked(0):
                        print("SELECTED", f)
                        self.directory = os.path.join(self.directory, f)

            elif os.path.isfile(os.path.join(self.directory, f)):
                # select file on single click
                imgui.bullet()
                imgui.same_line()
                if imgui.selectable(f, self.selected == f)[0]:
                    print("SELECTED", f)
                    self.selected = f

        imgui.end_child()
        imgui.end()
        return False, None

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        if show:
            if imgui_utils.button(self.title, enabled=self.enabled, width=self.width):
                self.open = True

            if self.open:
                return self.open_window()

        return False, None

