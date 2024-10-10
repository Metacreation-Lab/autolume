import os

import cv2
import imgui
import torch

from assets import LIGHTGRAY
from utils.gui_utils import imgui_utils, gl_utils
from utils.gui_utils import imgui_window



class SaveWidget():

    def __init__(self, parent, title, directory, extension, width=0, enabled=True,add_folder_button=True):
        self.add_folder_name = ""
        self.selected = ""
        self.file_name = ""
        self.parent = parent
        self.title = title
        self.directory = directory
        self.extension = extension
        self.width = width
        self.enabled = enabled
        self.add_folder_button = add_folder_button
        self.open = False

        # read as rgba
        self.folder = cv2.imread("assets/folder.png", cv2.IMREAD_UNCHANGED)
        self.folder = cv2.cvtColor(self.folder, cv2.COLOR_BGRA2RGBA)

        # in the alpha channel we put alpha to 0 where the image is black
        self.folder_texture = gl_utils.Texture(image=self.folder, width=self.folder.shape[1], height=self.folder.shape[0], channels=self.folder.shape[2])

    def resolve_selected(self):
        # returns a list of all selected files with their full path and if the file is a directory it recursively selects all files in it
        selected = []
        for f in self.selected:
            f = os.path.join(self.directory, f)
            if os.path.isdir(f):
                for root, dirs, files in os.walk(f):
                    for file in files:
                        selected.append(os.path.join(root, file))
            else:
                selected.append(f)

        print("Resolved Selection", selected)

        return selected


    def open_window(self):
        # Opens new window with file browser shows only files with given extensions in current directory and displays other directories by double clicking on a directory we enter said directory, single click selects files and if a directory is selected we recursively select all files in it

        imgui.set_next_window_size((self.parent.app.content_width *3)//4, self.parent.app.content_height//2)
        window_out = imgui.begin(self.title, True, flags=imgui.WINDOW_NO_COLLAPSE)
        self.open = window_out[1]

        # draw the top bar with the current directory and a button that goes up one directory
        imgui.begin_child("top_bar", 0, self.parent.app.font_size * 2, border=True, flags=imgui.WINDOW_NO_SCROLLBAR)

        if imgui.arrow_button("##up", imgui.DIRECTION_UP):
            self.directory = os.path.dirname(self.directory)

        imgui.same_line()
        _changed,directory = imgui.input_text("##directory", self.directory, 1000, imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
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
        # # draw the files
        for i, f in enumerate(self.files):
            # if the file is a directory draw it as a selectable or if it is a file and has the correct extension draw it as a selectable
            if os.path.isdir(os.path.join(self.directory, f)):
                imgui.image(self.folder_texture.gl_id, self.parent.app.font_size, self.parent.app.font_size, tint_color=(1, 1, 1, 1))
                imgui.same_line()
                # single click selects double clicks enters directory_popup
                if imgui.selectable(f, os.path.join(self.directory, f) in self.selected, imgui.SELECTABLE_ALLOW_DOUBLE_CLICK)[0]:
                    self.selected = os.path.join(self.directory, f)
                    if imgui.is_mouse_double_clicked(0):
                        self.directory = os.path.join(self.directory, f)
                        self.selected = ""


            elif os.path.isfile(os.path.join(self.directory, f)):
                # select file on single click
                imgui.bullet()
                imgui.same_line()
                imgui.text(f)



        #if self.add_folder_button add a + button that opens a popup to add a folder
        if self.add_folder_button:
            if imgui.button("+", self.parent.app.button_w):
                self.add_folder_name = ""
                imgui.open_popup("add_folder_popup")

            if imgui.begin_popup("add_folder_popup"):
                imgui.text("Add Folder")
                # add input text to add folder
                _changed, self.add_folder_name = imgui_utils.input_text("##add_folder_input", self.add_folder_name, 1024, help_text="Enter the name of the folder to add", flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE|imgui.INPUT_TEXT_AUTO_SELECT_ALL)

                # add button to confirm action and add folder to directory and a button to cancel
                if imgui_utils.button("Add", self.parent.app.font_size*2, enabled=self.add_folder_name != "") or _changed:
                    # add folder to directory
                    if not os.path.exists(os.path.join(self.directory, self.add_folder_name)):
                        os.mkdir(os.path.join(self.directory, self.add_folder_name))
                    # select created folder
                    else:
                        self.selected = os.path.join(self.directory, self.add_folder_name)
                    # close popup
                    imgui.close_current_popup()

                imgui.same_line()
                if imgui.button("Cancel", self.parent.app.button_w):
                    # close popup
                    imgui.close_current_popup()
                imgui.end_popup()

        imgui.end_child()

        # Draw the bottom bar with the select and cancel buttons.
        imgui.set_cursor_pos((imgui.get_cursor_pos()[0], imgui.get_window_height() - self.parent.app.font_size * 2.5))
        imgui.begin_child("bottom_bar", 0, border=True, flags=imgui.WINDOW_NO_SCROLLBAR)
        with imgui_utils.item_width(imgui.get_content_region_available_width() - self.parent.app.button_w * 3.25 -self.parent.app.button_w * 1.25 ):
            imgui.input_text("##selected_items",f"Selected: {self.selected}",1024, flags=imgui.INPUT_TEXT_READ_ONLY)
        imgui.same_line()

        with imgui_utils.item_width(self.parent.app.button_w * 1.25):
            _changed, self.file_name = imgui.input_text("##file_name", self.file_name, 1024, flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE|imgui.INPUT_TEXT_AUTO_SELECT_ALL)
        imgui.same_line()

        # add dropdown to select extensions
        imgui.text(self.extension)
        imgui.same_line()
        if imgui_utils.button("Save", self.parent.app.button_w, enabled=self.file_name != "") or (_changed and self.file_name != ""):
            self.open = False
            imgui.end_child()
            imgui.end()
            if self.selected:
                return True, os.path.join(self.directory, self.selected, self.file_name) + self.extension
            else:
                return True, os.path.join(self.directory, self.file_name) + self.extension
        imgui.same_line()
        if imgui.button("Cancel", self.parent.app.button_w):
            self.selected = []
            self.open = False

            imgui.end_child()
            imgui.end()
            return False, None

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

