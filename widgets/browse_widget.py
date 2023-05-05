import os

import imgui
from utils.gui_utils import imgui_utils
from utils.gui_utils import imgui_window

class BrowseWidget():

    def __init__(self, parent, title, directory, extensions, width=0, enabled=True, multiple=True, traverse_folders=True):
        self.parent = parent
        self.title = title
        self.directory = directory
        self.extensions = extensions
        self.extension = 0
        self.files = []
        self.selected = []
        self.width = width
        self.last_selected_idx = -1
        self.shift_idx = -1
        self.enabled = enabled
        self.multiple = multiple
        self.traverse_folders = traverse_folders
        self.open = False

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

        print("Resolbed Selection", selected)

        return selected


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
        for i, f in enumerate(self.files):
            # if the file is a directory draw it as a selectable or if it is a file and has the correct extension draw it as a selectable
            if os.path.isdir(os.path.join(self.directory, f)):
                imgui.text(u"\U0001F4C1")   # folder emoji
                imgui.same_line()
                # single click selects double clicks enters directory_popup
                if imgui.selectable(f, f in self.selected, imgui.SELECTABLE_ALLOW_DOUBLE_CLICK)[0]:
                    if self.multiple:
                        if f in self.selected:
                            self.selected.remove(f)
                        else:
                            self.selected.append(f)
                        if imgui.get_io().key_shift:
                            self.shift_idx = i
                            print("shift idx", self.last_selected_idx, self.shift_idx)
                            if self.last_selected_idx != -1 and self.shift_idx != -1:
                                if self.last_selected_idx < self.shift_idx:
                                    for j in range(self.last_selected_idx, self.shift_idx+1):
                                        if j == self.last_selected_idx or j == self.shift_idx:
                                            pass
                                        else:
                                            if self.files[j] not in self.selected:
                                                self.selected.append(self.files[j])
                                            else:
                                                self.selected.remove(self.files[j])
                                else:
                                    for j in range(self.shift_idx, self.last_selected_idx+1):
                                        if j == self.last_selected_idx or j == self.shift_idx:
                                            pass
                                        else:
                                            if self.files[j] not in self.selected:
                                                self.selected.append(self.files[j])
                                            else:
                                                self.selected.remove(self.files[j])
                            self.last_selected_idx = -1
                            self.shift_idx = -1
                        else:
                            self.last_selected_idx = i
                            print(self.last_selected_idx, self.shift_idx)
                    else:
                        self.selected = [f]
                    if imgui.is_mouse_double_clicked(0):
                        self.directory = os.path.join(self.directory, f)
                        if self.multiple:
                            if f in self.selected:
                                self.selected.remove(f)
                            else:
                                self.selected.append(f)
                        else:
                            self.selected = []


            elif os.path.isfile(os.path.join(self.directory, f)) and (self.extensions is None or not(len(self.extensions)) or f.endswith(self.extensions[self.extension]) or self.extensions[self.extension] == "*"):
                if not self.extensions[self.extension] == "":
                    # select file on single click
                    imgui.bullet()
                    imgui.same_line()
                    if imgui.selectable(f, f in self.selected, imgui.SELECTABLE_ALLOW_DOUBLE_CLICK)[0]:
                        if self.multiple:
                            if f in self.selected:
                                self.selected.remove(f)
                            else:
                                self.selected.append(f)
                            if imgui.get_io().key_shift:
                                self.shift_idx = i
                                print("shift idx", self.last_selected_idx, self.shift_idx)
                                if self.last_selected_idx != -1 and self.shift_idx != -1:
                                    if self.last_selected_idx < self.shift_idx:
                                        for j in range(self.last_selected_idx, self.shift_idx+1):
                                            if j == self.last_selected_idx or j == self.shift_idx:
                                                pass
                                            else:
                                                if self.files[j] not in self.selected:
                                                    self.selected.append(self.files[j])
                                                else:
                                                    self.selected.remove(self.files[j])
                                    else:
                                        for j in range(self.shift_idx, self.last_selected_idx+1):
                                            if j == self.last_selected_idx or j == self.shift_idx:
                                                pass
                                            else:
                                                if self.files[j] not in self.selected:
                                                    self.selected.append(self.files[j])
                                                else:
                                                    self.selected.remove(self.files[j])
                                self.last_selected_idx = -1
                                self.shift_idx = -1
                            else:
                                self.last_selected_idx = i
                                print(self.last_selected_idx, self.shift_idx)
                        else:
                            self.selected = [f]

        imgui.end_child()

        # Draw the bottom bar with the select and cancel buttons.
        imgui.set_cursor_pos((imgui.get_cursor_pos()[0], imgui.get_window_height() - self.parent.app.font_size * 2.5))
        imgui.begin_child("bottom_bar", 0, border=True, flags=imgui.WINDOW_NO_SCROLLBAR)
        with imgui_utils.item_width(imgui.get_content_region_available_width() - self.parent.app.button_w * 3.25):
            imgui.input_text("##selected_items",f"Selected: {', '.join(self.selected)}",1024, flags=imgui.INPUT_TEXT_READ_ONLY)
        imgui.same_line()
        # add dropdown to select extensions
        if self.extensions is not None:
            with imgui_utils.item_width(self.parent.app.button_w):
                _changed, self.extension = imgui.combo("##extensions", self.extension, self.extensions, len(self.extensions))
        imgui.same_line()
        if imgui.button("Select", self.parent.app.button_w):
            self.open = False
            imgui.end_child()
            imgui.end()
            if self.traverse_folders:
                return True, self.resolve_selected()
            else:
                return True, [os.path.join(self.directory, f) for f in self.selected]
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

