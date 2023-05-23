import os

import imgui

from utils.gui_utils import imgui_window
from utils.gui_utils import imgui_utils

#----------------------------------------------------------------------------
"""
A python implementation of the file dialog module using imgui, which allows users to select a single file or multiple files, or a single directory or multiple directories.
It recursively searches the directory for files and directories, and displays them in a tree structure.
The user can select a file or directory by clicking on it, and the path to the file or directory is returned after clicking the select button.
"""

# TODO - Add support for multiple extensions.
# TODO - Add support for selecting directories.
# TODO - Actually append to selected list.
# TODO - Add support for selecting multiple files.
# TODO - Make visually nice
# TODO - make back and forward buttons work
# instead of tree maybe just display selectable list of files and directories

class FileDialog():
    def __init__(self, parent, title, directory, extensions, show_hidden=False, width=0, enabled=True, multiple_files=True):
        self.parent = parent
        self.title = title
        self.directory = directory
        self.extensions = extensions
        self.extension = 0
        self.selected = []
        self.show_hidden = show_hidden
        self.history = []
        self.multiple_files = multiple_files
        self.current_history_idx = 0
        self.start_idx = -1
        self.end_idx = -1
        self.width = width
        self.enabled = enabled

        # Initialize window.
        self.open = False # Layout may change after first frame.

    @imgui_utils.scoped_by_object_id
    def __call__(self):
        if imgui_utils.button(self.title, width=self.width, enabled=self.enabled):
            self.open = True
        if self.open:
            return self._draw()

        return False, None

    def _draw_file_tree(self, directory):
        # Draw the directory.
        imgui.push_id(directory)
        if imgui.tree_node(directory):
            # Draw the files in the directory.
            for i, file in enumerate(os.listdir(directory)):
                # skip hidden files
                if not self.show_hidden and file.startswith("."):
                    continue

                #skip files that don't match the extension using regex in the form of *.extension where * means all files, if the file is a directory then check recursively
                if not file.endswith(self.extensions[self.extension]) and not self.extensions[self.extension] == "*" and not os.path.isdir(os.path.join(directory, file)):
                    continue

                file_path = os.path.join(directory, file)
                if os.path.isdir(file_path):
                    self._draw_file_tree(file_path)
                else:
                    if imgui.selectable(file, file_path in self.selected)[0]:
                        print("SELECTED INDEX", file_path, i)
                        # if shift is pressed down then select all files between start_idx and end_idx and reset start_idx and end_idx else only update start_idx and set end_idx -1
                        if imgui.get_io().key_shift and self.multiple_files:
                            if self.start_idx == -1:
                                self.start_idx = i
                                self.end_idx = -1
                            else:
                                print("SHIFT PRESSED", self.start_idx, i)
                                self.end_idx = i
                                if self.start_idx > self.end_idx:
                                    self.start_idx, self.end_idx = self.end_idx, self.start_idx
                                for j, tmp_file in enumerate(os.listdir(directory), start=self.start_idx):
                                    print("J", j, tmp_file)
                                    if not self.show_hidden and file.startswith("."):
                                        continue
                                    tmp_file_path = os.path.join(directory, tmp_file)
                                    if not os.path.isdir(tmp_file_path):
                                        if j <= self.end_idx:
                                            if tmp_file_path not in self.selected:
                                                print("ADDING", j, tmp_file_path, self.start_idx, self.end_idx)
                                                self.selected.append(tmp_file_path)
                                            else:
                                                print("REMOVING", j, tmp_file_path, self.start_idx, self.end_idx)
                                                self.selected.remove(tmp_file_path)
                                    if j == self.end_idx:
                                        break
                                self.start_idx = -1
                                self.end_idx = -1
                        else:
                            self.start_idx = i
                        print("SELECTED", file_path)
                        if self.multiple_files:
                            if file_path in self.selected:
                                self.selected.remove(file_path)
                                print("removed")
                            else:
                                self.selected.append(file_path)
                                print("added")
                        else:
                            self.selected = [file_path]
                            print("set")
            imgui.tree_pop()
        imgui.pop_id()

    def _draw(self):

        # imgui.set_next_window_position(self.parent.app.content_width // 8, self.parent.app.content_height // 4)
        imgui.set_next_window_size((self.parent.app.content_width * 3) // 4, self.parent.app.content_height // 2)
        window_out = imgui.begin(f"{self.title}##file_dialog", True)
        self.open = window_out[1]

        # Draw the top bar with the directory path and a button that goes up one directory.
        imgui.begin_child("top_bar", 0, self.parent.app.font_size * 2, border=True, flags=imgui.WINDOW_NO_SCROLLBAR)

        #if left arrow button pressed then go back in history
        if imgui_utils.button("<", self.parent.app.font_size * 1.5, enabled=self.current_history_idx > 0):
            if len(self.history) > 0:
                self.directory = self.history[self.current_history_idx]
                self.current_history_idx -= 1
                if self.current_history_idx < 0:
                    self.current_history_idx = 0
        imgui.same_line()
        # if right arrow button pressed then go forward in history
        if imgui_utils.button(">", self.parent.app.font_size * 1.5, enabled=self.current_history_idx < len(self.history) - 1):
            if len(self.history) > 0:
                self.directory = self.history[self.current_history_idx]
                self.current_history_idx += 1
                if self.current_history_idx >= len(self.history):
                    self.current_history_idx = len(self.history) - 1

        imgui.same_line()
        if imgui.button("^", self.parent.app.font_size * 1.5):
            self.directory = os.path.dirname(self.directory)
            self.history.append(self.directory)
            self.current_history_idx = len(self.history) - 1


        imgui.same_line()
        _changed, directory = imgui.input_text("##directory", self.directory, 1024, imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
        # if _changed then check if the directory exists and if so set it
        if _changed:
            if os.path.isdir(directory):
                self.directory = directory
            else:
                print(f"Directory {directory} does not exist")
        imgui.same_line()

        # if button pressed and subdirectories exist in the current directory then open a popup
        has_subdirectories = False
        for directory in os.listdir(self.directory):
            if os.path.isdir(os.path.join(self.directory, directory)):
                has_subdirectories = True
                break
        if imgui_utils.button("...", self.parent.app.font_size * 1.5, enabled=has_subdirectories):
            imgui.open_popup("##directory_popup")
        if imgui.begin_popup("##directory_popup"):
            for directory in os.listdir(self.directory):
                if os.path.isdir(os.path.join(self.directory, directory)):
                    print("directory", directory)
                    if imgui.selectable(directory)[1]:
                        self.directory = os.path.join(self.directory, directory)
            imgui.end_popup()
        imgui.end_child()


        # Draw the file tree.
        imgui.begin_child("file_tree", 0, - self.parent.app.font_size * 2.5, border=True)
        self._draw_file_tree(self.directory)
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
            return True, self.selected
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

