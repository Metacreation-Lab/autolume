"""Native OS file dialogs backed by filedialpy.

filedialpy opens the real platform dialog: NSOpenPanel via AppleScript on
macOS (out of process, so it cannot conflict with the GLFW run loop),
pywin32 on Windows, zenity/kdialog on Linux. tkinter is not used; Tk cannot
run inside this process on macOS because GLFW owns the NSApplication.
"""

import os
import sys
from typing import List, Optional, Tuple

try:
    import filedialpy
except ImportError:
    # On Linux without zenity/kdialog, filedialpy falls back to tkinter at
    # import time, which may be unavailable. Degrade to no-op dialogs.
    filedialpy = None


class NativeBrowserWidget:
    def __init__(self):
        # Image extensions
        self.image_extensions = [
            ('Image files', '*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp *.gif'),
            ('PNG files', '*.png'),
            ('JPEG files', '*.jpg *.jpeg'),
            ('BMP files', '*.bmp'),
            ('TIFF files', '*.tiff *.tif'),
            ('WebP files', '*.webp'),
            ('GIF files', '*.gif'),
            ('All files', '*.*')
        ]

        # Video file extensions
        self.video_extensions = [
            ('Video files', '*.mp4 *.avi *.mov *.mkv *.webm *.gif'),
            ('MP4 files', '*.mp4'),
            ('AVI files', '*.avi'),
            ('MOV files', '*.mov'),
            ('MKV files', '*.mkv'),
            ('WebM files', '*.webm'),
            ('GIF files', '*.gif'),
            ('All files', '*.*')
        ]

        # Optimized extension sets for fast lookup
        self._image_extensions_set = self._build_extension_set(self.image_extensions)
        self._video_extensions_set = self._build_extension_set(self.video_extensions)

    # --- Dialog primitives ---------------------------------------------------

    @staticmethod
    def _dialogs_available() -> bool:
        if filedialpy is None:
            print("Native file dialogs unavailable: filedialpy could not be imported "
                  "(on Linux, install zenity or kdialog)")
            return False
        return True

    @staticmethod
    def _filter_arg(patterns: str):
        # filedialpy's macOS backend only accepts a single space-separated
        # string (a list crashes it), while the Windows backend needs a list
        # to turn the patterns into valid ';'-separated Win32 filters.
        if sys.platform == 'darwin':
            return patterns
        return [patterns, '*']

    def _open_files(self, title: str, patterns: str) -> List[str]:
        """Open a native multi-file picker. Returns [] when cancelled."""
        if not self._dialogs_available():
            return []
        try:
            paths = filedialpy.openFiles(title=title, filter=self._filter_arg(patterns),
                                         initial_dir=os.getcwd())
        except Exception as e:
            print(f"Native file dialog failed: {e}")
            return []
        if isinstance(paths, str):
            paths = [paths] if paths else []
        # The macOS backend maps a cancelled dialog to the current directory;
        # keeping only real files also drops that artifact.
        return [p for p in paths if p and os.path.isfile(p)]

    def _open_directory(self, title: str) -> Optional[str]:
        """Open a native directory picker. Returns None when cancelled."""
        if not self._dialogs_available():
            return None
        try:
            path = filedialpy.openDir(title=title, initial_dir=os.getcwd())
        except Exception as e:
            print(f"Native directory dialog failed: {e}")
            return None
        if not path or not os.path.isdir(path):
            return None
        if sys.platform == 'darwin' and os.path.realpath(path) == os.path.realpath(''):
            # The macOS backend reports a cancelled dialog as the process
            # working directory; treat that as a cancel.
            return None
        return path

    # --- Helpers ------------------------------------------------------------

    def _build_extension_set(self, extensions_list: List[Tuple[str, str]]) -> set:
        """Build a set of extensions for O(1) lookup instead of nested loops."""
        ext_set = set()
        for _, patterns in extensions_list:
            for pattern in patterns.split():
                if pattern.startswith('*.'):
                    ext_set.add(pattern[1:].lower())  # Remove '*' and convert to lowercase
        return ext_set

    def _is_image_file(self, filename: str) -> bool:
        """Fast extension check using set lookup."""
        ext = os.path.splitext(filename)[1].lower()
        return ext in self._image_extensions_set

    def _is_video_file(self, filename: str) -> bool:
        """Fast extension check using set lookup."""
        ext = os.path.splitext(filename)[1].lower()
        return ext in self._video_extensions_set

    def _load_directory_files(self, directory: str) -> List[str]:
        """Return filenames (non-recursive) in directory; skips subdirectories."""
        try:
            return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        except (OSError, PermissionError) as e:
            print(f"Error reading directory {directory}: {e}")
            return []

    def _filter_files_by_type(self, files: List[str], file_type: str = "image") -> List[str]:
        """Filter files by type using optimized extension checking."""
        if file_type == "image":
            return [f for f in files if self._is_image_file(f)]
        elif file_type == "video":
            return [f for f in files if self._is_video_file(f)]
        else:
            return files

    # --- Public API ----------------------------------------------------------

    def select_images_from_folder(self) -> List[str]:
        """Open a native directory picker and return all image files (non-recursively)
        in the chosen folder as full, sorted paths. Returns [] when cancelled."""
        folder_path = self._open_directory("Select a folder containing images")
        if not folder_path:
            return []
        files = self._load_directory_files(folder_path)
        image_files = self._filter_files_by_type(files, "image")
        full_paths = [os.path.join(folder_path, f) for f in image_files]
        return sorted(full_paths)

    def select_video_files(self):
        """Select one or more video files using the native OS dialog."""
        return self._open_files("Select Video Files", self.video_extensions[0][1])

    def select_directory(self, title="Select Directory"):
        """
        Select a directory using native OS dialog.
        Returns the directory path or None if cancelled.
        """
        return self._open_directory(title)

    def select_directory_with_video_check(self, title="Select Directory"):
        """
        Select a directory and check if it contains video files.
        Returns (directory_path, has_video_files, video_files_list)
        """
        directory_path = self._open_directory(title)
        if not directory_path:
            return None, False, []

        # Check for video files in the directory
        video_files = []
        has_video_files = False

        try:
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    if self._is_video_file(file):
                        video_files.append(os.path.join(root, file))
                        has_video_files = True
        except (OSError, PermissionError) as e:
            print(f"Error scanning directory {directory_path}: {e}")
            return directory_path, False, []

        return directory_path, has_video_files, video_files

    def cleanup(self):
        """Kept for API compatibility; native dialogs hold no resources."""
        pass