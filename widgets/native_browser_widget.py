"""Native OS file dialogs backed by filedialpy.

filedialpy opens the real platform dialog: NSOpenPanel via AppleScript on
macOS (out of process, so it cannot conflict with the GLFW run loop),
pywin32 on Windows, zenity/kdialog on Linux. tkinter is not used; Tk cannot
run inside this process on macOS because GLFW owns the NSApplication.
"""

import logging
import os
import sys
from typing import List, Optional, Tuple

try:
    import filedialpy
except ImportError:
    # On Linux without zenity/kdialog, filedialpy falls back to tkinter at
    # import time, which may be unavailable. Degrade to no-op dialogs.
    filedialpy = None

logger = logging.getLogger(__name__)


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

        # Vector file extensions (latent vectors saved by the app)
        self.vector_extensions = [
            ('Vector files', '*.pt *.pth *.npy *.npz'),
            ('All files', '*.*')
        ]

        # Model file extensions (pickled StyleGAN checkpoints)
        self.model_extensions = [
            ('Model files', '*.pkl'),
            ('All files', '*.*')
        ]

        # LoRA weight files
        self.lora_extensions = [
            ('LoRA files', '*.safetensors'),
            ('All files', '*.*')
        ]

        # Optimized extension sets for fast lookup
        self._image_extensions_set = self._build_extension_set(self.image_extensions)
        self._video_extensions_set = self._build_extension_set(self.video_extensions)

    # --- Dialog primitives ---------------------------------------------------

    @staticmethod
    def _dialogs_available() -> bool:
        if filedialpy is None:
            logger.warning("Native file dialogs unavailable: filedialpy could not be imported "
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

    @staticmethod
    def _resolve_initial_dir(hint) -> str:
        """Resolve the folder a dialog should open in. `hint` may be a file path,
        a directory, an empty string, or a non-existent path. A file resolves to
        its parent folder; a missing path falls back to the user's home directory.
        `None` keeps the process working directory (the default when no path field
        is associated with the dialog)."""
        if hint is None:
            return os.getcwd()
        home = os.path.expanduser("~")
        if not hint:
            return home
        hint = os.path.expanduser(str(hint))
        if os.path.isdir(hint):
            return hint
        parent = os.path.dirname(hint)
        if parent and os.path.isdir(parent):
            return parent
        return home

    def _open_files(self, title: str, patterns: str, initial_dir=None) -> List[str]:
        """Open a native multi-file picker. Returns [] when cancelled."""
        if not self._dialogs_available():
            return []
        try:
            paths = filedialpy.openFiles(title=title, filter=self._filter_arg(patterns),
                                         initial_dir=self._resolve_initial_dir(initial_dir))
        except Exception as e:
            logger.error("Native file dialog failed: %s", e)
            return []
        if isinstance(paths, str):
            paths = [paths] if paths else []
        # The macOS backend maps a cancelled dialog to the current directory;
        # keeping only real files also drops that artifact.
        return [p for p in paths if p and os.path.isfile(p)]

    def _open_directory(self, title: str, initial_dir=None) -> Optional[str]:
        """Open a native directory picker. Returns None when cancelled."""
        if not self._dialogs_available():
            return None
        try:
            path = filedialpy.openDir(title=title, initial_dir=self._resolve_initial_dir(initial_dir))
        except Exception as e:
            logger.error("Native directory dialog failed: %s", e)
            return None
        if not path or not os.path.isdir(path):
            return None
        return path

    def _open_single_file(self, title: str, patterns: str, initial_dir=None) -> Optional[str]:
        """Open a native single-file picker. Returns None when cancelled."""
        if not self._dialogs_available():
            return None
        try:
            path = filedialpy.openFile(title=title, filter=self._filter_arg(patterns),
                                       initial_dir=self._resolve_initial_dir(initial_dir))
        except Exception as e:
            logger.error("Native file dialog failed: %s", e)
            return None
        # The macOS backend maps a cancelled dialog to the current directory.
        if not path or not os.path.isfile(path):
            return None
        return path

    def _save_file(self, title: str, patterns: str, initial_file: str, initial_dir=None) -> Optional[str]:
        """Open a native 'Save As' dialog. Returns the chosen path or None when
        cancelled. filedialpy performs its own overwrite confirmation."""
        if not self._dialogs_available():
            return None
        try:
            path = filedialpy.saveFile(title=title, filter=self._filter_arg(patterns),
                                       initial_dir=self._resolve_initial_dir(initial_dir), initial_file=initial_file)
        except Exception as e:
            logger.error("Native save dialog failed: %s", e)
            return None
        if not path:
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
            logger.warning("Error reading directory %s: %s", directory, e)
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

    def select_media_files(self, title="Select Input Files", initial_dir=None):
        """Select one or more image or video files. Returns [] when cancelled."""
        patterns = self.image_extensions[0][1] + ' ' + self.video_extensions[0][1]
        return self._open_files(title, patterns, initial_dir=initial_dir)

    def select_directory(self, title="Select Directory", initial_dir=None):
        """
        Select a directory using native OS dialog.
        Returns the directory path or None if cancelled.
        """
        return self._open_directory(title, initial_dir=initial_dir)

    def select_directory_with_video_check(self, title="Select Directory", initial_dir=None):
        """
        Select a directory and check if it contains video files.
        Returns (directory_path, has_video_files, video_files_list)
        """
        directory_path = self._open_directory(title, initial_dir=initial_dir)
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
            logger.warning("Error scanning directory %s: %s", directory_path, e)
            return directory_path, False, []

        return directory_path, has_video_files, video_files

    def select_vector_file(self, title="Select Vector", initial_dir=None):
        """Select a single latent vector file (.pt/.pth/.npy/.npz).
        Returns the path or None if cancelled."""
        return self._open_single_file(title, self.vector_extensions[0][1], initial_dir=initial_dir)

    def select_model_file(self, title="Select Model", initial_dir=None):
        """Select a single pickled model file (.pkl).
        Returns the path or None if cancelled."""
        return self._open_single_file(title, self.model_extensions[0][1], initial_dir=initial_dir)

    def select_lora_file(self, title="Select LoRA", initial_dir=None):
        """Select a single LoRA weights file (.safetensors).
        Returns the path or None if cancelled."""
        return self._open_single_file(title, self.lora_extensions[0][1], initial_dir=initial_dir)

    def select_image_file(self, title="Select Image", initial_dir=None):
        """Select a single image file. Returns the path or None if cancelled."""
        return self._open_single_file(title, self.image_extensions[0][1], initial_dir=initial_dir)

    def save_vector_file(self, initial_file="vector.pt", title="Save Vector", initial_dir=None):
        """Open a native 'Save As' dialog for a latent vector. Ensures a .pt
        extension. Returns the path or None if cancelled."""
        path = self._save_file(title, self.vector_extensions[0][1], initial_file, initial_dir=initial_dir)
        if path and not os.path.splitext(path)[1]:
            path += ".pt"
        return path

    def cleanup(self):
        """Kept for API compatibility; native dialogs hold no resources."""
        pass