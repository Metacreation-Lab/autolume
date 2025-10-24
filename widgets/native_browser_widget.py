import tkinter as tk
from tkinter import filedialog
import os
from typing import List, Optional, Tuple

class NativeBrowserWidget:
    def __init__(self):
        # Hide main tkinter window
        self.root = tk.Tk()
        self.root.withdraw()
        
        # Configure tkinter for better performance with large directories
        self.root.option_add('*Dialog.msg.font', 'TkDefaultFont')
        self.root.option_add('*Dialog.msg.wrapLength', '3i')
        
        # Set tkinter options to improve performance with large directories
        self.root.tk.call('tk', 'scaling', 1.0)  # Disable DPI scaling issues
        
        # Optimize for large file lists
        self.root.option_add('*Listbox.font', 'TkDefaultFont')
        self.root.option_add('*Listbox.selectBackground', 'lightblue')
        self.root.option_add('*Listbox.selectForeground', 'black')
        
        # Reduce visual effects for better performance
        self.root.option_add('*Button.relief', 'flat')
        self.root.option_add('*Button.borderWidth', '1')
        
        # Image extensions 
        self.image_extensions = [
            ('Image files', '*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp *.heic'),
            ('PNG files', '*.png'),
            ('JPEG files', '*.jpg *.jpeg'),
            ('BMP files', '*.bmp'),
            ('TIFF files', '*.tiff *.tif'),
            ('WebP files', '*.webp'),
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
            ('GIF files', '*.gif')
        ]
        
        self.all_media_extensions = [
            ('Media files', '*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.gif *.webp *.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.m4v *.3gp'),
            ('Image files', '*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.gif *.webp'),
            ('Video files', '*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.m4v *.3gp'),
            ('All files', '*.*')
        ]
        
        # Optimized extension sets for fast lookup
        self._image_extensions_set = self._build_extension_set(self.image_extensions)
        self._video_extensions_set = self._build_extension_set(self.video_extensions)
        
        # Lazy loading state
        self._current_directory = None
        self._all_files = []
        self._filtered_files = []
        self._page_size = 500  # Load files in chunks of 500
        self._current_page = 0
    
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
        """Load all files from directory (still needed for lazy loading)."""
        try:
            return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        except (OSError, PermissionError) as e:
            print(f"Error reading directory {directory}: {e}")
            return []
    
    def _get_files_page(self, files: List[str], page: int = 0, page_size: Optional[int] = None) -> List[str]:
        """Get a page of files for lazy loading."""
        if page_size is None:
            page_size = self._page_size
        
        start_idx = page * page_size
        end_idx = start_idx + page_size
        return files[start_idx:end_idx]
    
    def _filter_files_by_type(self, files: List[str], file_type: str = "image") -> List[str]:
        """Filter files by type using optimized extension checking."""
        if file_type == "image":
            return [f for f in files if self._is_image_file(f)]
        elif file_type == "video":
            return [f for f in files if self._is_video_file(f)]
        else:
            return files
    
    def select_image_directory(self):
        """Select a directory containing image files - returns directory path only for lazy loading."""
        folder_path = filedialog.askdirectory(
            title="Select a folder containing image dataset"
        )
        
        if not folder_path:
            return None
        
        # Store directory for lazy loading
        self._current_directory = folder_path
        self._all_files = self._load_directory_files(folder_path)
        self._filtered_files = self._filter_files_by_type(self._all_files, "image")
        self._current_page = 0
        
        return folder_path
    
    def get_image_files_lazy(self, page: int = 0, page_size: Optional[int] = None) -> Tuple[List[str], bool]:
        """
        Get image files using lazy loading (like desktop file browser).
        Returns (file_paths, has_more_pages)
        """
        if not self._current_directory or not self._filtered_files:
            return [], False
        
        if page_size is None:
            page_size = self._page_size
        
        page_files = self._get_files_page(self._filtered_files, page, page_size)
        full_paths = [os.path.join(self._current_directory, f) for f in page_files]
        
        has_more = (page + 1) * page_size < len(self._filtered_files)
        return full_paths, has_more
    
    def get_all_image_files(self) -> List[str]:
        """
        Get all image files from current directory (for backward compatibility).
        Use this only when you need all files at once.
        """
        if not self._current_directory:
            return []
        
        return [os.path.join(self._current_directory, f) for f in self._filtered_files]
    
    def select_image_files_native(self):
        """Select multiple image files using native OS dialog with enhanced performance."""
        try:
            # Configure tkinter for better performance with large directories
            self.root.update_idletasks()
            
            # Set tkinter options for better performance with large directories
            self.root.tk.call('tk', 'appname', 'ImageSelector')
            
            # Use native dialog with performance optimizations
            # Set initial directory to avoid scanning huge directories by default
            initial_dir = os.getcwd()
            
            image_files = filedialog.askopenfilenames(
                title="Select Image Files",
                filetypes=self.image_extensions,
                initialdir=initial_dir
            )
            return list(image_files)
            
        except Exception as e:
            print(f"Native dialog failed: {e}")
            return []
    

    def select_image_files(self):
        """Select one or more image files using native OS dialog."""
        return self.select_image_files_native()

    def select_video_files(self):
        """Select one or more video files using native OS dialog with enhanced performance."""
        try:
            # Configure tkinter for better performance with large directories
            self.root.update_idletasks()
            
            # Set tkinter options for better performance with large directories
            self.root.tk.call('tk', 'appname', 'VideoSelector')
            
            # Use native dialog with performance optimizations
            # Set initial directory to avoid scanning huge directories by default
            initial_dir = os.getcwd()
            
            video_files = filedialog.askopenfilenames(
                title="Select Video Files",
                filetypes=self.video_extensions,
                initialdir=initial_dir
            )
            return list(video_files)
            
        except Exception as e:
            print(f"Native video dialog failed: {e}")
            return []
    
    def select_video_directory(self):
        """Select a directory containing video files - returns directory path only for lazy loading."""
        folder_path = filedialog.askdirectory(
            title="Select a folder containing video dataset"
        )
        
        if not folder_path:
            return None
        
        # Store directory for lazy loading
        self._current_directory = folder_path
        self._all_files = self._load_directory_files(folder_path)
        self._filtered_files = self._filter_files_by_type(self._all_files, "video")
        self._current_page = 0
        
        return folder_path
    
    def get_video_files_lazy(self, page: int = 0, page_size: Optional[int] = None) -> Tuple[List[str], bool]:
        """
        Get video files using lazy loading (like desktop file browser).
        Returns (file_paths, has_more_pages)
        """
        if not self._current_directory or not self._filtered_files:
            return [], False
        
        if page_size is None:
            page_size = self._page_size
        
        page_files = self._get_files_page(self._filtered_files, page, page_size)
        full_paths = [os.path.join(self._current_directory, f) for f in page_files]
        
        has_more = (page + 1) * page_size < len(self._filtered_files)
        return full_paths, has_more
    
    def get_directory_file_count(self) -> int:
        """Get total number of files in current directory."""
        return len(self._filtered_files) if self._filtered_files else 0
    
    def reset_directory(self):
        """Reset directory state for new selection."""
        self._current_directory = None
        self._all_files = []
        self._filtered_files = []
        self._current_page = 0

    def select_directory(self, title="Select Directory"):
        """
        Select a directory using native OS dialog.
        Returns the directory path or None if cancelled.
        """
        try:
            # Configure tkinter for better performance
            self.root.update_idletasks()
            self.root.tk.call('tk', 'appname', 'DirectorySelector')
            
            # Use askdirectory to select a directory
            directory_path = filedialog.askdirectory(
                title=title,
                initialdir=os.getcwd()
            )
            
            return directory_path if directory_path else None
            
        except Exception as e:
            print(f"Error in select_directory: {e}")
            return None
    
    def select_directory_with_video_check(self, title="Select Directory"):
        """
        Select a directory and check if it contains video files.
        Returns (directory_path, has_video_files, video_files_list)
        """
        try:
            # Configure tkinter for better performance
            self.root.update_idletasks()
            self.root.tk.call('tk', 'appname', 'DirectorySelector')
            
            # Use askdirectory to select a directory
            directory_path = filedialog.askdirectory(
                title=title,
                initialdir=os.getcwd()
            )
            
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
            
        except Exception as e:
            print(f"Error in select_directory_with_video_check: {e}")
            return None, False, []

    def cleanup(self):
        """Clean up the tkinter root window."""
        try:
            self.root.destroy()
        except:
            pass 