import cv2
import os
import numpy as np
import imgui
from utils.gui_utils import gl_utils
import time
import gc

from utils.dataset_preprocessing_utils import DatasetPreprocessingUtils

class ThumbnailWidget:
    """Widget for handling image thumbnails with caching and display"""
    
    def __init__(self):
        self.thumbnail_size = 140 # Determines quality of thumbnails (resolution)
        self.thumbnails = {}  # Cache for thumbnail textures
        self.placeholder_textures = {}  # Cache for placeholder textures
        self.generate_thumbnails = False  # Default to performance mode (no thumbnails, only placeholders)
        self.selected_files = []
        self.last_selected_idx = None
        self.selected_indices = []  # For multi-selection
        self.last_mode_switch_time = 0  # For debouncing rapid toggles
        self.delete_pressed = False  # Flag for delete key press
    
    def create_placeholder_thumbnail(self, file_path):
        """Create a grey placeholder thumbnail with image name"""
        try:
            # Create grey background
            size = self.thumbnail_size
            canvas = np.full((size, size, 3), [128, 128, 128], dtype=np.uint8)  # Grey background
            
            # Get filename without extension for display
            filename = os.path.splitext(os.path.basename(file_path))[0]
            
            # Add text to the canvas
            canvas = self._add_text_to_canvas(canvas, filename, size)
            
            # Create texture
            texture = gl_utils.Texture(
                image=canvas,
                width=size,
                height=size,
                channels=3
            )
            
            return texture
        except Exception as e:
            print(f"Failed to create placeholder thumbnail for {file_path}: {e}")
            return None
    
    def _add_text_to_canvas(self, canvas, text, size):
        """Add text to the center of the thumbnail placeholder"""
        try:
            font_scale = 0.4
            font_thickness = 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
            
            # If text is too long, truncate it
            max_width = int(size * 0.9)  # Leave 10% margin
            if text_width > max_width:
                # Calculate how many characters we can fit
                char_width = text_width / len(text)
                max_chars = int(max_width / char_width) - 3  # Leave room for "..."
                text = text[:max_chars] + "..."
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
            
            # Calculate position to center the text
            x = (size - text_width) // 2
            y = (size + text_height) // 2
            
            # Add text to canvas
            cv2.putText(canvas, text, (x, y), font, font_scale, (255, 255, 255), font_thickness)
            
            return canvas
        except Exception as e:
            print(f"Failed to add text to canvas: {e}")
            return canvas
    
    def get_thumbnail(self, file_path):
        """Get thumbnail based on generate_thumbnails setting"""
        if self.generate_thumbnails:
            # Return actual thumbnail if available, otherwise placeholder
            if file_path in self.thumbnails:
                return self.thumbnails[file_path]
            else:
                # Return placeholder until background process creates the real thumbnail
                if file_path not in self.placeholder_textures:
                    self.placeholder_textures[file_path] = self.create_placeholder_thumbnail(file_path)
                return self.placeholder_textures[file_path]
        else:
            # Generate placeholders
            if file_path not in self.placeholder_textures:
                self.placeholder_textures[file_path] = self.create_placeholder_thumbnail(file_path)
            return self.placeholder_textures[file_path]
    
    def set_thumbnail_mode(self, generate_thumbnails):
        """Set thumbnail generation mode (placeholder or actual rendered thumbnails) and clear appropriate cache"""
        if self.generate_thumbnails != generate_thumbnails:
            current_time = time.time()
            
            # Debounce rapid toggling (prevent switches faster than 100ms)
            if current_time - self.last_mode_switch_time < 0.1:
                print("Debouncing rapid thumbnail mode toggle")
                return
            
            self.last_mode_switch_time = current_time
            print(f"Switching thumbnail mode: {self.generate_thumbnails} -> {generate_thumbnails}")
            
            # Set the mode first to prevent race conditions
            self.generate_thumbnails = generate_thumbnails
            
            # Clear ALL textures to prevent OpenGL conflicts
            self.clear_all_thumbnails()
            
            # Regenerate appropriate thumbnails (no blocking delay)
            if not generate_thumbnails and self.selected_files:
                # Switching to placeholders, regenerate placeholders
                self.update_thumbnails(self.selected_files)
    
    def update_thumbnails(self, file_paths):
        """Update thumbnails for all provided files"""
        self.selected_files = file_paths
        for file_path in file_paths:
            # Use get_thumbnail which handles both modes
            self.get_thumbnail(file_path)
    
    def clear_thumbnails(self):
        """Clear all thumbnails and free memory"""
        # Batch delete textures for better performance
        textures_to_delete = list(self.thumbnails.values())
        self.thumbnails.clear()  # Clear the dict first to prevent new references
        
        for texture in textures_to_delete:
            if texture is not None:
                try:
                    texture.delete()
                except Exception as e:
                    print(f"Error deleting texture: {e}")
    
    def clear_placeholder_thumbnails(self):
        """Clear all placeholder thumbnails and free memory"""
        # Batch delete textures for better performance
        textures_to_delete = list(self.placeholder_textures.values())
        self.placeholder_textures.clear()  # Clear the dict first to prevent new references
        
        for texture in textures_to_delete:
            if texture is not None:
                try:
                    texture.delete()
                except Exception as e:
                    print(f"Error deleting placeholder texture: {e}")
    
    def clear_all_thumbnails(self):
        """Clear both actual and placeholder thumbnails"""
        self.clear_thumbnails()
        self.clear_placeholder_thumbnails()
    
    def render_thumbnails(self, available_width, available_height):
        if not self.selected_files:
            # Calculate text size
            message = "No images imported"
            text_width = imgui.calc_text_size(message)[0]
            text_height = imgui.get_text_line_height()

            # Calculate centered position
            center_x = (available_width - text_width) / 2
            center_y = (available_height - text_height) / 2

            # Set cursor position
            imgui.set_cursor_pos_x(center_x)
            imgui.set_cursor_pos_y(center_y)

            imgui.text_colored(message, 0.5, 0.5, 0.5, 1.0)
            return

        min_thumb_size = 120
        max_thumb_size = 220
        spacing_x = 32  # horizontal space between thumbnails
        spacing_y = 32  # vertical space between rows
        n = len(self.selected_files)

        # Try to fit as many as possible, but not smaller than min_thumb_size
        thumbnails_per_row = max(1, int((available_width + spacing_x) // (min_thumb_size + spacing_x)))
        thumb_size = min(
            max_thumb_size,
            max(min_thumb_size, int((available_width - (thumbnails_per_row - 1) * spacing_x) // thumbnails_per_row))
        )

        # Center the row if there is extra space
        total_row_width = thumbnails_per_row * thumb_size + (thumbnails_per_row - 1) * spacing_x
        left_margin = max(0, (available_width - total_row_width) // 2)

        # Track selection
        if not hasattr(self, 'last_selected_idx'):
            self.last_selected_idx = None
        if not hasattr(self, 'selected_indices'):
            self.selected_indices = []

        for idx, file_path in enumerate(self.selected_files):
            # Get thumbnail (either actual or placeholder based on mode)
            texture = self.get_thumbnail(file_path)
            if texture is not None and hasattr(texture, 'gl_id') and texture.gl_id is not None:
                col = idx % thumbnails_per_row
                row = idx // thumbnails_per_row

                if col == 0:
                    imgui.dummy(left_margin, 0)
                elif col > 0:
                    imgui.same_line(spacing=spacing_x)

                # Draw thumbnail with clickable/hover effect
                imgui.begin_group()
                imgui.push_id(str(idx))
                is_hovered = False
                # Multi-selection support
                if self.selected_indices:
                    is_selected = (idx in self.selected_indices)
                else:
                    is_selected = (self.last_selected_idx == idx)

                cursor_pos = imgui.get_cursor_screen_pos()
                draw_list = imgui.get_window_draw_list()
                frame_color = (0.8, 0.4, 0.1, 1.0) if is_selected else (1.0, 0.8, 0.2, 0.7)
                border_thickness = 3 if is_selected else 2

                ctrl_down = imgui.is_key_down(341) or imgui.is_key_down(345)
                shift_down = imgui.is_key_down(340) or imgui.is_key_down(344)
                a_pressed = imgui.is_key_pressed(65)
                delete_pressed = imgui.is_key_pressed(261) or imgui.is_key_pressed(259) # Delete and backspace key

                if ctrl_down and a_pressed:
                    self.select_all()
                
                if delete_pressed and (self.selected_indices or self.last_selected_idx is not None):
                    self.delete_pressed = True

                if imgui.invisible_button("thumb", thumb_size, thumb_size):
                    if shift_down and self.last_selected_idx is not None:
                        start = min(self.last_selected_idx, idx)
                        end = max(self.last_selected_idx, idx)
                        self.selected_indices = list(range(start, end + 1))
                    elif ctrl_down:
                        if idx in self.selected_indices:
                            self.selected_indices.remove(idx)
                        else:
                            self.selected_indices.append(idx)
                        self.last_selected_idx = idx
                    else:
                        # Single selection
                        self.selected_indices = [idx]
                        self.last_selected_idx = idx

                is_hovered = imgui.is_item_hovered()

                imgui.set_cursor_screen_pos((cursor_pos[0], cursor_pos[1]))
                imgui.image(texture.gl_id, thumb_size, thumb_size)

                if is_hovered or is_selected:
                    x1, y1 = cursor_pos
                    x2, y2 = x1 + thumb_size, y1 + thumb_size
                    draw_list.add_rect(x1, y1, x2, y2, imgui.get_color_u32_rgba(*frame_color), rounding=6, thickness=border_thickness)

                filename = os.path.basename(file_path)
                if len(filename) > 15:
                    filename = filename[:12] + "..."
                text_width = imgui.calc_text_size(filename)[0]
                cursor_x = imgui.get_cursor_pos_x()
                imgui.set_cursor_pos_x(cursor_x + (thumb_size - text_width) / 2)
                imgui.text(filename)
                imgui.pop_id()
                imgui.end_group()

                if col == thumbnails_per_row - 1:
                    imgui.new_line()
                    imgui.dummy(0, spacing_y)
    
    def get_thumbnail_count(self):
        """Get the number of thumbnails"""
        return len(self.selected_files)
    
    def get_selected_files(self):
        """Get the list of selected files"""
        return self.selected_files.copy()

    # Select All button
    def select_all(self):
        """Select all images"""
        self.selected_indices = list(range(len(self.selected_files)))
        self.last_selected_idx = None

    def get_selected_indices(self):
        if self.selected_indices:
            return sorted(set(self.selected_indices))
        elif self.last_selected_idx is not None:
            return [self.last_selected_idx]
        return []
    
    def clear_selected(self):
        self.last_selected_idx = None
        self.selected_indices = []
    
    def is_delete_pressed(self):
        """Check if delete was pressed and reset the flag"""
        if self.delete_pressed:
            self.delete_pressed = False
            return True
        return False
    
    def update_thumbnail_from_data(self, file_path, thumbnail_data):
        """Update a specific thumbnail with processed data from background processing"""
        if thumbnail_data is not None:
            texture = self._create_texture_from_data(thumbnail_data)
            if texture is not None:
                self.thumbnails[file_path] = texture
    
    def _create_texture_from_data(self, thumbnail_data):
        """Create OpenGL texture from thumbnail data"""
        try:
            return gl_utils.Texture(
                image=thumbnail_data,
                width=thumbnail_data.shape[1],
                height=thumbnail_data.shape[0],
                channels=thumbnail_data.shape[2] if len(thumbnail_data.shape) > 2 else 1
            )
        except Exception as e:
            print(f"Error creating texture from thumbnail data: {e}")
            return None

    @staticmethod
    def process_thumbnails_background(queue, reply):
        """Process thumbnails in background - static method called by multiprocessing."""
        while True:
            try:
                request = queue.get(timeout=1.0)
                if request is None or request.get('type') == 'shutdown':
                    break
                
                if request['type'] == 'generate_thumbnails':
                    file_paths = request['file_paths']
                    thumbnail_size = request['thumbnail_size']
                    
                    # Process each thumbnail
                    for file_path in file_paths:
                        try:
                            # Create thumbnail data inline
                            thumbnail_data = None
                            
                            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
                            file_ext = os.path.splitext(file_path)[1].lower()
                            
                            if file_ext in video_extensions:
                                cap = cv2.VideoCapture(file_path)
                                if cap.isOpened():
                                    ret, frame = cap.read()
                                    cap.release()
                                    
                                    if ret:
                                        # Convert BGR to RGB
                                        if len(frame.shape) == 3 and frame.shape[2] == 3:
                                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                        elif len(frame.shape) == 3 and frame.shape[2] == 4:
                                            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
                                        
                                        # Create square canvas
                                        height, width = frame.shape[:2]
                                        max_dim = max(height, width)
                                        canvas = np.zeros((max_dim, max_dim, frame.shape[2] if len(frame.shape) > 2 else 1), dtype=frame.dtype)
                                        
                                        # Center the image
                                        y_offset = (max_dim - height) // 2
                                        x_offset = (max_dim - width) // 2
                                        canvas[y_offset:y_offset+height, x_offset:x_offset+width] = frame
                                        
                                        # Resize to thumbnail size
                                        thumbnail_data = cv2.resize(canvas, (thumbnail_size, thumbnail_size), interpolation=cv2.INTER_AREA)
                            else:
                                utils = DatasetPreprocessingUtils()
                                img = utils.load_images(file_path)
                                
                                # Create square canvas
                                height, width = img.shape[:2]
                                max_dim = max(height, width)
                                padding_colour = 26  # Match thumbnail background color #1A1A1A
                                canvas = np.full((max_dim, max_dim, img.shape[2] if len(img.shape) > 2 else 1), padding_colour, dtype=img.dtype)
                                
                                # Center the image
                                y_offset = (max_dim - height) // 2
                                x_offset = (max_dim - width) // 2
                                canvas[y_offset:y_offset+height, x_offset:x_offset+width] = img
                                
                                # Resize to thumbnail size
                                thumbnail_data = cv2.resize(canvas, (thumbnail_size, thumbnail_size), interpolation=cv2.INTER_AREA)
                            
                            # Send thumbnail result
                            reply.put({
                                'type': 'thumbnail',
                                'file_path': file_path,
                                'thumbnail_data': thumbnail_data
                            })
                            
                        except Exception as e:
                            print(f"Error creating thumbnail for {file_path}: {e}")
                            reply.put({
                                'type': 'thumbnail',
                                'file_path': file_path,
                                'thumbnail_data': None
                            })
                    
                    reply.put({'type': 'completed'})
                
            except:
                continue  

    def cleanup(self):
        """Clean up OpenGL textures and resources"""
        try:
            # Stop any background processing first
            self.generate_thumbnails = False
            
            # Clear all thumbnail textures
            for texture in self.thumbnails.values():
                if texture is not None and hasattr(texture, 'delete') and callable(texture.delete):
                    try:
                        if texture.gl_id is not None:
                            texture.delete()
                    except Exception as e:
                        print(f"Error deleting thumbnail texture: {e}")
            self.thumbnails.clear()
            
            # Clear placeholder textures
            for texture in self.placeholder_textures.values():
                if texture is not None and hasattr(texture, 'delete') and callable(texture.delete):
                    try:
                        if texture.gl_id is not None:
                            texture.delete()
                    except Exception as e:
                        print(f"Error deleting placeholder texture: {e}")
            self.placeholder_textures.clear()
            
            # Clear file lists and reset state
            self.selected_files = []
            self.selected_indices = []
            self.last_selected_idx = None
            
            # Force garbage collection to ensure cleanup
            gc.collect()
            
        except Exception as e:
            print(f"Warning: Error during thumbnail widget cleanup: {e}")

    def __del__(self):
        pass 