import os
import cv2
import numpy as np
from utils.gui_utils import gl_utils
import imgui

from utils.dataset_preprocessing_utils import DatasetPreprocessingUtils

class ImagePreviewWidget:
    """Widget for displaying a preview of the selected image."""
    def __init__(self, preview_size=320):
        self.preview_size = preview_size
        self.current_file = None
        self.texture = None
        self.image_shape = None

    def update_preview(self, file_path, settings, preview_original=False):
        """Update the preview to show the image at file_path."""
        self.current_file = file_path
        
        # Clear current texture
        if self.texture is not None:
            self.texture.delete()
            self.texture = None
        
        # Handle no file case
        if file_path is None or not os.path.exists(file_path):
            self.image_shape = None
            self.texture = None
            return
        
        # Load and process image
        try:
            utils = DatasetPreprocessingUtils()
            img = utils.load_images(file_path)
            # Process image if settings are provided and not previewing original
            if settings is not None and not preview_original:
                processed_img = DatasetPreprocessingUtils.resize_image_np(img, settings)
                self._update_texture(processed_img)
            else:
                # When original preview is toggled on, load the image as-is
                self._update_texture(img)
                
        except Exception as e:
            print(f"Error processing image for preview: {e}")
            return
    
    def _update_texture(self, img):
        """Update the texture with processed image."""
        self.image_shape = img.shape
        self.texture = gl_utils.Texture(
            image=img,
            width=img.shape[1],
            height=img.shape[0],
            channels=img.shape[2]
        )

    def render(self, available_width, available_height):
        """Render the image preview."""
        if self.texture is None or self.image_shape is None:
            # Draw placeholder centered both horizontally and vertically
            message = "No image selected"
            text_width = imgui.calc_text_size(message)[0] if hasattr(imgui, 'calc_text_size') else 120
            text_height = imgui.get_text_line_height_with_spacing()

            center_x = (available_width - text_width) / 2
            center_y = (available_height - text_height) / 2
            
            imgui.set_cursor_pos_x(center_x)
            imgui.set_cursor_pos_y(center_y)
            imgui.text_colored(message, 0.5, 0.5, 0.5, 1.0)
            return

        img_h, img_w = self.image_shape[:2]
        scale = min(available_width / img_w, available_height / img_h, 1.0)
        disp_w = int(img_w * scale)
        disp_h = int(img_h * scale)
        
        # Center the image both horizontally and vertically
        center_x = (available_width - disp_w) / 2
        center_y = (available_height - disp_h) / 2
        
        imgui.set_cursor_pos_x(center_x)
        imgui.set_cursor_pos_y(center_y)
        if self.texture.gl_id is not None:
            imgui.image(self.texture.gl_id, disp_w, disp_h)
        else:
            # Draw placeholder if texture is invalid
            imgui.dummy(disp_w, disp_h)

    def cleanup(self):
        """Clean up OpenGL textures and resources"""
        try:
            # Clean up texture
            if self.texture is not None and self.texture.gl_id is not None:
                self.texture.delete()
                self.texture = None
            
            # Reset state
            self.current_file = None
            self.image_shape = None
            
        except Exception as e:
            print(f"Warning: Error during image preview widget cleanup: {e}")
