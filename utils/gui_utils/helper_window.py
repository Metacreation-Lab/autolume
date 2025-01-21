import os
import imgui
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import imageio
from . import gl_utils

class HelperWindow:
    """Helper window component that supports displaying text descriptions, images, GIFs and videos"""
    
    def __init__(self):
        self.help_contents = {}  # Store help contents from Excel
        self.media_textures = {}  # Store media textures
        self.current_frame = {}  # Store current GIF frame
        self.gif_readers = {}  # Store GIF readers
        self.last_update_time = {}  # Store last update time for each GIF
        self.frame_duration = {}  # Store GIF frame duration
        
    def load_help_contents(self, excel_path):
        """Load help contents from Excel
        
        Excel format requirements:
        - key: Unique identifier for help item
        - text: Help text content
        - image_path: Path to image (optional)
        - gif_path: Path to GIF (optional)
        - video_path: Path to video (optional)
        """
        try:
            df = pd.read_excel(excel_path)
            for _, row in df.iterrows():
                key = str(row['key'])  # Ensure key is string
                self.help_contents[key] = {
                    'text': str(row['text']),
                    'image_path': str(row['image_path']) if pd.notna(row.get('image_path', '')) else None,
                    'gif_path': str(row['gif_path']) if pd.notna(row.get('gif_path', '')) else None,
                    'video_path': str(row['video_path']) if pd.notna(row.get('video_path', '')) else None
                }
                
                # Preload media files
                self._load_media(key)
                
        except Exception as e:
            print(f"Error loading help contents: {str(e)}")

    def _load_media(self, key):
        """Load media files"""
        content = self.help_contents[key]
        
        # Load image
        if content['image_path'] and os.path.exists(content['image_path']):
            try:
                img = cv2.imread(content['image_path'], cv2.IMREAD_UNCHANGED)
                if img is not None:
                    if img.shape[2] == 3:  # BGR to RGBA
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
                    else:  # BGRA to RGBA
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
                    self.media_textures[f"{key}_image"] = gl_utils.Texture(
                        image=img, 
                        width=img.shape[1],
                        height=img.shape[0], 
                        channels=img.shape[2]
                    )
            except Exception as e:
                print(f"Failed to load image {content['image_path']}: {str(e)}")

        # Load GIF
        if content['gif_path'] and os.path.exists(content['gif_path']):
            try:
                self.gif_readers[key] = imageio.get_reader(content['gif_path'])
                self.current_frame[key] = 0
                self.last_update_time[key] = 0
                # Get GIF frame duration
                self.frame_duration[key] = self.gif_readers[key].get_meta_data().get('duration', 100)  # Default 100ms
                
                # Load first frame
                first_frame = self.gif_readers[key].get_data(0)
                if len(first_frame.shape) == 2:  # Grayscale to RGBA
                    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_GRAY2RGBA)
                elif first_frame.shape[2] == 3:  # RGB to RGBA
                    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGB2RGBA)
                
                self.media_textures[f"{key}_gif"] = gl_utils.Texture(
                    image=first_frame,
                    width=first_frame.shape[1],
                    height=first_frame.shape[0],
                    channels=first_frame.shape[2]
                )
            except Exception as e:
                print(f"Failed to load GIF {content['gif_path']}: {str(e)}")

        # Load video first frame
        if content['video_path'] and os.path.exists(content['video_path']):
            try:
                cap = cv2.VideoCapture(content['video_path'])
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                    self.media_textures[f"{key}_video"] = gl_utils.Texture(
                        image=frame,
                        width=frame.shape[1],
                        height=frame.shape[0],
                        channels=frame.shape[2]
                    )
                cap.release()
            except Exception as e:
                print(f"Failed to load video {content['video_path']}: {str(e)}")

    def update_gif_frame(self, key):
        """Update GIF frame"""
        if key in self.gif_readers:
            try:
                current_time = imgui.get_time() * 1000  # Convert to milliseconds
                if current_time - self.last_update_time[key] >= self.frame_duration[key]:
                    reader = self.gif_readers[key]
                    self.current_frame[key] = (self.current_frame[key] + 1) % reader.get_length()
                    frame = reader.get_data(self.current_frame[key])
                    
                    # Ensure frame is RGBA format
                    if len(frame.shape) == 2:  # Grayscale to RGBA
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGBA)
                    elif frame.shape[2] == 3:  # RGB to RGBA
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
                    
                    # Update texture
                    if f"{key}_gif" in self.media_textures:
                        self.media_textures[f"{key}_gif"].update(frame)
                    
                    self.last_update_time[key] = current_time
                    
            except Exception as e:
                print(f"Error updating GIF frame: {str(e)}")

    def show_help_marker(self, key, width=200, height=200):
        """Display help marker and popup window
        
        Args:
            key: Unique identifier for help content
            width: Media display width
            height: Media display height
        """
        if key not in self.help_contents:
            return
            
        imgui.same_line()
        imgui.text_disabled("(?)")
        
        if imgui.is_item_hovered():
            imgui.begin_tooltip()
            content = self.help_contents[key]
            
            # Display text description
            imgui.push_text_wrap_pos(imgui.get_font_size() * 35.0)
            imgui.text_unformatted(content['text'])
            imgui.pop_text_wrap_pos()
            
            # Display image
            if f"{key}_image" in self.media_textures:
                imgui.text("Example Image:")
                imgui.image(self.media_textures[f"{key}_image"].gl_id, width, height)
            
            # Display GIF
            if f"{key}_gif" in self.media_textures:
                imgui.text("Dynamic Demo:")
                imgui.image(self.media_textures[f"{key}_gif"].gl_id, width, height)
                self.update_gif_frame(key)
            
            # Display video first frame
            if f"{key}_video" in self.media_textures:
                imgui.text("Video Preview:")
                imgui.image(self.media_textures[f"{key}_video"].gl_id, width, height)
            
            imgui.end_tooltip()

    def cleanup(self):
        """Clean up resources"""
        # Clean up GIF readers
        for reader in self.gif_readers.values():
            reader.close()
        
        # Clean up textures
        for texture in self.media_textures.values():
            texture.cleanup()
