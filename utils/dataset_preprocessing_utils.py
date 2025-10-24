import os
from pathlib import Path

import numpy as np
import cv2
import PIL.Image
import PIL.ImageOps
import subprocess
import torchvision.transforms as transforms
import ffmpeg

class DatasetPreprocessingUtils:
    """Utility class to support dataset preprocessing functions."""
    def __init__(self):
        self.images = []
        self.resizeMode = 0 # 0 = stretch, 1 = center crop
        self.size = 128
        self.nonSquare = False
        self.fps = 10
        self.nonSquareSettings = {
            "widthRatio": 16,
            "heightRatio": 9,
            "paddingMode": 0 # 0 = black, 1 = white, 2 = bleeding
        }
        self.augmentationSettings = {
            "xFlip": False,
            "yFlip": False
        }
        self.folder_name = "training_dataset"
        self.output_path = str(Path.cwd() / "data").replace('\\', '/')


    def load_images(self, image_path):
        """Load image with color space handling and EXIF orientationfor dataset processing."""
        if isinstance(image_path, str):
            pil_image = PIL.Image.open(image_path)
            
            # Handle EXIF orientation data
            pil_image = PIL.ImageOps.exif_transpose(pil_image)
            
            # Handle 16-bit and high bit-depth images
            if pil_image.mode == 'I':  # 16-bit grayscale
                print(f"Converting 16-bit grayscale image to 8-bit: {image_path}")
                pil_image = pil_image.convert('L')  # Convert to 8-bit grayscale
            elif pil_image.mode == 'F':  # 32-bit float
                print(f"Converting 32-bit float image to 8-bit: {image_path}")
                pil_image = pil_image.convert('L')  # Convert to 8-bit grayscale
            elif pil_image.mode == 'LA':  # 16-bit grayscale + alpha
                print(f"Converting 16-bit grayscale+alpha image to RGB: {image_path}")
                pil_image = pil_image.convert('RGB')  # Convert to RGB, dropping alpha
            elif pil_image.mode == 'I;16':  # 16-bit grayscale (alternative format)
                print(f"Converting 16-bit grayscale image to 8-bit: {image_path}")
                pil_image = pil_image.convert('L')  # Convert to 8-bit grayscale
            elif pil_image.mode == 'I;16B':  # 16-bit grayscale big-endian
                print(f"Converting 16-bit grayscale image to 8-bit: {image_path}")
                pil_image = pil_image.convert('L')  # Convert to 8-bit grayscale
            elif pil_image.mode == 'I;16L':  # 16-bit grayscale little-endian
                print(f"Converting 16-bit grayscale image to 8-bit: {image_path}")
                pil_image = pil_image.convert('L')  # Convert to 8-bit grayscale
            
            # Convert to RGB (handles palette mode, CMYK, LAB, HSV, etc.)
            if pil_image.mode == 'P':
                pil_image = pil_image.convert('RGB')  # Convert limited palette images to RGB
            elif pil_image.mode == 'L':  # 8-bit grayscale
                pil_image = pil_image.convert('RGB')  # Convert grayscale to RGB
            elif pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')  # Convert other modes (CMYK, LAB, HSV, etc.) to RGB
            
            # Convert to numpy array
            image = np.array(pil_image)
        else:
            # Assume it's already a numpy array
            image = image_path
            
            # Convert numpy array to PIL Image for better color space handling
            if len(image.shape) == 2:
                # Grayscale image
                pil_image = PIL.Image.fromarray(image, mode='L')
            elif len(image.shape) == 3:
                if image.shape[2] == 1:
                    # Single channel grayscale
                    pil_image = PIL.Image.fromarray(image[:, :, 0], mode='L')
                elif image.shape[2] == 3:
                    # RGB image
                    pil_image = PIL.Image.fromarray(image, mode='RGB')
                elif image.shape[2] == 4:
                    # RGBA image
                    pil_image = PIL.Image.fromarray(image, mode='RGBA')
                else:
                    # Other formats, try to use first 3 channels
                    if image.shape[2] > 3:
                        print(f"Warning: Image has {image.shape[2]} channels, using first 3 channels")
                        pil_image = PIL.Image.fromarray(image[:, :, :3], mode='RGB')
                    else:
                        raise ValueError(f"Unsupported image format: shape={image.shape}")
            else:
                raise ValueError(f"Unexpected image format: shape={image.shape}")
            
            # Convert to RGB (handles palette mode, CMYK, LAB, HSV, etc.)
            if pil_image.mode == 'P':
                pil_image = pil_image.convert('RGB')  # Convert limited palette images to RGB
            elif pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')  # Convert other modes (CMYK, LAB, HSV, etc.) to RGB
            
            # Convert back to numpy array
            image = np.array(pil_image)
        
        return image

    @staticmethod
    def get_image_data(image_path):
        """Get comprehensive image data including dimensions, size, and metadata."""
        image_data = {
            'filename': None,
            'full_path': None,
            'file_size': None,
            'width': None,
            'height': None,
            'format': None,
            'mode': None,
            'orientation': None,
            'error': None
        }
        
        try:
            path = Path(image_path)
            image_data['filename'] = path.stem
            image_data['full_path'] = str(path).replace('\\', '/')
            
            try:
                file_size_bytes = path.stat().st_size
                image_data['file_size'] = round(file_size_bytes / (1024 * 1024), 2)  # Convert to MB
            except OSError:
                image_data['file_size'] = None
            
            # Image metadata
            with PIL.Image.open(image_path) as img:
                image_data['width'], image_data['height'] = img.size
                image_data['format'] = img.format
                image_data['mode'] = img.mode
                
        except Exception as e:
            image_data['error'] = str(e)
            
        return image_data

    def augment_image(self, image, settings):
        """Augment image according to augmentation settings."""
        augmented_images = []
        
        if settings.augmentationSettings.get('xFlip', False):
            xflipped_image = np.fliplr(image)
            augmented_images.append(xflipped_image)
        
        if settings.augmentationSettings.get('yFlip', False):
            yflipped_image = np.flipud(image)
            augmented_images.append(yflipped_image)
        
        return augmented_images
    
    @staticmethod
    def get_video_fps(video_path):
        probe = ffmpeg.probe(video_path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        
        fps_num, fps_den = map(int, video_info['avg_frame_rate'].split('/'))
        video_fps = fps_num / fps_den if fps_den != 0 else 0
        
        return video_fps

    @staticmethod
    def calculate_expected_video_frames(video_path, fps=10):
        probe = ffmpeg.probe(video_path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')

        duration = float(video_info['duration'])
        expected_frames = int(duration * fps)

        return expected_frames

    @staticmethod
    def extract_videos(video_paths, fps, queue_in, queue_out):
        if not queue_in.empty():
            if queue_in.get() == "cancel":
                return
        
        results = []
        total_videos = len(video_paths)
        
        for i, video_path in enumerate(video_paths):
            # Send progress update for progress bar
            progress_data = {
                'type': 'progress',
                'current': i,
                'total': total_videos,
                'current_file': Path(video_path).name
            }
            queue_out.put(progress_data)
            
            # Check for cancel
            if not queue_in.empty():
                if queue_in.get() == "cancel":
                    return
            
            # Extract frames for this video
            video_path_obj = Path(video_path)
            video_dir = video_path_obj.parent
            video_name = video_path_obj.stem
            save_path = video_dir / f"{video_name}_frames @ {fps} fps"
            save_path.mkdir(parents=True, exist_ok=True)

            output_pattern = str(save_path / f"{video_name}_frame_%05d.jpg")
            cmd = [
                "ffmpeg",
                "-i", video_path,
                "-vf", f"fps={fps}",
                output_pattern
            ]

            try:
                subprocess.run(cmd, check=True)
                results.append(str(save_path))
            except subprocess.CalledProcessError as e:
                print(f"FFmpeg failed for {video_path}: {e}")
                continue
        
        # Send completion signal
        queue_out.put({'type': 'completed', 'results': results})
    
    @staticmethod
    def resize_image_np(image: np.ndarray, settings):
        target_size = settings.size
        resize_mode = settings.resizeMode
        
        # Handle non-square settings
        if hasattr(settings, 'nonSquare') and settings.nonSquare:
            image = DatasetPreprocessingUtils.non_square(image, settings)
        else:
            # Square image processing
            if resize_mode == 0: # stretch resize mode
                pil_image = PIL.Image.fromarray(image)
                resized_pil = pil_image.resize((target_size, target_size), PIL.Image.LANCZOS)
                image = np.array(resized_pil)
            else: # center crop resize mode
                pil_image = PIL.Image.fromarray(image)
                resize_transform = transforms.Resize(target_size)
                resized_image = resize_transform(pil_image)
                crop_transform = transforms.CenterCrop((target_size, target_size))
                cropped_image = crop_transform(resized_image)
                image = np.array(cropped_image)

        return image
        
    @staticmethod
    def non_square(image, settings):
        """Process image with non-square aspect ratio and padding to square"""
        # Extract settings
        target_size = settings.size
        resize_mode = settings.resizeMode
        width_ratio = settings.nonSquareSettings["widthRatio"]
        height_ratio = settings.nonSquareSettings["heightRatio"]
        padding_mode = settings.nonSquareSettings["paddingMode"]
        
        target_ratio = float(width_ratio) / float(height_ratio)
        
        if resize_mode == 0:  # stretch mode
            if target_ratio > 1:
                frame_height = target_size
                frame_width = int(target_size * target_ratio)
            else:
                frame_width = target_size
                frame_height = int(target_size / target_ratio)
            
            resized_content = cv2.resize(image, dsize=(frame_width, frame_height), 
                                       interpolation=cv2.INTER_CUBIC)
        else:  # center crop mode
            current_ratio = image.shape[1] / image.shape[0]
            
            if current_ratio > target_ratio:
                new_width = int(image.shape[0] * target_ratio)
                start_x = (image.shape[1] - new_width) // 2
                resized_content = image[:, start_x:start_x+new_width]
            else:
                new_height = int(image.shape[1] / target_ratio)
                start_y = (image.shape[0] - new_height) // 2
                resized_content = image[start_y:start_y+new_height, :]
        
        canvas_size = target_size
        
        if resized_content.shape[0] > canvas_size or resized_content.shape[1] > canvas_size:
            scale_factor = min(canvas_size / resized_content.shape[0], canvas_size / resized_content.shape[1])
            new_height = int(resized_content.shape[0] * scale_factor)
            new_width = int(resized_content.shape[1] * scale_factor)
            resized_content = cv2.resize(resized_content, dsize=(new_width, new_height), 
                                       interpolation=cv2.INTER_CUBIC)
        
        if padding_mode == 2:  # Bleeding mode
            canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
            y_offset = (canvas_size - resized_content.shape[0]) // 2
            x_offset = (canvas_size - resized_content.shape[1]) // 2
            
            canvas[y_offset:y_offset+resized_content.shape[0], x_offset:x_offset+resized_content.shape[1]] = resized_content
            
            # Apply bleeding effect
            if resized_content.shape[1] > resized_content.shape[0]:  
                # Vertical bleeding
                for y in range(y_offset):
                    canvas[y] = canvas[y_offset] if y % 2 == 0 else canvas[y_offset+1]
                for y in range(y_offset+resized_content.shape[0], canvas_size):
                    canvas[y] = canvas[y_offset+resized_content.shape[0]-2] if y % 2 == 0 else canvas[y_offset+resized_content.shape[0]-1]
            else: 
                # Horizontal bleeding
                for x in range(x_offset):
                    canvas[:, x] = canvas[:, x_offset] if x % 2 == 0 else canvas[:, x_offset+1]
                for x in range(x_offset+resized_content.shape[1], canvas_size):
                    canvas[:, x] = canvas[:, x_offset+resized_content.shape[1]-2] if x % 2 == 0 else canvas[:, x_offset+resized_content.shape[1]-1]
        else:
            # Black or white padding
            padding_value = 255 if padding_mode == 1 else 0
            canvas = np.full((canvas_size, canvas_size, 3), padding_value, dtype=np.uint8)
            y_offset = (canvas_size - resized_content.shape[0]) // 2
            x_offset = (canvas_size - resized_content.shape[1]) // 2
            canvas[y_offset:y_offset+resized_content.shape[0], x_offset:x_offset+resized_content.shape[1]] = resized_content
        
        return canvas
    
    @staticmethod
    def create_training_dataset(queue, reply):
        """Preprocess curated dataset prepared for training."""
        settings = queue.get()
        images = settings.images
        size = settings.size
        resizeMode = settings.resizeMode
        nonSquare = settings.nonSquare
        nonSquareSettings = settings.nonSquareSettings
        augmentationSettings = settings.augmentationSettings
        output_path = settings.output_path
        
        # Create output (data) directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Debug print settings
        print("=== DATASET PREPROCESSING SETTINGS ===")
        print(f"Images: {len(images)} files")
        print(f"Resolution: {size}x{size}")
        print(f"Resize Mode: {resizeMode}")
        print(f"Non-square: {nonSquare}")
        if nonSquare:
            print(f"  Width Ratio: {nonSquareSettings['widthRatio']}")
            print(f"  Height Ratio: {nonSquareSettings['heightRatio']}")
            print(f"  Padding Mode: {nonSquareSettings['paddingMode']}")
        print(f"X-Flip Augmentation: {augmentationSettings['xFlip']}")
        print(f"Y-Flip Augmentation: {augmentationSettings['yFlip']}")
        print("=====================================")
        
        # Process each image
        processed_count = 0
        total_images = len(images)
        utils = DatasetPreprocessingUtils()
        
        for i, image_path in enumerate(images):
            try:
                # Check for cancel signal at the start of each image
                if not queue.empty():
                    try:
                        signal = queue.get_nowait()
                        if signal == 'cancel':
                            print("Batch preprocessing cancelled by user")
                            reply.put(['Batch preprocessing cancelled', True])
                            return None
                    except:
                        pass
                
                image = utils.load_images(image_path)
                
                # PIPELINE: Augmentation → Non-square → Resize
                
                # Create list of images to process (original + augmented versions)
                images_to_process = [image]
                
                # Apply augmentation
                if any(settings.augmentationSettings.values()): 
                    augmented_images = utils.augment_image(image, settings)
                    images_to_process.extend(augmented_images)
                
                # Process all images (original + augmented)
                for img_idx, img_to_process in enumerate(images_to_process):
                    # Check for cancel signal before processing each image
                    if not queue.empty():
                        try:
                            signal = queue.get_nowait()
                            if signal == 'cancel':
                                print("Batch preprocessing cancelled by user")
                                reply.put(['Batch preprocessing cancelled', True])
                                return None
                        except:
                            pass
                    
                    processed_image = utils.resize_image_np(img_to_process, settings)
                    
                    if img_idx == 0:
                        output_filename = f"image_{i:05d}.png"
                    else:
                        # Augmented image
                        output_filename = f"image_{i:05d}_augmented{img_idx}.png"
                    
                    # Save processed image
                    output_filepath = Path(output_path) / output_filename
                    pil_image = PIL.Image.fromarray(processed_image)
                    pil_image.save(str(output_filepath), 'PNG', compress_level=1)
                    processed_count += 1
                
                # Send progress update
                progress_data = {
                    'type': 'progress',
                    'current': i + 1,
                    'total': total_images,
                    'percentage': ((i + 1) / total_images) * 100,
                    'current_file': Path(image_path).name
                }
                reply.put(progress_data)
                
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(images)} images...")
                    
            except Exception as e:
                print(f"Error processing image {i}: {str(e)}")
                continue
        
        print(f"Dataset processing completed. {processed_count} images processed and saved to {output_path}")

        # Send completion signal
        completion_data = {
            'type': 'completed',
            'processed_count': processed_count,
            'output_path': output_path
        }
        reply.put(completion_data)
        
        return output_path