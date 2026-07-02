import logging
import os
from pathlib import Path

import numpy as np
import cv2
import PIL.Image
import PIL.ImageOps
import torchvision.transforms as transforms
import ffmpeg

from utils.user_data import data_path

logger = logging.getLogger(__name__)


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
        self.output_path = data_path("datasets")


    def load_images(self, image_path):
        """Load image, normalize color space to RGB, and handle EXIF orientation."""
        pil_image = None
        if isinstance(image_path, str):
            pil_image = PIL.Image.open(image_path)
            pil_image = PIL.ImageOps.exif_transpose(pil_image)
        elif isinstance(image_path, PIL.Image.Image):
            pil_image = image_path
        else:
            image = image_path
            if not isinstance(image, np.ndarray):
                raise TypeError("load_images expects path, PIL.Image, or numpy.ndarray")
            if image.ndim == 2:
                pil_image = PIL.Image.fromarray(image, mode='L')
            elif image.ndim == 3:
                channels = image.shape[2]
                if channels >= 3:
                    pil_image = PIL.Image.fromarray(image[:, :, :3])
                elif channels == 1:
                    pil_image = PIL.Image.fromarray(image[:, :, 0], mode='L')
                else:
                    raise ValueError(f"Unsupported image format: shape={image.shape}")
            else:
                raise ValueError(f"Unexpected image format: shape={image.shape}")

        if pil_image.mode == 'P':
            pil_image = pil_image.convert('RGB')

        image = np.array(pil_image)
        if image.ndim == 2:
            image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        elif image.ndim == 3:
            if image.shape[2] == 4 or image.shape[2] > 3:
                image = image[:, :, :3]

        pil_image = PIL.Image.fromarray(image)
        pil_image = pil_image.convert('RGB')
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
    def calculate_video_duration(video_path):
        probe = ffmpeg.probe(video_path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')

        duration = None
        
        if 'duration' in video_info and video_info['duration']:
            try:
                duration = float(video_info['duration'])
            except (ValueError, TypeError):
                pass
        
        if duration is None and 'tags' in video_info:
            tags = video_info['tags']
            if 'DURATION' in tags:
                try:
                    time_str = tags['DURATION']
                    parts = time_str.split(':')
                    if len(parts) == 3:
                        hours = float(parts[0])
                        minutes = float(parts[1])
                        seconds = float(parts[2])
                        duration = hours * 3600 + minutes * 60 + seconds
                except (ValueError, TypeError, IndexError):
                    pass
        
        if duration is None and 'format' in probe:
            format_info = probe['format']
            if 'duration' in format_info and format_info['duration']:
                try:
                    duration = float(format_info['duration'])
                except (ValueError, TypeError):
                    pass
        
        if duration is None:
            logger.warning("Could not determine duration for video %s, using default estimate", video_path)
            duration = 0
        
        return duration

    @staticmethod
    def calculate_expected_video_frames(video_path, fps=10):
        duration = DatasetPreprocessingUtils.calculate_video_duration(video_path)
        return int(duration * fps) if duration > 0 else 0

    @staticmethod
    def extract_videos(video_paths, fps, queue_in, queue_out):
        if not queue_in.empty():
            if queue_in.get() == "cancel":
                return
        
        results = []
        total_videos = len(video_paths)

        expected_per_video = []
        for video_path in video_paths:
            try:
                expected_per_video.append(
                    DatasetPreprocessingUtils.calculate_expected_video_frames(video_path, fps)
                )
            except Exception as e:
                logger.warning("Could not estimate frame count for %s: %s", video_path, e)
                expected_per_video.append(0)
        total_expected = sum(expected_per_video)
        frames_done_prior = 0

        for i, video_path in enumerate(video_paths):
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

            if total_expected > 0:
                start_pct = min(frames_done_prior / total_expected * 100.0, 99.0)
            else:
                start_pct = (i / total_videos * 100.0) if total_videos > 0 else 0.0
            queue_out.put({
                'type': 'progress',
                'current': i,
                'total': total_videos,
                'current_file': video_path_obj.name,
                'percentage': start_pct,
            })

            process = (
                ffmpeg
                .input(video_path)
                .output(output_pattern, vf=f"fps={fps}")
                .global_args('-progress', 'pipe:1', '-nostats')
                .run_async(pipe_stdout=True)
            )

            cancelled = False
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                line = line.decode('utf-8', errors='ignore').strip()
                if line.startswith('frame='):
                    try:
                        cur_frame = int(line.split('=', 1)[1])
                    except ValueError:
                        cur_frame = 0
                    if total_expected > 0:
                        done = frames_done_prior + min(cur_frame, expected_per_video[i])
                        pct = min(done / total_expected * 100.0, 99.0)
                    else:
                        pct = (i / total_videos * 100.0) if total_videos > 0 else 0.0
                    queue_out.put({
                        'type': 'progress',
                        'current': i,
                        'total': total_videos,
                        'current_file': video_path_obj.name,
                        'percentage': pct,
                    })

                if not queue_in.empty() and queue_in.get() == "cancel":
                    process.terminate()
                    cancelled = True
                    break

            if cancelled:
                return

            retcode = process.wait()
            frames_done_prior += expected_per_video[i]
            if retcode != 0:
                logger.error("FFmpeg failed for %s (exit code %s)", video_path, retcode)
                continue
            results.append(str(save_path))

        queue_out.put({'type': 'completed', 'results': results})
    
    @staticmethod
    def resize_image_np(image: np.ndarray, settings):
        target_size = settings.size
        resize_mode = settings.resizeMode
        
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
        
        os.makedirs(output_path, exist_ok=True)
        
        logger.info("Dataset preprocessing: %d images, resolution=%dx%d, resize_mode=%s, "
                    "non_square=%s, xflip=%s, yflip=%s",
                    len(images), size, size, resizeMode, nonSquare,
                    augmentationSettings['xFlip'], augmentationSettings['yFlip'])
        
        processed_count = 0
        total_source_images = len(images)
        utils = DatasetPreprocessingUtils()
        
        num_augmentations = sum(1 for v in augmentationSettings.values() if v)
        total_images = total_source_images * (1 + num_augmentations)
        
        # Update progress more frequently for better responsiveness
        update_interval = max(1, min(10, total_images // 500))
        
        for i, image_path in enumerate(images):
            try:
                if not queue.empty():
                    try:
                        if queue.get_nowait() == 'cancel':
                            logger.info("Batch preprocessing cancelled by user")
                            reply.put(['Batch preprocessing cancelled', True])
                            return None
                    except:
                        pass
                
                image = utils.load_images(image_path)
                
                images_to_process = [image]
                if any(settings.augmentationSettings.values()): 
                    images_to_process.extend(utils.augment_image(image, settings))
                
                for img_idx, img_to_process in enumerate(images_to_process):
                    if not queue.empty():
                        try:
                            if queue.get_nowait() == 'cancel':
                                logger.info("Batch preprocessing cancelled by user")
                                reply.put(['Batch preprocessing cancelled', True])
                                return None
                        except:
                            pass
                    
                    processed_image = utils.resize_image_np(img_to_process, settings)
                    output_filename = f"image_{i:05d}.png" if img_idx == 0 else f"image_{i:05d}_augmented{img_idx}.png"
                    output_filepath = Path(output_path) / output_filename
                    PIL.Image.fromarray(processed_image).save(str(output_filepath), 'PNG', compress_level=1)
                    
                    processed_count += 1
                    
                    if processed_count % update_interval == 0 or processed_count == total_images:
                        reply.put({
                            'type': 'progress',
                            'current': processed_count,
                            'total': total_images,
                            'percentage': (processed_count / total_images * 100) if total_images > 0 else 0,
                            'current_file': Path(image_path).name
                        })
                    
            except Exception:
                logger.exception("Error processing image %d", i)
                continue
        
        logger.info("Dataset processing completed: %d images saved to %s", processed_count, output_path)

        completion_data = {
            'type': 'completed',
            'processed_count': processed_count,
            'output_path': output_path
        }
        reply.put(completion_data)
        
        return output_path