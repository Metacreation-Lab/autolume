import logging
import os
import queue
from pathlib import Path

import numpy as np
import cv2
import PIL.Image
import PIL.ImageOps
import torchvision.transforms as transforms

from utils import video_io
from utils.user_data import data_path

logger = logging.getLogger(__name__)


class DatasetPreprocessingUtils:
    """Utility class to support dataset preprocessing functions."""
    def __init__(self):
        self.images = []
        self.resizeMode = 1 # 0 = stretch, 1 = center crop
        self.size = 128
        self.nonSquare = False
        self.extraction_interval = 1.0
        self.nonSquareSettings = {
            "widthRatio": 16,
            "heightRatio": 9,
            "paddingMode": 0 # 0 = black, 1 = white, 2 = bleeding
        }
        self.augmentationSettings = {
            "xFlip": False,
            "yFlip": False
        }
        self.upscaleSettings = {
            "aiUpscale": False,
            "model": "RealPLKSR"
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
        """Duration of a video in seconds, or 0 when it cannot be determined."""
        try:
            return video_io.probe(video_path).duration
        except video_io.VideoIOError as e:
            logger.warning("Could not determine duration for video %s: %s", video_path, e)
            return 0

    @staticmethod
    def estimate_extracted_frames(info, interval):
        """Upper bound on the frames extraction will write for one video."""
        if info.duration <= 0:
            return 0
        if interval > 0:
            return int(info.duration / interval) + 1
        return int(info.duration * info.fps)

    @staticmethod
    def extract_videos(video_paths, interval, queue_in, queue_out):
        """Extract JPEG frames from each video into a sibling directory.

        Progress on ``queue_out`` is weighted by video duration so long files
        advance the bar smoothly; stops when ``queue_in`` says "cancel".
        """
        cancelled = False

        def should_cancel():
            nonlocal cancelled
            while not cancelled and not queue_in.empty():
                try:
                    cancelled = queue_in.get_nowait() == "cancel"
                except queue.Empty:
                    break
            return cancelled

        if should_cancel():
            return

        total_videos = len(video_paths)
        durations = [DatasetPreprocessingUtils.calculate_video_duration(p)
                     for p in video_paths]
        total_duration = sum(durations)
        seconds_done = 0.0
        last_reported = -1.0
        results = []

        for i, video_path in enumerate(video_paths):
            if should_cancel():
                return

            source = Path(video_path)
            save_path = source.parent / f"{source.stem}_frames @ {interval:g}s"

            def report(seconds, index=i, name=source.name, force=False):
                nonlocal last_reported
                if total_duration > 0:
                    percentage = min(seconds / total_duration * 100.0, 100.0)
                else:
                    percentage = (index / total_videos * 100.0) if total_videos else 0.0
                if not force and percentage - last_reported < 0.5:
                    return
                last_reported = percentage
                queue_out.put({
                    'type': 'progress',
                    'current': index,
                    'total': total_videos,
                    'current_file': name,
                    'percentage': percentage,
                })

            def on_progress(written, timestamp, base=seconds_done):
                report(base + timestamp)

            report(seconds_done, force=True)
            try:
                video_io.extract_frames(video_path, interval, str(save_path),
                                        source.stem,
                                        on_progress=on_progress,
                                        should_cancel=should_cancel)
            except video_io.VideoIOError as e:
                logger.error("Frame extraction failed for %s: %s", video_path, e)
                continue

            seconds_done += durations[i]
            if cancelled:
                return
            report(seconds_done, force=True)
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
        augmentationSettings = settings.augmentationSettings
        output_path = settings.output_path

        import upscale

        upscale_settings = getattr(settings, 'upscaleSettings', None) or {}
        ai_upscale = upscale_settings.get('aiUpscale', False)
        upscale_model = upscale_settings.get('model')
        if upscale_model not in upscale.PREPARE_MODELS:
            upscale_model = upscale.PREPARE_DEFAULT_MODEL
        upscaler = None
        if ai_upscale:
            try:
                upscaler = upscale.load_upscaler(upscale_model)
            except Exception:
                logger.exception("Could not load the upscaler")
                upscaler = None
            if upscaler is None:
                logger.warning("Upscaler unavailable, falling back to standard resizing")
                ai_upscale = False

        os.makedirs(output_path, exist_ok=True)

        logger.info("Dataset preprocessing: %d images, resolution=%dx%d, resize_mode=%s, "
                    "non_square=%s, xflip=%s, yflip=%s, ai_upscale=%s, upscale_model=%s",
                    len(images), size, size, resizeMode, nonSquare,
                    augmentationSettings['xFlip'], augmentationSettings['yFlip'],
                    ai_upscale, upscale_model)
        
        processed_count = 0
        total_source_images = len(images)
        utils = DatasetPreprocessingUtils()
        
        num_augmentations = sum(1 for v in augmentationSettings.values() if v)
        total_images = total_source_images * (1 + num_augmentations)
        
        # Update progress more frequently for better responsiveness
        update_interval = 1 if ai_upscale else max(1, min(10, total_images // 500))
        
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

                if upscaler is not None:
                    h, w = image.shape[:2]
                    if upscale.needs_upscale(w, h, size):
                        # An upscale failure (out of memory above all) must not
                        # drop the image: keep it and let the resize handle it.
                        try:
                            image = upscale.upscale_to_target(image, upscaler, size)
                        except Exception:
                            logger.exception("Upscale failed for %s, falling back to resize",
                                             image_path)

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