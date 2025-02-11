import os
import shutil
import cv2
import numpy as np
import PIL.Image
import ffmpeg
from pathlib import Path
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

def process_non_square_dataset(
    input_path,          # Path to input dataset
    output_path,         # Path to output processed dataset
    crop_ratio,          # (width, height) e.g. (16, 9)
    padding_color=0,     # 0=black padding, 1=white padding, 2=bleeding
    resize_mode="stretch",
    num_workers=None,    # Number of worker processes (defaults to CPU count)
):
    """Preprocess dataset to specified aspect ratio with padding to square"""
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    print(f'\n=== Starting non-square dataset processing ===')
    print(f'Input path: {input_path}')
    print(f'Output path: {output_path}')
    print(f'Target ratio: {crop_ratio[0]}:{crop_ratio[1]}')
    print(f'Padding color value: {padding_color} (type: {type(padding_color)})')
    print(f'Resize mode: {resize_mode}')
    print(f'Number of workers: {num_workers}')

    if padding_color == 2:
        print(f'Padding mode: bleeding')
    else:
        print(f'Padding color: {"white" if padding_color == 1 else "black"}')
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Supported file extensions
    image_extensions = {'.png', '.jpg', '.jpeg'}
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    
    # Collect all files
    files_to_process = []
    if input_path.is_dir():
        for file_path in input_path.rglob('*'):
            if file_path.suffix.lower() in image_extensions | video_extensions | {'.gif'}:
                files_to_process.append(file_path)
    else:
        raise IOError('Input path must be a directory')
    
    if len(files_to_process) == 0:
        raise IOError('No supported files found in the input path')
    
    print(f'Found {len(files_to_process)} files to process')
    
    # Process each file type
    target_ratio = float(crop_ratio[0]) / float(crop_ratio[1])
    if padding_color != 2:
        padding_value = 255 if padding_color == 1 else 0
        print(f'Using padding value: {padding_value}')
    
    frame_count = 0
    
    # Group files by type for batch processing
    image_files = [f for f in files_to_process if f.suffix.lower() in image_extensions]
    gif_files = [f for f in files_to_process if f.suffix.lower() == '.gif']
    video_files = [f for f in files_to_process if f.suffix.lower() in video_extensions]
    
    # Process images in batches
    if image_files:
        print(f'\nProcessing {len(image_files)} images in parallel...')
        batch_size = max(1, len(image_files) // (num_workers * 4))
        image_batches = []
        total_processed = 0
        
        for i in range(0, len(image_files), batch_size):
            batch = image_files[i:i + batch_size]
            image_batches.append((
                batch,
                output_path,
                frame_count + i,
                target_ratio,
                padding_color,
                resize_mode
            ))
        
        progress_bar = tqdm(
            total=len(image_files),
            desc="Processing images",
            unit="images",
            dynamic_ncols=True
        )
        
        with mp.Pool(num_workers) as pool:
            for count in pool.imap(process_images_batch, image_batches):
                total_processed += count
                progress_bar.update(count)
        
        progress_bar.close()
        frame_count += total_processed
    
    # Process GIFs
    for gif_idx, fname in enumerate(gif_files):
        print(f'\nProcessing GIF {gif_idx+1}/{len(gif_files)}: {fname}')
        try:
            frame_count += process_gif_parallel(
                fname, output_path, target_ratio, padding_color, resize_mode, 
                num_workers, frame_count
            )
        except Exception as e:
            print(f'Error processing GIF {fname}: {e}')
            continue
    
    # Process videos
    for video_idx, fname in enumerate(video_files):
        print(f'\nProcessing video {video_idx+1}/{len(video_files)}: {fname}')
        try:
            frame_count += process_video_parallel(
                fname, output_path, target_ratio, padding_color, resize_mode, num_workers
            )
        except Exception as e:
            print(f'Error processing video {fname}: {e}')
            continue
    
    print(f'\nDataset preprocessing completed. Total frames processed: {frame_count}')
    return output_path

def process_frame_crop(frame, target_ratio, padding_color):
    """Process a single frame"""
    current_ratio = frame.shape[1] / frame.shape[0]
    
    # Crop to target ratio
    if current_ratio > target_ratio:
        new_width = int(frame.shape[0] * target_ratio)
        start_x = (frame.shape[1] - new_width) // 2
        frame = frame[:, start_x:start_x+new_width]
    else:
        new_height = int(frame.shape[1] / target_ratio)
        start_y = (frame.shape[0] - new_height) // 2
        frame = frame[start_y:start_y+new_height, :]
    
    # Create square canvas
    target_size = max(frame.shape)
    
    if padding_color == 2:  # Bleeding mode
        canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        y_offset = (target_size - frame.shape[0]) // 2
        x_offset = (target_size - frame.shape[1]) // 2
        canvas[y_offset:y_offset+frame.shape[0], x_offset:x_offset+frame.shape[1]] = frame
        
        cropped_ratio = frame.shape[1] / frame.shape[0]
        if cropped_ratio > target_ratio:
            # Vertical bleeding
            for y in range(y_offset):
                canvas[y] = frame[0] if y % 2 == 0 else frame[1]
            for y in range(y_offset+frame.shape[0], target_size):
                canvas[y] = frame[-2] if y % 2 == 0 else frame[-1]
        else:
            # Horizontal bleeding
            for x in range(x_offset):
                canvas[:, x] = frame[:, 0] if x % 2 == 0 else frame[:, 1]
            for x in range(x_offset+frame.shape[1], target_size):
                canvas[:, x] = frame[:, -2] if x % 2 == 0 else frame[:, -1]
    else:
        padding_value = 255 if padding_color == 1 else 0
        canvas = np.full((target_size, target_size, 3), padding_value, dtype=np.uint8)
        y_offset = (target_size - frame.shape[0]) // 2
        x_offset = (target_size - frame.shape[1]) // 2
        canvas[y_offset:y_offset+frame.shape[0], x_offset:x_offset+frame.shape[1]] = frame
    
    return canvas

def process_frame(frame, target_ratio, padding_color, resize_mode="stretch"):
    """Process a single frame"""
    current_ratio = frame.shape[1] / frame.shape[0]
    target_size = max(frame.shape)
    
    if target_ratio > 1:  
        frame_height = target_size
        frame_width = int(target_size * target_ratio)
    else:  
        frame_width = target_size
        frame_height = int(target_size / target_ratio)
    
    if resize_mode == "stretch":
        resized_content = cv2.resize(frame, dsize=(frame_width, frame_height), 
                                   interpolation=cv2.INTER_CUBIC)
    else:
        return process_frame_crop(frame, target_ratio, padding_color)
    
    canvas_size = max(frame_width, frame_height)
    if padding_color == 2:  # Bleeding mode
        canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
        y_offset = (canvas_size - frame_height) // 2
        x_offset = (canvas_size - frame_width) // 2
        
        canvas[y_offset:y_offset+frame_height, x_offset:x_offset+frame_width] = resized_content
        
        if frame_width > frame_height:  
            for y in range(y_offset):  
                canvas[y] = canvas[y_offset] if y % 2 == 0 else canvas[y_offset+1]
            for y in range(y_offset+frame_height, canvas_size):  # 下部bleeding
                canvas[y] = canvas[y_offset+frame_height-2] if y % 2 == 0 else canvas[y_offset+frame_height-1]
        else:  
            for x in range(x_offset):  
                canvas[:, x] = canvas[:, x_offset] if x % 2 == 0 else canvas[:, x_offset+1]
            for x in range(x_offset+frame_width, canvas_size):  
                canvas[:, x] = canvas[:, x_offset+frame_width-2] if x % 2 == 0 else canvas[:, x_offset+frame_width-1]
    else:
        padding_value = 255 if padding_color == 1 else 0
        canvas = np.full((canvas_size, canvas_size, 3), padding_value, dtype=np.uint8)
        y_offset = (canvas_size - frame_height) // 2
        x_offset = (canvas_size - frame_width) // 2
        canvas[y_offset:y_offset+frame_height, x_offset:x_offset+frame_width] = resized_content
    
    return canvas

def save_frame(frame, output_path, index):
    """Save a processed frame"""
    out_fname = os.path.join(output_path, f"{index:08d}.png")
    PIL.Image.fromarray(frame).save(out_fname)

def process_images_batch(args):
    """Process a batch of images in parallel"""
    image_paths, output_path, start_idx, target_ratio, padding_color, resize_mode = args
    count = 0
    for i, image_path in enumerate(image_paths):
        try:
            image = PIL.Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = np.array(image)
            processed_frame = process_frame(image, target_ratio, padding_color, resize_mode)
            save_frame(processed_frame, output_path, start_idx + i)
            count += 1
        except Exception as e:
            print(f'Error processing image {image_path}: {e}')
    return count

def process_video_parallel(video_path, output_path, target_ratio, padding_color, resize_mode, num_workers):
    """Process video frames in parallel"""
    # Get video info
    probe = ffmpeg.probe(video_path)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    
    fps_num, fps_den = map(int, video_info['avg_frame_rate'].split('/'))
    fps = fps_num / fps_den if fps_den != 0 else 0
    duration = float(video_info['duration'])
    total_frames = int(duration * fps)
    
    print(f'Video info: {total_frames} frames @ {fps} fps')
    
    # Create temporary directory for frames
    temp_dir = os.path.join(output_path, 'temp_frames')
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Extract frames using ffmpeg
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(stream, os.path.join(temp_dir, 'frame%d.png'),
                            r=fps, loglevel='error')
        ffmpeg.run(stream, overwrite_output=True)
        
        # Prepare batches for parallel processing
        frame_files = sorted(os.listdir(temp_dir))
        batch_size = max(1, len(frame_files) // (num_workers * 4))  # Divide work into smaller batches
        batches = []
        total_processed = 0
        
        for i in range(0, len(frame_files), batch_size):
            batch_frames = []
            batch_files = frame_files[i:i + batch_size]
            for frame_file in batch_files:
                frame_path = os.path.join(temp_dir, frame_file)
                frame = np.array(PIL.Image.open(frame_path))
                batch_frames.append(frame)
                os.remove(frame_path)  # Remove temporary frame
            
            batches.append((
                batch_frames,
                output_path,
                i,
                target_ratio,
                padding_color,
                resize_mode
            ))
        
        # Process batches in parallel with frame-level progress
        progress_bar = tqdm(
            total=len(frame_files),
            desc="Processing video frames",
            unit="frames",
            dynamic_ncols=True
        )
        
        with mp.Pool(num_workers) as pool:
            for count in pool.imap(process_frames_batch, batches):
                total_processed += count
                progress_bar.update(count)
        
        progress_bar.close()
        return total_processed
    
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def process_gif_parallel(gif_path, output_path, target_ratio, padding_color, resize_mode, num_workers, start_idx=0):
    """Process GIF frames in parallel"""
    gif = PIL.Image.open(gif_path)
    total_frames = gif.n_frames
    print(f'GIF info: {total_frames} frames')
    
    # Prepare batches
    batch_size = max(1, total_frames // (num_workers * 4))
    batches = []
    current_frames = []
    total_processed = 0
    
    for frame_idx in range(total_frames):
        gif.seek(frame_idx)
        frame = gif.convert('RGB')
        current_frames.append(np.array(frame))
        
        if len(current_frames) >= batch_size:
            batches.append((
                current_frames,
                output_path,
                start_idx + len(batches) * batch_size,
                target_ratio,
                padding_color,
                resize_mode
            ))
            current_frames = []
    
    if current_frames:
        batches.append((
            current_frames,
            output_path,
            start_idx + len(batches) * batch_size,
            target_ratio,
            padding_color,
            resize_mode
        ))
    
    # Process batches in parallel with frame-level progress
    progress_bar = tqdm(
        total=total_frames,
        desc="Processing GIF frames",
        unit="frames",
        dynamic_ncols=True
    )
    
    with mp.Pool(num_workers) as pool:
        for count in pool.imap(process_frames_batch, batches):
            total_processed += count
            progress_bar.update(count)
    
    progress_bar.close()
    return total_processed

def process_frames_batch(args):
    """Process a batch of frames in parallel"""
    frames, output_path, start_idx, target_ratio, padding_color, resize_mode = args
    results = []
    for i, frame in enumerate(frames):
        processed = process_frame(frame, target_ratio, padding_color, resize_mode)
        out_fname = os.path.join(output_path, f"{start_idx + i:08d}.png")
        PIL.Image.fromarray(processed).save(out_fname)
    return len(frames)

