
import os
import shutil
import cv2
import numpy as np
import PIL.Image
import ffmpeg

def process_non_square_dataset(
    input_path,          # Path to input dataset
    output_path,         # Path to output processed dataset
    crop_ratio,          # (width, height) e.g. (16, 9)
    padding_color=0,     # 0=black padding, 1=white padding, 2=bleeding
    resize_mode="stretch",
):
    """Preprocess dataset to specified aspect ratio with padding to square"""
    print(f'\n=== Starting non-square dataset processing ===')
    print(f'Input path: {input_path}')
    print(f'Output path: {output_path}')
    print(f'Target ratio: {crop_ratio[0]}:{crop_ratio[1]}')
    print(f'Padding color value: {padding_color} (type: {type(padding_color)})')
    print(f'Resize mode: {resize_mode}')

    if padding_color == 2:
        print(f'Padding mode: bleeding')
    else:
        print(f'Padding color: {"white" if padding_color == 1 else "black"}')
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # 支持的文件格式
    image_extensions = ['.png', '.jpg', '.jpeg']
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MOV', '.MP4', '.AVI', '.MKV']
    
    # Collect all files
    files_to_process = []
    if os.path.isdir(input_path):
        for root, _dirs, files in os.walk(input_path):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext in image_extensions + video_extensions + ['.gif']:
                    files_to_process.append(os.path.join(root, fname))
    else:
        raise IOError('Input path must be a directory')
    
    if len(files_to_process) == 0:
        raise IOError('No supported files found in the input path')
    
    print(f'Found {len(files_to_process)} files to process')
    
    # Process each file
    target_ratio = float(crop_ratio[0]) / float(crop_ratio[1])
    if padding_color != 2:
        padding_value = 255 if padding_color == 1 else 0
        print(f'Using padding value: {padding_value}')
    
    frame_count = 0
    for file_idx, fname in enumerate(files_to_process):
        ext = os.path.splitext(fname)[1].lower()
        print(f'\nProcessing file {file_idx+1}/{len(files_to_process)}: {fname}')
        
        if ext in image_extensions:
            # Process single image
            frame_count = process_image(fname, output_path, frame_count, target_ratio, padding_color,resize_mode)
            
        elif ext == '.gif':
            # Process GIF
            try:
                gif = PIL.Image.open(fname)
                print(f'Processing GIF with {gif.n_frames} frames')
                
                for frame_idx in range(gif.n_frames):
                    gif.seek(frame_idx)
                    frame = gif.convert('RGB')
                    frame_array = np.array(frame)
                    processed_frame = process_frame(frame_array, target_ratio, padding_color)
                    save_frame(processed_frame, output_path, frame_count)
                    frame_count += 1
                    if frame_idx % 10 == 0:
                        print(f'Processed {frame_idx+1}/{gif.n_frames} frames')
                        
            except Exception as e:
                print(f'Error processing GIF {fname}: {e}')
                
        elif ext in video_extensions:
            # Process video
            try:
                # Get video info using ffmpeg
                probe = ffmpeg.probe(fname)
                video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
                
                # Get original fps
                if 'avg_frame_rate' in video_info:
                    fps_num, fps_den = map(int, video_info['avg_frame_rate'].split('/'))
                    fps = fps_num / fps_den if fps_den != 0 else 0
                else:
                    fps = eval(video_info['r_frame_rate'])
                
                print(f'Video FPS: {fps}')
                duration = float(video_info['duration'])
                total_frames = int(duration * fps)
                print(f'Total frames to extract: {total_frames}')
                
                # Create temporary directory for frames
                temp_dir = os.path.join(output_path, 'temp_frames')
                os.makedirs(temp_dir, exist_ok=True)
                
                try:
                    # Extract frames using ffmpeg
                    stream = ffmpeg.input(fname)
                    stream = ffmpeg.output(stream, os.path.join(temp_dir, 'frame%d.png'),
                                        r=fps, loglevel='error')
                    ffmpeg.run(stream, overwrite_output=True)
                    
                    # Process each frame
                    frame_files = sorted(os.listdir(temp_dir))
                    for i, frame_file in enumerate(frame_files):
                        frame_path = os.path.join(temp_dir, frame_file)
                        frame = np.array(PIL.Image.open(frame_path))
                        processed_frame = process_frame(frame, target_ratio, padding_color)
                        save_frame(processed_frame, output_path, frame_count)
                        frame_count += 1
                        os.remove(frame_path)  # Remove temporary frame
                        
                        if i % 10 == 0:
                            print(f'Processed {i+1}/{len(frame_files)} frames')
                            
                finally:
                    # Clean up temporary directory
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                        
            except Exception as e:
                print(f'Error processing video {fname}: {e}')
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
    
    print(f'\nDataset preprocessing completed. Total frames processed: {frame_count}')
    return output_path

def process_image(fname, output_path, frame_count, target_ratio, padding_color,resize_mode):
    """Process a single image file"""
    print(f'Loading image...')
    image = PIL.Image.open(fname)
    if image.mode != 'RGB':
        print(f'Converting image from {image.mode} to RGB')
        image = image.convert('RGB')
    image = np.array(image)
    print(f'Original image shape: {image.shape}')
    
    processed_frame = process_frame(image, target_ratio, padding_color, resize_mode)
    save_frame(processed_frame, output_path, frame_count)
    return frame_count + 1

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
        # if current_ratio > target_ratio:
        #     new_width = int(frame.shape[0] * target_ratio)
        #     start_x = (frame.shape[1] - new_width) // 2
        #     frame = frame[:, start_x:start_x+new_width]
        # else:
        #     new_height = int(frame.shape[1] / target_ratio)
        #     start_y = (frame.shape[0] - new_height) // 2
        #     frame = frame[start_y:start_y+new_height, :]
        # resized_content = frame
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

