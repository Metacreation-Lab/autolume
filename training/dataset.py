# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import shutil


import cv2
import numpy as np
import zipfile
import PIL.Image
import torchvision.transforms
import json
import torch
import dnnlib
import ffmpeg
import traceback

try:
    import pyspng
except ImportError:
    pyspng = None


def calc_res(shape):
    base0 = 2 ** int(np.log2(shape[0]))
    base1 = 2 ** int(np.log2(shape[1]))
    base = min(base0, base1)
    min_res = min(shape[0], shape[1])

    def int_log2(xs, base):
        return [x * 2 ** (2 - int(np.log2(base))) % 1 == 0 for x in xs]

    if min_res != base or max(*shape) / min(*shape) >= 2:
        if np.log2(base) < 10 and all(int_log2(shape, base * 2)):
            base = base * 2

    return base  # , [shape[0]/base, shape[1]/base]

def calc_init_res(shape, resolution=None):
    if len(shape) == 1:
        shape = [shape[0], shape[0], 1]
    elif len(shape) == 2:
        shape = [*shape, 1]
    size = shape[:2] if shape[2] < min(*shape[:2]) else shape[1:] # fewer colors than pixels
    if resolution is None:
        resolution = calc_res(size)
    res_log2 = int(np.log2(resolution))
    init_res = [int(s * 2**(2-res_log2)) for s in size]
    return init_res, resolution, res_log2

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)

        if image.shape[0] == 1:
            image = np.repeat(image, 3, axis=0)

        if image.shape[0] == 4:
            image = image[:3, :, :]
        image_shape = (3, self.width, self.height) if self.height is not None and self.width is not None else self.image_shape
        
        if list(image.shape) != self.image_shape:
            image = cv2.resize(image.transpose(1,2,0), dsize=self.image_shape[-2:], interpolation=cv2.INTER_CUBIC).transpose(2,0,1)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

        # if list(image.shape) != image_shape:
        #     if self.resize_mode == "stretch":
        #         # 检查图像是否包含padding(通过检查边缘像素值是否为纯黑或纯白)
        #         img_transposed = image.transpose(1, 2, 0)
        #         is_padded = False
                
        #         # 检查上下左右边缘是否存在padding
        #         edges = [
        #             img_transposed[0, :],  # top
        #             img_transposed[-1, :],  # bottom
        #             img_transposed[:, 0],  # left
        #             img_transposed[:, -1]   # right
        #         ]
                
        #         for edge in edges:
        #             if np.all(edge == 0) or np.all(edge == 255):
        #                 is_padded = True
        #                 break
                
        #         if is_padded:
        #             mask = np.any(img_transposed != 0, axis=2) & np.any(img_transposed != 255, axis=2)
        #             rows = np.any(mask, axis=1)
        #             cols = np.any(mask, axis=0)
        #             y_min, y_max = np.where(rows)[0][[0, -1]]
        #             x_min, x_max = np.where(cols)[0][[0, -1]]
                    
        #             content = img_transposed[y_min:y_max+1, x_min:x_max+1]
        #             resized_content = cv2.resize(content, 
        #                                     dsize=(image_shape[2], image_shape[1]), 
        #                                     interpolation=cv2.INTER_CUBIC)
                    
        #             padding_value = 0 if np.all(img_transposed[0,0] == 0) else 255
        #             result = np.full((image_shape[1], image_shape[2], 3), padding_value, dtype=np.uint8)
                    
        #             y_start = (image_shape[1] - resized_content.shape[0]) // 2
        #             x_start = (image_shape[2] - resized_content.shape[1]) // 2
        #             result[y_start:y_start+resized_content.shape[0], 
        #                 x_start:x_start+resized_content.shape[1]] = resized_content
                    
        #             image = result.transpose(2, 0, 1)
        #         else:
        #             image = cv2.resize(image.transpose(1,2,0), 
        #                             dsize=image_shape[-2:], 
        #                             interpolation=cv2.INTER_CUBIC).transpose(2,0,1)
        #     else:
        #         image = image.transpose(1, 2, 0)
        #         pil_image = PIL.Image.fromarray(image.astype(np.uint8))
        #         resize_transform = torchvision.transforms.Resize(min(self.height, self.width))
        #         resized_image = resize_transform(pil_image)
        #         crop_transform = torchvision.transforms.CenterCrop((self.height, self.width))
        #         cropped_image = crop_transform(resized_image)
        #         image = np.array(cropped_image)
        #         image = image.transpose(2,0,1)
                
        # assert list(image.shape) == self.image_shape
        # assert image.dtype == np.uint8
        # if self._xflip[idx]:
        #     assert image.ndim == 3 # CHW
        #     image = image[:, :, ::-1]
        # return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        max_res = calc_res(self.image_shape[1:])
        return max_res

    # !!! custom init res
    @property
    def res_log2(self):
        return int(np.ceil(np.log2(self.resolution)))

    # !!! custom init res
    @property
    def init_res(self):
        return [int(s * 2 ** (2 - self.res_log2)) for s in self.image_shape[1:]]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    # def __init__(self,
    #     path,                   # Path to directory or zip.
    #     resolution      = None, # Ensure specific resolution, None = highest available.
    #     height = None,
    #     width   = None, # Override resolution.
    #     resize_mode = "stretch",
    #     fps = 10,
    #     **super_kwargs,         # Additional arguments for the Dataset base class.
    # ):
    #     self._path = path
    #     self._zipfile = None
    #     self.height = height
    #     self.width = width
    #     self.resize_mode = resize_mode
    #     # self.has_frames_folder = False
    #     self.frame_path = set()

    #     if os.path.isdir(self._path):
    #         self._type = 'dir'
    #         self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
    #     elif self._file_ext(self._path) == '.zip':
    #         self._type = 'zip'
    #         self._all_fnames = set(self._get_zipfile().namelist())
    #     # elif self._file_ext(self._path) == '.mp4' or self._file_ext(self._path) == '.avi':
    #     elif self._file_ext(self._path) in ['.mp4', '.avi', '.gif']:
    #         self._type = 'video'
    #         self._all_fnames = [self._path]
    #     else:
    #         raise IOError('Path must point to a directory or zip')

    #     found_video = False
    #     # if any file in self__all_fnames is a video create a new subfolder where we save the frames based on fps using ffmpeg
    #     for fname in self._all_fnames:
    #         # if fname.endswith('.mp4') or fname.endswith('.avi'):
    #         if fname.endswith(('.mp4', '.avi', '.gif')):
    #             found_video = True
    #             # if self._type is video or zip we have to create a new folder where we save the frames
    #             if self._type == 'video' or self._type == 'zip':
    #                 # extract the name of the video
    #                 video_name = os.path.splitext(fname)[0]
    #                 save_name = video_name + '_frames'
    #                 # update self._path to be the new folder which we create based on cwd
    #                 save_path = os.path.join(os.getcwd(), save_name)
    #                 # if file exists we add a number to the end of the folder name
    #                 i = 1
    #                 while os.path.exists(save_path):
    #                     save_path = os.path.join(os.getcwd(), save_name + str(i))
    #                     i += 1
    #                 # create the folder
    #                 os.makedirs(save_path)
    #                 self._path = save_path
    #                 video_path = os.path.join(fname)
    #                 # self.frame_path.add(save_path)
    #             else:
    #                 # make dir with the name of the video + _frames
    #                 video_name = os.path.splitext(fname)[0]
    #                 save_name = video_name + '_frames'
    #                 save_path = os.path.join(self._path, save_name)
    #                 # self.has_frames_folder = True
    #                 if not os.path.exists(save_path):
    #                     os.makedirs(save_path)
    #                 # extract frames from video using ffmpeg
    #                 video_path = os.path.join(self._path, fname)
    #             cmd = 'ffmpeg -i {} -vf fps={} {}/%04d.jpg'.format(video_path, fps, save_path)
    #             self.frame_path.add(save_path)
    #             os.system(cmd)

    #     # if any of the files were videos we need to update the all_fnames list
    #     if found_video:
    #         if os.path.isdir(self._path):
    #             self._type = 'dir'
    #             self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files
    #                                 in os.walk(self._path) for fname in files}
    #         elif self._file_ext(self._path) == '.zip':
    #             self._type = 'zip'
    #             self._all_fnames = set(self._get_zipfile().namelist())
    #         else:
    #             raise IOError('Path must point to a directory or zip')





    #     PIL.Image.init()
    #     self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)

    #     if len(self._image_fnames) == 0:
    #         raise IOError('No image files found in the specified path')

    #     name = os.path.splitext(os.path.basename(self._path))[0]
    #     img_shape = [3, self.height,self.width]  if self.width is not None and self.height is not None else list(self._load_raw_image(0).shape)
    #     raw_shape = [len(self._image_fnames)] + img_shape
    #     super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        height = None,
        width   = None, # Override resolution.
        resize_mode = "stretch",
        fps = 10,
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = os.path.abspath(path)
        self._zipfile = None
        self.height = height
        self.width = width
        self.resize_mode = resize_mode
        self.frame_path = set()
        
        if not os.path.exists(self._path):
            raise IOError(f'Path does not exist: {self._path}')
            
        if os.path.isdir(self._path):
            self._type = 'dir'
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._zipfile = zipfile.ZipFile(self._path)
        else:
            raise IOError('Path must point to a directory or zip')

        video_files = []
        if self._type == 'dir':
            for root, _, files in os.walk(self._path):
                for fname in files:
                    if fname.endswith(('.mp4', '.avi', '.gif','.MOV','.mov','.mkv')):
                        video_files.append(os.path.join(root, fname))
        elif self._type == 'zip':
            for fname in self._zipfile.namelist():
                if fname.endswith(('.mp4', '.avi', '.gif','.MOV','.mov','.mkv')):
                    video_files.append(fname)

        frames_extracted = False
        if video_files:
            print(f"Found {len(video_files)} video file(s), extracting frames...")
            for video_path in video_files:
                try:
                    if self._type == 'zip':
                        # 对于zip文件中的视频，先解压到临时目录
                        temp_dir = tempfile.mkdtemp()
                        video_data = self._zipfile.read(video_path)
                        temp_video_path = os.path.join(temp_dir, os.path.basename(video_path))
                        with open(temp_video_path, 'wb') as f:
                            f.write(video_data)
                        video_path = temp_video_path

                    video_name = os.path.splitext(os.path.basename(video_path))[0]
                    save_path = os.path.join(self._path, f"{video_name}_frames")
                    
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    
                    cmd = f'ffmpeg -i "{video_path}" -vf fps={fps} "{save_path}/%04d.jpg"'
                    print(f"Executing command: {cmd}")
                    result = os.system(cmd)
                    
                    if result == 0:
                        print(f"Successfully extracted frames from {video_path}")
                        self.frame_path.add(save_path)
                        frames_extracted = True
                        
                        extracted_frames = [f for f in os.listdir(save_path) if f.endswith(('.jpg', '.png'))]
                        if not extracted_frames:
                            print(f"Warning: No frames were extracted from {video_path}")
                        else:
                            print(f"Extracted {len(extracted_frames)} frames from {video_path}")
                    else:
                        print(f"Failed to extract frames from {video_path}, ffmpeg returned {result}")
                        
                except Exception as e:
                    print(f"Error processing video {video_path}: {str(e)}")
                    traceback.print_exc()
                finally:
                    if self._type == 'zip' and 'temp_dir' in locals():
                        import shutil
                        shutil.rmtree(temp_dir)

        if self._type == 'dir':
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) 
                               for root, _dirs, files in os.walk(self._path) 
                               for fname in files}
        else:  # zip
            self._all_fnames = set(self._zipfile.namelist())

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames 
                                  if self._file_ext(fname) in PIL.Image.EXTENSION)
        
        if len(self._image_fnames) == 0:
            if video_files:
                if frames_extracted:
                    raise IOError('Failed to find any extracted frames after processing videos')
                else:
                    raise IOError('Failed to extract frames from any of the videos')
            else:
                raise IOError('No image files found in the specified path')

        print(f"Found {len(self._image_fnames)} image files")
        
        name = os.path.splitext(os.path.basename(self._path))[0]
        img_shape = [3, self.height, self.width] if self.width is not None and self.height is not None else list(self._load_raw_image(0).shape)
        raw_shape = [len(self._image_fnames)] + img_shape
        
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None
    


    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

    @property
    def heightandwidth(self):
        return self._load_raw_image(0).shape[1:]

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        if image.shape[0] == 1:
            image = np.repeat(image, 3, axis=0)
        if image.shape[0] == 4:
            image = image[:3, :, :]
        image_shape = (3, self.width, self.height) if self.height is not None and self.width is not None else self.image_shape
        if list(image.shape) != image_shape:
            if self.resize_mode == "stretch":
                image = cv2.resize(image.transpose(1,2,0), dsize=image_shape[-2:], interpolation=cv2.INTER_CUBIC).transpose(2,0,1)
            else:
                image = image.transpose(1, 2, 0)
                pil_image = PIL.Image.fromarray(image.astype(np.uint8))  # Convert NumPy array to PIL Image
                resize_transform = torchvision.transforms.Resize(min(self.height, self.width))  # 先等比例缩放
                resized_image = resize_transform(pil_image)  # 应用 Resize 变换
                crop_transform  = torchvision.transforms.CenterCrop((self.height, self.width))  # Target size
                cropped_image = crop_transform(resized_image )  # Perform the center crop
                image = np.array(cropped_image)  # Convert back to NumPy array
                image = image.transpose(2,0,1)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)
    
    def save_resized(self, path):
        for idx in np.arange(self.__len__()):
            img, label = self.__getitem__(idx)
            img = PIL.Image.fromarray(img.astype(np.uint8).transpose(1,2,0), 'RGB')
            if not os.path.exists(path+str('/resized_images')):
                os.mkdir(path+str('/resized_images'))
            img.save(path+str('/resized_images/')+str(idx)+'.png', 'PNG')
    

    # def move_frames_folders(self, output_dir):
    #     for frame_path in self.frame_path:  # 遍历所有帧文件夹路径
    #         if os.path.exists(frame_path):
    #             new_path = os.path.join(output_dir, os.path.basename(frame_path))
    #             os.rename(frame_path, new_path)  # 移动文件夹
    #             print(f"Moved frames folder to: {new_path}")

    def copy_frames_folders(self, output_dir):
        for frame_path in self.frame_path:  # 遍历所有帧文件夹路径
            if os.path.exists(frame_path):
                new_path = os.path.join(output_dir, os.path.basename(frame_path))
                print(f"Copying folder: {frame_path} to {new_path}")
                try:
                    shutil.copytree(frame_path, new_path, dirs_exist_ok=True)  # 复制文件夹，允许目标目录已存在
                    print(f"Copied frames folder to: {new_path}")
                except Exception as e:
                    print(f"Failed to copy {frame_path} to {new_path}: {e}")
            else:
                print(f"Frame folder does not exist: {frame_path}")

    @property
    def resolution(self):
        image_shape = (
        3, self.height, self.width) if self.height is not None and self.width is not None else self.image_shape
        assert len(image_shape) == 3  # CHW
        max_res = calc_res(image_shape[1:])
        return max_res

#----------------------------------------------------------------------------

# def process_non_square_dataset(
#     input_path,           # Path to input dataset
#     output_path,         # Path to output processed dataset
#     crop_ratio,          # (width, height) e.g. (16, 9)
#     padding_color=0,     # 0=black padding, 1=white padding
# ):
#     """Preprocess dataset to specified aspect ratio with padding to square"""
#     print(f'Starting non-square dataset processing...')
#     print(f'Input path: {input_path}')
#     print(f'Output path: {output_path}')
#     print(f'Target ratio: {crop_ratio[0]}:{crop_ratio[1]}')
#     print(f'Padding color: {"white" if padding_color == 1 else "black"}')
    
#     # Create output directory
#     os.makedirs(output_path, exist_ok=True)
    
#     # Collect all image files
#     image_fnames = []
#     if os.path.isdir(input_path):
#         for root, _dirs, files in os.walk(input_path):
#             for fname in files:
#                 if os.path.splitext(fname)[1].lower() in ['.png', '.jpg', '.jpeg']:
#                     image_fnames.append(os.path.join(root, fname))
#     else:
#         raise IOError('Input path must be a directory')
    
#     if len(image_fnames) == 0:
#         raise IOError('No image files found in the input path')
    
#     # Process each image
#     target_ratio = float(crop_ratio[0]) / float(crop_ratio[1])
#     padding_value = 255 if padding_color == 1 else 0  # Set padding color
    
#     print(f'Using padding value: {padding_value} ({"white" if padding_value == 255 else "black"})')
    
#     for idx, fname in enumerate(image_fnames):
#         out_fname = os.path.join(output_path, f"{idx:08d}.png")
        
#         # Load image
#         image = PIL.Image.open(fname)
#         if image.mode != 'RGB':
#             image = image.convert('RGB')
#         image = np.array(image)
        
#         # Calculate current ratio
#         current_ratio = image.shape[1] / image.shape[0]
        
#         # Crop to target ratio
#         if current_ratio > target_ratio:
#             # Image too wide, crop sides
#             new_width = int(image.shape[0] * target_ratio)
#             start_x = (image.shape[1] - new_width) // 2
#             image = image[:, start_x:start_x+new_width]
#         else:
#             # Image too tall, crop top/bottom
#             new_height = int(image.shape[1] / target_ratio)
#             start_y = (image.shape[0] - new_height) // 2
#             image = image[start_y:start_y+new_height, :]
        
#         # Create square canvas with padding
#         target_size = max(image.shape)
#         canvas = np.full((target_size, target_size, 3), padding_value, dtype=np.uint8)
        
#         # Place cropped image in center of canvas
#         y_offset = (target_size - image.shape[0]) // 2
#         x_offset = (target_size - image.shape[1]) // 2
#         canvas[y_offset:y_offset+image.shape[0], x_offset:x_offset+image.shape[1]] = image
        
#         # Save processed image
#         PIL.Image.fromarray(canvas).save(out_fname)
        
#         if idx % 10 == 0:
#             print(f'Processed {idx+1}/{len(image_fnames)} images')
    
#     print('Dataset preprocessing completed successfully.')
#     return output_path

# def process_non_square_dataset(
#     input_path,           # Path to input dataset
#     output_path,         # Path to output processed dataset
#     crop_ratio,          # (width, height) e.g. (16, 9)
#     padding_color=0,     # 0=black padding, 1=white padding, 2=bleeding
# ):
#     """Preprocess dataset to specified aspect ratio with padding to square"""
#     print(f'\n=== Starting non-square dataset processing ===')
#     print(f'Input path: {input_path}')
#     print(f'Output path: {output_path}')
#     print(f'Target ratio: {crop_ratio[0]}:{crop_ratio[1]}')
#     print(f'Padding color value: {padding_color} (type: {type(padding_color)})')
    
#     if padding_color == 2:
#         print(f'Padding mode: bleeding')
#     else:
#         print(f'Padding color: {"white" if padding_color == 1 else "black"}')
    
#     # Create output directory
#     os.makedirs(output_path, exist_ok=True)
    
#     # Collect all image files
#     image_fnames = []
#     if os.path.isdir(input_path):
#         for root, _dirs, files in os.walk(input_path):
#             for fname in files:
#                 if os.path.splitext(fname)[1].lower() in ['.png', '.jpg', '.jpeg']:
#                     image_fnames.append(os.path.join(root, fname))
#     else:
#         raise IOError('Input path must be a directory')
    
#     if len(image_fnames) == 0:
#         raise IOError('No image files found in the input path')
    
#     print(f'Found {len(image_fnames)} images to process')
    
#     # Process each image
#     target_ratio = float(crop_ratio[0]) / float(crop_ratio[1])
#     print(f'Target aspect ratio: {target_ratio:.4f}')
    
#     if padding_color != 2:
#         padding_value = 255 if padding_color == 1 else 0
#         print(f'Using padding value: {padding_value} ({"white" if padding_value == 255 else "black"})')
    
#     for idx, fname in enumerate(image_fnames):
#         print(f'\nProcessing image {idx+1}/{len(image_fnames)}: {fname}')
#         out_fname = os.path.join(output_path, f"{idx:08d}.png")
        
#         # Load image
#         print(f'Loading image...')
#         image = PIL.Image.open(fname)
#         if image.mode != 'RGB':
#             print(f'Converting image from {image.mode} to RGB')
#             image = image.convert('RGB')
#         image = np.array(image)
#         print(f'Original image shape: {image.shape}')
        
#         # Calculate current ratio
#         current_ratio = image.shape[1] / image.shape[0]
#         print(f'Current aspect ratio: {current_ratio:.4f}')
        
#         # Crop to target ratio
#         if current_ratio > target_ratio:
#             print('Image too wide, cropping sides')
#             new_width = int(image.shape[0] * target_ratio)
#             start_x = (image.shape[1] - new_width) // 2
#             image = image[:, start_x:start_x+new_width]
#             print(f'Cropped image shape: {image.shape}')
#         else:
#             print('Image too tall, cropping top/bottom')
#             new_height = int(image.shape[1] / target_ratio)
#             start_y = (image.shape[0] - new_height) // 2
#             image = image[start_y:start_y+new_height, :]
#             print(f'Cropped image shape: {image.shape}')
        
#         # Create square canvas
#         target_size = max(image.shape)
#         print(f'Target square size: {target_size}')
        
#         if padding_color == 2:  # Bleeding mode
#             print('Applying bleeding mode padding')
#             # Create initial canvas
#             canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
#             print(f'Created canvas shape: {canvas.shape}')
            
#             # Place cropped image in center
#             y_offset = (target_size - image.shape[0]) // 2
#             x_offset = (target_size - image.shape[1]) // 2
#             print(f'Image placement offsets - x: {x_offset}, y: {y_offset}')
            
#             canvas[y_offset:y_offset+image.shape[0], x_offset:x_offset+image.shape[1]] = image
#             print('Placed image in canvas')
            
#             cropped_ratio = image.shape[1] / image.shape[0]
#             print(f'Cropped image ratio: {cropped_ratio:.4f}')

#             if cropped_ratio > target_ratio:
#                 print('Applying vertical bleeding effect')
#                 # 需要填充上下
#                 for y in range(y_offset):
#                     canvas[y] = image[0] if y % 2 == 0 else image[1]
                
#                 for y in range(y_offset+image.shape[0], target_size):
#                     canvas[y] = image[-2] if y % 2 == 0 else image[-1]
#                 print('Applied vertical bleeding')
#             else:
#                 print('Applying horizontal bleeding effect')
#                 # 需要填充左右
#                 for x in range(x_offset):
#                     canvas[:, x] = image[:, 0] if x % 2 == 0 else image[:, 1]
                
#                 for x in range(x_offset+image.shape[1], target_size):
#                     canvas[:, x] = image[:, -2] if x % 2 == 0 else image[:, -1]
#                 print('Applied horizontal bleeding')
#         else:
#             print(f'Applying regular padding with value {padding_value}')
#             # Original black/white padding
#             canvas = np.full((target_size, target_size, 3), padding_value, dtype=np.uint8)
#             y_offset = (target_size - image.shape[0]) // 2
#             x_offset = (target_size - image.shape[1]) // 2
#             canvas[y_offset:y_offset+image.shape[0], x_offset:x_offset+image.shape[1]] = image
        
#         print(f'Final canvas shape: {canvas.shape}')
#         print(f'Canvas value range: min={canvas.min()}, max={canvas.max()}')
        
#         # Save processed image
#         PIL.Image.fromarray(canvas).save(out_fname)
#         print(f'Saved processed image to: {out_fname}')
        
#         if idx % 10 == 0:
#             print(f'\nProgress: {idx+1}/{len(image_fnames)} images processed')
    
#     print('\nDataset preprocessing completed successfully.')
#     return output_path


def process_non_square_dataset(
    input_path,           # Path to input dataset
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

