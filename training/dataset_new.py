# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os

import cv2
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
import ffmpeg

import sys

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
        if list(image.shape) != self.image_shape:
            image = cv2.resize(image.transpose(1,2,0), dsize=self.image_shape[-2:], interpolation=cv2.INTER_CUBIC).transpose(2,0,1)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

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
    def _preprocess_images(self):
        # Initialize an array to store all preprocessed images
        self._image_array = np.zeros((len(self._image_fnames), self.height, 3, self.width), dtype=np.uint8)

        for raw_idx in range(len(self._image_fnames)):
            image = self._load_raw_image(raw_idx)
            image_shape = (3, self.width, self.height) if self.height is not None and self.width is not None else self.image_shape

            if list(image.shape) != image_shape:
                image = cv2.resize(image.transpose(1, 2, 0), dsize=image_shape[-2:], interpolation=cv2.INTER_CUBIC).transpose(2, 0, 1)

            assert list(image.shape) == self.image_shape
            assert image.dtype == np.uint8

            if self._xflip[raw_idx]:
                assert image.ndim == 3 # CHW
                image = image[:, :, ::-1]

            self._image_array[raw_idx] = image.transpose(2, 0, 1)
    
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        height = None,
        width   = None, # Override resolution.
        fps = 10,
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None
        self.height = height
        self.width = width

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        elif self._file_ext(self._path) == '.mp4' or self._file_ext(self._path) == '.avi':
            self._type = 'video'
            self._all_fnames = [self._path]
        else:
            raise IOError('Path must point to a directory or zip')

        found_video = False
        # if any file in self__all_fnames is a video create a new subfolder where we save the frames based on fps using ffmpeg
        for fname in self._all_fnames:
            if fname.endswith('.mp4') or fname.endswith('.avi'):
                found_video = True
                # if self._type is video or zip we have to create a new folder where we save the frames
                if self._type == 'video' or self._type == 'zip':
                    # extract the name of the video
                    video_name = os.path.splitext(fname)[0]
                    save_name = video_name + '_frames'
                    # update self._path to be the new folder which we create based on cwd
                    save_path = os.path.join(os.getcwd(), save_name)
                    # if file exists we add a number to the end of the folder name
                    i = 1
                    while os.path.exists(save_path):
                        save_path = os.path.join(os.getcwd(), save_name + str(i))
                        i += 1
                    # create the folder
                    os.makedirs(save_path)
                    self._path = save_path
                    video_path = os.path.join(fname)
                else:
                    # make dir with the name of the video + _frames
                    video_name = os.path.splitext(fname)[0]
                    save_name = video_name + '_frames'
                    save_path = os.path.join(self._path, save_name)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    # extract frames from video using ffmpeg
                    video_path = os.path.join(self._path, fname)
                cmd = 'ffmpeg -i {} -vf fps={} {}/%04d.jpg'.format(video_path, fps, save_path)
                os.system(cmd)

        # if any of the files were videos we need to update the all_fnames list
        if found_video:
            if os.path.isdir(self._path):
                self._type = 'dir'
                self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files
                                    in os.walk(self._path) for fname in files}
            elif self._file_ext(self._path) == '.zip':
                self._type = 'zip'
                self._all_fnames = set(self._get_zipfile().namelist())
            else:
                raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)

        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        img_shape = [3, self.height,self.width]  if self.width is not None and self.height is not None else list(self._load_raw_image(0).shape)
        raw_shape = [len(self._image_fnames)] + img_shape

        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

        # Check if preprocessed images exist, if not, preprocess and save them
        self._preprocess_images()

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
            # if pyspng is not None and self._file_ext(fname) == '.png':
            #     image = pyspng.load(f.read())
            # else:
            image = PIL.Image.open(f)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = np.array(image)
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
        image = self._image_array[self._raw_idx[idx]]
        image = image.transpose(1, 2, 0)
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        return image.copy(), self.get_label(idx)

    @property
    def resolution(self):
        image_shape = (
        3, self.height, self.width) if self.height is not None and self.width is not None else self.image_shape
        assert len(image_shape) == 3  # CHW
        max_res = calc_res(image_shape[1:])
        return max_res

#----------------------------------------------------------------------------
