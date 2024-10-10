import numpy as np
import librosa
from scipy.ndimage import gaussian_filter1d
import torch

from utils import Deque


class FeatureExtractor(object):

    def __init__(self, n_fft, n_chroma, sr=44100):
        self.n_fft = n_fft
        self.n_chroma = n_chroma
        self.sr = sr

        self.fft_buffer = Deque(2, samples_per_window=n_fft, data_dimensions=2)
        self.fft_smoothing = Deque(5, samples_per_window=n_fft, data_dimensions=2)
        self.chroma_buffer = Deque(2, samples_per_window=n_chroma, data_dimensions=2)
        self.chroma_smoothing = Deque(5, samples_per_window=n_chroma, data_dimensions=2)

        self.lo_rms = 0
        self.hi_rms = 0
        self.counter = 0

        self.lo_grad = 0
        self.hi_grad = 0
        self.onset = False

        self.low_onset_range = (0,150)
        self.high_onset_range = (500,self.n_fft)

    def extract(self, data):
        fft = np.abs(librosa.stft(data, n_fft=self.n_fft * 2 - 1)).T

        self.fft_smoothing.append_data(fft.mean(axis=0))
        
        smoothed_fft = gaussian_filter1d(self.fft_smoothing.get_buffer_data()[self.fft_smoothing.index_order], 6,axis=0)
        self.fft_buffer.append_data(smoothed_fft[-1])

        chroma = librosa.feature.chroma_stft(S=fft.T**2, tuning=True, n_chroma=self.n_chroma)
        chroma /= chroma.sum()
        self.chroma_smoothing.append_data(chroma.T.mean(axis=0))
        smoothed_chroma = gaussian_filter1d(self.chroma_smoothing.get_buffer_data()[self.chroma_smoothing.index_order], 3, axis=0)

        self.chroma_buffer.append_data(smoothed_chroma[-1])


        self.counter += 1
        self.lo_rms += self.fft_buffer.get_most_recent(1)[0][self.low_onset_range[0]:self.low_onset_range[1]].mean()
        self.hi_rms += self.fft_buffer.get_most_recent(1)[0][self.high_onset_range[0]:self.high_onset_range[1]].mean()


