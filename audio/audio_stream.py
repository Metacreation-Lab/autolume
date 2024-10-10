from utils.utils import Deque

import pyaudio
import sys
import numpy as np
import time
from functools import partial


class AudioStream(object):

    def __init__(self,
                 pa,
                 device=None,
                 rate=None,
                 buffer_size=6,
                 callback = None
                 ):

        self.rate = rate
        self.buffer_size = buffer_size
        self.pa = pa
        self.callback = callback

        # Temporary variables #hacks!
        self.update_window_n_frames = 1024  # Don't remove this, needed for device testing!
        self.data_buffer = None
        self.stream_start_time = 0

        self.device = device
        if self.device is None:
            self.device = self.input_device()
        if self.rate is None:
            self.rate = self.valid_low_rate(self.device)

        self.info = self.pa.get_device_info_by_index(self.device)

        self.new_data = False

        self.stream = self.pa.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.update_window_n_frames,
            stream_callback=partial(self.non_blocking_stream_read, self.callback)
        )

        print("\n##################################################################################################")
        print("\nDefaulted to using first working mic, Running on:")
        self.print_mic_info(self.device)

    def non_blocking_stream_read(self, func, in_data, frame_count, time_info, status):
        data = np.frombuffer(in_data, dtype=np.float32)
        if self.data_buffer is not None:
            self.data_buffer.append_data(data)
            self.new_data = True
        if func:
            func(data)

        return in_data, pyaudio.paContinue

    def stream_start(self):

        self.data_buffer = Deque(self.buffer_size, self.update_window_n_frames)

        print("\n--🎙  -- Starting live audio stream...\n")
        self.stream.start_stream()
        self.stream_start_time = time.time()

    def terminate(self):
        print("👋  Sending stream termination command...")
        self.stream.stop_stream()
        self.stream.close()

    def valid_low_rate(self, device, test_rates=(44100, 22050)):
        """Set the rate to the lowest supported audio rate."""
        for testrate in test_rates:
            if self.test_device(device, rate=testrate):
                return testrate

        # If none of the test_rates worked, try the default rate:
        self.info = self.pa.get_device_info_by_index(device)
        default_rate = int(self.info["defaultSampleRate"])

        if self.test_device(device, rate=default_rate):
            return default_rate

        print("SOMETHING'S WRONG! I can't figure out a good sample-rate for DEVICE =>", device)
        return default_rate

    def test_device(self, device, rate=None):
        """given a device ID and a rate, return True/False if it's valid."""
        try:
            self.info = self.pa.get_device_info_by_index(device)
            if not self.info["maxInputChannels"] > 0:
                return False

            if rate is None:
                rate = int(self.info["defaultSampleRate"])

            stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=1,
                input_device_index=device,
                frames_per_buffer=self.update_window_n_frames,
                rate=rate,
                input=True)
            stream.close()
            return True
        except Exception as e:
            # print(e)
            return False

    def input_device(self):
        """
        See which devices can be opened for microphone input.
        Return the first valid device
        """
        mics = []
        for device in range(self.pa.get_device_count()):
            if self.test_device(device):
                mics.append(device)

        if len(mics) == 0:
            print("No working microphone devices found!")
            sys.exit()

        print("Found %d working microphone device(s): " % len(mics))
        for mic in mics:
            self.print_mic_info(mic)

        return mics[0]

    def print_mic_info(self, mic):
        mic_info = self.pa.get_device_info_by_index(mic)
        print('\nMIC %s:' % (str(mic)))
        for k, v in sorted(mic_info.items()):
            print("%s: %s" % (k, v))
