import glob
import multiprocessing
import os
import shutil
import sys

import torch

from modules.autolume_live import Autolume


def get_runtime_bin_dir():
    # PyInstaller frozen app
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return sys._MEIPASS

    # Development mode
    base = os.path.dirname(os.path.abspath(__file__))
    bin_root = os.path.join(base, "bin")
    for root, dirs, files in os.walk(bin_root):
        if "ffmpeg.exe" in files:
            return root
    return bin_root

BIN_DIR = get_runtime_bin_dir()
os.environ["PATH"] = BIN_DIR + os.pathsep + os.environ.get("PATH", "")


def main():
    torch_extension_pattern = 'C:/Users/*/AppData/Local/torch_extensions'
    matches = sorted(glob.glob(torch_extension_pattern))
    if len(matches):
        print("Found torch extension at", matches)
        shutil.rmtree(matches[-1])
        print("Deleted", matches[-1])

    app = Autolume()

    while not app.should_close():
        app.draw_frame()
    app.close()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)
    multiprocessing.set_start_method("spawn", force=True)
    main()
