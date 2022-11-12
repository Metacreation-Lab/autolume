import multiprocessing
from modules.autolume_live import Autolume
import torch

def main():
    """Interactive model visualizer.

    Optional PATH argument can be used specify which .pkl file to load.
    """

    app = Autolume()

    # Run.
    while not app.should_close():
        app.draw_frame()
    app.close()

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)
    multiprocessing.set_start_method("spawn", force=True)
    main()