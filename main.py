import glob
import shutil
import multiprocessing
from modules.autolume_live import Autolume
from torch_utils.ops import params
import torch



def main():
    """Interactive model visualizer.

    Optional PATH argument can be used specify which .pkl file to load.
    """
    torch_extension_pattern = 'C:/Users/*/AppData/Local/torch_extensions'
    
    matches = sorted(glob.glob(torch_extension_pattern))
    if len(matches):
        print("Found torch extension at", matches)
        shutil.rmtree(matches[-1])
        print("Deleted", matches[-1])

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