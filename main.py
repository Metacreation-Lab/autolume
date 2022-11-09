import multiprocessing
from modules.autolume_live import Autolume

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
    multiprocessing.set_start_method("spawn", force=True)
    main()