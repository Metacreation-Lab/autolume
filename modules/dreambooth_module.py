import os
import multiprocessing as mp
import imgui
import dnnlib
from dnnlib import EasyDict
from utils.gui_utils import imgui_utils
from streamdiffusion.train_dreambooth import main as train_main
from widgets.browse_widget import BrowseWidget


class DreamboothModule:
    def __init__(self, menu):
        cwd = os.getcwd()
        self.args = EasyDict()


        if not os.path.exists(os.path.abspath(os.path.join(os.getcwd(), "data"))):
            os.makedirs(os.path.abspath(os.path.join(os.getcwd(), "data")))

        self.training_active = False
        self.progress_message = ""
        self.current_step = 0
        self.total_steps = 0

        self.use_model_name = True
        # self.model_path_dialog = BrowseWidget(self, "Browse", os.path.abspath(os.getcwd()),
        #                                       ["*"], multiple=False, traverse_folders=False,
        #                                       width=menu.app.button_w)
        self.model_path_dialog = BrowseWidget(self, "Select Model", os.path.abspath(os.getcwd()),
                                              ["*"], multiple=False, traverse_folders=False,
                                              width=menu.app.button_w)

        self.dataset_dir_dialog = BrowseWidget(menu, "Class Data Dir", os.path.abspath(os.path.join(os.getcwd(), "data")),
                                               ["*", ""],
                                               multiple=False, traverse_folders=False, width=menu.app.button_w)
        self.app = menu.app

        self.output_dir_dialog = BrowseWidget(self, "Save Path", os.path.abspath(os.getcwd()), [""], multiple=False,
                                              traverse_folders=False, add_folder_button=True, width=menu.app.button_w)

        self.instance_data_dir_dialog = BrowseWidget(self, "Instance Data Dir", os.path.abspath(os.getcwd()), [""], multiple=False,
                                              traverse_folders=False, add_folder_button=True, width=menu.app.button_w)

        self.menu = menu
        self.predefined_configs = {
            "16gb_gpu": {
                "pretrained_model_name_or_path": "CompVis/stable-diffusion-v1-4",
                "instance_data_dir": "",
                "class_data_dir": "",
                "output_dir": os.path.join(cwd, "training-runs"),
                "with_prior_preservation": True,
                "prior_loss_weight": 1.0,
                "instance_prompt": "a photo of sks dog",
                "class_prompt": "a photo of dog",
                "resolution": 512,
                "train_batch_size": 1,
                "gradient_accumulation_steps": 2,
                "gradient_checkpointing": True,
                "use_8bit_adam": True,
                "learning_rate": 0.005,
                "lr_scheduler": "constant",
                "lr_warmup_steps": 0,
                "num_class_images": 200,
                "max_train_steps": 800,
            },
            "12gb_gpu": {
                "pretrained_model_name_or_path": "CompVis/stable-diffusion-v1-4",
                "instance_data_dir": "path/to/instance/data",
                "class_data_dir": "path-to-class-images",
                "output_dir": os.path.join(cwd, "training-runs"),
                "with_prior_preservation": True,
                "prior_loss_weight": 1.0,
                "instance_prompt": "a photo of sks dog",
                "class_prompt": "a photo of dog",
                "resolution": 512,
                "train_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "gradient_checkpointing": True,
                "use_8bit_adam": True,
                "enable_xformers_memory_efficient_attention": True,
                "set_grads_to_none": True,
                "learning_rate": 2e-6,
                "lr_scheduler": "constant",
                "lr_warmup_steps": 0,
                "num_class_images": 200,
                "max_train_steps": 800,
            }
        }
        self.queue = mp.Queue()
        self.reply = mp.Queue()
        self.message = ""
        self.done = False
        self.training_process = None
        self.current_config = "16gb_gpu"
        self.args = EasyDict(self.predefined_configs[self.current_config])
        # self.args.report_to = "tensorboard"  # Default to tensorboard



    def update_training_status(self):
        updated = False
        while self.reply.qsize() > 0:
            self.message, self.done = self.reply.get()
            self.training_active = not self.done
            updated = True
            print(f"Received update: {self.message}, done: {self.done}")  # Debug print
        return updated

    def stop_training(self):
        if hasattr(self, 'training_process') and self.training_process.is_alive():
            self.training_process.terminate()
            self.training_process.join()
        self.training_active = False
        self.message = "Training stopped by user"

    @imgui_utils.scoped_by_object_id
    def __call__(self):
        if self.reply.qsize() > 0:
            self.message, self.done = self.reply.get()
            while self.reply.qsize() > 0:
                self.message, self.done = self.reply.get()
            print(self.message, self.done)

        # Configuration selection
        config_options = list(self.predefined_configs.keys())
        with imgui_utils.item_width(-(self.app.button_w + self.app.spacing) * 1.7):
            _, selected_config = imgui.combo(
                "GPU Configuration",
                config_options.index(self.current_config),
                config_options
            )

        if config_options[selected_config] != self.current_config:
            self.current_config = config_options[selected_config]
            self.args = EasyDict(self.predefined_configs[self.current_config])
            print(f"Selected configuration: {self.current_config}")

        # Model selection
        changed, self.args.pretrained_model_name_or_path = imgui_utils.input_text(
            "##SRModel",
            self.args.pretrained_model_name_or_path,
            1024,
            flags=0,  # No special flags
            width=-(self.app.button_w + self.app.spacing) * 1.7,
            help_text="Model Path or Name"
        )
        imgui.same_line()
        _clicked, model_path = self.model_path_dialog(self.app.button_w)
        if _clicked:
            if len(model_path) > 0:
                self.args.pretrained_model_name_or_path = model_path[0]
                print(f"Selected model path: {self.args.pretrained_model_name_or_path}")
            else:
                print("No model path selected")

        # Output directory selection
        imgui_utils.input_text(
            "##SROUTPUT PATH",
            self.args.output_dir,
            1024,
            flags=imgui.INPUT_TEXT_READ_ONLY,
            width=-(self.app.button_w + self.app.spacing) * 1.7,
            help_text="Output Path"
        )
        imgui.same_line()
        _clicked, save_path = self.output_dir_dialog(self.app.button_w)
        if _clicked:
            if len(save_path) > 0:
                self.args.output_dir = save_path[0]
                print(f"Selected output directory: {self.args.output_dir}")
            else:
                self.args.output_dir = ""
                print("No output path selected")


        # Instance Data Directory selection
        imgui_utils.input_text(
            "##SRInstance Data Directory",
            self.args.instance_data_dir,
            1024,
            flags=imgui.INPUT_TEXT_READ_ONLY,
            width=-(self.app.button_w + self.app.spacing) * 1.7,
            help_text="Instance Data Directory"
        )
        imgui.same_line()
        _clicked, instance_data_dir = self.instance_data_dir_dialog(self.app.button_w)
        if _clicked:
            if len(instance_data_dir) > 0:
                self.args.instance_data_dir = instance_data_dir[0]
                print(f"Selected Instance Data Directory: {self.args.instance_data_dir}")
            else:
                self.args.instance_data_dir = ""
                print("No Instance Data Directory selected")

        # class data directory selection
        imgui_utils.input_text(
            "##SRclass data directory",
            self.args.class_data_dir,
            1024,
            flags=imgui.INPUT_TEXT_READ_ONLY,
            width=-(self.app.button_w + self.app.spacing) *1.7,
            help_text="Class Data Directory"
        )
        imgui.same_line()
        _clicked, class_data_dir = self.dataset_dir_dialog(self.app.button_w)
        if _clicked:
            if len(class_data_dir) > 0:
                self.args.class_data_dir = class_data_dir[0]
                print(f"Selected Instance Data Directory: {self.args.class_data_dir}")
            else:
                self.args.class_data_dir = ""
                print("No Instance Data Directory selected")

        # Training parameters
        with imgui_utils.item_width(-(self.app.button_w + self.app.spacing) *1.7):
            _, self.args.instance_prompt = imgui.input_text("Instance Prompt", self.args.instance_prompt, 1024)
            _, self.args.resolution = imgui.input_int("Resolution", self.args.resolution)
            _, self.args.train_batch_size = imgui.input_int("Train Batch Size", self.args.train_batch_size)

            _, self.args.learning_rate = imgui.input_float("Learning Rate", self.args.learning_rate, format="%.10f")
            _, self.args.max_train_steps = imgui.input_int("Max Train Steps", self.args.max_train_steps)

            lr_scheduler_options = ["constant", "linear", "cosine", "cosine_with_restarts", "polynomial",
                                    "constant_with_warmup"]
            _, selected_lr_scheduler = imgui.combo("LR Scheduler", lr_scheduler_options.index(self.args.lr_scheduler),
                                                   lr_scheduler_options)
            self.args.lr_scheduler = lr_scheduler_options[selected_lr_scheduler]

            _, self.args.lr_warmup_steps = imgui.input_int("LR Warmup Steps", self.args.lr_warmup_steps)
        with imgui_utils.item_width(-(self.app.button_w + self.app.spacing) * 1.9):
            _, self.args.gradient_accumulation_steps = imgui.input_int("Gradient Accumulation Steps",
                                                                   self.args.gradient_accumulation_steps)
        # Advanced options
        _, self.args.with_prior_preservation = imgui.checkbox("With Prior Preservation",
                                                              self.args.get("with_prior_preservation", False))
        if self.args.with_prior_preservation:
            _, self.args.prior_loss_weight = imgui.input_float("Prior Loss Weight",
                                                               self.args.get("prior_loss_weight", 1.0))
            _, self.args.class_prompt = imgui.input_text("Class Prompt", self.args.get("class_prompt", ""), 1024)
            _, self.args.num_class_images = imgui.input_int("Number of Class Images",
                                                            self.args.get("num_class_images", 200))

        _, self.args.use_8bit_adam = imgui.checkbox("Use 8-bit Adam", self.args.get("use_8bit_adam", False))
        imgui.same_line()
        _, self.args.gradient_checkpointing = imgui.checkbox("Gradient Checkpointing",
                                                             self.args.get("gradient_checkpointing", False))
        _, self.args.enable_xformers_memory_efficient_attention = imgui.checkbox(
            "Enable xFormers Memory Efficient Attention",
            self.args.get("enable_xformers_memory_efficient_attention", False))
        _, self.args.set_grads_to_none = imgui.checkbox("Set Grads to None", self.args.get("set_grads_to_none", False))

        # Training popup
        imgui.set_next_window_size(self.menu.app.content_width // 2, (self.menu.app.content_height // 2), imgui.ONCE)
        if imgui.begin_popup_modal("Training")[0]:
            # Check for updates
            self.update_training_status()

            imgui.text(self.message)

            # Add 'Done' button
            if imgui_utils.button("Done", enabled=1):
                self.stop_training()
                imgui.close_current_popup()
                self.message = ''

            # Keep the existing 'Close' button for when training is not active
            if not self.training_active:
                if imgui_utils.button("Close", enabled=1):
                    imgui.close_current_popup()
                    self.message = ''

            imgui.end_popup()


        if imgui.button("Train", width=-1):
            imgui.open_popup("Training")
            print("Starting training...")

            self.queue = mp.Queue()
            self.reply = mp.Queue()
            self.training_process = mp.Process(target=train_main, args=(self.queue, self.reply), name='TrainingProcess')
            self.training_active = True
            self.progress_message = "Initializing training..."
            self.queue.put(self.args)
            self.training_process.start()

