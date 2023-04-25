import os
import time

import cv2
import imgui
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np

from modules.filedialog import FileDialog
from utils.gui_utils import imgui_utils
from super_res import main as super_res_main, load_model, get_resolution, check_width_height, get_audio, Reader, Writer
#from super_res import base_args
from dnnlib import EasyDict
import multiprocessing as mp

args = EasyDict(result_path="", input_path=[""], model_type="Balance",
                outscale=3, width=4096, height=4096, sharpen_scale=1, scale_mode=0)
scale_factor = ['1', '2', '3', '4', '5', '6', '7', '8']


class SuperResModule:
    def __init__(self, menu):
        self.result_path=args.result_path
        self.input_path = args.input_path
        self.models = ['Quality','Balance','Fast']
        self.model_selected = 0
        self.model_type = self.models[self.model_selected]
        self.width = args.width
        self.height = args.height
        self.out_scale = args.outscale
        self.sharpen = args.sharpen_scale
        self.menu = menu
        self.app = menu.app
        self.file_dialog = FileDialog(self, "Videos and Images", os.path.abspath(os.getcwd()), ["*", ".mp4", ".avi", ".jpg", ".png", ".jpeg", ".bmp"])
        self.scale_mode = 0
        self.running = False
        self.progress = {"num_files": 0, "current_file": 0, "current_file_name": "", "current_file_progress": 0, "eta":0}
        self.queue = mp.Queue()
        self.reply = mp.Queue()
        self.super_res_process = mp.Process(target=super_res_mp, args=(self.queue, self.reply))


    def display_progress(self):
        if self.reply.qsize() > 0:
            self.progress, self.running = self.reply.get()
            while self.reply.qsize() > 0:
                self.progress, self.running = self.reply.get()
        imgui.begin("Super Resolution", False)
        imgui.text('Super Resolution...')
        imgui.text("Files: " + str(self.progress["current_file"]) + "/" + str(self.progress["num_files"]))
        imgui.text("Current File: " + self.progress["current_file_name"])
        imgui.text("Progress: " + str(self.progress["current_file_progress"]) + "%" + " | ETA: " + str(self.progress["eta"]) + "s")
        imgui.end()

    @imgui_utils.scoped_by_object_id
    def __call__(self):
        if self.running:
            self.display_progress()

        joined = '\n'.join(self.input_path)
        imgui_utils.input_text("##SRINPUT", joined, 1024, flags=imgui.INPUT_TEXT_READ_ONLY, width=-self.app.button_w * 3, help_text="Input Files")
        imgui.same_line()
        _clicked, input = self.file_dialog()
        if _clicked:
            self.input_path = input
            print(self.input_path)
        _, self.result_path = imgui.input_text("Destination Folder", self.result_path, 1024)
        self.models = ['Quality','Balance','Fast']
        if len(self.models) > 0:
            _, self.model_selected = imgui.combo("Model", self.model_selected, self.models)
            self.model_type = self.models[self.model_selected]
        clicked, self.scale_mode = imgui.combo("Scale Mode", self.scale_mode, ["Custom", "Scale"])
        if clicked:
            print(self.scale_mode)
        if self.scale_mode:
            _, self.out_scale = imgui.combo("Scale Factor", self.out_scale, scale_factor)
        else:
            _, self.height = imgui.input_int("Height", self.height)
            _, self.width = imgui.input_int("Width", self.width)
        _, self.sharpen = imgui.input_int("Sharpening Factor", self.sharpen)
        if self.sharpen<1:
            self.sharpen=1
        if imgui.is_item_hovered():
            imgui.set_tooltip("Additional sharpening performed after super resolution")
        try:
            if imgui.button("Super Resolution"):
                self.running = True
                print("Super Resolution")
                args.result_path = self.result_path
                args.input_path = self.input_path
                args.model_type = self.model_type
                args.outscale = self.out_scale + 1
                args.out_height = self.height
                args.out_width = self.width
                args.sharpen_scale = self.sharpen
                args.scale_mode = self.scale_mode
                self.queue.put(args)
                print("Starting Super Resolution")
                self.super_res_process.start()

        except Exception as e:
            print("SRR ERROR", e)

def super_res_mp(queue, reply):
    start = time.time()
    eta = -1

    args = queue.get()
    print("in super_res_mp", args)
    if args.model_type == "Quality":
        model_path = "./sr_models/Quality.pth"
    elif args.model_type == "Balance":
        model_path = "./sr_models/Balance.pth"
    elif args.model_type == "Fast":
        model_path = "./sr_models/Fast.pt"

    upsampler = load_model(args.model_type, model_path)
    list_file = args.input_path
    # if args output path does not exist
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    reply.put(({"num_files": len(list_file), "current_file": 0, "current_file_name": "", "current_file_progress": 0, "eta":eta}, True))
    print(list_file)
    for i, file in enumerate(list_file):
        start = time.time()
        reply.put(({"num_files": len(list_file), "current_file": i, "current_file_name": file, "current_file_progress": 0, "eta":eta}, True))
        print(f'working on {file}')
        head, tail = os.path.split(file)
        if file[-3:] == 'jpg' or file[-3:] == 'png':
            data_transformer = transforms.Compose([transforms.ToTensor()])
            image = cv2.imread(file)
            input_width, input_height = image.shape[0], image.shape[1]
            print("INPUT DIMENSIONS", input_width, input_height, image.shape)
            image = data_transformer(image).to('cuda')
            input = torch.unsqueeze(image, 0)

            with torch.inference_mode():
                output = upsampler(input)
                print("OUTPUT DIMENSIONS", output.shape)
                output = F.adjust_sharpness(output, args.sharpen_scale) * 255

                output = output[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                if args.scale_mode:
                    if args.outscale != 4:
                        output = cv2.resize(
                            output, (
                                int(input_width * args.outscale),
                                int(input_width * args.outscale),
                            ), interpolation=cv2.INTER_LINEAR)


                else:
                    output = cv2.resize(
                        output, (
                            int(args.out_width),
                            int(args.out_height),
                        ), interpolation=cv2.INTER_LINEAR)

            if args.scale_mode:
                print("USING these params", input_width, input_height, args.outscale)
                path = os.path.join(args.result_path,
                                    tail[
                                    :-4] + f'_result_{args.model_type}_{int(input_width * args.outscale)}x{int(input_height * args.outscale)}_Sharpness{args.sharpen_scale}.jpg')

            else:
                path = os.path.join(args.result_path,
                                    tail[
                                    :-4] + f'_result_{args.model_type}_{int(args.out_width)}x{int(args.out_height)}_Sharpness{args.sharpen_scale}.jpg')

            print("Saving image to {}".format(path))
            cv2.imwrite(path, output)
            eta = (time.time() - start) * (len(list_file) - i)
            reply.put(
               ({"num_files": len(list_file), "current_file": i, "current_file_name": file, "current_file_progress": 1, "eta":eta}, True))
        if file[-3:] == 'mp4' or file[-3:] == 'avi' or file[-3:] == 'mov':
            reply.put(({"num_files": len(list_file), "current_file": len(list_file), "current_file_name": file, "current_file_progress":0, "eta":eta}, True))
            width, height = get_resolution(file)

            if args.outscale > 4 or (
                    check_width_height(args) and (args.out_width > 4 * width or args.out_height > 4 * height)):
                print(
                    'warning: Any super-res scale larger than x4 required non-model inference with interpolation and can be slower')

            audio = get_audio(file)
            if args.scale_mode:
                video_save_path = os.path.join(args.result_path, tail[
                                                                 :-4] + f'_result_{args.model_type}_{int(width * args.outscale)}x{int(height * args.outscale)}_Sharpness{args.sharpen_scale}.mp4')
            else:
                video_save_path = os.path.join(args.result_path, tail[
                                                                 :-4] + f'_result_{args.model_type}_x{int(args.out_width)}x{int(args.out_height)}_Sharpness{args.sharpen_scale}.mp4')

            cap = cv2.VideoCapture(file)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            reader = Reader(width, height, file)
            writer = Writer(args, audio, height, width, video_save_path, fps=fps)
            current_frame = 0
            while True:
                current_frame += 1
                eta = (time.time() - start) * (frame_count - current_frame) * (len(list_file) - i)
                reply.put(({"num_files": len(list_file), "current_file": len(list_file), "current_file_name": file, "current_file_progress":current_frame/frame_count, "eta":eta}, True))
                start = time.time()
                img = reader.get_frame()
                if img is not None:
                    input = torch.tensor(img).permute(2, 0, 1).float().to('cuda') / 255
                    input = torch.unsqueeze(input, 0)
                    with torch.inference_mode():
                        output = upsampler(input)
                        output = F.adjust_sharpness(output, args.sharpen_scale) * 255

                        output = output[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)

                        if args.scale_mode:
                            if args.outscale != 4:
                                output = cv2.resize(
                                    output, (
                                        int(width * args.outscale),
                                        int(height * args.outscale),
                                    ), interpolation=cv2.INTER_LINEAR)


                        else:
                            output = cv2.resize(
                                output, (
                                    int(args.out_width),
                                    int(args.out_height),
                                ), interpolation=cv2.INTER_LINEAR)

                    writer.write_frame(output)
                    ret, img = cap.read()

                else:
                    print('break')
                    break

            writer.close()

    reply.put(({"num_files": len(list_file), "current_file": len(list_file), "current_file_name": "", "current_file_progress": 0, "eta":eta}, False))

