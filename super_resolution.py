import gc
import os

import click
import numpy as np
from tqdm.auto import tqdm
import ffmpeg
import cv2
from super_res.net_base import SRVGGNetPlus, SRVGGNetCompact, RRDBNet
import torch
from torchvision import transforms
import torchvision.transforms.functional as F


def get_audio(video_path):
    probe = ffmpeg.probe(video_path)
    has_audio = any(stream['codec_type'] == 'audio' for stream in probe['streams'])
    audio = ffmpeg.input(video_path).audio if has_audio else None
    return audio


class Reader:
    def __init__(self, width, height, video_path):
        self.width = width
        self.height = height
        self.stream_reader = (
            ffmpeg.input(video_path).output('pipe:', format='rawvideo', pix_fmt='bgr24',
                                            loglevel='error').run_async(
                pipe_stdin=True, pipe_stdout=True))

    def get_frame_from_stream(self):
        img_bytes = self.stream_reader.stdout.read(self.width * self.height * 3)  # 3 bytes for one pixel
        if not img_bytes:
            return None
        img = np.frombuffer(img_bytes, np.uint8).reshape([self.height, self.width, 3])
        return img

    def get_frame(self):
        return self.get_frame_from_stream()


class Writer:

    def __init__(self, audio, height, width, video_save_path, fps, scale_mode="Factor", out_width=None, out_height=None,
                 outscale=None):
        if scale_mode == "Factor":
            assert (outscale is not None)  # outscale should be specify when scale_mode is Factor
            out_width, out_height = int(width * outscale), int(height * outscale)
        else:
            assert (out_width is not None and out_height is not None)  # width and height should be specify together

            out_width, out_height = int(out_width), int(out_width)

        if audio is not None:
            self.stream_writer = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{out_width}x{out_height}',
                             framerate=fps).output(
                    audio,
                    video_save_path,
                    pix_fmt='yuv420p',
                    vcodec='libx264',
                    loglevel='error',
                    acodec='copy').overwrite_output().run_async(
                    pipe_stdin=True, pipe_stdout=True, cmd='ffmpeg'))
        else:
            self.stream_writer = (
                ffmpeg.input('pipe:', format='rawvideo',
                             pix_fmt='bgr24',
                             s=f'{out_width}x{out_height}',
                             framerate=fps).output(
                    video_save_path, pix_fmt='yuv420p', vcodec='libx264',
                    loglevel='error').overwrite_output().run_async(
                    pipe_stdin=True, pipe_stdout=True, cmd='ffmpeg'))

    def write_frame(self, frame):
        frame = frame.tobytes()
        self.stream_writer.stdin.write(frame)

    def close(self):
        self.stream_writer.stdin.close()
        self.stream_writer.wait()


def load_model(choice, path):
    if choice == 'Quality':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4).to('cuda')
        model_sd = torch.load(path)['params_ema']
        model.load_state_dict(model_sd)
        return model

    if choice == 'Balance':
        model = model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4,
                                        act_type='prelu').to('cuda')
        model_sd = torch.load(path)['params']
        model.load_state_dict(model_sd)
        return model

    if choice == 'Fast':
        model = SRVGGNetPlus(num_in_ch=3, num_out_ch=3, num_feat=48, upscale=4, act_type='prelu').to('cuda')
        model_sd = torch.load(path)
        model.load_state_dict(model_sd)
        return model


@click.command()
@click.option('--input', '-i', "input_dir", required=True, type=click.Path(), help='Input Path',
              prompt="Absolute Path to the images/ videos that should be upscaled")
@click.option('--output', '-o', "output_dir", required=True, type=click.Path(), help='Output Path',
              prompt="Absolute Path to the output folder where the results should be stored")
@click.option('--scale_mode', '-s', required=True, type=click.Choice(["Factor", "Custom"]), help='Scale Mode',
              prompt="Scale Mode", default="Factor")
@click.option('--sharpening_factor', '-shf', required=True, type=click.INT, help='Sharpening Factor',
              prompt="Sharpening Factor", default=1)
@click.option('--model', '-m', required=True, type=click.Choice(["Quality", "Balanced", "Fast"]), help='Model Type',
              prompt="Model Type", default="Quality")
@click.option('--scale_factor', '-sf', required=False, type=click.IntRange(2, 8), default=None, help='Scale Factor to be used, should be between 2 and 8.')
@click.option('--out_width', '-ow', required=False, type=click.INT, default=None, help='Output Width')
@click.option('--out_height', '-oh', required=False, type=click.INT, default=None, help='Output Height')
def super_res_main(input_dir, output_dir, scale_mode, sharpening_factor, model, scale_factor=None, out_width=None,out_height=None):

    msg = "Running Super Resolution on " + ", ".join(input_dir) + " storing Results at" + output_dir + " with " + model + " model" + " and " + scale_mode + " scale mode"
    if scale_mode == "Factor" and scale_factor is None:
        scale_factor = click.prompt("Scale Factor to be used, should be between 2 and 8. Default: ",
                                    type=click.IntRange(2, 8), default=4, show_default=True)
        out_height = None
        out_width = None
        msg += " with scale factor " + str(scale_factor)
    elif scale_mode == "Custom" and (out_width is None or out_height is None):
        out_width = click.prompt("Output Width", type=click.INT, default=1920, show_default=True)
        out_height = click.prompt("Output Height", type=click.INT, default=1080, show_default=True)
        scale_factor = None
        msg += " with output width " + str(out_width) + " and output height " + str(out_height)

    msg += " and sharpening factor " + str(sharpening_factor)

    print(msg)

    if model == "Quality":
        model_path = "./sr_models/Quality.pth"
    elif model == "Balance":
        model_path = "./sr_models/Balance.pth"
    else:
        model_path = "./sr_models/Fast.pt"

    print("Loading Model", model_path)
    super_res_model = load_model(model, model_path)

    # if output directory does not exist create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    low_res_files = []
    for f in input_dir:
        f = os.path.abspath(f)
        if os.path.isdir(f):
            for root, dirs, files in os.walk(f):
                for file in files:
                    low_res_files.append(os.path.join(root, file))
        else:
            low_res_files.append(f)

    print("Found", len(low_res_files), "files to upscale:", low_res_files)
    print("Starting Super Resolution")
    for file in tqdm(low_res_files, desc="Super Res", unit="files", position=0, leave=True,):
        # if file is an image perform super res on it
        if file.endswith(('.jpg', '.jpeg', '.png')):
            data_transformer = transforms.Compose([transforms.ToTensor()])
            image = cv2.imread(file)
            input_width, input_height = image.shape[0], image.shape[1]
            image = data_transformer(image).to('cuda')
            input = torch.unsqueeze(image, 0)

            with torch.inference_mode():
                output = super_res_model(input)
                output = F.adjust_sharpness(output, sharpening_factor) * 255

                output = output[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                if scale_mode == "Factor":
                    if scale_factor != 4:
                        output = cv2.resize(
                            output, (
                                int(input_width * scale_factor),
                                int(input_width * scale_factor),
                            ), interpolation=cv2.INTER_LINEAR)


                else:
                    output = cv2.resize(
                        output, (
                            int(out_width),
                            int(out_height),
                        ), interpolation=cv2.INTER_LINEAR)

            head, tail = os.path.split(file)
            if scale_mode == "Factor":
                path = os.path.join(output_dir,
                                    tail[
                                    :-4] + f'_result_{model}_{int(input_width * scale_factor)}x{int(input_height * scale_factor)}_Sharpness{sharpening_factor}.jpg')

            else:
                path = os.path.join(output_dir,
                                    tail[
                                    :-4] + f'_result_{model}_{int(out_width)}x{int(out_height)}_Sharpness{sharpening_factor}.jpg')
            cv2.imwrite(path, output)
        elif file.endswith((".mp4", ".avi", ".mov")):
            audio = get_audio(file)
            video = cv2.VideoCapture(file)
            fps = video.get(cv2.CAP_PROP_FPS)
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            head, tail = os.path.split(file)
            if scale_mode == "Factor":
                video_save_path = os.path.join(output_dir, tail[
                                                       :-4] + f'_result_{model}_{int(video_width * scale_factor)}x{int(video_height * scale_factor)}_Sharpness{sharpening_factor}.mp4')
            else:
                video_save_path = os.path.join(output_dir, tail[
                                                       :-4] + f'_result_{model}_x{int(out_width)}x{int(out_height)}_Sharpness{sharpening_factor}.mp4')

            writer = Writer(audio, video_height, video_width,
                            video_save_path=video_save_path, fps=fps, scale_mode=scale_mode, out_height=out_height,
                            out_width=out_width, outscale=scale_factor)
            reader = Reader(video_width, video_height, file)
            for i in tqdm(range(total_frames), desc="Super Res", unit="frames", position=0, leave=True,):
                img = reader.get_frame()
                if img is None:
                    break
                with torch.inference_mode():
                    sr_input = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().to('cuda') / 255
                    sr_output = super_res_model(sr_input)
                    sr_output = F.adjust_sharpness(sr_output, sharpening_factor) * 255

                    sr_output = sr_output[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)

                    if scale_mode == "Factor":
                        if scale_factor != 4:
                            sr_output = cv2.resize(
                                sr_output, (
                                    int(video_width * scale_factor),
                                    int(video_height * scale_factor),
                                ), interpolation=cv2.INTER_LINEAR)


                    else:
                        sr_output = cv2.resize(
                            sr_output, (
                                int(out_width),
                                int(out_height),
                            ), interpolation=cv2.INTER_LINEAR)
                    writer.write_frame(sr_output)
                    ret, img = video.read()
            writer.close()
            torch.cuda.empty_cache()
            gc.collect()
        else:
            print("File type not supported. \n Supported file types are: \n jpg, jpeg, png, mp4, avi, mov")


if __name__ == "__main__":
    super_res_main()
