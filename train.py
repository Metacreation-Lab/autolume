# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Train a GAN using the techniques described in the paper
"Alias-Free Generative Adversarial Networks"."""

import os
import click
import re
import json
import tempfile
import torch
import traceback

import dnnlib as dnnlib
from training import training_loop
from metrics import metric_main
from torch_utils import training_stats, custom_ops
from queue import Empty


#----------------------------------------------------------------------------

def subprocess_fn(rank, c, temp_dir, queue, reply):
    dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    try:
        # Init torch.distributed.
        if c.num_gpus > 1:
            init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
            if os.name == 'nt':
                init_method = 'file:///' + init_file.replace('\\', '/')
                torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=c.num_gpus)
            else:
                init_method = f'file://{init_file}'
                torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=c.num_gpus)

        # Init torch_utils.
        sync_device = torch.device('cuda', rank) if c.num_gpus > 1 else None
        training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
        print(rank)
        if rank != 0:
            custom_ops.verbosity = 'none'

        # Execute training loop.
        training_loop.training_loop(rank=rank, **c, queue=queue, reply=reply)
    except Exception as e:
        print(f"Caught an exception of type: {type(e).__name__}")
        print(f"Exception message: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        reply.put(['Exception occured in subprocess_fn...', True])

#----------------------------------------------------------------------------

def launch_training(c, desc, outdir, dry_run, queue, reply):
    dnnlib.util.Logger(should_flush=True)
    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    c.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
    assert not os.path.exists(c.run_dir)

    # Print options.
    print()
    print('Training options:')
    reply.put(['Training options:' + json.dumps(c, indent=2), False])
    print(json.dumps(c, indent=2))
    print()
    print(f'Output directory:    {c.run_dir}')
    print(f'Number of GPUs:      {c.num_gpus}')
    print(f'Batch size:          {c.batch_size} images')
    print(f'Training duration:   {c.total_kimg} kimg')
    print(f'Dataset path:        {c.training_set_kwargs.path}')
    print(f'Dataset size:        {c.training_set_kwargs.max_size} images')
    print(f'Dataset resolution:  {c.training_set_kwargs.resolution}')
    print(f'Dataset Height and width:  {c.training_set_kwargs.height} {c.training_set_kwargs.width}')
    print(f'Dataset labels:      {c.training_set_kwargs.use_labels}')
    print(f'Dataset x-flips:     {c.training_set_kwargs.xflip}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(c.run_dir)
    with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(c, f, indent=2)

    # Launch processes.
    print('Launching processes...')
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess_fn(rank=0, c=c, temp_dir=temp_dir, queue=queue, reply=reply)
    except:
        reply.put(['Exception occured in launch_training...', True])

#----------------------------------------------------------------------------

def init_dataset_kwargs(data, resolution=None, height = None, width = None, fps=10):
    try:
        print("RESOLUTION: ", resolution, height, width)
        dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data, use_labels=True, max_size=None, xflip=False, resolution=resolution, height=height, width=width, fps=fps)
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # Subclass of training.dataset.Dataset.
        dataset_kwargs.resolution = dataset_obj.resolution # Be explicit about resolution.
        dataset_kwargs.use_labels = dataset_obj.has_labels # Be explicit about labels.
        dataset_kwargs.max_size = len(dataset_obj) # Be explicit about dataset size.
        dataset_kwargs.height = dataset_obj.height
        dataset_kwargs.width = dataset_obj.width
        return dataset_kwargs, dataset_obj.name, dataset_obj.init_res
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

#----------------------------------------------------------------------------

def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

#----------------------------------------------------------------------------

@click.command()

# Required.
@click.option('--outdir',       help='Where to save the results', metavar='DIR',                required=True)
@click.option('--cfg',          help='Base configuration',                                      type=click.Choice(['stylegan3-t', 'stylegan3-r', 'stylegan2']), required=True)
@click.option('--data',         help='Training data', metavar='[ZIP|DIR]',                      type=str, required=True)
@click.option('--gpus',         help='Number of GPUs to use', metavar='INT',                    type=click.IntRange(min=1), required=True)
@click.option('--batch',        help='Total batch size', metavar='INT',                         type=click.IntRange(min=1), required=True)
@click.option('--resolution',   help='Dataset resolution', metavar='TUPLE',                     type=tuple, default=None)
@click.option('--gamma',        help='R1 regularization weight', metavar='FLOAT',               type=click.FloatRange(min=0), required=True)
@click.option('--topk', help='Enable topk training [default: None]', type=float, metavar='FLOAT')

# Generator Options
@click.option('--z_dim', help='Override z dimension', type=int, metavar='INT', default=512)
@click.option('--w_dim', help='Override w dimension', type=int, metavar='INT', default=512)

# Optional features.
@click.option('--cond',         help='Train conditional model', metavar='BOOL',                 type=bool, default=False, show_default=True)
@click.option('--mirror', help='Enable dataset x-flips [default: false]', type=bool, metavar='BOOL')
@click.option('--aug', help='Augmentation mode [default: ada]', type=click.Choice(['noaug', 'ada', 'fixed']))
@click.option('--augpipe', help='Augmentation pipeline [default: bgc]', type=click.Choice(['blit', 'geom', 'color', 'filter', 'noise', 'cutout', 'bg', 'bgc', 'bgcf', 'bgcfn', 'bgcfnc']))
@click.option('--resume',       help='Resume from given network pickle', metavar='[PATH|URL]',  type=str)
@click.option('--freezed',      help='Freeze first layers of D', metavar='INT',                 type=click.IntRange(min=0), default=0, show_default=True)

# Misc hyperparameters.
@click.option('--p',            help='Probability for --aug=fixed', metavar='FLOAT',            type=click.FloatRange(min=0, max=1), default=0.2, show_default=True)
@click.option('--target',       help='Target value for --aug=ada', metavar='FLOAT',             type=click.FloatRange(min=0, max=1), default=0.6, show_default=True)
@click.option('--batch-gpu',    help='Limit batch size per GPU', metavar='INT',                 type=click.IntRange(min=1))
@click.option('--cbase',        help='Capacity multiplier', metavar='INT',                      type=click.IntRange(min=1), default=32768, show_default=True)
@click.option('--cmax',         help='Max. feature maps', metavar='INT',                        type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--glr',          help='G learning rate  [default: varies]', metavar='FLOAT',     type=click.FloatRange(min=0))
@click.option('--dlr',          help='D learning rate', metavar='FLOAT',                        type=click.FloatRange(min=0), default=0.002, show_default=True)
@click.option('--map-depth',    help='Mapping network depth  [default: varies]', metavar='INT', type=click.IntRange(min=1))
@click.option('--mbstd-group',  help='Minibatch std group size', metavar='INT',                 type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--initstrength', help='Override ADA strength at start', type=float, default=None)
@click.option('--projected', help='Use Projected GAN Discriminator', type=bool, default=False)
@click.option('--DiffAugment', help='Comma-separated list of DiffAugment policy [default: None]', type=str, default=None)

# Misc settings.
@click.option('--desc',         help='String to include in result dir name', metavar='STR',     type=str)
@click.option('--metrics',      help='Quality metrics', metavar='[NAME|A,B,C|none]',            type=parse_comma_separated_list, default='fid50k_full', show_default=True)
@click.option('--kimg',         help='Total training duration', metavar='KIMG',                 type=click.IntRange(min=1), default=25000, show_default=True)
@click.option('--nkimg',  help='Override starting count', type=int, metavar='INT', default=None)
@click.option('--tick',         help='How often to print progress', metavar='KIMG',             type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--snap',         help='How often to save snapshots', metavar='TICKS',            type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--seed',         help='Random seed', metavar='INT',                              type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--fp32',         help='Disable mixed-precision', metavar='BOOL',                 type=bool, default=False, show_default=True)
@click.option('--nobench',      help='Disable cuDNN benchmarking', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--workers',      help='DataLoader worker processes', metavar='INT',              type=click.IntRange(min=1), default=3, show_default=True)
@click.option('-n','--dry-run', help='Print training options and exit',                         is_flag=True)
@click.option('--kd_l1_lambda', help='Influence of l1', type=float, default=0)
@click.option('--kd_lpips_lambda', help='Influence of lpips', type=float, default=0)
@click.option('--kd_mode', help='Choose KD MOde',  type=click.Choice(['Output_Only', 'Intermediate']), default='Output_Only')
@click.option('--content_aware_kd', help='Use content aware KD', type=bool, default=True)
@click.option('--custom', help='Use specialized Architecture', type=bool, default=True)
@click.option('--LPIPS_IMAGE_SIZE', help='LPIPS image size', type=float, default=256)

# Distillation Options
@click.option('--teacher',       help='Resume from given network pickle', metavar='[PATH|URL]',  type=str, default=None)
def clickmain(**kwargs):
    main(**kwargs)

def main(queue, reply):
    try:
        """Train a GAN using the techniques described in the paper
        "Alias-Free Generative Adversarial Networks".

        Examples:

        \b
        # Train StyleGAN3-T for AFHQv2 using 8 GPUs.
        python train.py --outdir=~/training-runs --cfg=stylegan3-t --data=~/datasets/afhqv2-512x512.zip \\
            --gpus=8 --batch=32 --gamma=8.2 --mirror=1

        \b
        # Fine-tune StyleGAN3-R for MetFaces-U using 1 GPU, starting from the pre-trained FFHQ-U pickle.
        python train.py --outdir=~/training-runs --cfg=stylegan3-r --data=~/datasets/metfacesu-1024x1024.zip \\
            --gpus=8 --batch=32 --gamma=6.6 --mirror=1 --kimg=5000 --snap=5 \\
            --resume=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-1024x1024.pkl

        \b
        # Train StyleGAN2 for FFHQ at 1024x1024 resolution using 8 GPUs.
        python train.py --outdir=~/training-runs --cfg=stylegan2 --data=~/datasets/ffhq-1024x1024.zip \\
            --gpus=8 --batch=32 --gamma=10 --mirror=1 --aug=noaug
        """

        # Initialize config.
        kwargs = queue.get()
        print("kwargs", kwargs)

        opts = dnnlib.EasyDict(**kwargs) # Command line arguments.
        c = dnnlib.EasyDict() # Main config dict.
        reply.put(['Configuring Models...', False])
        c.G_kwargs = dnnlib.EasyDict(class_name=None, z_dim=opts.z_dim, w_dim=opts.w_dim, mapping_kwargs=dnnlib.EasyDict())
        c.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0, 0.99], eps=1e-8)
        c.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0, 0.99], eps=1e-8)
        c.loss_kwargs = dnnlib.EasyDict(class_name='training.losses.scratch_loss.StyleGAN2Loss', kd_l1_lambda=opts.kd_l1_lambda, kd_lpips_lambda=opts.kd_lpips_lambda, kd_mode=opts.kd_mode,
                                        content_aware_KD=opts.content_aware_kd, LPIPS_IMAGE_SIZE=opts.lpips_image_size)
        c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2)

        # Training set.
        c.training_set_kwargs, dataset_name, init_res = init_dataset_kwargs(data=opts.data, resolution=opts.resolution, height=opts.resolution[1], width=opts.resolution[0], fps=opts.fps)
        if opts.cond and not c.training_set_kwargs.use_labels:
            raise click.ClickException('--cond=True requires labels specified in dataset.json')
        c.training_set_kwargs.use_labels = opts.cond
        c.training_set_kwargs.xflip = opts.mirror

        # Hyperparameters & settings.

        c.num_gpus = opts.gpus
        c.batch_size = opts.batch
        c.batch_gpu = opts.batch_gpu or opts.batch // opts.gpus
        c.G_kwargs.channel_base = opts.cbase
        c.G_kwargs.channel_max = opts.cmax
        c.G_kwargs.mapping_kwargs.num_layers = (8 if opts.cfg == 'stylegan2' else 2) if opts.map_depth is None else opts.map_depth
        if opts.teacher is not None:
            reply.put(['Loading Teacher...', False])
            assert isinstance(opts.teacher, str)
            c.teacher = opts.teacher
        c.projected = opts.projected
        if opts.projected:
            reply.put(['Using Projected Discriminator...', False])
            c.D_kwargs = dnnlib.EasyDict(
                class_name='architectures.pg_modules.discriminator.ProjectedDiscriminator',
                diffaug=True,
                interp224=(c.training_set_kwargs.resolution < 224),
                backbone_kwargs=dnnlib.EasyDict(),
            )
            c.D_kwargs.backbone_kwargs.cout = 64
            c.D_kwargs.backbone_kwargs.expand = True
            c.D_kwargs.backbone_kwargs.proj_type = 2
            c.D_kwargs.backbone_kwargs.num_discs = 4
            c.D_kwargs.backbone_kwargs.separable = True
            c.D_kwargs.backbone_kwargs.cond = opts.cond
            c.loss_kwargs.r1_gamma = opts.gamma
        else:
            c.D_kwargs = dnnlib.EasyDict(class_name='architectures.custom_stylegan2.Discriminator',
                                        block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(),
                                        epilogue_kwargs=dnnlib.EasyDict())
            c.D_kwargs.channel_base = opts.cbase
            c.D_kwargs.block_kwargs.freeze_layers = opts.freezed
            c.D_kwargs.epilogue_kwargs.mbstd_group_size = opts.mbstd_group
            c.D_kwargs.channel_max = opts.cmax
            c.loss_kwargs.r1_gamma = opts.gamma
        c.G_opt_kwargs.lr = (0.002 if opts.cfg == 'stylegan2' else 0.0025) if opts.glr is None else opts.glr
        c.D_opt_kwargs.lr = opts.dlr
        c.metrics = opts.metrics
        c.total_kimg = opts.kimg
        c.kimg_per_tick = opts.tick
        c.image_snapshot_ticks = c.network_snapshot_ticks = opts.snap
        c.random_seed = c.training_set_kwargs.random_seed = opts.seed
        c.data_loader_kwargs.num_workers = opts.workers


        if list(init_res) != [4, 4]:
            print(' custom init resolution', init_res)
            c.G_kwargs.init_res = c.D_kwargs.init_res = list(init_res)

        # Sanity checks.
        if c.batch_size % c.num_gpus != 0:
            raise click.ClickException('--batch must be a multiple of --gpus')
        if c.batch_size % (c.num_gpus * c.batch_gpu) != 0:
            raise click.ClickException('--batch must be a multiple of --gpus times --batch-gpu')
        if not opts.projected:
            if c.batch_gpu < c.D_kwargs.epilogue_kwargs.mbstd_group_size:
                raise click.ClickException('--batch-gpu cannot be smaller than --mbstd')
        if any(not metric_main.is_valid_metric(metric) for metric in c.metrics):
            raise click.ClickException('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))

        # Base configuration.
        c.ema_kimg = c.batch_size * 10 / 32
        if opts.cfg == 'stylegan2':
            c.G_kwargs.class_name = 'architectures.custom_stylegan2.Generator' if opts.custom else 'architectures.networks_stylegan2.Generator'
            c.loss_kwargs.style_mixing_prob = 0.9 # Enable style mixing regularization.
            c.loss_kwargs.pl_weight = 2 # Enable path length regularization.
            c.G_reg_interval = 4 # Enable lazy regularization for G.
            c.G_kwargs.fused_modconv_default = 'inference_only' # Speed up training by using regular convolutions instead of grouped convolutions.
            c.loss_kwargs.pl_no_weight_grad = True # Speed up path length regularization by skipping gradient computation wrt. conv2d weights.
        else:
            c.G_kwargs.class_name = 'architectures.networks_stylegan3.Generator'
            c.G_kwargs.magnitude_ema_beta = 0.5 ** (c.batch_size / (20 * 1e3))
            if opts.cfg == 'stylegan3-r':
                c.G_kwargs.conv_kernel = 1 # Use 1x1 convolutions.
                c.G_kwargs.channel_base *= 2 # Double the number of feature maps.
                c.G_kwargs.channel_max *= 2
                c.G_kwargs.use_radial_filters = True # Use radially symmetric downsampling filters.
                c.loss_kwargs.blur_init_sigma = 10 # Blur the images seen by the discriminator.
                c.loss_kwargs.blur_fade_kimg = c.batch_size * 200 / 32 # Fade out the blur during the first N kimg.

        if opts.topk is not None:
            print("topking-------")
            assert isinstance(opts.topk, float)
            c.loss_kwargs.G_top_k = True
            c.loss_kwargs.G_top_k_gamma = opts.topk
            c.loss_kwargs.G_top_k_frac = 0.5

        augpipe_specs = {
            'blit': dict(xflip=1, rotate90=1, xint=1),
            'geom': dict(scale=1, rotate=1, aniso=1, xfrac=1),
            'color': dict(brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
            'filter': dict(imgfilter=1),
            'noise': dict(noise=1),
            'cutout': dict(cutout=1),
            'bg': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1),
            'bgc': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1,
                        lumaflip=1, hue=1, saturation=1),
            'bgcf': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1,
                        lumaflip=1, hue=1, saturation=1, imgfilter=1),
            'bgcfn': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1,
                        lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1),
            'bgcfnc': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1,
                        lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1, cutout=1),
        }

        # Augmentation.
        aug = opts.aug
        if opts.aug is None and opts.diffaugment is None:
            aug = 'ada'
        elif opts.diffaugment:
            c.loss_kwargs.diffaugment = opts.diffaugment
            aug = 'noaug'



        if aug != 'noaug':
            assert opts.augpipe is None or isinstance(opts.augpipe, str)
            augpipe = opts.augpipe
            if augpipe is None:
                augpipe = 'bgc'
            c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', **augpipe_specs[augpipe])
            if opts.aug == 'ada':
                c.ada_target = opts.target
            if opts.aug == 'fixed':
                c.augment_p = opts.p
            if opts.initstrength is not None:
                assert isinstance(opts.initstrength, float)
                c.augment_p = opts.initstrength

        # Resume.
        if opts.resume is not None:
            c.resume_pkl = opts.resume
            c.ada_kimg = 100 # Make ADA react faster at the beginning.
            c.ema_rampup = None # Disable EMA rampup.
            c.loss_kwargs.blur_init_sigma = 0 # Disable blur rampup.

        # Performance-related toggles.
        if opts.fp32:
            c.G_kwargs.num_fp16_res = c.D_kwargs.num_fp16_res = 0
            c.G_kwargs.conv_clamp = c.D_kwargs.conv_clamp = None
        if opts.nobench:
            c.cudnn_benchmark = False

        if opts.nkimg is not None:
            assert isinstance(opts.nkimg, int)
            c.nimg = opts.nkimg * 1000

        # Description string.
        desc = f'{opts.cfg:s}-{dataset_name:s}-gpus{c.num_gpus:d}-batch{c.batch_size:d}-gamma{c.loss_kwargs.r1_gamma:g}'
        if opts.desc is not None:
            desc += f'-{opts.desc}'

        # Launch.
        reply.put(["Launching...", False])
        if not queue.empty():
            if queue.get(block=False) == 'done':
                reply.put(['Training Process Aborted... Please close this window.', True])
        launch_training(c=c, desc=desc, outdir=opts.outdir, dry_run=opts.dry_run, queue=queue, reply=reply)
    except Exception as e:
        print(f"Caught an exception of type: {type(e).__name__}")
        print(f"Exception message: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        reply.put(['Training Process Could not Start... Please close this window.', True])

#----------------------------------------------------------------------------

if __name__ == "__main__":
    clickmain() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
