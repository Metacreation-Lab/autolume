import click
import multiprocessing as mp
from projection.bayle_projection import run_projection
import os

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--target', 'target_fname', help='Target image file to project to', default=None)
@click.option('--text', 'target_text', help='Target text to project to', default=None)
@click.option('--initial-latent', help='Initial latent vector to start from (.npy/.npz file)', default=None)
@click.option('--outdir', help='Where to save the output images', required=True, metavar='DIR')
@click.option('--save-video', help='Save an mp4 video of optimization progress', is_flag=True, default=False)
@click.option('--seed', help='Random seed', type=int, default=300)
@click.option('--lr', help='Learning rate', type=float, default=0.1)
@click.option('--num-steps', 'num_steps', help='Number of optimization steps', type=int, default=1000)
@click.option('--no-vgg', 'skip_vgg', help='Disable VGG perceptual loss', is_flag=True, default=False)
@click.option('--no-clip', 'skip_clip', help='Disable CLIP loss', is_flag=True, default=False)
@click.option('--no-pixel', 'skip_pixel', help='Disable pixel loss', is_flag=True, default=False)
@click.option('--no-penalty', 'skip_penalty', help='Disable penalty loss', is_flag=True, default=False)
@click.option('--no-center', 'skip_center', help='Disable center crop loss', is_flag=True, default=False)
@click.option('--no-kmeans', 'skip_kmeans', help='Disable kmeans clustering', is_flag=True, default=False)
def main(
    network_pkl: str,
    target_fname: str,
    target_text: str,
    initial_latent: str,
    outdir: str,
    save_video: bool,
    seed: int,
    lr: float,
    num_steps: int,
    skip_vgg: bool,
    skip_clip: bool,
    skip_pixel: bool,
    skip_penalty: bool,
    skip_center: bool,
    skip_kmeans: bool,
):
    """Project given image and/or text to the latent space of pretrained network pickle.

    Examples:

    \b
    # Project image:
    python projector.py --outdir=out --target=target.png --network=network.pkl

    \b
    # Project text:
    python projector.py --outdir=out --text="a photo of a smiling person" --network=network.pkl

    \b
    # Project both:
    python projector.py --outdir=out --target=target.png --text="a photo of a smiling person" --network=network.pkl
    """
    
    if target_fname is None and target_text is None:
        raise click.BadParameter("Either --target or --text must be specified")

    if target_fname and not os.path.isfile(target_fname):
        raise click.BadParameter(f"Image file not found: {target_fname}")

    if initial_latent and not os.path.isfile(initial_latent):
        raise click.BadParameter(f"Initial latent file not found: {initial_latent}")

    # Create queues for communication with the projection process
    queue = mp.Queue()
    reply_queue = mp.Queue()

    # Pack all parameters into the queue
    queue.put((
        network_pkl,
        target_fname,
        target_text,
        initial_latent,
        outdir,
        save_video,
        seed,
        lr,
        num_steps,
        not skip_vgg,
        not skip_clip,
        not skip_pixel,
        not skip_penalty,
        not skip_center,
        not skip_kmeans
    ))

    # Run the projection in a separate process
    process = mp.Process(target=run_projection, args=(queue, reply_queue))
    process.start()

    try:
        while True:
            if not reply_queue.empty():
                message, image, done, video_done = reply_queue.get()
                if message:
                    click.echo(message)
                if (not save_video and done) or video_done:
                    break
            
    except KeyboardInterrupt:
        click.echo("\nInterrupted by user. Stopping projection...")
        queue.put(True)
        
    finally:
        process.join(timeout=3)
        if process.is_alive():
            process.terminate()
        queue.close()
        reply_queue.close()
        queue.join_thread()
        reply_queue.join_thread()

if __name__ == "__main__":
    main()
