"""Universal StyleGAN2 generator latency benchmark.

Picks the best available device via utils.device_utils (CUDA, then MPS, then
CPU) and times a batch-1 forward pass. Works unchanged on every platform.

Usage:
    uv run bench.py path/to/network.pkl --label baseline
    uv run bench.py path/to/network.pkl --label experiment --breakdown
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import dnnlib
from torch_utils import legacy
from utils import device_utils


def load_generator(pkl_path, device):
    with dnnlib.util.open_url(pkl_path, verbose=False) as f:
        data = legacy.load_network_pkl(f, custom=True)
    G = data.get("G_ema") or data.get("G")
    return G.to(device).eval().requires_grad_(False)


def stats(times_ms):
    a = np.asarray(times_ms)
    return {
        "min_ms": float(a.min()),
        "median_ms": float(np.median(a)),
        "p95_ms": float(np.percentile(a, 95)),
        "max_ms": float(a.max()),
        "mean_ms": float(a.mean()),
    }


def wrap_blocks(G, device, block_times):
    """Wrap each SynthesisBlock forward in a synchronized timer."""
    synthesis = G.synthesis
    for res in synthesis.block_resolutions:
        block = getattr(synthesis, f"b{res}")
        name = f"b{res}"
        block_times[name] = []
        orig_forward = block.forward

        def make_wrapper(orig, name):
            def wrapper(*args, **kwargs):
                device_utils.synchronize(device)
                t0 = time.perf_counter()
                out = orig(*args, **kwargs)
                device_utils.synchronize(device)
                block_times[name].append((time.perf_counter() - t0) * 1e3)
                return out
            return wrapper

        block.forward = make_wrapper(orig_forward, name)


def timed_pass(G, z, c, device):
    device_utils.synchronize(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        G(z, c, truncation_psi=1)
    device_utils.synchronize(device)
    return (time.perf_counter() - t0) * 1e3


def main():
    parser = argparse.ArgumentParser(description="Universal StyleGAN2 latency benchmark")
    parser.add_argument("pkl", help="Path to a StyleGAN2 generator pickle")
    parser.add_argument("--label", default=None, help="Run label stored in the JSON output")
    parser.add_argument("--breakdown", action="store_true", help="Time each synthesis block separately")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--passes", type=int, default=100)
    args = parser.parse_args()

    device = device_utils.get_device()
    print(f"Loading {args.pkl} on {device}...")
    G = load_generator(args.pkl, device)

    torch.manual_seed(0)
    z = torch.randn([1, G.z_dim], device=device)
    c = None if G.c_dim == 0 else torch.zeros([1, G.c_dim], device=device)

    block_times = {}
    if args.breakdown:
        wrap_blocks(G, device, block_times)

    print(f"Warming up ({args.warmup} passes)...")
    for _ in range(args.warmup):
        timed_pass(G, z, c, device)
    if args.breakdown:
        for times in block_times.values():
            del times[:args.warmup]

    print(f"Benchmarking ({args.passes} passes)...")
    frame_times = [timed_pass(G, z, c, device) for _ in range(args.passes)]

    result = {
        "label": args.label or Path(args.pkl).stem,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "pkl": str(args.pkl),
        "device": device.type,
        "torch_version": torch.__version__,
        "img_resolution": G.img_resolution,
        "num_fp16_res": getattr(G.synthesis, "num_fp16_res", None),
        "warmup": args.warmup,
        "passes": args.passes,
        "overall": stats(frame_times),
        "blocks": {name: stats(times) for name, times in block_times.items()},
    }

    out_dir = Path("bench_results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    out_path.write_text(json.dumps(result, indent=2))

    o = result["overall"]
    print(f"min {o['min_ms']:.2f} ms | median {o['median_ms']:.2f} ms | "
          f"p95 {o['p95_ms']:.2f} ms | max {o['max_ms']:.2f} ms")
    for name, s in result["blocks"].items():
        print(f"  {name}: median {s['median_ms']:.2f} ms | p95 {s['p95_ms']:.2f} ms")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
