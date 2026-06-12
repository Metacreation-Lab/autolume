# StyleGAN training on Apple Silicon (MPS): feasibility report

Status: analysis only, no implementation yet. Companion to [PORTING_MACOS.md](PORTING_MACOS.md).

## Verdict

Feasible with a small, fully device-gated change set (5 files, roughly 40 lines). The training stack is closer to MPS-ready than it looks: metrics are dormant, snapshots already detour through CPU, the architectures already force fp32 off CUDA, and the custom CUDA kernels already fall back to reference PyTorch ops. There is exactly one hard API incompatibility (float64 in `training_stats`), a handful of `torch.cuda.*` calls to guard, and one genuine unknown (the backward kernel of `grid_sample` used by ADA's geometric augmentations). Expected end state: correct fp32 training with ADA on MPS, suitable for fine-tuning pretrained models at 256 to 512; training from scratch remains impractical for time reasons, not correctness ones.

The earlier working-but-slow attempt is consistent with this analysis: fp32 plus silent CPU fallbacks in the hot path (most likely the augmentation backward) is exactly the configuration that runs correctly and disappoints in throughput.

## How training executes today

UI Train button → `train_main(queue, reply)` in an `mp.Process` → `launch_training` → `subprocess_fn(rank=0)` in-process → `training_loop()`. Single process, single device: `num_gpus=1` from the UI, `torch.distributed` is commented out, and `training_stats.init_multiprocessing` gets `sync_device=None`. The UI passes `metrics=[]` and the in-loop FID evaluation is commented out entirely, so the Inception pipeline is out of scope. This simplifies the port to: make one process train on `mps` instead of `cuda`.

## Hard blockers (crash today, must change)

| # | Location | Problem | Gated fix |
|---|----------|---------|-----------|
| 1 | [training/training_loop.py](training/training_loop.py) line 132 | `device = torch.device('cuda', rank)` | `device_utils.get_device()` when `num_gpus == 1`; unchanged expression otherwise |
| 2 | [torch_utils/training_stats.py](torch_utils/training_stats.py) line 93 | `moments.to(torch.float64)` runs on the value's device; MPS has no float64 at the storage level, and `PYTORCH_ENABLE_MPS_FALLBACK` cannot rescue a dtype the backend cannot represent. Every `training_stats.report()` from the loss crashes | In `report()`: when `moments.device.type == 'mps'`, move moments to CPU before the float64 cast. Counters then accumulate on CPU, which is exactly where `_sync` reads them in single-process mode (line 245) |
| 3 | [training/training_loop.py](training/training_loop.py) lines 263 to 267, 364, 390, 521 to 524 | `torch.cuda.Event` phase timing created unconditionally at rank 0, recorded against `torch.cuda.current_stream` | Create events only when the device is CUDA; on MPS report wall-clock per phase via `time.perf_counter` around the phase (with a `torch.mps.synchronize` only at tick boundaries), or skip phase timing initially |
| 4 | [training/training_loop.py](training/training_loop.py) lines 437 to 439 | `torch.cuda.max_memory_allocated / max_memory_reserved / reset_peak_memory_stats` in the status line | Guard by device; report `torch.mps.current_allocated_memory()` (no peak API) or 0 |
| 5 | [training/training_loop.py](training/training_loop.py) line 355 and [train.py](train.py) line 253 | explicit `.pin_memory()` on labels; `pin_memory=True` in DataLoader kwargs | Skip pinning when the training device is not CUDA (same pattern as the renderer port) |

One environment detail: `PYTORCH_ENABLE_MPS_FALLBACK=1` is inherited by the training subprocess when launched from the app (set in [main.py](main.py)), but the standalone CLI path needs the same setdefault at the top of [train.py](train.py) before `import torch`.

## Already safe (no changes needed)

- Custom CUDA kernels (`bias_act`, `upfirdn2d`, `filtered_lrelu`): compile only for CUDA tensors, otherwise use their reference PyTorch implementations. The optimization is lost on MPS, as expected.
- Second-order gradients for R1 and path-length regularization: the important finding of this analysis. `conv2d_gradfix._should_use_custom_op` returns False on torch >= 1.11 on every platform (see the `_use_pytorch_1_11_api` short-circuit), so current CUDA training already uses PyTorch's native double-backward, which is implemented as a composite of ordinary convolutions and is device-agnostic. MPS takes the identical path. A side effect with parity on all platforms: `pl_no_weight_grad` and `no_weight_gradients()` are already no-ops everywhere. This still needs one empirical confirmation on-device (Phase 0), but it is not expected to need code.
- fp16: `use_fp16 ... and x.device.type == 'cuda'` in the architectures means MPS automatically trains in fp32. Slower, but the numerically safest starting point; StyleGAN's fp16 scheme is tuned for CUDA.
- Network snapshots are deep-copied to CPU before pickling, so `.pkl` files from a Mac run load anywhere, including the live renderer.
- Dataset pipeline: `pyspng` is an optional import, DataLoader workers use spawn, `InfiniteSampler` is pure Python.
- `nan_to_num`, EMA updates, ADA's `p` update via `misc.constant`: device-agnostic.
- tf32 and cudnn flags: harmless no-ops off CUDA.
- KD / teacher / LPIPS / face parsing: only constructed when a teacher is set, which is the compression path; it stays gated on macOS and out of scope.
- codecarbon's EmissionsTracker runs degraded on Apple Silicon (estimates instead of RAPL); worst case is warnings.

## The one real unknown: ADA geometric augmentations

[training/augment.py](training/augment.py) line 300 samples images through `grid_sample_gradfix`, which (unlike the conv counterpart) is active on torch 2.x and dispatches `aten::grid_sampler_2d_backward` as a forward op inside its backward. If that kernel has no MPS implementation, the fallback env var runs it on CPU: correct results, but the augmentation backward of every D phase crosses to CPU, and the default `bgc` pipeline applies geometric transforms to every real and fake batch. This is the prime suspect for the "worked but performed badly" experience. The `erfinv` calls in augment.py are debug-only paths and do not matter.

Resolution order: probe whether the kernel is native on current torch (Phase 0); if native, nothing to do; if CPU-fallback, measure the cost and offer an MPS default of `blit` plus `color` pipelines (no geometric stage) while keeping `bgc` selectable. DiffAugment's policies are plain tensor ops and should be unaffected, also covered by the probe.

## Isolation guarantees

Every change is conditioned on the resolved device type, sourced from [utils/device_utils.py](utils/device_utils.py), with the CUDA branch textually unchanged. The only vendored file touched is `training_stats.py` with a three-line, device-gated detour, following the precedent of the `custom_ops.py` fix. Verification of non-impact on CUDA: run a short fixed-seed training before and after the change set on a CUDA machine and diff `stats.jsonl`; the gating predicts byte-identical output.

## Performance expectations and levers

Honest framing, to be validated in Phase 2: fp32 single-device StyleGAN2 training on an M-series Max class GPU should land in the minutes-per-kimg range at 512 (versus seconds-per-kimg on a modern high-end NVIDIA card), putting a 200 to 600 kimg fine-tune in the hours-to-a-day band and a 25000 kimg from-scratch run out of practical reach. Levers, in expected order of impact: keep the augmentation pipeline free of CPU fallbacks (see above); size `batch_gpu` to fill unified memory at the target resolution; prefer 256 or 512 fine-tunes over 1024; reduce snapshot frequency (each snapshot deep-copies G, D, and G_ema); keep 2 to 4 DataLoader workers. Two experiments worth trying after correctness is established, both behind explicit flags: allowing fp16 layers on MPS (extend the dtype condition in the architectures; risk of instability since StyleGAN's clamping scheme was tuned on CUDA) and `torch.compile` with the experimental Metal backend in recent PyTorch.

## Phased plan

Phase 0, op probe (about an hour, mostly runtime): a small script that exercises each suspect op on `mps` and reports native / cpu-fallback / fail: float64 casts, `grid_sampler_2d_backward` through the gradfix Function, an R1 micro-step on a two-layer conv discriminator (double backward), a path-length micro-step, lognormal sampling used by ADA, DiffAugment policies. Its output decides the augmentation default and confirms the conv double-backward assumption before any code changes.

Phase 1, hard blockers plus correctness: implement the five gated fixes, then on the Mac run a 100-to-200-image dataset at 256 for a few kimg from scratch and a short fine-tune from a pretrained pickle. Pass criteria: finite losses, `r1_penalty` and `pl_penalty` statistics in normal ranges, ADA `p` adapting, visibly improving fakes grid, snapshot loading in the live renderer on both macOS and a CUDA machine.

Phase 2, performance: sec/kimg sweeps over batch size and resolution, an MPS fallback audit over one tick (the fallback env var warns once per op), geometric-augs on/off comparison, then pick macOS defaults for the training UI (suggested batch, augpipe, snapshot cadence) and record measured numbers in this document.

Phase 3, CUDA parity and UI: fixed-seed before/after diff on a CUDA box, surface an "experimental on Apple Silicon" notice plus the chosen defaults in the training pane, update PORTING_MACOS.md.

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| `grid_sampler_2d_backward` not native on MPS | medium | performance only (CPU fallback is correct) | non-geometric default pipeline on macOS |
| conv double-backward gap on MPS (R1, PL) | low | high if real (CPU fallback for the regularizers) | Phase 0 probe before committing to the plan |
| fp32 instability | very low | medium | fp32 is the stable mode; fp16 stays experimental |
| Unified memory pressure at 1024 | medium | run fails or thrashes | recommend 256/512; document `batch_gpu` guidance |
| Hidden float64 elsewhere | low | crash, trivially locatable | Phase 0 probe plus first smoke run |
| Behavioral drift on CUDA | designed against | high | device-gated changes only; fixed-seed stats diff |

## Recommendation

Proceed. The blocker list is short, the dangerous-sounding parts (second-order gradients) turn out to already run on the code path MPS needs, and the workflow that proved out the renderer port (small gated change, run on the Mac, iterate) applies directly. Implement Phase 0 and Phase 1 together on this branch when ready.
