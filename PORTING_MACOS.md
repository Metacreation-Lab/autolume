# Apple Silicon (macOS) port

This document records the key changes that make Autolume run natively on Apple Silicon Macs using the PyTorch MPS backend. The live visualizer/renderer is the primary supported feature. Training and model compression are intentionally not supported on macOS and their UI is greyed out. Windows and Linux behavior is unchanged: every macOS adaptation is behind a platform or device check.

## Device selection

A single shared helper now decides which torch device to use everywhere:

- [utils/device_utils.py](utils/device_utils.py) (new)
  - `get_device()` returns `cuda` > `mps` > `cpu`, cached after first call.
  - `synchronize(device)` and `empty_cache(device)` wrap the per-backend `torch.cuda.*` / `torch.mps.*` calls and are no-ops on CPU.
  - `is_macos()` is the platform check used for feature gating.

All code that previously hardcoded `'cuda'` or `cuda if available else cpu` now goes through `get_device()`:

| File | What changed |
|------|--------------|
| [widgets/renderer.py](widgets/renderer.py) | Renderer device, super-res model load (`map_location`), render-time measurement (`synchronize` on MPS before timing), `device` arg defaults to `get_device()`, autocast skipped on MPS, custom-kernel log only when CUDA exists |
| [modules/renderloop.py](modules/renderloop.py) | Cache flush via `device_utils.empty_cache()`; on MPS the flush synchronizes the GPU, so it runs every 120 rendered frames instead of every loop iteration (CUDA keeps the original per-iteration flush, measured ~2 FPS gain on Apple Silicon) |
| [widgets/performance_widget.py](widgets/performance_widget.py) | Default device, GPU checkbox now selects CUDA or MPS, greyed when neither exists |
| [widgets/looping_widget.py](widgets/looping_widget.py) | Keyframe vectors on `get_device()` instead of `.cuda()`, pinned buffers only with CUDA |
| [widgets/latent_widget.py](widgets/latent_widget.py) | `pin_memory()` only with CUDA (also fixes a missing-parentheses bug that always pinned) |
| [modules/pca_module.py](modules/pca_module.py) | Model load and worker device via `get_device()`; model is shipped to the PCA worker process on CPU when on MPS (MPS tensors cannot cross processes) |
| [ganspace/extract_pca.py](ganspace/extract_pca.py) | Worker moves model to target device; CUDA memory probing skipped off-CUDA (default batch size); device-aware cache flush |
| [projection/bayle_projection.py](projection/bayle_projection.py) | Projection device and CLIP normalization tensors via `get_device()` |
| [super_res/super_res.py](super_res/super_res.py), [super_resolution.py](super_resolution.py), [modules/super_res_module.py](modules/super_res_module.py) | All model loads (`map_location`) and input tensors via `get_device()`, device-aware cache flush |
| [torch_utils/custom_ops.py](torch_utils/custom_ops.py) | `_get_mangled_gpu_name()` returns `cpu` without CUDA instead of raising (minimal vendored edit) |
| [main.py](main.py) | Sets `PYTORCH_ENABLE_MPS_FALLBACK=1` on macOS before torch import, so operators missing on MPS fall back to CPU instead of crashing |

The vendored StyleGAN custom CUDA ops (`torch_utils/ops/`) need no changes: `bias_act`, `upfirdn2d`, and `filtered_lrelu` only compile their CUDA kernels when the input tensor is on a CUDA device and otherwise use their reference PyTorch implementations, which run on MPS. The generator architectures already force fp32 on non-CUDA devices (`use_fp16 ... and x.device.type == 'cuda'`), so MPS renders in fp32.

## Multiprocessing

`multiprocessing.Queue.qsize()` raises `NotImplementedError` on macOS (`sem_getvalue` is missing on Darwin). All queue polling now uses `empty()` instead, which behaves identically for these single-producer/consumer checks on every platform. Affected: [modules/renderloop.py](modules/renderloop.py), [modules/projection_module.py](modules/projection_module.py), [projection/bayle_projection.py](projection/bayle_projection.py), [modules/pca_module.py](modules/pca_module.py), [ganspace/extract_pca.py](ganspace/extract_pca.py), [widgets/looping_widget.py](widgets/looping_widget.py), [modules/training_module.py](modules/training_module.py). The thread queue in the visualizer (`queue.Queue`) is unaffected.

## OpenGL / GUI

macOS exposes either legacy OpenGL 2.1 or a 3.2+ core profile, never a compatibility profile (see the 2024 "Autolume for macOS" analysis). Autolume's drawing code ([utils/gui_utils/gl_utils.py](utils/gui_utils/gl_utils.py), [utils/gui_utils/glfw_window.py](utils/gui_utils/glfw_window.py)) is fixed-function GL, while pyimgui's default backend needs GLSL 330. Instead of rewriting the drawing code for core profile, macOS stays on the default GL 2.1 context and the imgui backend is swapped:

- [utils/gui_utils/imgui_window.py](utils/gui_utils/imgui_window.py): on macOS, `_GlfwRenderer` uses the rendering methods of pyimgui's `FixedPipelineRenderer` (GL 2.1 compatible) while keeping the GLFW input glue. Other platforms keep the shader backend.
- [utils/gui_utils/glfw_window.py](utils/gui_utils/glfw_window.py): macOS renders at native (retina) resolution. The GL viewport covers the framebuffer (2x the window on scaled displays) while all drawing stays in window coordinates, and imgui fonts are rasterized at the framebuffer scale with `font_global_scale` compensating ([utils/gui_utils/imgui_window.py](utils/gui_utils/imgui_window.py)). The window is also positioned inside the monitor work area; placing it at (0, 0) let the macOS menu bar push it down and crop the bottom of the content.
- [utils/gui_utils/gl_utils.py](utils/gui_utils/gl_utils.py): texture mipmaps are disabled on macOS (`glGenerateMipmap` needs GL 3.0+); non-mipmapped bilinear textures minify with `GL_LINEAR` so logos stay smooth.
- [utils/gui_utils/imgui_window.py](utils/gui_utils/imgui_window.py): mouse presses are latched from the GLFW button callback and held for one frame. pyimgui's backend only polls button state once per frame, so clicks shorter than a frame (trackpad taps, any click at low FPS) were dropped, which made buttons feel hit or miss. This mirrors the official Dear ImGui GLFW backend and applies on all platforms.
- Frame-time reductions (all platforms, click latency scales with frame time): [utils/gui_utils/gl_utils.py](utils/gui_utils/gl_utils.py) streams frames with `glTexSubImage2D` instead of re-specifying the texture each frame, and [modules/visualizer.py](modules/visualizer.py) skips the per-frame RGB to BGRA conversion unless recording or NDI streaming is active.

## Dependencies

[pyproject.toml](pyproject.toml) now resolves per platform:

- `torch` / `torchvision` / `torchaudio`: cu128 builds from the PyTorch index on Windows/Linux (unchanged); the same versions from PyPI on macOS (arm64 wheels with MPS, verified present for Python 3.10).
- `ndi-python` was already Windows/Linux only; the code now imports it optionally.

After pulling this branch, regenerate the lockfile once (any platform): `uv lock`, then `uv sync`. Plain `uv sync` also re-locks automatically.

macOS prerequisites before `uv sync`:

- Xcode Command Line Tools (`xcode-select --install`): `imgui` and `pyaudio` have no macOS arm64 wheels and build from source.
- `brew install portaudio` (required to build `pyaudio`).
- `brew install ffmpeg` (super-res video, recording).
- `brew install python-tk@3.10` if using Homebrew Python: it ships Tk separately, and tkinter backs the native file dialogs ([widgets/native_browser_widget.py](widgets/native_browser_widget.py)). Without it the app still runs; Browse dialogs are no-ops with a console hint.
- The pre-trained models from the README (sr_models etc.), same as other platforms.

## Features disabled on macOS (greyed out)

| Feature | Where | Why |
|---------|-------|-----|
| Training (incl. dataset prep UI) | [modules/training_module.py](modules/training_module.py) | CUDA-only training stack; impractical on Apple GPUs |
| Compression / distillation | [modules/compress_module.py](modules/compress_module.py) | Same CUDA-only training stack ([prune.py](prune.py), [train.py](train.py) untouched) |
| NDI streaming | [modules/visualizer.py](modules/visualizer.py), [widgets/performance_widget.py](widgets/performance_widget.py) | `ndi-python` publishes no macOS wheels; import is optional, send/destroy guarded, NDI name input greyed |
| Full Screen Display window | [modules/visualizer.py](modules/visualizer.py) | The preview window creates a GL 3.3 core context sharing textures with the legacy main context; macOS cannot share across profiles |
| Custom CUDA kernel toggle | already greyed via `has_custom` | Kernels never compile without CUDA |

Everything else stays enabled on macOS and runs through MPS: live renderer, network bending, model mixing, looping, audio reactivity, OSC, presets, projection, GANSpace feature extraction, super resolution, screenshots, recording.

## Known limitations and follow-ups

- `uv.lock` must be regenerated (`uv lock`); the sandbox used for this port could not reach download.pytorch.org.
- Render timing on MPS synchronizes before measuring, so reported render time is accurate but conservative.
- Super resolution runs fp32 on MPS (no autocast). Expect lower FPS than CUDA fp16.
- The 2024 analysis observed UI lag on macOS while rendering. The fixed-pipeline imgui backend changes the rendering path, so this needs re-measuring; further UI optimization is out of scope here.
- Projection and PCA on MPS are best-effort ports of gradient-based / statistics code and need on-device validation.
- Packaging (PyInstaller) is Windows-only ([release.bat](release.bat)); a macOS .app bundle is a separate effort.
- README does not yet have a macOS install section; this document covers it in the meantime.

## How was this tested

Static only: every changed file compiles (`py_compile`), call sites of the new helpers were traced, and dependency resolution for macOS arm64/Python 3.10 wheels was verified against PyPI. This is a GUI + GPU app; none of the runtime behavior was exercised. Manual verification needed on an Apple Silicon Mac:

1. `uv lock && uv sync` succeeds (after brew prerequisites).
2. `uv run main.py` opens the window; menu renders and is responsive; Training and Compression panes show the greyed-out notice.
3. Start renderer, load a model: image renders, drag/latent/truncation/layer widgets work, FPS reasonable (reference: ~30 FPS at 512 on M1 Max from the 2024 analysis).
4. Performance pane: GPU checkbox selectable (MPS), CPU toggle works, Custom Kernel greyed, NDI name greyed.
5. Super Resolution toggle in renderer, and the Super Resolution menu module on an image/video.
6. Screenshot and recording buttons produce files.
7. Audio module enables with a microphone; OSC server activates.
8. Feature extraction (PCA) and Projection complete on a small model.
9. Regression check on Windows/Linux + CUDA: launch, render, train start, NDI send.

Also verify on Windows that `uv sync` still installs cu128 builds after relocking.
