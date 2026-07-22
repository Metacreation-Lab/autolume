# AGENTS.md

Guidance for AI coding assistants working on Autolume. Humans should read [CONTRIBUTING.md](CONTRIBUTING.md) instead — most of this file is the same information distilled into a denser form.

## Project snapshot

Autolume is a no-coding generative AI system for real-time visual performances using StyleGAN. Target users are artists who train, craft, and perform with their own generative models. Built on Python 3.12 + PyTorch 2.8 (CUDA 12.8), with an imgui-based desktop UI. Windows is the primary platform; Linux (Ubuntu 24.04) is supported.

## Commands

| Command | Purpose |
|---------|---------|
| `uv sync` | Install or update the locked dependency set |
| `uv run main.py` | Launch the application |
| `uv run zensical build` | Build the documentation site to `site/` |
| `uv run zensical serve` | Serve docs locally at http://127.0.0.1:8000 |
| `uv run release.py` | Build the PyInstaller release (cross-platform): `dist/Autolume/` on Windows/Linux, `dist/Autolume.app` on macOS. Run it on the target OS — PyInstaller cannot cross-compile |

Initial setup also requires CUDA 12.8, MSVC build tools (Windows), `portaudio19-dev` + `ffmpeg` (Linux), pre-trained Real-ESRGAN/face-parsing models, and an FFmpeg binary on Windows. Full details in the [README](README.md#development-instructions).

## Architecture

```
autolume/
  main.py                     # Application entry point
  pyproject.toml              # Dependencies, Python pin (3.12), CUDA torch index
  release.py                  # Cross-platform release script (drives PyInstaller)

  architectures/              # Generator/discriminator network definitions
  audio/                      # Audio capture and analysis (librosa, pyaudio)
  bending/                    # Network bending operators applied at inference
  dnnlib/                     # Vendored NVIDIA dnnlib (StyleGAN2/3)
  ganspace/                   # GANSpace PCA latent direction discovery
  metrics/                    # FID, KID, and related quality metrics
  modules/                    # UI modules and live-performance widgets
  widgets/                    # Reusable imgui widgets
  projection/                 # Image-to-latent projection
  sr_models/                  # Super-resolution model weights (gitignored)
  super_res/                  # Real-ESRGAN super-resolution wrappers
  torch_utils/                # Vendored NVIDIA torch utilities (custom ops, persistence)
  training/                   # Model training and distillation
  utils/                      # Shared helpers (datasets, GUI, paths, version)
  assets/                     # Bundled images and icons

  docs/                       # User documentation (zensical, versioned with mike)
  .github/                    # Issue/PR templates, workflows
```

## Key Files

- [main.py](main.py) — entry point; wires the UI and rendering pipeline.
- [pyproject.toml](pyproject.toml) — pinned dependencies; `torch==2.8.0+cu128` is a hard requirement, do not relax it.
- [release.py](release.py) — cross-platform release script (`uv run release.py`); detects the host OS, resolves package locations via `importlib`, and assembles the PyInstaller `--add-binary`/`--add-data` flags per platform, then copies `sr_models/` and creates runtime dirs. **This is where to add new runtime files** — the auto-generated `Autolume.spec` is gitignored and rebuilt every release. Windows/Linux bundle the runtime JIT toolchain (torch headers/libs, ninja); macOS skips it (ops fall back to reference PyTorch on MPS) and produces `Autolume.app`. Requires a repo-root `.env` with the crash report endpoint values (baked in at build time) or it fails fast — see `.env.example` and [tools/crash_endpoint/README.md](tools/crash_endpoint/README.md); `--disable-crash-reporting` builds without one (for forks).
- [utils/user_data.py](utils/user_data.py) — user preferences and writable data paths. Preferences (data root, UI font size, …) persist to a JSON file at `~/.config/autolume/config.json` (`XDG_CONFIG_HOME` honored); the Settings modal ([modules/settings.py](modules/settings.py)) is their UI. **Add new user-facing preferences here** (a `pref()`/`set_pref()` accessor pair following the existing ones), not as ad-hoc files.
- [utils/gui_utils/dpi.py](utils/gui_utils/dpi.py) — all display-scale math (monitor DPI scale, font atlas raster scale, 1x text sharpening). The UI is sized in DPI-independent units so it keeps the same physical size and layout on every monitor and platform; start here for any scaling/blurriness issue.
- [.github/workflows/docs.yml](.github/workflows/docs.yml) — only CI workflow; publishes versioned docs.
- [zensical.toml](zensical.toml) — docs site config (Material theme variant).

## Code Style

- Project predates a formal style enforcer; match the surrounding code.
- Wrap `Path` / `WindowsPath` objects with `str()` before passing them anywhere that needs JSON-serializable values (training kwargs, OSC payloads, preset files). This is a recurring source of bugs.
- Imports follow standard Python order; vendored NVIDIA modules (`dnnlib`, `torch_utils`) keep their upstream style — do not reformat them.
- Keep new comments rare and load-bearing; do not paraphrase code.

## Testing

There is **no automated test suite**, and Autolume is a desktop GUI app (imgui + GLFW + GPU). An agent cannot drive the UI, see rendered frames, or verify visual output. Be explicit about this limit instead of pretending to have tested.

What an agent **can and should** do:

1. **Static checks**
   - Confirm imports resolve and types are sane using whatever static analysis is available to you.
   - Read the surrounding code paths and trace how the change interacts with callers.
2. **Smoke launch (optional, only when the user is at the desktop)**
   - `uv run main.py` will attempt to open a GLFW window. It will fail outright in any headless or sandboxed shell. If you do try, run it in the background, watch stderr for a crash within ~5–10 seconds, then stop it. A successful launch only proves the app starts — it proves nothing about the changed feature.
3. **Hand off GUI verification to a human**
   - In the PR's "How was this tested?" section, write what you *did* verify (e.g. "imports resolve, pyright clean, traced callers in `modules/X`") and explicitly call out what still needs human testing (e.g. "needs manual verification: load a model, change the noise widget, save preset, reload — confirm OSC mapping persists").

Do not claim to have tested a UI workflow you could not actually perform. Honest hand-off is more useful to maintainers than a fabricated checklist.

## Conventions

- **Conventional Commits without scope.** Format: `<type>: <subject>`. Types: `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `chore`, `ci`, `build`, `style`, `revert`. Subject is imperative, lowercase, no trailing period, ≤ 72 chars. Never use `feat(scope): …`. Full spec in [CONTRIBUTING.md](CONTRIBUTING.md#commit-message-convention).
- **PRs** use [the PR template](.github/PULL_REQUEST_TEMPLATE.md). Title must match the commit convention.
- **One logical change per PR.**

## Gotchas

- **Windows-first project.** Many code paths assume Windows; `Path` objects come back as `WindowsPath` and break JSON serialization (e.g. training kwargs, preset save). Wrap as `str()` defensively. See commits `d77d944`, `1000e66`, `b75b68d` for prior fixes.
- **CUDA 12.8 is mandatory.** `pyproject.toml` pins `torch==2.8.0+cu128` against a custom uv index. Do not suggest CPU torch, do not relax the pin to test a fix locally — it will break the release.
- **Custom CUDA ops are JIT-compiled** under `torch_utils/ops/`. Cache busting and runtime MSVC autoresolution exist (commits `8f806ee`, `da2b384`) — do not re-introduce env-var-based MSVC selection or break the cache key.
- **PyInstaller bundle is the shipped artifact.** New runtime files (help texts, models, assets) must be added to [release.py](release.py) — either to the shared `datas`/`binaries` lists, a per-platform branch, or the `post_build()` copy step. The auto-generated `Autolume.spec` is gitignored — never edit it, the change will be lost on the next build. The bundle also needs to load resources conditionally depending on whether the app is running from source or bundled — use the `utils/resource_paths.py` helpers. Note: app code that loads files via hardcoded `./relative/...` paths (e.g. `sr_models/`) resolves against the CWD, which is unreliable inside `Autolume.app` — prefer `resource_paths.resource_path(...)`.
- **No test suite** — relying on type checks or static analysis alone is not enough; you must exercise the UI manually.
- **Vendored NVIDIA code** in `dnnlib/` and `torch_utils/` carries the [Nvidia Source Code License](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/LICENSE.txt). Treat those modules as upstream — do not refactor them, and minimize edits.

## Workflow

- Use the [PR template](.github/PULL_REQUEST_TEMPLATE.md) — the checklist mirrors the gotchas above (`release.py` updates for new runtime files, manual test description, screenshots for UI work).
- For docs changes, edit Markdown files in [docs/](docs/) and verify locally with `uv run zensical serve`. The site is rebuilt on push to `main`.
- Do not put contributor-facing docs in [docs/](docs/) — that directory is for end-user documentation only. Contributor docs belong in [CONTRIBUTING.md](CONTRIBUTING.md) or this file.

## Pointers

- [README.md](README.md) — installation, model downloads, release process.
- [CONTRIBUTING.md](CONTRIBUTING.md) — full contribution guide for humans.
- [docs/](docs/) — user documentation source.
- [https://metacreation-lab.github.io/autolume](https://metacreation-lab.github.io/autolume) — published docs site.
