# AGENTS.md

Guidance for AI coding assistants working on Autolume. Humans should read [CONTRIBUTING.md](CONTRIBUTING.md) instead.

**Maintenance principle**: only instructions an agent cannot discover from the repo — constraints, conventions, gotchas, commands. No overviews or restated docs; they add cost without helping ([arXiv:2602.11988](https://arxiv.org/abs/2602.11988)).

## Project snapshot

Autolume is a no-coding generative AI system for real-time visual performances using StyleGAN. Target users are artists who train, craft, and perform with their own generative models. Built on Python 3.12 + PyTorch 2.8 (CUDA 12.8), with an imgui-based desktop UI. Windows is the primary platform; Linux (Ubuntu 24.04) is supported.

## Commands

| Command | Purpose |
|---------|---------|
| `uv sync` | Install or update the locked dependency set |
| `uv run main.py` | Launch the application |
| `uv run pytest` | Run the unit test suite (headless logic only) |
| `uv run zensical build` | Build the documentation site to `site/` |
| `uv run zensical serve` | Serve docs locally at http://127.0.0.1:8000 |
| `uv run release.py` | Build the PyInstaller release (cross-platform): `dist/Autolume/` on Windows/Linux, `dist/Autolume.app` on macOS. Run it on the target OS — PyInstaller cannot cross-compile |

Initial setup also requires CUDA 12.8, MSVC build tools (Windows), `ffmpeg` (Linux), pre-trained Real-ESRGAN/face-parsing models, and an FFmpeg binary on Windows. Full details in the [README](README.md#development-instructions).

There is no architecture overview here on purpose (see the maintenance principle above) — explore the tree directly; [CONTRIBUTING.md](CONTRIBUTING.md#project-layout) has a directory table if you want one.

## Key Files

- [main.py](main.py) — entry point; wires the UI and rendering pipeline.
- [pyproject.toml](pyproject.toml) — pinned dependencies; `torch==2.8.0+cu128` is a hard requirement, do not relax it.
- [release.py](release.py) — cross-platform release script (`uv run release.py`); detects the host OS, resolves package locations via `importlib`, and assembles the PyInstaller `--add-binary`/`--add-data` flags per platform, then copies `sr_models/` and creates runtime dirs. **This is where to add new runtime files** — the auto-generated `Autolume.spec` is gitignored and rebuilt every release. Windows/Linux bundle the runtime JIT toolchain (torch headers/libs, ninja); macOS skips it (ops fall back to reference PyTorch on MPS) and produces `Autolume.app`. Requires a repo-root `.env` with the crash report endpoint values (baked in at build time) or it fails fast — see `.env.example` and [tools/crash_endpoint/README.md](tools/crash_endpoint/README.md); `--disable-crash-reporting` builds without one (for forks).
- [utils/user_data.py](utils/user_data.py) — user preferences and writable data paths. Preferences (data root, UI font size, …) persist to a JSON file at `~/.config/autolume/config.json` (`XDG_CONFIG_HOME` honored); the Settings modal ([modules/settings.py](modules/settings.py)) is their UI. **Add new user-facing preferences here** (a `pref()`/`set_pref()` accessor pair following the existing ones), not as ad-hoc files.
- [utils/gui_utils/dpi.py](utils/gui_utils/dpi.py) — all display-scale math (monitor DPI scale, font atlas raster scale, 1x text sharpening). The UI is sized in DPI-independent units so it keeps the same physical size and layout on every monitor and platform; start here for any scaling/blurriness issue.
- [.github/workflows/](.github/workflows/) — `docs.yml` publishes versioned docs; `tests.yml` runs the pytest suite on every PR.
- [zensical.toml](zensical.toml) — docs site config (Material theme variant).

## Code Style

- Project predates a formal style enforcer; match the surrounding code.
- Wrap `Path` / `WindowsPath` objects with `str()` before passing them anywhere that needs JSON-serializable values (training kwargs, OSC payloads, preset files). This is a recurring source of bugs.
- Imports follow standard Python order; vendored NVIDIA modules (`dnnlib`, `torch_utils`) keep their upstream style — do not reformat them.
- Keep new comments rare and load-bearing; do not paraphrase code.

## Testing

A pytest suite exists in `tests/` and runs via `uv run pytest`; CI ([.github/workflows/tests.yml](.github/workflows/tests.yml)) runs it on every PR. It covers headless logic only — tests must not require a display, CUDA, audio hardware, or model weights. Autolume is a desktop GUI app (imgui + GLFW + GPU), and an agent cannot drive the UI, see rendered frames, or verify visual output. Be explicit about this limit instead of pretending to have tested.

What an agent **can and should** do:

1. **Write unit tests for new modules**
   - When you add a module with headless-testable logic (state, parsing, math, protocols, file I/O), add a matching `tests/test_<module>.py` covering its core behavior and edge cases. Structure new code so that logic is separated from GUI/GPU glue and stays testable.
   - When you fix a bug in headless logic, add a regression test that reproduces it.
   - Run `uv run pytest` and confirm the suite passes before declaring the change done.
2. **Static checks**
   - Confirm imports resolve and types are sane using whatever static analysis is available to you.
   - Read the surrounding code paths and trace how the change interacts with callers.
3. **Smoke launch (optional, only when the user is at the desktop)**
   - `uv run main.py` will attempt to open a GLFW window. It will fail outright in any headless or sandboxed shell. If you do try, run it in the background, watch stderr for a crash within ~5–10 seconds, then stop it. A successful launch only proves the app starts — it proves nothing about the changed feature.
4. **Hand off GUI verification to a human**
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
- **Tests cover headless logic only** — a passing suite says nothing about the GUI; type checks and static analysis alone are not enough either. UI changes must be exercised manually.
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
