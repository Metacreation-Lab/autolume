# Contributing to Autolume

Thanks for your interest in improving Autolume! This document explains how to file issues, set up a development environment, and submit changes.

If you are looking for the user guide, see the [documentation site](https://metacreation-lab.github.io/autolume).

## Reporting issues

Please file all reports through the issue templates in [.github/ISSUE_TEMPLATE/](.github/ISSUE_TEMPLATE/):

- [Bug report](.github/ISSUE_TEMPLATE/bug_report.md) — defects, crashes, unexpected behavior.
- [Feature request](.github/ISSUE_TEMPLATE/feature_request.md) — new features or enhancements.

Please include a Autolume version, OS, GPU, and CUDA version for any bug — many issues are environment-specific.

## Development setup

The full setup (CUDA, `uv`, model downloads, FFmpeg, platform-specific dependencies) is documented in the [README's "Development instructions" section](README.md#development-instructions). Please follow it as written; the environment is sensitive (Python 3.12, CUDA 12.8, pinned PyTorch 2.8 wheel).

Quick reference once your environment is ready:

```bash
uv sync                  # install/update dependencies
uv run main.py           # run the application
uv run zensical serve    # serve the docs locally at http://127.0.0.1:8000
```

## Project layout

| Path | Purpose |
|------|---------|
| `main.py` | Application entry point |
| `architectures/` | Generator and discriminator network definitions |
| `bending/` | Network bending operators applied at inference |
| `dnnlib/` | Vendored NVIDIA dnnlib utilities (StyleGAN2/3) |
| `ganspace/` | GANSpace PCA-based latent direction discovery |
| `metrics/` | FID, KID, and related quality metrics |
| `modules/` | UI modules and live performance widgets |
| `widgets/` | Reusable imgui widgets |
| `projection/` | Image-to-latent projection |
| `super_res/`, `sr_models/` | Real-ESRGAN super-resolution pipeline |
| `torch_utils/` | Vendored NVIDIA torch utilities (custom ops, persistence) |
| `training/` | Model training and distillation |
| `utils/` | Shared helpers (datasets, GUI, paths, version) |
| `assets/` | Bundled images and icons |
| `docs/` | User documentation (rendered with zensical, versioned with mike) |
| `release.py` | Cross-platform release script (`uv run release.py`; drives PyInstaller and copies assets) |

## Making changes

- **Branch** off `main`. Pick a short, descriptive branch name.
- **One logical change per PR.** Smaller PRs land faster and are easier to review.
- **Update the docs** in [docs/](docs/) when you change user-visible behavior. The site is rebuilt automatically on push to `main`.
- **Update [release.py](release.py)** if you add new runtime files (help texts, models, assets) — add them to the shared `datas`/`binaries` lists, a per-platform branch, or the `post_build()` copy step, otherwise they will be missing from the packaged release. The auto-generated `Autolume.spec` is gitignored; do not edit it.
- **Put user preferences in [utils/user_data.py](utils/user_data.py)** if your change needs a persistent, user-facing setting (e.g. the data folder or UI font size). Add an accessor pair following the existing ones — preferences persist to a single JSON file (`~/.config/autolume/config.json`) — and expose the control in the Settings modal ([modules/settings.py](modules/settings.py)) rather than inventing a new config file.
- **No automated test suite exists.** Verify your change manually by running `uv run main.py` and exercising the affected UI path. Describe what you tested in the PR.

## Commit message convention

Autolume uses [Conventional Commits](https://www.conventionalcommits.org/) **without scopes**.

### Format

```
<type>: <subject>

[optional body]

[optional footer(s)]
```

- **No scopes.** Write `feat: …`, never `feat(audio): …`.
- **Subject** — imperative mood, lowercase first letter, no trailing period, ≤ 72 characters.
- **Body** (optional) — separated from the subject by a blank line; wrap at 100 characters; explain *why*, not *what*.
- **Breaking changes** — append `!` after the type and add a `BREAKING CHANGE:` footer:
  ```
  feat!: drop OSC v1 mapping format

  BREAKING CHANGE: presets saved before 2.16 must be re-exported with the new mapping schema.
  ```

### Allowed types

| Type | When to use |
|------|-------------|
| `feat` | New user-facing feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `refactor` | Code change that neither adds a feature nor fixes a bug |
| `perf` | Performance improvement |
| `test` | Adding or fixing tests |
| `chore` | Tooling, dependencies, repo maintenance |
| `ci` | CI/CD configuration |
| `build` | Build system, release packaging, PyInstaller spec |
| `style` | Formatting, whitespace; no behavior change |
| `revert` | Reverts a previous commit |

### Examples

Good:

```
fix: wrap Path objects as strings for training kwargs
feat: add lazy loading for thumbnail rendering
docs: document Linux installation steps
ci: publish versioned docs with mike
```

Bad:

```
feat(osc): add lazy loading            ← scope is not allowed
Update link to latest release 2.16.2   ← missing type prefix
Fixed bug.                             ← past tense, capitalized, no type
```

## Pull requests

- The PR title must follow the commit convention above (`<type>: <subject>`, no scope).
- Link the related issue with `Closes #N` or `Refs #N`.
- Fill out [the PR template](.github/PULL_REQUEST_TEMPLATE.md) — especially the "How was this tested?" section.
- Attach a screenshot or short clip for any UI change.
- Expect review feedback; we may ask for changes before merging.

## AI-assisted contributions

You are welcome to use AI coding assistants when contributing. We ask that:

- The author of the PR understands and stands behind every change.
- AI agents working in this repo should read [AGENTS.md](AGENTS.md) for project-specific conventions, gotchas, and verification steps.
- Generated content (commit messages, PR descriptions, code comments) should be reviewed and edited by a human before submission.
