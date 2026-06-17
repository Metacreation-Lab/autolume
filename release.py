"""Cross-platform PyInstaller build for Autolume (Windows / macOS / Linux).

Run with ``uv run release.py``. Detects the host platform and assembles the
right PyInstaller invocation:

- Windows / Linux bundle the runtime JIT toolchain (torch headers + libs, ninja,
  python headers) because the custom StyleGAN ops are compiled on first use when
  running on CUDA. Windows additionally ships ``python310.lib``.
- macOS skips that toolchain entirely (the ops fall back to reference PyTorch on
  MPS, see PORTING_MACOS.md) and produces an ``Autolume.app`` bundle instead of a
  plain folder.

ffmpeg/ffprobe are bundled on every platform via ffmpeg-downloader so the
artifact is self-contained.

PyInstaller cannot cross-compile: each artifact must be built on its own OS.
"""

import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import ffmpeg_downloader as ffdl

REPO = Path(__file__).resolve().parent
SYSTEM = platform.system()  # 'Windows' | 'Darwin' | 'Linux'
IS_WINDOWS = SYSTEM == "Windows"
IS_MACOS = SYSTEM == "Darwin"
IS_LINUX = SYSTEM == "Linux"

# Windows and Linux compile the custom CUDA ops at runtime; macOS does not.
NEEDS_JIT_TOOLCHAIN = IS_WINDOWS or IS_LINUX


def fail(message: str) -> "NoReturn":
    print(f"error: {message}", file=sys.stderr)
    sys.exit(1)


def package_dir(name: str) -> Path:
    """Locate an installed package's directory without hardcoding site-packages."""
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.submodule_search_locations:
        fail(f"could not locate the '{name}' package in the current environment")
    return Path(spec.submodule_search_locations[0])


def spec_arg(src: Path, dest: str) -> str:
    """Build a PyInstaller add-data/add-binary value using the host separator."""
    return f"{src}{os.pathsep}{dest}"


def ensure_ffmpeg() -> tuple[Path, Path]:
    """Make sure ffmpeg + ffprobe are downloaded and return their paths."""
    if not ffdl.installed():
        print("Installing ffmpeg via ffmpeg-downloader...")
        cmd = [sys.executable, "-m", "ffmpeg_downloader", "install", "-y"]
        if not IS_WINDOWS:
            cmd.append("--no-simlinks")  # don't touch the user's ~/.local/bin
        subprocess.run(cmd, check=True)
    ffmpeg = ffdl.ffmpeg_path and Path(ffdl.ffmpeg_path)
    ffprobe = ffdl.ffprobe_path and Path(ffdl.ffprobe_path)
    if not ffmpeg or not ffmpeg.exists() or not ffprobe or not ffprobe.exists():
        fail("ffmpeg/ffprobe not found after install; run `uv run ffdl install` manually")
    return ffmpeg, ffprobe


def glfw_native_lib() -> Path:
    """The GLFW shared library shipped inside the glfw package, per platform."""
    glfw = package_dir("glfw")
    patterns = {
        "Windows": "glfw3.dll",
        "Darwin": "libglfw*.dylib",
        "Linux": "libglfw*.so*",
    }
    matches = sorted(glfw.glob(patterns[SYSTEM]))
    if not matches:
        fail(f"no GLFW native library matching '{patterns[SYSTEM]}' in {glfw}")
    return matches[0]


def ninja_binary() -> Path:
    binary = package_dir("ninja") / "data" / "bin" / ("ninja.exe" if IS_WINDOWS else "ninja")
    if not binary.exists():
        fail(f"ninja binary not found at {binary}")
    return binary


def icon_path() -> Path | None:
    candidates = {
        "Windows": REPO / "assets" / "metacreation-logo.ico",
        "Darwin": REPO / "assets" / "metacreation-logo.icns",
        "Linux": REPO / "assets" / "metacreation-logo.png",
    }
    icon = candidates[SYSTEM]
    if not icon.exists():
        print(f"warning: icon {icon.name} not found; building without an icon")
        return None
    return icon


def build_args() -> list[str]:
    args = [sys.executable, "-m", "PyInstaller", "main.py", "--name", "Autolume", "--noconfirm"]

    icon = icon_path()
    if icon:
        args += ["--icon", str(icon)]

    if IS_MACOS:
        # Emits dist/Autolume.app
        args.append("--windowed")

    # --- Binaries shipped on every platform -------------------------------
    # ffmpeg/ffprobe (and ninja below) go in a bin/ subdir, not the bundle root:
    # on macOS/Linux the extensionless `ffmpeg` binary would otherwise collide
    # with the `ffmpeg` (ffmpeg-python) package PyInstaller packs at the root.
    # main.py adds this bin/ dir to PATH at runtime.
    ffmpeg, ffprobe = ensure_ffmpeg()
    binaries = [(ffmpeg, "bin"), (ffprobe, "bin"), (glfw_native_lib(), ".")]

    # --- Data files shipped on every platform -----------------------------
    clip = package_dir("clip")
    datas = [
        (REPO / "pyproject.toml", "."),
        (REPO / "models.csv", "."),
        (REPO / "modules" / "help_texts.csv", "modules"),
        (REPO / "architectures", "architectures"),
        (REPO / "assets", "assets"),
        (REPO / "training", "training"),
        (REPO / "torch_utils", "torch_utils"),
        (REPO / "sr_models", "sr_models"),
        (clip / "bpe_simple_vocab_16e6.txt.gz", "clip"),
    ]

    # --- Runtime JIT toolchain (Windows + Linux only) ---------------------
    if NEEDS_JIT_TOOLCHAIN:
        binaries.append((ninja_binary(), "bin"))  # same collision risk as ffmpeg

        torch_lib = package_dir("torch") / "lib"
        link_libs = "*.lib" if IS_WINDOWS else "*.so*"
        for lib in sorted(torch_lib.glob(link_libs)):
            binaries.append((lib, "torch/lib"))

        datas.append((package_dir("torch") / "include", "torch/include"))

        py_include = Path(sysconfig.get_path("include"))
        if py_include.exists():
            datas.append((py_include, "include"))

        if IS_WINDOWS:
            python_lib = Path(sys.base_prefix) / "libs" / "python310.lib"
            if python_lib.exists():
                binaries.append((python_lib, "libs"))
            else:
                print(f"warning: {python_lib} not found; runtime op compilation may fail")

    for src, dest in binaries:
        args += ["--add-binary", spec_arg(src, dest)]
    for src, dest in datas:
        if not src.exists():
            fail(f"missing data file: {src}")
        args += ["--add-data", spec_arg(src, dest)]

    args += ["--collect-all", "lpips", "--collect-all", "codecarbon"]
    return args


def clean() -> None:
    for name in ("dist", "build"):
        target = REPO / name
        if target.exists():
            print(f"Removing {name}/ ...")
            shutil.rmtree(target)


def post_build() -> None:
    # Read-only resources are bundled and resolved via utils.resource_paths, so
    # nothing needs to be copied next to the executable. Writable user-output
    # dirs (screenshots, recordings, presets, models, ...) are created lazily by
    # the features that use them, so nothing is seeded here.
    if IS_MACOS:
        print("Built dist/Autolume.app")
    else:
        print(f"Release created in {REPO / 'dist' / 'Autolume'}")


def main() -> None:
    print(f"Building Autolume for {SYSTEM}...")
    clean()
    args = build_args()
    print("Running PyInstaller...")
    subprocess.run(args, check=True, cwd=REPO)
    post_build()


if __name__ == "__main__":
    main()
