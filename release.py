"""Cross-platform PyInstaller build for Autolume (Windows / macOS / Linux).

Run with ``uv run release.py``. Detects the host platform and assembles the
right PyInstaller invocation:

- Windows / Linux precompile the custom StyleGAN CUDA ops for compute
  capabilities 7.5/8.6/8.9/12.0 (RTX 20/30/40/50 series) via
  ``scripts/precompile_ops.py`` and ship them in the bundle, so users need no
  compiler or CUDA toolkit. They also bundle the runtime JIT toolchain (torch
  headers + libs, ninja, python headers) so cards with other compute
  capabilities can fall back to compiling the ops on first use (this requires
  MSVC/GCC + CUDA nvcc on the user's machine). Windows additionally ships the
  Python import library (e.g. ``python312.lib``).
- macOS skips all of that (the ops fall back to reference PyTorch on MPS) and
  produces an ``Autolume.app`` bundle instead of a plain folder. The app is
  signed with the Developer ID Application identity from the keychain when one
  is present (hardened runtime + ``entitlements.plist``), falling back to
  ad-hoc signing with a warning. ``--package`` also signs the ``.dmg``.
  Notarization is a separate, explicit step: ``--notarize`` submits the
  packaged ``.dmg`` to Apple and staples the ticket, using the
  ``autolume-notary`` keychain profile (create it once with ``xcrun notarytool
  store-credentials``). It can be combined with ``--package`` or run alone
  against an already-built ``.dmg``.

ffmpeg/ffprobe are bundled on every platform via ffmpeg-downloader so the
artifact is self-contained (Windows uses gyan.dev's essentials build).

Pass ``--package`` to additionally wrap the build into the platform's
distributable format: ``.AppImage`` (zstd) on Linux, ``.dmg`` (lzma) on macOS,
an Inno Setup installer (lzma2) on Windows. Linux and Windows also get a
portable no-install ``.tar.xz``.
``--package-only`` skips the
PyInstaller build and packages an existing ``dist/`` output.

Pass ``--split`` to slice any ``dist/`` artifact over GitHub's 2 GiB
release-asset limit into ``<name>.partNN`` pieces (recombine with
``cat <name>.part* > <name>``). It runs after packaging when combined with
``--package``, or standalone against an existing ``dist/``.

PyInstaller cannot cross-compile: each artifact must be built on its own OS.
"""

import argparse
import functools
import importlib.util
import os
import platform
import plistlib
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import ffmpeg_downloader as ffdl

from utils.resource_paths import get_version

REPO = Path(__file__).resolve().parent
SYSTEM = platform.system()  # 'Windows' | 'Darwin' | 'Linux'
IS_WINDOWS = SYSTEM == "Windows"
IS_MACOS = SYSTEM == "Darwin"
IS_LINUX = SYSTEM == "Linux"

# Windows and Linux ship precompiled custom CUDA ops with a runtime JIT
# fallback for other compute capabilities; macOS uses neither.
NEEDS_JIT_TOOLCHAIN = IS_WINDOWS or IS_LINUX

# Staging dir for the precompiled ops bundled into the release.
PRECOMPILED_OPS_DIR = REPO / "build" / "torch_extensions"

APPIMAGETOOL_URL = "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage"

# GitHub rejects release assets larger than 2 GiB. --split slices anything over
# that into <name>.partNN pieces safely under the limit; recombine with
# `cat <name>.part* > <name>` (Linux/macOS) or `copy /b` on Windows.
GITHUB_ASSET_LIMIT = 2 * 1024**3
SPLIT_CHUNK = 1900 * 1024**2

# torch 2.8 wheels ship a libomp.dylib stamped minos 14.0 despite the
# macosx_11_0_arm64 tag (pytorch/pytorch#177140), so the bundle cannot load on
# anything older. Declaring it here turns a startup crash on old systems into
# a clear "requires macOS 14" dialog. Verify with check_macos_compat.py after
# bumping torch.
MACOS_MIN_VERSION = "14.0"

# Hardened-runtime exceptions for the signed macOS build.
ENTITLEMENTS = REPO / "entitlements.plist"

# notarytool keychain profile holding the App Store Connect API key.
NOTARY_PROFILE = "autolume-notary"


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


# Windows bundles gyan.dev's "essentials" ffmpeg (~97 MB/exe vs ~136 MB for
# the default full build); libx264 + aac is all Autolume uses. The major
# version is pinned because "release@essentials" resolves the latest version
# across all providers, usually a btbn post-release with no essentials build.
FFMPEG_WIN_SPEC = "8@essentials"


def ensure_ffmpeg() -> tuple[Path, Path]:
    """Make sure ffmpeg + ffprobe are downloaded and return their paths."""
    if IS_WINDOWS and ffdl.installed():
        banner = subprocess.run(
            [ffdl.ffmpeg_path, "-version"], capture_output=True, text=True
        ).stdout.partition("\n")[0]
        if "essentials_build" not in banner:
            print(f"Cached ffmpeg is not the essentials build ({banner}); replacing...")
            subprocess.run(
                [sys.executable, "-m", "ffmpeg_downloader", "uninstall", "-y"], check=True
            )
    if not ffdl.installed():
        print("Installing ffmpeg via ffmpeg-downloader...")
        cmd = [sys.executable, "-m", "ffmpeg_downloader", "install", "-y"]
        if IS_WINDOWS:
            cmd.append(FFMPEG_WIN_SPEC)
        else:
            cmd.append("--no-simlinks")  # don't touch the user's ~/.local/bin
        subprocess.run(cmd, check=True)
    ffmpeg = ffdl.ffmpeg_path and Path(ffdl.ffmpeg_path)
    ffprobe = ffdl.ffprobe_path and Path(ffdl.ffprobe_path)
    if not ffmpeg or not ffmpeg.exists() or not ffprobe or not ffprobe.exists():
        fail("ffmpeg/ffprobe not found after install; run `uv run ffdl install` manually")
    return ffmpeg, ffprobe


def bake_crash_endpoint() -> str:
    """Generate _endpoint_baked.py from the repo-root .env; return its dir.

    The endpoint URL and token live gitignored in .env on the release
    machine (see .env.example). Baking them into a generated module ships
    them inside the PYZ bytecode archive instead of a plaintext data file;
    XOR with a per-build key keeps them out of naive string scans. This is
    obfuscation, not protection. Failing the build here guarantees a
    release never silently ships without crash reporting.

    Written to a temp dir outside the repo so no secret-bearing file lands
    in the working tree; build_args wires it in via --paths and deletes it.
    """
    env_file = REPO / ".env"
    if not env_file.exists():
        fail(".env missing at the repo root; see .env.example and "
             "tools/crash_endpoint/README.md, or pass --disable-crash-reporting "
             "to build without it")
    values = {}
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            value = value.partition(" #")[0].partition("\t#")[0]
            values[key.strip()] = value.strip().strip("'\"")
    url = values.get("AUTOLUME_CRASH_REPORT_URL", "")
    token = values.get("AUTOLUME_CRASH_REPORT_TOKEN", "")
    if not url.startswith("https://") or not token:
        fail(".env malformed: expected AUTOLUME_CRASH_REPORT_URL (https) and "
             "AUTOLUME_CRASH_REPORT_TOKEN; see .env.example")
    key = os.urandom(16)
    def enc(s: str) -> bytes:
        return bytes(c ^ key[i % len(key)] for i, c in enumerate(s.encode()))
    tmp_dir = tempfile.mkdtemp(prefix="autolume-endpoint-")
    Path(tmp_dir, "_endpoint_baked.py").write_text(
        "# Generated by release.py at build time, baked into the frozen bundle.\n"
        f"_K = {key!r}\n"
        f"_U = {enc(url)!r}\n"
        f"_T = {enc(token)!r}\n\n\n"
        "def endpoint():\n"
        "    def d(b):\n"
        "        return bytes(c ^ _K[i % len(_K)] for i, c in enumerate(b)).decode()\n"
        "    return d(_U), d(_T)\n",
        encoding="utf-8")
    return tmp_dir


def glfw_native_libs() -> list[tuple[Path, str]]:
    """GLFW shared libraries to bundle, as (src, dest_dir) pairs for PyInstaller.

    On Linux the pyglfw wheel ships x11/ and wayland/ variants in subdirectories.
    The frozen glfw loader resolves the correct one via PYGLFW_LIBRARY_VARIANT or
    XDG_SESSION_TYPE, so both must be bundled preserving that directory structure.
    """
    glfw = package_dir("glfw")
    if IS_LINUX:
        result = []
        for variant in ("x11", "wayland"):
            matches = sorted((glfw / variant).glob("libglfw*.so*"))
            if not matches:
                fail(f"no GLFW native library in {glfw / variant}")
            result.append((matches[0], f"glfw/{variant}"))
        return result
    patterns = {"Windows": "glfw3.dll", "Darwin": "libglfw*.dylib"}
    matches = sorted(glfw.glob(patterns[SYSTEM]))
    if not matches:
        fail(f"no GLFW native library matching '{patterns[SYSTEM]}' in {glfw}")
    # Keep alongside library.py in _internal/glfw/ so pyglfw's frozen loader
    # finds it without env-var workarounds. On Windows this also ensures
    # MSVCR120.dll is pre-loaded before glfw3.dll (required on fresh installs).
    return [(matches[0], "glfw")]


def sounddevice_native_libs() -> list[tuple[Path, str]]:
    """PortAudio shared library sounddevice loads from its own package data.

    Windows/macOS wheels bundle it in _sounddevice_data/portaudio-binaries,
    a package next to (not inside) the sounddevice module, which is why it
    needs its own collection step rather than PyInstaller's normal module
    tracing. Linux has no such data dir: sounddevice there talks to the
    system libportaudio2 package instead, so there is nothing to bundle.
    """
    if IS_LINUX:
        return []
    data_dir = package_dir("_sounddevice_data") / "portaudio-binaries"
    patterns = {"Windows": "libportaudio*.dll", "Darwin": "libportaudio*.dylib"}
    matches = sorted(data_dir.glob(patterns[SYSTEM]))
    if not matches:
        fail(f"no PortAudio native library matching '{patterns[SYSTEM]}' in {data_dir}")
    return [(match, "_sounddevice_data/portaudio-binaries") for match in matches]


@functools.cache
def signing_identity() -> str | None:
    """Find a Developer ID Application identity in the keychain, or None."""
    result = subprocess.run(
        ["security", "find-identity", "-v", "-p", "codesigning"],
        capture_output=True, text=True,
    )
    for line in result.stdout.splitlines():
        if "Developer ID Application" in line:
            identity = line.split('"')[1]
            print(f"Signing with identity: {identity}")
            return identity
    print(
        "warning: no Developer ID Application identity in the keychain; "
        "signing ad-hoc (users will hit Gatekeeper warnings)",
        file=sys.stderr,
    )
    return None


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


def build_args(disable_crash_reporting: bool = False) -> tuple[list[str], str | None]:
    args = [sys.executable, "-m", "PyInstaller", "main.py", "--name", "Autolume", "--noconfirm", "--windowed"]

    icon = icon_path()
    if icon:
        args += ["--icon", str(icon)]

    if IS_MACOS:
        identity = signing_identity()
        if identity:
            # PyInstaller deep-signs every nested binary with the hardened
            # runtime, as notarization requires.
            args += [
                "--codesign-identity", identity,
                "--osx-entitlements-file", str(ENTITLEMENTS),
            ]

    # --- Binaries shipped on every platform -------------------------------
    # ffmpeg/ffprobe (and ninja below) go in a bin/ subdir, not the bundle root:
    # on macOS/Linux the extensionless `ffmpeg` binary would otherwise collide
    # with the `ffmpeg` (ffmpeg-python) package PyInstaller packs at the root.
    # main.py adds this bin/ dir to PATH at runtime.
    ffmpeg, ffprobe = ensure_ffmpeg()
    binaries = [
        (ffmpeg, "bin"), (ffprobe, "bin"), *glfw_native_libs(), *sounddevice_native_libs(),
    ]

    # --- Data files shipped on every platform -----------------------------
    clip = package_dir("clip")
    if disable_crash_reporting:
        endpoint_dir = None
        print("warning: building without crash reporting (--disable-crash-reporting)")
    else:
        endpoint_dir = bake_crash_endpoint()
    datas = [
        (REPO / "pyproject.toml", "."),
        (REPO / "models.csv", "."),
        (REPO / "modules" / "help_texts.csv", "modules"),
        (REPO / "architectures", "architectures"),
        (REPO / "assets", "assets"),
        (REPO / "training", "training"),
        (REPO / "torch_utils", "torch_utils"),
        (REPO / "sr_models" / "Fast.pt", "sr_models"),  # Quality/Balance download on first use
        (clip / "bpe_simple_vocab_16e6.txt.gz", "clip"),
    ]

    # --- Precompiled CUDA ops + runtime JIT toolchain (Windows + Linux) ---
    if NEEDS_JIT_TOOLCHAIN:
        datas.append((PRECOMPILED_OPS_DIR, "torch_extensions"))

        binaries.append((ninja_binary(), "bin"))  # same collision risk as ffmpeg

        torch_lib = package_dir("torch") / "lib"
        if IS_WINDOWS:
            # Only the import libraries that torch.utils.cpp_extension links
            # (see _prepare_ldflags there). The other *.lib in torch/lib are
            # static libraries the JIT never uses; dnnl.lib alone is 2.2 GB.
            jit_link_libs = ["c10.lib", "c10_cuda.lib", "torch.lib",
                             "torch_cpu.lib", "torch_cuda.lib", "torch_python.lib"]
            for name in jit_link_libs:
                lib = torch_lib / name
                if not lib.exists():
                    fail(f"expected torch import library not found: {lib}")
                binaries.append((lib, "torch/lib"))
        else:
            for lib in sorted(torch_lib.glob("*.so*")):
                binaries.append((lib, "torch/lib"))

        datas.append((package_dir("torch") / "include", "torch/include"))

        py_include = Path(sysconfig.get_path("include"))
        if py_include.exists():
            datas.append((py_include, "include"))

        if IS_WINDOWS:
            python_lib = (
                Path(sys.base_prefix)
                / "libs"
                / f"python{sys.version_info.major}{sys.version_info.minor}.lib"
            )
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

    # --paths finds the temp-dir module; --hidden-import forces its inclusion
    # (crash_report imports it inside a function, untraceable). None for a
    # --disable-crash-reporting build.
    if endpoint_dir is not None:
        args += ["--paths", endpoint_dir, "--hidden-import", "_endpoint_baked"]

    # PyOpenGL resolves its platform backend (GLX/EGL) via dynamic import at
    # startup. PyInstaller misses these because there's no static import to trace.
    if IS_LINUX:
        args += [
            "--hidden-import", "OpenGL.platform.glx",
            "--hidden-import", "OpenGL.platform.egl",
        ]

    return args, endpoint_dir


def precompile_ops() -> None:
    """Build the custom CUDA ops for the supported archs and stage them for bundling."""
    print("Precompiling custom CUDA ops...")
    subprocess.run(
        [sys.executable, str(REPO / "scripts" / "precompile_ops.py"), "--export", str(PRECOMPILED_OPS_DIR)],
        check=True,
        cwd=REPO,
    )


def clean() -> None:
    for name in ("dist", "build"):
        target = REPO / name
        if target.exists():
            print(f"Removing {name}/ ...")
            shutil.rmtree(target)


# Collected binaries the app can never reach, deleted from the bundle after
# PyInstaller runs (globs relative to dist/Autolume).
PRUNE_PATTERNS = [
    # PyInstaller copies nvrtc to the bundle root as an import of
    # caffe2_nvrtc.dll, but the loader resolves the torch/lib copy via torch's
    # add_dll_directory; the root copy is an 83 MB dead duplicate.
    "_internal/nvrtc64_*.dll",
    # Multi-GPU cusolver backend (150 MB); torch_cuda.dll does not import it
    # and nothing in Autolume reaches the cusolverMg APIs.
    "_internal/torch/lib/cusolverMg64_*.dll",
    # imageio-ffmpeg's own ffmpeg (62 MB); main.py points IMAGEIO_FFMPEG_EXE
    # at the ffmpeg already shipped in bin/.
    "_internal/imageio_ffmpeg/binaries/ffmpeg-*",
    # cuRAND (69 MB); nothing in the bundle references it, torch uses its own
    # Philox RNG on CUDA. Verified by training + inference with it removed.
    "_internal/torch/lib/curand64_*.dll",
    # cuDNN's RNN/attention sub-library (269 MB), loaded lazily by the cudnn
    # shim; StyleGAN is conv-only. Verified by training + inference with it
    # removed.
    "_internal/torch/lib/cudnn_adv64_*.dll",
    # Alternate nvrtc build (83 MB); nothing references it and the primary
    # nvrtc ships alongside.
    "_internal/torch/lib/nvrtc64_*.alt.dll",
]


def prune_bundle() -> None:
    bundle = REPO / "dist" / "Autolume"
    freed = 0
    for pattern in PRUNE_PATTERNS:
        for path in bundle.glob(pattern):
            freed += path.stat().st_size
            path.unlink()
            print(f"Pruned {path.relative_to(bundle)}")
    print(f"Pruning freed {freed / 1024 / 1024:.0f} MB")


def post_build() -> None:
    # Read-only resources are bundled and resolved via utils.resource_paths, so
    # nothing needs to be copied next to the executable. Writable user-output
    # dirs (screenshots, recordings, presets, models, ...) are created lazily by
    # the features that use them, so nothing is seeded here.
    if IS_MACOS:
        app = REPO / "dist" / "Autolume.app"
        plist = app / "Contents" / "Info.plist"
        info = plistlib.loads(plist.read_bytes())
        info["LSMinimumSystemVersion"] = MACOS_MIN_VERSION
        info["CFBundleShortVersionString"] = get_version()
        info["NSHumanReadableCopyright"] = (
            "Metacreation Lab for Creative AI\nmetacreation.net/autolume"
        )
        info["NSMicrophoneUsageDescription"] = (
            "Autolume uses the microphone for audio-reactive visuals."
        )
        plist.write_bytes(plistlib.dumps(info))
        # Editing Info.plist invalidates the bundle signature; re-sign the
        # outer bundle (nested binaries are untouched and stay valid).
        identity = signing_identity()
        if identity:
            subprocess.run(
                ["codesign", "--force", "--options", "runtime", "--timestamp",
                 "--entitlements", str(ENTITLEMENTS), "--sign", identity, str(app)],
                check=True,
            )
        else:
            subprocess.run(["codesign", "--force", "--sign", "-", str(app)], check=True)
        print(f"Built dist/Autolume.app (requires macOS {MACOS_MIN_VERSION}+)")
    else:
        prune_bundle()
        print(f"Release created in {REPO / 'dist' / 'Autolume'}")


def artifact_name(ext: str) -> str:
    return f"autolume-{get_version()}-{SYSTEM.lower()}-{platform.machine().lower()}{ext}"


def package_portable_archive() -> None:
    """Create the portable no-install ``.tar.xz`` from ``dist/Autolume``.

    tar preserves the symlinks and executable bits in the Linux bundle, which
    zip would not. Windows 11 23H2+ extracts .tar.xz natively (File Explorer
    and bsdtar); older Windows needs 7-Zip. Prefers system tar piped into
    multithreaded xz (minutes); falls back to stdlib lzma, which is
    single-threaded and takes ~20 minutes for the ~5 GB bundle.
    """
    archive = REPO / "dist" / artifact_name(".tar.xz")
    tar_bin, xz_bin = shutil.which("tar"), shutil.which("xz")
    if tar_bin and xz_bin:
        print(f"Creating {archive.name} (multithreaded xz)...")
        with open(archive, "wb") as out:
            tar_proc = subprocess.Popen(
                [tar_bin, "-C", str(REPO / "dist"), "-cf", "-", "Autolume"],
                stdout=subprocess.PIPE,
            )
            xz_proc = subprocess.Popen([xz_bin, "-T0"], stdin=tar_proc.stdout, stdout=out)
            tar_proc.stdout.close()
            xz_failed = xz_proc.wait() != 0
            tar_failed = tar_proc.wait() != 0
        if tar_failed or xz_failed:
            archive.unlink(missing_ok=True)
            fail("tar | xz pipeline failed")
    else:
        print(f"Creating {archive.name} (single-threaded xz; this takes a while)...")
        with tarfile.open(archive, "w:xz") as tar:
            tar.add(REPO / "dist" / "Autolume", arcname="Autolume")
    print(f"Packaged {archive}")


def appimagetool_path() -> Path:
    on_path = shutil.which("appimagetool")
    if on_path:
        return Path(on_path)
    tool = REPO / "build" / "tools" / "appimagetool-x86_64.AppImage"
    if not tool.exists():
        print("Downloading appimagetool...")
        tool.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(APPIMAGETOOL_URL, tool)
        tool.chmod(0o755)
    return tool


def package_linux() -> None:
    appdir = REPO / "build" / "Autolume.AppDir"
    if appdir.exists():
        shutil.rmtree(appdir)
    shutil.copytree(REPO / "dist" / "Autolume", appdir / "usr" / "bin", symlinks=True)

    apprun = appdir / "AppRun"
    apprun.write_text(
        '#!/bin/sh\n'
        'HERE="$(dirname "$(readlink -f "$0")")"\n'
        'exec "$HERE/usr/bin/Autolume" "$@"\n'
    )
    apprun.chmod(0o755)

    (appdir / "Autolume.desktop").write_text(
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=Autolume\n"
        "Exec=Autolume\n"
        "Icon=autolume\n"
        "Categories=Graphics;\n"
        "Terminal=false\n"
    )
    shutil.copy2(REPO / "assets" / "metacreation-logo.png", appdir / "autolume.png")

    output = REPO / "dist" / artifact_name(".AppImage")
    # APPIMAGE_EXTRACT_AND_RUN lets appimagetool run without FUSE on the build host.
    env = {**os.environ, "ARCH": "x86_64", "APPIMAGE_EXTRACT_AND_RUN": "1"}
    # appimagetool's bundled mksquashfs only supports zstd; level 22 is max.
    subprocess.run(
        [str(appimagetool_path()), "--comp", "zstd",
         "--mksquashfs-opt", "-Xcompression-level", "--mksquashfs-opt", "22",
         str(appdir), str(output)],
        check=True, env=env,
    )
    print(f"Packaged {output}")

    package_portable_archive()


def package_macos() -> None:
    staging = REPO / "build" / "dmg"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    # ditto preserves the code signature; shutil.copytree can break it.
    subprocess.run(
        ["ditto", str(REPO / "dist" / "Autolume.app"), str(staging / "Autolume.app")],
        check=True,
    )
    os.symlink("/Applications", staging / "Applications")

    output = REPO / "dist" / artifact_name(".dmg")
    subprocess.run(
        ["hdiutil", "create", "-volname", "Autolume", "-srcfolder", str(staging),
         "-format", "ULMO", "-ov", str(output)],
        check=True,
    )

    identity = signing_identity()
    if identity:
        subprocess.run(
            ["codesign", "--timestamp", "--sign", identity, str(output)], check=True
        )
    else:
        print("warning: DMG is unsigned; Gatekeeper will block it", file=sys.stderr)
    print(f"Packaged {output} (pass --notarize to submit it to Apple)")


def notarize(dmg: Path) -> None:
    """Submit the DMG to Apple's notary service and staple the ticket.

    Notarizing the DMG covers the .app inside it, which must already carry a
    hardened-runtime Developer ID signature.
    """
    print("Notarizing (Apple usually takes a few minutes)...")
    result = subprocess.run(
        ["xcrun", "notarytool", "submit", str(dmg),
         "--keychain-profile", NOTARY_PROFILE, "--wait"],
    )
    if result.returncode != 0:
        fail(
            "notarization failed; the DMG is signed but Gatekeeper will "
            f"block it on other machines. If the '{NOTARY_PROFILE}' keychain "
            "profile is missing, create it once with `xcrun notarytool "
            f"store-credentials {NOTARY_PROFILE} --key <AuthKey.p8> "
            "--key-id <key-id> --issuer-id <issuer-id>` using an App Store "
            "Connect API key (see README). If the submission was rejected, "
            "inspect it with `xcrun notarytool log <submission-id> "
            f"--keychain-profile {NOTARY_PROFILE}`."
        )
    subprocess.run(["xcrun", "stapler", "staple", str(dmg)], check=True)
    print("Notarized and stapled.")


def iscc_path() -> Path:
    on_path = shutil.which("ISCC")
    if on_path:
        return Path(on_path)
    # Inno Setup never adds itself to PATH. Version 7 defaults to a per-user
    # install under LOCALAPPDATA\Programs; 6 (and machine-wide 7) go to
    # Program Files (x86).
    roots = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs",
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")),
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")),
    ]
    for root in roots:
        for version in ("Inno Setup 7", "Inno Setup 6"):
            candidate = root / version / "ISCC.exe"
            if candidate.exists():
                return candidate
    fail("Inno Setup (ISCC.exe) not found; install it from https://jrsoftware.org/isdl.php")


def package_windows() -> None:
    version = get_version()
    output_base = artifact_name("-setup")
    # Inno Setup caps single-file output at ~2 GB; if a build ever exceeds it,
    # add DiskSpanning=yes (splits the installer into .bin slices).
    iss = REPO / "build" / "autolume.iss"
    iss.parent.mkdir(parents=True, exist_ok=True)
    iss.write_text(f"""\
[Setup]
AppName=Autolume
AppId=Autolume
AppVersion={version}
AppPublisher=Metacreation Lab
CloseApplications=yes
DefaultDirName={{autopf}}\\Autolume
DefaultGroupName=Autolume
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
Compression=lzma2
SolidCompression=yes
SetupIconFile={REPO / 'assets' / 'metacreation-logo.ico'}
UninstallDisplayIcon={{app}}\\Autolume.exe
OutputDir={REPO / 'dist'}
OutputBaseFilename={output_base}

[Tasks]
Name: "desktopicon"; Description: "{{cm:CreateDesktopIcon}}"; Flags: unchecked

[InstallDelete]
Type: filesandordirs; Name: "{{app}}\\_internal"

[Files]
Source: "{REPO / 'dist' / 'Autolume'}\\*"; DestDir: "{{app}}"; Flags: recursesubdirs createallsubdirs

[Icons]
Name: "{{group}}\\Autolume"; Filename: "{{app}}\\Autolume.exe"
Name: "{{autodesktop}}\\Autolume"; Filename: "{{app}}\\Autolume.exe"; Tasks: desktopicon
""")
    subprocess.run([str(iscc_path()), str(iss)], check=True)
    print(f"Packaged {REPO / 'dist' / (output_base + '.exe')}")

    package_portable_archive()


def package() -> None:
    if IS_LINUX:
        package_linux()
    elif IS_MACOS:
        package_macos()
    else:
        package_windows()


def split_large_artifacts() -> None:
    """Split any dist/ file over GitHub's 2 GiB asset limit into upload-ready parts.

    Only top-level files in dist/ are considered (the unpacked bundle folder is
    left alone). Each oversized file is sliced into <name>.partNN pieces under
    the limit; the original is kept (upload only the parts). Streamed in a small
    buffer so it never loads a whole part into memory.
    """
    dist = REPO / "dist"
    if not dist.exists():
        fail(f"{dist} not found; build or package first")

    oversized = [
        f for f in sorted(dist.iterdir())
        if f.is_file() and f.stat().st_size > GITHUB_ASSET_LIMIT
    ]
    if not oversized:
        print(f"No dist/ artifact exceeds {GITHUB_ASSET_LIMIT // 1024**3} GiB; nothing to split")
        return

    buffer_size = 8 * 1024**2
    for artifact in oversized:
        size = artifact.stat().st_size
        parts = -(-size // SPLIT_CHUNK)  # ceil division
        print(f"Splitting {artifact.name} ({size / 1024**3:.1f} GiB) into {parts} parts...")
        with open(artifact, "rb") as src:
            index = 0
            part = None
            written = 0
            while chunk := src.read(buffer_size):
                if part is None or written >= SPLIT_CHUNK:
                    if part is not None:
                        part.close()
                    index += 1
                    part = open(artifact.with_name(f"{artifact.name}.part{index:03d}"), "wb")
                    written = 0
                part.write(chunk)
                written += len(chunk)
            if part is not None:
                part.close()
        print(f"  split into {index} parts; recombine with `cat {artifact.name}.part* > {artifact.name}`")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Autolume PyInstaller release.")
    parser.add_argument(
        "--package", action="store_true",
        help="wrap the build into a distributable (.AppImage / .dmg / Inno Setup "
             "installer, plus a portable .tar.xz on Linux and Windows)",
    )
    parser.add_argument(
        "--package-only", action="store_true",
        help="skip the PyInstaller build and only package an existing dist/ output",
    )
    parser.add_argument(
        "--notarize", action="store_true",
        help="submit the packaged .dmg to Apple's notary service and staple the "
             "ticket (macOS only); combine with --package or run alone against "
             "an existing .dmg",
    )
    parser.add_argument(
        "--split", action="store_true",
        help="slice any dist/ artifact over GitHub's 2 GiB asset limit into "
             ".partNN pieces (keeping the original); combine with --package or "
             "run alone against an existing dist/",
    )
    parser.add_argument(
        "--disable-crash-reporting", action="store_true",
        help="build without a crash report endpoint (no .env required); crash "
             "dialogs and uploads are disabled. For forks without their own "
             "endpoint",
    )
    opts = parser.parse_args()

    if opts.notarize and not IS_MACOS:
        fail("--notarize only applies to the macOS .dmg")

    # --notarize and --split are post-steps that run against an existing dist/;
    # requesting only those (without --package) must not trigger a rebuild.
    post_only = opts.notarize or opts.split

    if opts.package_only:
        bundle = REPO / "dist" / ("Autolume.app" if IS_MACOS else "Autolume")
        if not bundle.exists():
            fail(f"{bundle} not found; run `uv run release.py` first")
        package()
    elif opts.package or not post_only:
        print(f"Building Autolume for {SYSTEM}...")
        clean()
        if NEEDS_JIT_TOOLCHAIN:
            precompile_ops()
        args, endpoint_dir = build_args(opts.disable_crash_reporting)
        print("Running PyInstaller...")
        try:
            subprocess.run(args, check=True, cwd=REPO)
        finally:
            if endpoint_dir is not None:
                shutil.rmtree(endpoint_dir, ignore_errors=True)
        post_build()
        if opts.package:
            package()

    if opts.notarize:
        dmg = REPO / "dist" / artifact_name(".dmg")
        if not dmg.exists():
            fail(f"{dmg} not found; run `uv run release.py --package` first")
        notarize(dmg)

    if opts.split:
        split_large_artifacts()


if __name__ == "__main__":
    main()
