"""Build and send crash reports with user consent.

Two catch paths feed this module:
- Path A: ``main.py``'s top-level ``except BaseException`` calls
  :func:`handle_fatal_exception`, which shows a native OS dialog (imgui is
  dead at that point) and uploads on consent.
- Path B: a ``logs/last_run`` marker left behind by a hard native crash
  (segfault, driver death) is detected on the next launch by
  :func:`process_startup`; the imgui popup in ``modules/autolume_live``
  offers to send the previous run's logs.

Reports go to a Google Apps Script endpoint as JSON (metadata + base64
zip). The endpoint URL and token come from ``AUTOLUME_CRASH_REPORT_URL``
/ ``AUTOLUME_CRASH_REPORT_TOKEN`` environment variables, the repo-root
``.env`` in development (gitignored; see ``.env.example``), or the
``_endpoint_baked`` module ``release.py`` bakes into frozen
builds. Without any of them, ``reporting_available()`` is false and every
crash path becomes a no-op: no dialog, no popup, no upload.

Everything here must be safe to call from a dying process: no non-stdlib
imports, and no public function may raise.
"""
import base64
import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
import threading
import traceback
import urllib.request
import uuid
import zipfile
from pathlib import Path

from utils import user_data

logger = logging.getLogger(__name__)

MODE_ASK = "ask"
MODE_ALWAYS = "always"
MODE_NEVER = "never"

LOG_TAIL_BYTES = 500_000
CRASHES_TAIL_BYTES = 1_000_000
MAX_PAYLOAD_BYTES = 10_000_000

_endpoint_override = None
_endpoint_cache = None
_pending = None


def _parse_env(path):
    """Minimal KEY=VALUE parser for the repo-root .env file."""
    values = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            value = value.partition(" #")[0].partition("\t#")[0]
            values[key.strip()] = value.strip().strip("'\"")
    return values


def _endpoint_config():
    """(url, token) for the upload endpoint, or (None, None).

    Priority: environment variables, then the repo-root .env (development),
    then the _endpoint_baked module release.py bakes into frozen builds
    (present only inside a bundle, so it never shadows dev values).
    """
    global _endpoint_cache
    if _endpoint_override is not None:
        return _endpoint_override
    if _endpoint_cache is None:
        url = token = None
        try:
            url = os.environ.get("AUTOLUME_CRASH_REPORT_URL")
            token = os.environ.get("AUTOLUME_CRASH_REPORT_TOKEN")
            if not (url and token):
                env_file = Path(__file__).resolve().parent.parent / ".env"
                if env_file.exists():
                    values = _parse_env(env_file)
                    url = values.get("AUTOLUME_CRASH_REPORT_URL")
                    token = values.get("AUTOLUME_CRASH_REPORT_TOKEN")
            if not (url and token):
                import _endpoint_baked
                url, token = _endpoint_baked.endpoint()
        except Exception:
            pass
        if url and url.startswith("https://") and token:
            _endpoint_cache = (url, token)
        else:
            _endpoint_cache = (None, None)
    return _endpoint_cache


def reporting_available():
    """True if an upload endpoint is configured in this build."""
    return _endpoint_config()[0] is not None


def install_id():
    """Random per-installation id, stored beside config.json (not a pref)."""
    try:
        path = user_data.config_dir() / "install_id"
        if path.exists():
            value = path.read_text(encoding="utf-8").strip()
            if value:
                return value
        value = uuid.uuid4().hex
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
        return value
    except Exception:
        return "unknown"


def _read_tail(path, max_bytes):
    """Last ``max_bytes`` of a text file as a string, '' on any error."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fp:
            if size > max_bytes:
                fp.seek(size - max_bytes)
            return fp.read().decode("utf-8", "replace")
    except Exception:
        return ""


def _gpu_name():
    """GPU name via OS tools. Torch-free so it works in early startup
    crashes where torch never loaded, and cannot touch a wedged driver."""
    try:
        if sys.platform == "darwin":
            proc = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True, text=True, timeout=10)
            names = [line.split(":", 1)[1].strip()
                     for line in proc.stdout.splitlines()
                     if "Chipset Model:" in line]
        elif sys.platform == "win32":
            # wmic is removed from current Windows 11 builds.
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_VideoController).Name"],
                capture_output=True, text=True, timeout=15,
                creationflags=subprocess.CREATE_NO_WINDOW)
            names = [line.strip() for line in proc.stdout.splitlines()
                     if line.strip()]
        else:
            # lspci resolves names from the local pci.ids database, which
            # often lags behind new GPUs (codenames like "GN21-X11"); prefer
            # the driver's own name via nvidia-smi for NVIDIA devices.
            nv_names = []
            try:
                proc = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=10)
                if proc.returncode == 0:
                    nv_names = [line.strip() for line in proc.stdout.splitlines()
                                if line.strip()]
            except Exception:
                pass
            names = list(nv_names)
            proc = subprocess.run(
                ["lspci", "-mm"], capture_output=True, text=True, timeout=10)
            for line in proc.stdout.splitlines():
                parts = line.split('"')
                if len(parts) > 5 and parts[1] in (
                        "VGA compatible controller", "3D controller"):
                    if nv_names and parts[3].startswith("NVIDIA"):
                        continue
                    names.append(parts[3] + " " + parts[5])
        if names:
            return ", ".join(names)
    except Exception:
        pass
    return "unknown"


def _system_meta(crash_type, exception_text, comment):
    try:
        from utils.resource_paths import get_version
        version = get_version()
    except Exception:
        version = "unknown"
    return {
        "app_version": version,
        "os": sys.platform,
        "os_version": platform.platform(),
        "python": sys.version.split()[0],
        "frozen": bool(getattr(sys, "frozen", False)),
        "gpu": _gpu_name(),
        "crash_type": crash_type,
        "exception": exception_text[:20_000],
        "comment": comment[:5_000],
        "install_id": install_id(),
    }


def build_report(crash_type, exception_text, comment="", snapshot=None):
    """Assemble the report zip in a temp dir. Returns (zip path, meta) or None.

    ``snapshot`` (from :func:`check_unclean_exit`) supplies the log tails for
    path B; without it the current log files are read directly (path A).
    """
    try:
        meta = _system_meta(crash_type, exception_text, comment)
        log_dir = user_data.data_path("logs")
        if snapshot is not None:
            log_tail = snapshot.get("log_tail", "")
            crashes = snapshot.get("crashes", "")
        else:
            log_tail = _read_tail(log_dir / "autolume.log", LOG_TAIL_BYTES)
            crashes = _read_tail(log_dir / "crashes.log", CRASHES_TAIL_BYTES)

        tmp_dir = Path(tempfile.gettempdir()) / "autolume"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        zip_path = tmp_dir / ("crash-report-%s.zip" % uuid.uuid4().hex[:8])
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("report.json", json.dumps(meta, indent=2))
            if log_tail:
                zf.writestr("autolume.log.txt", log_tail)
            if crashes:
                zf.writestr("crashes.log.txt", crashes)
            config = user_data.config_file()
            if config.exists():
                zf.write(config, "config.json")
        return zip_path, meta
    except Exception:
        return None


def send_report(zip_path, meta):
    """Upload a report. Synchronous, ~20s timeout. True on success."""
    try:
        url, token = _endpoint_config()
        if url is None:
            return False
        body = json.dumps({
            "token": token,
            "meta": meta,
            "zip_b64": base64.b64encode(Path(zip_path).read_bytes()).decode(),
        }).encode("utf-8")
        if len(body) > MAX_PAYLOAD_BYTES:
            return False
        request = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=20) as response:
            result = response.read(1024).decode("utf-8", "replace")
        ok = '"ok"' in result
        if not ok:
            logger.warning("Crash report rejected: %s", result[:200])
        return ok
    except Exception:
        try:
            logger.warning("Crash report upload failed", exc_info=True)
        except Exception:
            pass
        return False


def _marker_path():
    return user_data.data_path("logs", "last_run")


def mark_running():
    """Arm the unclean-exit marker for this run."""
    try:
        path = user_data.ensure_data_path("logs") / "last_run"
        path.write_text(str(os.getpid()), encoding="utf-8")
    except Exception:
        pass


def mark_clean_exit():
    """Clear the marker: this run ended cleanly or its crash was handled."""
    try:
        _marker_path().unlink(missing_ok=True)
    except Exception:
        pass


def check_unclean_exit():
    """Detect a marker left by the previous run and snapshot its log tails.

    Must run before :func:`mark_running` and early in startup, so the tails
    predate most of this run's log output. Returns the snapshot (also kept
    as module state for :func:`pending_unclean_report`) or None.
    """
    global _pending
    try:
        marker = _marker_path()
        if not marker.exists():
            return None
        log_dir = marker.parent
        log_tail = _read_tail(log_dir / "autolume.log", LOG_TAIL_BYTES)
        if len(log_tail) < 50_000:
            # Fresh file after midnight rotation: the crashed run's output
            # is mostly in the newest dated backup.
            backups = sorted(log_dir.glob("autolume.log.*"))
            if backups:
                log_tail = _read_tail(
                    backups[-1], LOG_TAIL_BYTES - len(log_tail)) + log_tail
        _pending = {
            "log_tail": log_tail,
            "crashes": _read_tail(log_dir / "crashes.log", CRASHES_TAIL_BYTES),
        }
        return _pending
    except Exception:
        _pending = None
        return None


def pending_unclean_report():
    """Snapshot from :func:`check_unclean_exit`, or None."""
    return _pending


def clear_pending():
    global _pending
    _pending = None


def _send_snapshot(snapshot, comment):
    result = build_report("unclean_exit", "", comment=comment, snapshot=snapshot)
    if result is not None:
        send_report(*result)


def send_pending_report(comment=""):
    """Upload the pending unclean-exit report on a background thread."""
    snapshot = _pending
    clear_pending()
    if snapshot is None:
        return
    try:
        threading.Thread(
            target=_send_snapshot, args=(snapshot, comment), daemon=True).start()
    except Exception:
        pass


def process_startup():
    """Startup hook: detect an unclean previous exit, then arm the marker.

    In always mode the pending report is sent silently; in ask mode it is
    left for the imgui popup in modules/autolume_live.
    """
    try:
        # Sweep leftover report zips (uploads are synchronous per run).
        try:
            tmp_dir = Path(tempfile.gettempdir()) / "autolume"
            for stale in tmp_dir.glob("crash-report-*.zip"):
                try:
                    stale.unlink()
                except OSError:
                    pass
        except Exception:
            pass

        snapshot = check_unclean_exit()
        if (snapshot is not None
                and user_data.crash_report_mode() == MODE_ALWAYS
                and reporting_available()):
            send_pending_report()
        mark_running()
    except Exception:
        pass


DIALOG_TITLE = "Autolume"
DIALOG_MESSAGE = ("Autolume has crashed. You can send a crash report to the "
                  "developers. The report includes recent logs and system "
                  "information.")
BTN_SEND = "Send Report"
BTN_ALWAYS = "Always Send"
BTN_NO = "Don't Send"


def _dialog_windows():
    import ctypes
    from ctypes import wintypes

    # CommCtrl.h declares the TaskDialog structs with 1 byte packing
    # (pshpack1.h), so _pack_ = 1 is required, not an error.
    class TASKDIALOG_BUTTON(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("nButtonID", ctypes.c_int),
                    ("pszButtonText", wintypes.LPCWSTR)]

    class TASKDIALOGCONFIG(ctypes.Structure):
        _pack_ = 1
        _fields_ = [
            ("cbSize", wintypes.UINT),
            ("hwndParent", wintypes.HWND),
            ("hInstance", wintypes.HINSTANCE),
            ("dwFlags", wintypes.UINT),
            ("dwCommonButtons", wintypes.UINT),
            ("pszWindowTitle", wintypes.LPCWSTR),
            ("hMainIcon", ctypes.c_void_p),
            ("pszMainInstruction", wintypes.LPCWSTR),
            ("pszContent", wintypes.LPCWSTR),
            ("cButtons", wintypes.UINT),
            ("pButtons", ctypes.POINTER(TASKDIALOG_BUTTON)),
            ("nDefaultButton", ctypes.c_int),
            ("cRadioButtons", wintypes.UINT),
            ("pRadioButtons", ctypes.c_void_p),
            ("nDefaultRadioButton", ctypes.c_int),
            ("pszVerificationText", wintypes.LPCWSTR),
            ("pszExpandedInformation", wintypes.LPCWSTR),
            ("pszExpandedControlText", wintypes.LPCWSTR),
            ("pszCollapsedControlText", wintypes.LPCWSTR),
            ("hFooterIcon", ctypes.c_void_p),
            ("pszFooter", wintypes.LPCWSTR),
            ("pfCallback", ctypes.c_void_p),
            ("lpCallbackData", ctypes.c_void_p),
            ("cxWidth", wintypes.UINT),
        ]

    ids = {100: "send", 101: "always", 102: "no"}
    try:
        buttons = (TASKDIALOG_BUTTON * 3)(
            TASKDIALOG_BUTTON(100, BTN_SEND),
            TASKDIALOG_BUTTON(101, BTN_ALWAYS),
            TASKDIALOG_BUTTON(102, BTN_NO))
        config = TASKDIALOGCONFIG()
        config.cbSize = ctypes.sizeof(TASKDIALOGCONFIG)
        config.dwFlags = 0x0008  # TDF_ALLOW_DIALOG_CANCELLATION
        config.pszWindowTitle = DIALOG_TITLE
        config.pszMainInstruction = "Autolume has crashed"
        config.pszContent = DIALOG_MESSAGE
        config.cButtons = 3
        config.pButtons = buttons
        config.nDefaultButton = 100
        pressed = ctypes.c_int(0)
        result = ctypes.windll.comctl32.TaskDialogIndirect(
            ctypes.byref(config), ctypes.byref(pressed), None, None)
        if result == 0:
            return ids.get(pressed.value, "no")
    except Exception:
        pass
    try:
        # Fallback: two-choice MessageBoxW (always mode stays in Settings).
        MB_YESNO, MB_ICONERROR, IDYES = 0x4, 0x10, 6
        answer = ctypes.windll.user32.MessageBoxW(
            None, DIALOG_MESSAGE, DIALOG_TITLE, MB_YESNO | MB_ICONERROR)
        return "send" if answer == IDYES else "no"
    except Exception:
        return None


def _dialog_macos():
    labels = {BTN_NO: "no", BTN_ALWAYS: "always", BTN_SEND: "send"}
    script = (
        'display dialog "%s" with title "%s" buttons {"%s", "%s", "%s"} '
        'default button "%s" with icon caution'
        % (DIALOG_MESSAGE, DIALOG_TITLE, BTN_NO, BTN_ALWAYS, BTN_SEND, BTN_SEND))
    try:
        proc = subprocess.run(["osascript", "-e", script],
                              capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            return "no" if "-128" in proc.stderr else None  # -128: user cancel
        for label, choice in labels.items():
            if label in proc.stdout:
                return choice
        return None
    except Exception:
        return None


def _dialog_linux():
    try:
        proc = subprocess.run(
            ["zenity", "--question", "--title", DIALOG_TITLE,
             "--text", DIALOG_MESSAGE,
             "--ok-label", BTN_SEND, "--cancel-label", BTN_NO,
             "--extra-button", BTN_ALWAYS],
            capture_output=True, text=True, timeout=300)
        if proc.returncode == 0:
            return "send"
        if BTN_ALWAYS in proc.stdout:
            return "always"
        return "no"
    except Exception:
        return None


def show_crash_dialog():
    """Native three-choice consent dialog. None if none could be shown."""
    if sys.platform == "win32":
        return _dialog_windows()
    if sys.platform == "darwin":
        return _dialog_macos()
    return _dialog_linux()


def handle_fatal_exception():
    """Path A: called from main.py's except block while exc_info is live.

    Flushes logging first so the zipped log tail includes the traceback,
    then prompts (or honors the always/never pref) and uploads. Clears the
    unclean-exit marker when the incident was handled so the next launch
    does not prompt again. Never raises.
    """
    try:
        exception_text = "".join(traceback.format_exception(*sys.exc_info()))
        try:
            from utils.app_logging import shutdown_logging
            shutdown_logging()  # idempotent; main.py's finally call is a no-op
        except Exception:
            pass
        mode = user_data.crash_report_mode()
        if mode == MODE_NEVER or not reporting_available():
            mark_clean_exit()
            return
        choice = "send" if mode == MODE_ALWAYS else show_crash_dialog()
        if choice is None:
            return  # no dialog possible: leave the marker for path B
        if choice == "always":
            user_data.set_crash_report_mode(MODE_ALWAYS)
        if choice in ("send", "always"):
            result = build_report("exception", exception_text)
            if result is not None:
                send_report(*result)
        mark_clean_exit()
    except Exception:
        try:
            traceback.print_exc(file=sys.__stderr__)
        except Exception:
            pass
