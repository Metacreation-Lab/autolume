"""Build and send crash reports with user consent.

Two catch paths feed this module:
- Path A: ``main.py``'s top-level ``except BaseException`` calls
  :func:`handle_fatal_exception`, which shows a native OS dialog (imgui is
  dead at that point) and uploads on consent.
- Path B: a per-process marker under ``logs/runs`` left behind by a hard
  native crash (segfault, driver death) is detected on the next launch by
  :func:`process_startup`; the imgui popup in ``modules/autolume_live``
  offers to send the previous run's logs.

Many things besides a crash can kill the process uncleanly — an OS shutdown,
a closed terminal, Task Manager — so path B only reports on positive crash
evidence: faulthandler (armed in ``utils/app_logging``) appends to
``logs/crashes.log`` at the moment of a native fault, and a dead run's marker
turns into a report only when a fault dump landed there during the run. Session
ends are additionally recognised on their own (a session-end hook stamps the
marker, a heartbeat correlates the death with the OS shutdown clock) so they
can be logged as such; every other evidence-free death is dropped with a log
line, the same way engines like Unreal only report when their crash handler
actually fired.

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
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
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

MARKER_VERSION = 2
HEARTBEAT_SECONDS = 15
# How far from the last heartbeat the OS may record a shutdown and still
# explain the death: clock jitter on the early side, the time the rest of the
# session takes to tear down on the late side. A run that stopped breathing
# well before the machine went down died of something else.
SHUTDOWN_MATCH_BEFORE = 60
SHUTDOWN_MATCH_AFTER = 300
# How long a recorded session end stays believable while the run carries on.
SESSION_END_GRACE = 120

_endpoint_override = None
_endpoint_cache = None
_pending = None

_marker_lock = threading.Lock()
_marker_base = None
_exit_reason = None
_exit_reason_at = 0.0
_heartbeat_stop = threading.Event()


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


def _runs_dir():
    return user_data.data_path("logs", "runs")


def _marker_path():
    """This run's own marker. One file per process, so concurrent instances
    never overwrite each other's heartbeat."""
    return _runs_dir() / ("%d.json" % os.getpid())


def _own_create_time():
    """This process's OS-assigned start time, or None if psutil is broken."""
    try:
        import psutil
        return psutil.Process().create_time()
    except Exception:
        return None


def _still_running(marker):
    """True only when the marker's process is verifiably still alive: same
    pid *and* same creation time, so a recycled pid cannot make a crashed run
    look alive. Anything unverifiable counts as dead and gets classified."""
    pid = marker.get("pid")
    create = marker.get("create")
    if not isinstance(pid, int) or not isinstance(create, (int, float)):
        return False
    try:
        import psutil
        return abs(psutil.Process(pid).create_time() - create) < 1.0
    except Exception:
        return False


def _write_marker():
    """Rewrite the marker with a fresh heartbeat. Atomic, so a kill mid-write
    cannot leave a truncated file behind. No-op once the marker is cleared."""
    try:
        with _marker_lock:
            if _marker_base is None:
                return
            payload = dict(_marker_base, heartbeat=time.time(), exit=_exit_reason)
            path = _marker_path()
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(tmp, path)
    except Exception:
        pass


def _read_marker(path=None):
    """Marker contents as a dict, or None if there is no marker.

    Markers written before the heartbeat existed hold a bare pid; report those
    as version 1 so they classify as a crash, exactly as they used to.
    """
    try:
        data = json.loads((path or _marker_path()).read_text(encoding="utf-8"))
    except OSError:
        return None
    except ValueError:
        return {"version": 1}
    return data if isinstance(data, dict) else {"version": 1}


def _heartbeat_tick():
    if _exit_reason and time.time() - _exit_reason_at > SESSION_END_GRACE:
        # Windows kills what is left of a session within seconds, so still
        # running this long after being asked to go means the shutdown was
        # called off. Drop the stamp rather than let it suppress a real crash
        # later in the run.
        _set_exit_reason(None)
    else:
        _write_marker()


def _heartbeat_loop():
    while not _heartbeat_stop.wait(HEARTBEAT_SECONDS):
        _heartbeat_tick()


def mark_running():
    """Arm the unclean-exit marker for this run and start its heartbeat."""
    global _marker_base
    try:
        user_data.ensure_data_path("logs", "runs")
        # Recording the shutdown clock as it stands now is what lets the next
        # launch tell a *new* shutdown from the one before this run started.
        base = {"version": MARKER_VERSION, "pid": os.getpid(),
                "create": _own_create_time(),
                "started": time.time(),
                "os_shutdown": _last_os_shutdown_time(),
                "crashes_size": _crashes_log_size()}
        with _marker_lock:
            _marker_base = base
        _write_marker()
        _heartbeat_stop.clear()
        threading.Thread(target=_heartbeat_loop, daemon=True).start()
    except Exception:
        pass


def mark_clean_exit():
    """Clear the marker: this run ended cleanly or its crash was handled."""
    global _marker_base
    try:
        _heartbeat_stop.set()
        with _marker_lock:
            _marker_base = None
            _marker_path().unlink(missing_ok=True)
    except Exception:
        pass


def _set_exit_reason(reason):
    """Stamp why this run is ending into the marker. Called from a WndProc, a
    console control handler and a signal handler, so it must never raise."""
    global _exit_reason, _exit_reason_at
    try:
        _exit_reason = reason
        _exit_reason_at = time.time()
        _write_marker()
    except Exception:
        pass


def _last_os_shutdown_time():
    """When the OS last finished shutting down, or None if unknown.

    Windows only: it is the one platform where the shipped build is windowed,
    gets no SIGTERM, and can be killed at session end before the message pump
    has had a chance to run. The value is an 8-byte FILETIME the session
    manager writes at the end of every clean shutdown, readable unelevated.
    """
    if sys.platform != "win32":
        return None
    try:
        import winreg
        with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SYSTEM\CurrentControlSet\Control\Windows") as key:
            raw, kind = winreg.QueryValueEx(key, "ShutdownTime")
        if kind != winreg.REG_BINARY or len(raw) != 8:
            return None
        # FILETIME: 100ns ticks since 1601-01-01.
        return struct.unpack("<Q", raw)[0] / 10_000_000.0 - 11_644_473_600.0
    except Exception:
        return None


def _crashes_log_size():
    """Current size of the native crash log, or 0 if there is none."""
    try:
        return os.path.getsize(user_data.data_path("logs", "crashes.log"))
    except OSError:
        return 0


def _crash_evidence(marker):
    """True when faulthandler dumped a fault while the marker's run was alive.

    The crash log is append-only but not fault-only: every process appends a
    start banner (see utils/app_logging), and each run spawns workers that do
    the same, so growth alone proves nothing. What counts is the content
    faulthandler wrote after the run armed its marker: its dumps always open
    with "Fatal Python error" or "Windows fatal exception". A kill that
    leaves no such trace (Task Manager, power loss, a frozen app put down by
    the user) produces no report: only what the handler caught gets offered,
    the way engine-style crash reporters behave.
    """
    offset = marker.get("crashes_size")
    if not isinstance(offset, int):
        return False
    try:
        with open(user_data.data_path("logs", "crashes.log"), "rb") as fh:
            fh.seek(0, os.SEEK_END)
            if fh.tell() < offset:
                # Truncated or replaced since the run armed; nothing left in
                # it can be attributed to the run.
                return False
            fh.seek(offset)
            added = fh.read(4 * 1024 * 1024)
    except OSError:
        return False
    return b"fatal" in added.lower()


def _shutdown_evidence(marker):
    """Why the previous run should be read as an OS shutdown, or None.

    Returning None means "no shutdown was involved": a missing heartbeat, a
    platform with no shutdown clock, a shutdown that does not line up with
    the moment the run stopped breathing. :func:`_crash_evidence` then
    decides whether the death is worth a report.
    """
    if marker.get("exit") == "shutdown":
        return "the session end was signalled"
    heartbeat = marker.get("heartbeat")
    if not isinstance(heartbeat, (int, float)):
        return None
    shutdown = _last_os_shutdown_time()
    if shutdown is None or shutdown == marker.get("os_shutdown"):
        # No shutdown has been recorded since the run started, so whatever
        # ended it, the machine going down was not it.
        return None
    delta = shutdown - heartbeat
    if -SHUTDOWN_MATCH_BEFORE <= delta <= SHUTDOWN_MATCH_AFTER:
        return "it was still alive %.0fs before the OS finished shutting down" % delta
    return None


GWLP_WNDPROC = -4
WM_QUERYENDSESSION = 0x0011
WM_ENDSESSION = 0x0016

_wndproc_hwnd = None
_wndproc_old = None
_wndproc_new = None


def _install_windows_wndproc(hwnd):
    """Subclass the window so WM_QUERYENDSESSION reaches us."""
    global _wndproc_hwnd, _wndproc_old, _wndproc_new
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    LRESULT = ctypes.c_ssize_t
    WPARAM = ctypes.c_size_t
    LPARAM = ctypes.c_ssize_t
    WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, ctypes.c_uint,
                                 WPARAM, LPARAM)

    # SetWindowLongW truncates a 64-bit proc pointer, and the Ptr variant is
    # only exported on 64-bit, where it is the one that must be used.
    set_long = getattr(user32, "SetWindowLongPtrW", None) or user32.SetWindowLongW
    set_long.restype = LRESULT
    set_long.argtypes = [wintypes.HWND, ctypes.c_int, LRESULT]
    user32.CallWindowProcW.restype = LRESULT
    user32.CallWindowProcW.argtypes = [LRESULT, wintypes.HWND, ctypes.c_uint,
                                       WPARAM, LPARAM]
    user32.DefWindowProcW.restype = LRESULT
    user32.DefWindowProcW.argtypes = [wintypes.HWND, ctypes.c_uint, WPARAM, LPARAM]

    def dispatch(hwnd_, msg, wparam, lparam):
        try:
            if msg == WM_QUERYENDSESSION:
                _set_exit_reason("shutdown")
            elif msg == WM_ENDSESSION:
                # A false wParam means the session end was called off again.
                _set_exit_reason("shutdown" if wparam else None)
        except Exception:
            pass
        old = _wndproc_old
        if not old:
            return user32.DefWindowProcW(hwnd_, msg, wparam, lparam)
        return user32.CallWindowProcW(old, hwnd_, msg, wparam, lparam)

    # GLFW keeps driving the window, so its proc must stay in the chain, and
    # the callback object has to outlive this call or the pump lands on freed
    # memory.
    _wndproc_new = WNDPROC(dispatch)
    _wndproc_old = set_long(
        hwnd, GWLP_WNDPROC, ctypes.cast(_wndproc_new, ctypes.c_void_p).value)
    _wndproc_hwnd = hwnd


def _install_posix_signals():
    """macOS and Linux announce session end with SIGTERM."""
    def on_terminate(signum, frame):
        # Stamp, then die the way the OS expects.
        _set_exit_reason("shutdown")
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for name in ("SIGTERM", "SIGHUP"):
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                signal.signal(sig, on_terminate)
            except (ValueError, OSError):
                pass


_console_handler = None


def _install_console_ctrl_handler():
    """Closing the terminal kills the process with no window message and no
    signal; the console control event is the only notice Windows gives. Dev
    runs only in practice: the shipped build is windowed and has no console,
    so there the handler simply never fires."""
    global _console_handler
    import ctypes

    HANDLER = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint)

    def on_console_event(event):
        if event in (2, 5, 6):  # CTRL_CLOSE, CTRL_LOGOFF, CTRL_SHUTDOWN
            _set_exit_reason("shutdown")
        return False  # keep Python's own Ctrl+C handling in the chain

    # The callback must outlive the process or the console host calls into
    # freed memory.
    _console_handler = HANDLER(on_console_event)
    ctypes.windll.kernel32.SetConsoleCtrlHandler(_console_handler, True)


def install_session_end_hook(window_handle):
    """Hook the window so a Windows session end is recorded before the kill.

    Needs the window to exist, so it is installed from the app rather than at
    startup; the gap in between is covered by the heartbeat correlation in
    :func:`_shutdown_evidence`. ``window_handle`` is an HWND, or None on the
    platforms that get a signal instead.
    """
    try:
        if sys.platform == "win32" and window_handle:
            _install_windows_wndproc(window_handle)
    except Exception:
        pass


def remove_session_end_hook():
    """Restore the original window proc before the window is destroyed.

    ``_wndproc_new`` is deliberately kept alive: if anything subclassed on top
    of us it still holds our address, and freeing the thunk would leave the
    message pump calling into released memory.
    """
    global _wndproc_hwnd, _wndproc_old
    try:
        if _wndproc_hwnd is not None and _wndproc_old:
            import ctypes
            user32 = ctypes.windll.user32
            set_long = (getattr(user32, "SetWindowLongPtrW", None)
                        or user32.SetWindowLongW)
            set_long(_wndproc_hwnd, GWLP_WNDPROC, _wndproc_old)
    except Exception:
        pass
    _wndproc_hwnd = _wndproc_old = None


def _dead_markers():
    """Yield (path, marker) for every marker whose process is gone.

    Includes the single shared marker older versions wrote at logs/last_run,
    so the first launch after an upgrade still classifies it.
    """
    legacy = user_data.data_path("logs", "last_run")
    paths = [legacy] if legacy.exists() else []
    try:
        paths += sorted(_runs_dir().glob("*.json"))
    except OSError:
        pass
    for path in paths:
        marker = _read_marker(path)
        if marker is not None and not _still_running(marker):
            yield path, marker


def check_unclean_exit():
    """Classify the markers left by dead runs and snapshot the log tails.

    Markers left by an OS shutdown are dropped here rather than downstream, so
    that ``always`` mode's silent upload is suppressed along with the popup.
    Markers whose process is still running belong to a concurrent instance and
    are left alone.

    Must run before :func:`mark_running` and early in startup, so the tails
    predate most of this run's log output. Returns the snapshot (also kept
    as module state for :func:`pending_unclean_report`) or None.
    """
    global _pending
    try:
        try:
            for stale in _runs_dir().glob("*.tmp"):  # kills mid-write
                stale.unlink()
        except OSError:
            pass
        crashed = False
        for path, marker in _dead_markers():
            shutdown = _shutdown_evidence(marker)
            if shutdown is not None:
                logger.info("Previous run ended with the OS session (%s); "
                            "not reporting it as a crash", shutdown)
            elif _crash_evidence(marker):
                crashed = True
            else:
                logger.info("Previous run ended without a trace in the crash "
                            "log (killed?); not reporting it as a crash")
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        if not crashed:
            _pending = None
            return None
        log_dir = user_data.data_path("logs")
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
    left for the imgui popup in modules/autolume_live. The session-end hooks
    that need no window go up here too, so a shutdown during the long import
    phase is still recorded.
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
        if sys.platform == "win32":
            _install_console_ctrl_handler()
        else:
            _install_posix_signals()
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
        # The dialog below can sit open for minutes. Freeze the heartbeat at
        # the moment of the crash so that shutting down while it is up does
        # not make the next launch read this as an OS shutdown.
        _heartbeat_stop.set()
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
