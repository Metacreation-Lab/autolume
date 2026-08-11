import json
import os

import pytest

from utils import crash_report


@pytest.fixture
def marker(tmp_path, monkeypatch):
    """Point markers at a temp config dir, logs at a temp data root, and
    reset the module's run state."""
    runs = tmp_path / "config" / "runs"
    runs.mkdir(parents=True)
    (tmp_path / "logs").mkdir()
    monkeypatch.setattr(crash_report.user_data, "config_dir",
                        lambda: tmp_path / "config")
    monkeypatch.setattr(crash_report.user_data, "data_path",
                        lambda *parts: tmp_path.joinpath(*parts))
    monkeypatch.setattr(crash_report, "_marker_base", None)
    monkeypatch.setattr(crash_report, "_exit_reason", None)
    monkeypatch.setattr(crash_report, "_exit_reason_at", 0.0)
    crash_report._heartbeat_stop.clear()
    yield runs / ("%d.json" % os.getpid())
    crash_report._heartbeat_stop.set()
    crash_report._marker_base = None


@pytest.fixture
def shutdown_clock(monkeypatch):
    """Replace the OS shutdown clock with a value the test controls."""
    def install(value):
        monkeypatch.setattr(crash_report, "_last_os_shutdown_time", lambda: value)
    return install


def test_reads_back_what_it_wrote(marker, tmp_path):
    crash_report.mark_running()
    info = crash_report._read_marker()
    assert info["version"] == crash_report.MARKER_VERSION
    assert info["exit"] is None
    assert info["heartbeat"] >= info["started"]
    assert info["crashes_size"] == 0
    assert info["logs"] == str(tmp_path / "logs")


def test_marker_lives_in_the_config_dir_not_the_data_root(marker, tmp_path):
    # The data root may be a slow, removable or fragile mount; per-run state
    # written every heartbeat and scanned at startup must stay local.
    crash_report.mark_running()
    assert marker.exists()
    assert marker.is_relative_to(tmp_path / "config")
    assert not marker.is_relative_to(tmp_path / "logs")


def test_legacy_pid_marker_is_not_mistaken_for_session_data(marker):
    # json.loads("12345") succeeds and yields an int, so an isinstance check is
    # the only thing standing between the old format and an attribute error.
    marker.write_text("12345", encoding="utf-8")
    assert crash_report._read_marker() == {"version": 1}


@pytest.mark.parametrize("content", ["", "{not json", "[1, 2]", "null"])
def test_unreadable_marker_reads_as_legacy(marker, content):
    marker.write_text(content, encoding="utf-8")
    assert crash_report._read_marker() == {"version": 1}


def test_no_marker_reads_as_none(marker):
    assert crash_report._read_marker() is None


def test_clean_exit_removes_the_marker(marker):
    crash_report.mark_running()
    assert marker.exists()
    crash_report.mark_clean_exit()
    assert not marker.exists()


def test_a_late_heartbeat_cannot_resurrect_a_cleared_marker(marker):
    # A tick already past its wait and blocked on the lock still runs after
    # mark_clean_exit; if it wrote, every later launch would show a phantom
    # crash popup.
    crash_report.mark_running()
    crash_report.mark_clean_exit()
    crash_report._write_marker()
    assert not marker.exists()


def test_exit_reason_is_stamped_into_the_marker(marker):
    crash_report.mark_running()
    crash_report._set_exit_reason("shutdown")
    assert crash_report._read_marker()["exit"] == "shutdown"
    crash_report._set_exit_reason(None)
    assert crash_report._read_marker()["exit"] is None


def test_a_session_end_that_never_happened_stops_counting(marker, monkeypatch):
    # A shutdown request that is abandoned without the cancelling WM_ENDSESSION
    # would otherwise leave the stamp on for the rest of the run, suppressing
    # any real crash after it.
    crash_report.mark_running()
    crash_report._set_exit_reason("shutdown")
    monkeypatch.setattr(crash_report, "_exit_reason_at",
                        crash_report._exit_reason_at - crash_report.SESSION_END_GRACE - 1)
    crash_report._heartbeat_tick()
    assert crash_report._read_marker()["exit"] is None


def test_a_fresh_session_end_survives_a_heartbeat(marker):
    crash_report.mark_running()
    crash_report._set_exit_reason("shutdown")
    crash_report._heartbeat_tick()
    assert crash_report._read_marker()["exit"] == "shutdown"


def test_marker_write_is_atomic(marker):
    crash_report.mark_running()
    crash_report._write_marker()
    # A reader must never catch a half-written file, so the payload only ever
    # appears under the final name.
    assert json.loads(marker.read_text(encoding="utf-8"))["version"] == 2


class TestCrashEvidence:
    """A death only becomes a report when the crash handler left a trace."""

    def test_a_grown_crash_log_is_evidence(self, marker, tmp_path):
        crash_report.mark_running()
        info = crash_report._read_marker()
        (tmp_path / "logs" / "crashes.log").write_text(
            "Windows fatal exception: access violation", encoding="utf-8")
        assert crash_report._crash_evidence(info)

    def test_an_untouched_crash_log_is_not_evidence(self, marker):
        crash_report.mark_running()
        assert not crash_report._crash_evidence(crash_report._read_marker())

    def test_process_start_banners_are_not_evidence(self, marker, tmp_path):
        # Normal runs append banners for themselves and their workers, so the
        # log growing is business as usual, not a fault.
        crash_report.mark_running()
        info = crash_report._read_marker()
        (tmp_path / "logs" / "crashes.log").write_text(
            "--- pid 123 (MainProcess) started ---\n"
            "--- pid 456 (renderer) started ---\n", encoding="utf-8")
        assert not crash_report._crash_evidence(info)

    def test_a_fault_after_banners_is_evidence(self, marker, tmp_path):
        crash_report.mark_running()
        info = crash_report._read_marker()
        (tmp_path / "logs" / "crashes.log").write_text(
            "--- pid 123 (renderer) started ---\n"
            "Windows fatal exception: access violation\n", encoding="utf-8")
        assert crash_report._crash_evidence(info)

    def test_a_truncated_crash_log_is_not_evidence(self, marker, tmp_path):
        (tmp_path / "logs" / "crashes.log").write_text(
            "old fatal content from before this run", encoding="utf-8")
        crash_report.mark_running()
        info = crash_report._read_marker()
        (tmp_path / "logs" / "crashes.log").write_text(
            "short", encoding="utf-8")
        assert not crash_report._crash_evidence(info)

    def test_a_marker_without_a_size_is_not_evidence(self, marker):
        # Markers from builds that predate the evidence gate: at worst one
        # leftover crash goes unreported once, at the upgrade.
        assert not crash_report._crash_evidence({"version": 1})

    def test_evidence_follows_the_recorded_logs_dir(self, marker, tmp_path,
                                                    monkeypatch):
        # The marker outlives the run that wrote it; if the data root pref
        # changes in between, evidence must be read from where that run
        # actually logged, not from the new root.
        crash_report.mark_running()
        info = crash_report._read_marker()
        (tmp_path / "logs" / "crashes.log").write_text(
            "Fatal Python error: Segmentation fault", encoding="utf-8")
        moved = tmp_path / "elsewhere"
        (moved / "logs").mkdir(parents=True)
        monkeypatch.setattr(crash_report.user_data, "data_path",
                            lambda *parts: moved.joinpath(*parts))
        assert crash_report._crash_evidence(info)


class TestShutdownClassification:
    """A shutdown is only recognised on positive evidence of one."""

    def test_a_signalled_session_end_is_a_shutdown(self, shutdown_clock):
        shutdown_clock(None)
        assert crash_report._shutdown_evidence({"exit": "shutdown"})

    def test_dying_just_before_a_new_shutdown_is_a_shutdown(self, shutdown_clock):
        shutdown_clock(2_000_000.0)
        assert crash_report._shutdown_evidence(
            {"heartbeat": 1_999_988.0, "os_shutdown": 1_000_000.0})

    def test_dying_long_before_the_shutdown_is_not_a_shutdown(self, shutdown_clock):
        shutdown_clock(2_000_000.0)
        assert crash_report._shutdown_evidence(
            {"heartbeat": 1_990_000.0, "os_shutdown": 1_000_000.0}) is None

    def test_an_unchanged_shutdown_clock_is_not_a_shutdown(self, shutdown_clock):
        # Nothing has shut down since this run started, so whatever ended it,
        # the machine going down was not it.
        shutdown_clock(2_000_000.0)
        assert crash_report._shutdown_evidence(
            {"heartbeat": 2_000_500.0, "os_shutdown": 2_000_000.0}) is None

    def test_no_shutdown_clock_is_not_a_shutdown(self, shutdown_clock):
        # macOS and Linux have no cheap equivalent, and they get a signal
        # instead; the crash-evidence gate decides what happens next.
        shutdown_clock(None)
        assert crash_report._shutdown_evidence(
            {"heartbeat": 2_000_000.0, "os_shutdown": None}) is None

    def test_a_marker_without_a_heartbeat_is_not_a_shutdown(self, shutdown_clock):
        shutdown_clock(2_000_000.0)
        assert crash_report._shutdown_evidence({"version": 1}) is None


def test_shutdown_marker_is_dropped_without_a_pending_report(
        marker, shutdown_clock, monkeypatch):
    shutdown_clock(None)
    crash_report.mark_running()
    crash_report._set_exit_reason("shutdown")
    crash_report._marker_base = None  # the run is over; the file remains
    monkeypatch.setattr(crash_report, "_still_running", lambda m: False)

    assert crash_report.check_unclean_exit() is None
    assert crash_report.pending_unclean_report() is None
    # Dropped here rather than downstream, so "always send" mode has nothing
    # left to upload silently.
    assert not marker.exists()


def test_a_native_fault_still_produces_a_pending_report(
        marker, shutdown_clock, monkeypatch, tmp_path):
    shutdown_clock(None)
    crash_report.mark_running()
    (tmp_path / "logs" / "crashes.log").write_text(
        "Windows fatal exception: access violation", encoding="utf-8")
    crash_report._marker_base = None
    monkeypatch.setattr(crash_report, "_still_running", lambda m: False)

    assert crash_report.check_unclean_exit() is not None
    assert crash_report.pending_unclean_report() is not None
    assert not marker.exists()


def test_a_kill_without_crash_evidence_is_dropped(
        marker, shutdown_clock, monkeypatch):
    # Task Manager, power loss, a frozen app put down by the user: no trace
    # in the crash log, no report.
    shutdown_clock(None)
    crash_report.mark_running()
    crash_report._marker_base = None
    monkeypatch.setattr(crash_report, "_still_running", lambda m: False)

    assert crash_report.check_unclean_exit() is None
    assert crash_report.pending_unclean_report() is None
    assert not marker.exists()


class TestLiveness:
    """A marker is only "alive" when pid and creation time both match."""

    def test_our_own_marker_reads_as_alive(self, marker):
        crash_report.mark_running()
        assert crash_report._still_running(crash_report._read_marker())

    def test_a_recycled_pid_does_not_look_alive(self):
        # Windows reuses pids; the creation time is what tells the crashed
        # run's pid from the unrelated process now wearing it.
        assert not crash_report._still_running(
            {"pid": os.getpid(), "create": 123.0})

    def test_a_marker_without_a_create_time_counts_as_dead(self):
        assert not crash_report._still_running({"pid": os.getpid()})

    def test_a_dead_pid_counts_as_dead(self):
        # A pid beyond the OS's range can never be running.
        assert not crash_report._still_running(
            {"pid": 2 ** 31 - 1, "create": 123.0})


def test_a_live_instances_marker_is_left_alone(marker, shutdown_clock):
    shutdown_clock(None)
    crash_report.mark_running()  # a real marker for a really-running process
    crash_report._heartbeat_stop.set()

    assert crash_report.check_unclean_exit() is None
    assert crash_report.pending_unclean_report() is None
    assert marker.exists()


def test_legacy_shared_marker_is_swept_without_a_report(
        marker, shutdown_clock, tmp_path):
    # Pre-upgrade markers carry no crash-log size, so they cannot clear the
    # evidence gate; the file still has to go or it would be rescanned forever.
    shutdown_clock(None)
    legacy = tmp_path / "logs" / "last_run"
    legacy.write_text("12345", encoding="utf-8")

    assert crash_report.check_unclean_exit() is None
    assert not legacy.exists()


def test_stale_tmp_files_are_swept(marker, shutdown_clock):
    shutdown_clock(None)
    stale = marker.parent / "999.json.tmp"
    stale.write_text("{", encoding="utf-8")

    crash_report.check_unclean_exit()
    assert not stale.exists()
