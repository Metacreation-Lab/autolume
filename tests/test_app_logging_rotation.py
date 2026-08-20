"""Log rotation contention must not take logging down (autolume#63).

On Windows the midnight rename of autolume.log fails with WinError 32 while
any other process holds the file (a second instance, antivirus, backup). The
stock TimedRotatingFileHandler then drops every record and retries the rename
per record, and its handleError reports to sys.stderr — which the app
redirects back into logging, snowballing one failure into an unbounded record
storm that exhausted the reporter's machine.
"""
import logging
import time

import pytest

from utils.app_logging import (_ResilientTimedRotatingFileHandler,
                               _mute_handler_feedback)


def record(msg="hello"):
    return logging.LogRecord("test", logging.INFO, __file__, 1, msg, None, None)


@pytest.fixture
def handler(tmp_path):
    h = _ResilientTimedRotatingFileHandler(
        str(tmp_path / "autolume.log"), when="midnight", backupCount=3,
        encoding="utf-8", delay=True)
    yield h
    h.close()


def test_rotation_failure_falls_back_to_current_file(handler, tmp_path, monkeypatch):
    handler.emit(record("before"))
    handler.rolloverAt = int(time.time()) - 3600
    monkeypatch.setattr(handler, "rotate", lambda src, dst: (_ for _ in ()).throw(
        PermissionError(32, "held by another process")))

    handler.emit(record("during contention"))

    text = (tmp_path / "autolume.log").read_text(encoding="utf-8")
    assert "during contention" in text
    assert handler.rolloverAt > int(time.time())


def test_rotation_failure_is_reported_once_not_per_record(handler, monkeypatch, caplog):
    handler.emit(record("before"))
    handler.rolloverAt = int(time.time()) - 3600
    monkeypatch.setattr(handler, "rotate", lambda src, dst: (_ for _ in ()).throw(
        PermissionError(32, "held by another process")))

    with caplog.at_level(logging.WARNING, logger="autolume"):
        for i in range(50):
            handler.emit(record("record %d" % i))

    postponed = [r for r in caplog.records if "rotation postponed" in r.message]
    assert len(postponed) == 1


def test_foreign_rotation_reschedules_the_next_rollover(handler, tmp_path):
    # When another instance already created the dated backup, the stock
    # doRollover returns early with rolloverAt still in the past, re-entering
    # rotation on every subsequent record forever.
    handler.emit(record("before"))
    handler.rolloverAt = int(time.time()) - 3600
    suffix = time.strftime(handler.suffix,
                           time.localtime(handler.rolloverAt - handler.interval))
    (tmp_path / ("autolume.log." + suffix)).touch()

    handler.emit(record("after foreign rotation"))

    assert handler.rolloverAt > int(time.time())
    text = (tmp_path / "autolume.log").read_text(encoding="utf-8")
    assert "after foreign rotation" in text


def test_handler_failures_stay_out_of_stderr(monkeypatch):
    # sys.stderr is _StreamToLogger in every Autolume process: anything a
    # handler writes there re-enters the logging pipeline as new records.
    class Broken:
        def write(self, text):
            raise OSError("disk full")

        def flush(self):
            pass

    writes = []

    class Probe:
        def write(self, text):
            writes.append(text)

        def flush(self):
            pass

    import sys
    monkeypatch.setattr(sys, "stderr", Probe())
    h = _mute_handler_feedback(logging.StreamHandler(Broken()))

    h.handle(record("boom"))  # must neither raise nor touch sys.stderr

    assert writes == []
