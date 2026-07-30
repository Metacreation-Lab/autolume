"""Where the development crash-report endpoint is read from.

The `.env` lives at the repo root, and this module resolves it two levels up
from itself now that it sits in `src/utils/`. Getting that arithmetic wrong is
silent: an endpoint that cannot be found is an ordinary "reporting not
configured" state, so every crash path simply becomes a no-op with no dialog,
no popup and no log line. That is what these pin, in the same spirit as
`test_resource_paths.py::test_repo_root_is_repo_not_src` — the same repo-root
computation, its second consumer.
"""
import sys

from utils import crash_report, resource_paths

URL = "https://example.invalid/report"
TOKEN = "t0ken"


def _isolate(monkeypatch, repo):
    """Resolve `.env` under `repo` and remove every other endpoint source.

    Only the path arithmetic is faked: `resource_root()` runs for real, and so
    does everything in `_endpoint_config`.
    """
    (repo / "src" / "utils").mkdir(parents=True)
    monkeypatch.setattr(resource_paths, "_REPO_ROOT", repo)
    monkeypatch.setattr(resource_paths, "_SRC_ROOT", repo / "src")
    monkeypatch.delenv("AUTOLUME_CRASH_REPORT_URL", raising=False)
    monkeypatch.delenv("AUTOLUME_CRASH_REPORT_TOKEN", raising=False)
    monkeypatch.setattr(crash_report, "_endpoint_override", None)
    monkeypatch.setattr(crash_report, "_endpoint_cache", None)
    # A None entry makes `import _endpoint_baked` raise, so a frozen build's
    # baked endpoint can never stand in for the file being tested.
    monkeypatch.setitem(sys.modules, "_endpoint_baked", None)


def _write_env(path):
    path.write_text(
        f"AUTOLUME_CRASH_REPORT_URL={URL}\nAUTOLUME_CRASH_REPORT_TOKEN={TOKEN}\n",
        encoding="utf-8",
    )


def test_a_repo_root_env_configures_reporting_in_a_source_run(tmp_path, monkeypatch):
    _isolate(monkeypatch, tmp_path)
    _write_env(tmp_path / ".env")

    assert crash_report.reporting_available()
    assert crash_report._endpoint_config() == (URL, TOKEN)


def test_an_env_under_src_is_not_the_one_that_counts(tmp_path, monkeypatch):
    """The exact file the pre-fix arithmetic pointed at.

    `<repo>/src/.env` is not a location anything writes, so finding one there
    would mean the resolution had walked up one level too few again.
    """
    _isolate(monkeypatch, tmp_path)
    _write_env(tmp_path / "src" / ".env")

    assert not crash_report.reporting_available()


def test_environment_variables_still_win_over_the_file(tmp_path, monkeypatch):
    _isolate(monkeypatch, tmp_path)
    _write_env(tmp_path / ".env")
    monkeypatch.setenv("AUTOLUME_CRASH_REPORT_URL", "https://other.invalid/report")
    monkeypatch.setenv("AUTOLUME_CRASH_REPORT_TOKEN", "other")

    assert crash_report._endpoint_config() == ("https://other.invalid/report", "other")
