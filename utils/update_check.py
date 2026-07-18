"""Check GitHub releases for a newer Autolume version in the background.

Started once at launch (see ``main.py``). The request runs on a daemon thread
so a slow or absent network never blocks startup; any failure is logged and
the app simply reports no update.
"""
import logging
import threading

logger = logging.getLogger(__name__)

RELEASES_API_URL = "https://api.github.com/repos/Metacreation-Lab/autolume/releases?per_page=10"
RELEASES_PAGE_URL = "https://github.com/Metacreation-Lab/autolume/releases/latest"

_result = None
_started = False


def start_update_check():
    """Fetch the latest release on a background thread."""
    global _started
    if _started:
        return
    _started = True
    threading.Thread(target=_check, name="update-check", daemon=True).start()


def _check():
    global _result
    try:
        import requests
        from packaging.version import InvalidVersion, Version

        from utils.resource_paths import get_version

        response = requests.get(RELEASES_API_URL, timeout=10)
        response.raise_for_status()
        current = Version(get_version())

        # Stable builds are only offered stable releases; prerelease builds
        # (rc) are also offered newer prereleases.
        candidates = []
        for release in response.json():
            if release.get("draft"):
                continue
            if release.get("prerelease") and not current.is_prerelease:
                continue
            tag = str(release.get("tag_name") or "").lstrip("v")
            try:
                candidates.append((Version(tag), tag, release))
            except InvalidVersion:
                continue

        if not candidates:
            logger.info("Update check found no releases to compare against")
            return
        latest, tag, release = max(candidates, key=lambda c: c[0])
        _result = {"version": tag,
                   "url": release.get("html_url") or RELEASES_PAGE_URL,
                   "newer": latest > current}
        if _result["newer"]:
            logger.info("A newer Autolume release is available: %s", tag)
        else:
            logger.info("Autolume is up to date (latest release is %s)", tag)
    except Exception as e:
        logger.info("Update check did not complete: %s", e)


def latest_release():
    """Latest release as ``{"version", "url", "newer"}``, or None if the check
    has not completed successfully."""
    return _result
