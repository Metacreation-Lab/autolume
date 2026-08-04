"""GUI-free download helpers.

Kept separate from the imgui download widget so headless code (super-resolution
worker process, CLI entry points) can fetch files without importing imgui.
"""
import csv
import html
import io
import logging
import os
import re
import time

import requests

from utils.resource_paths import is_frozen, resource_path

CHUNK_SIZE = 1024 * 1024

logger = logging.getLogger(__name__)
REQUIRED_COLUMNS = ('name', 'resolution', 'architecture', 'author', 'license', 'filename', 'url')

# Raw copy of models.csv on the default branch. Frozen builds pull this first so
# the curated model list can be refreshed without shipping a new release.
CATALOG_URL = 'https://raw.githubusercontent.com/Metacreation-Lab/autolume/main/models.csv'


def _parse_catalog(f):
    """Parse an open models.csv file object into a list of catalog entries."""
    rows = []
    for row in csv.DictReader(f):
        if all(row.get(col) for col in REQUIRED_COLUMNS):
            rows.append({col: row[col].strip() for col in REQUIRED_COLUMNS})
        else:
            logger.warning('Skipping malformed models.csv row: %s', row)
    return rows


def _load_bundled_catalog():
    csv_path = resource_path('models.csv')
    try:
        with open(csv_path, newline='', encoding='utf-8') as f:
            return _parse_catalog(f)
    except OSError as e:
        logger.error('Could not load models.csv: %s', e)
        return []


def _fetch_remote_catalog():
    response = requests.get(CATALOG_URL, timeout=(10, 30))
    response.raise_for_status()
    rows = _parse_catalog(io.StringIO(response.text))
    if not rows:
        raise ValueError('remote models.csv contained no usable rows')
    return rows


def load_catalog():
    """Parse the models.csv catalog into a list of entries.

    Frozen builds fetch the copy on GitHub first so the curated list can be
    updated without a new release, falling back to the bundled file on any
    failure. Source checkouts always read the bundled file.
    """
    if is_frozen():
        try:
            return _fetch_remote_catalog()
        except Exception as e:
            logger.warning('Could not fetch models.csv from GitHub (%s); using bundled copy', e)
    return _load_bundled_catalog()


def _resolve_google_drive(session, response):
    """Follow the Google Drive interstitial page to the actual file response."""
    page = response.text
    if 'Google Drive - Quota exceeded' in page:
        raise IOError('Google Drive download quota exceeded -- please try again later')
    # Modern interstitial: a form pointing at drive.usercontent.google.com with hidden params.
    match = re.search(r'<form[^>]*id="download-form"[^>]*action="([^"]+)"', page)
    if match:
        action = html.unescape(match.group(1))
        params = {name: html.unescape(value) for name, value in
                  re.findall(r'<input type="hidden" name="([^"]+)" value="([^"]*)"', page)}
        return session.get(action, params=params, stream=True, timeout=(10, 30))
    # Legacy interstitial: scrape the export=download confirmation link.
    links = [html.unescape(link) for link in page.split('"') if 'export=download' in link]
    if len(links) == 1:
        return session.get(requests.compat.urljoin(response.url, links[0]), stream=True, timeout=(10, 30))
    raise IOError('Could not resolve Google Drive download link')


RETRY_DELAY = 3


def _stream_to_part(url, part_path, start, cancel_event, progress_cb):
    """One transfer attempt. False if cancelled, raises on failure."""
    with requests.Session() as session:
        headers = {'Range': f'bytes={start}-'} if start else None
        response = session.get(url, stream=True, timeout=(10, 30), headers=headers)
        response.raise_for_status()
        if response.headers.get('Content-Type', '').startswith('text/html'):
            response = _resolve_google_drive(session, response)
            response.raise_for_status()
            if response.headers.get('Content-Type', '').startswith('text/html'):
                raise IOError('Could not resolve download link (quota exceeded or page layout changed)')
        # a server that ignores Range answers 200 with the whole file, so the
        # bytes already on disk are worthless and the transfer starts over
        resumed = start and response.status_code == 206
        if not resumed:
            start = 0
        total = int(response.headers.get('Content-Length', 0)) + start
        done = start
        with open(part_path, 'ab' if resumed else 'wb') as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if cancel_event.is_set():
                    return False
                f.write(chunk)
                done += len(chunk)
                progress_cb(done, total)
    if total and done < total:
        raise IOError(f'Incomplete download: {done} of {total} bytes')
    return True


def download_file(url, dest_path, cancel_event, progress_cb, resume=False, tries=1):
    """Stream url into dest_path. Returns False if cancelled, raises on failure.

    Data is written to dest_path + '.part' and atomically renamed on success,
    so dest_path only ever exists as a complete file.

    With resume=True a broken transfer leaves its partial file in place and the
    next attempt continues from it with a Range request. Both Civitai and
    HuggingFace drop multi-gigabyte transfers routinely, so anything that size
    cannot finish without it.
    """
    os.makedirs(os.path.dirname(dest_path) or '.', exist_ok=True)
    part_path = dest_path + '.part'
    keep_part = False
    try:
        for attempt in range(max(1, tries)):
            start = (os.path.getsize(part_path)
                     if resume and os.path.exists(part_path) else 0)
            try:
                if not _stream_to_part(url, part_path, start, cancel_event, progress_cb):
                    return False
            except Exception:
                if not resume or attempt == max(1, tries) - 1:
                    raise
                keep_part = True          # the next attempt continues from it
                logger.warning('Download of %s broke, retrying', url, exc_info=True)
                time.sleep(RETRY_DELAY)
                continue
            os.replace(part_path, dest_path)
            return True
    finally:
        if not keep_part and os.path.exists(part_path):
            try:
                os.remove(part_path)
            except OSError:
                pass
