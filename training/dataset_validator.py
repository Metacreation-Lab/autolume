import os
from collections import Counter
from pathlib import Path

import numpy as np
import PIL.Image

try:
    import pyspng
except ImportError:
    pyspng = None


MAX_REPORTED_ERRORS = 10


def _file_ext(fname: str) -> str:
    return os.path.splitext(fname)[1].lower()


def _load_image_array(path: str) -> np.ndarray:
    """Load an image to a CHW numpy array."""
    with open(path, 'rb') as f:
        if pyspng is not None and _file_ext(path) == '.png':
            image = pyspng.load(f.read())
        else:
            pil_image = PIL.Image.open(f)
            if pil_image.mode == 'P':
                pil_image = pil_image.convert('RGB')
            image = np.array(pil_image)

    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    image = image.transpose(2, 0, 1)

    if image.shape[0] == 1:
        image = np.repeat(image, 3, axis=0)
    if image.shape[0] == 4:
        image = image[:3, :, :]
    return image


def _discover_image_files(path: str) -> list[str]:
    """Find every image file under `path` using PIL's registered extensions."""
    PIL.Image.init()
    root_path = Path(path)
    if not root_path.is_dir():
        return []
    files: list[str] = []
    for root, _dirs, names in os.walk(root_path):
        for name in names:
            if _file_ext(name) in PIL.Image.EXTENSION:
                files.append(os.path.join(root, name))
    files.sort()
    return files


def _check_absolute(channels: int, height: int, width: int, dtype) -> list[str]:
    """Per-image checks that need no reference: square, power-of-2, dtype."""
    errors: list[str] = []

    if height != width:
        errors.append(
            f"Not square ({width}x{height}). Training only accepts square images."
        )
    if width <= 0 or (width & (width - 1)) != 0:
        errors.append(
            f"Resolution {width}x{height} is not a power of 2 (e.g. 256, 512, 1024)."
        )
    if dtype != np.uint8:
        errors.append(f"Has dtype {dtype}, expected uint8.")
    return errors


EXAMPLE_FILENAMES_PER_GROUP = 5


def _emit_completion(reply, *, is_valid: bool, errors, consistency, total: int,
                     checked: int):
    reply.put({
        'type': 'completed',
        'is_valid': is_valid,
        'errors': list(errors),
        'consistency': list(consistency),
        'total': total,
        'checked': checked,
    })


def _emit_progress(reply, *, current: int, total: int, filename: str):
    reply.put({
        'type': 'progress',
        'current': current,
        'total': total,
        'percentage': (current / total * 100) if total else 0,
        'current_file': filename,
    })


def _group_record_errors(record) -> dict | None:
    """Collapse one image's issues into a single {filename, message} entry."""
    messages = record['errors']
    if not messages:
        return None
    if len(messages) == 1:
        message = messages[0]
    else:
        message = "\n".join(f"- {m}" for m in messages)
    return {'filename': record['filename'], 'message': message}


def _summarise_groups(counter, loaded_records, key_fn, label_fn) -> str:
    """Build a single consistency block: per-distinct-value counts plus example
    filenames for the minority (non-majority) groups."""
    ordered = counter.most_common()
    majority_value = ordered[0][0]
    lines = []
    for value, count in ordered:
        lines.append(f"  - {label_fn(value)}: {count} image(s)")
        if value != majority_value:
            examples = [
                r['filename'] for r in loaded_records if key_fn(r) == value
            ]
            shown = examples[:EXAMPLE_FILENAMES_PER_GROUP]
            suffix = ""
            if len(examples) > EXAMPLE_FILENAMES_PER_GROUP:
                suffix = f", ...and {len(examples) - EXAMPLE_FILENAMES_PER_GROUP} more"
            if shown:
                lines.append(f"      {', '.join(shown)}{suffix}")
    return "\n".join(lines)


def _build_consistency_report(loaded_records) -> list[str]:
    """Dataset-level report of mixed resolutions / channel counts."""
    report: list[str] = []
    if not loaded_records:
        return report

    res_counter = Counter((r['width'], r['height']) for r in loaded_records)
    if len(res_counter) > 1:
        summary = _summarise_groups(
            res_counter, loaded_records,
            key_fn=lambda r: (r['width'], r['height']),
            label_fn=lambda wh: f"{wh[0]}x{wh[1]}",
        )
        report.append(
            "Mixed resolutions found. Training requires all images to share one "
            "resolution:\n" + summary
        )

    chan_counter = Counter(r['channels'] for r in loaded_records)
    if len(chan_counter) > 1:
        summary = _summarise_groups(
            chan_counter, loaded_records,
            key_fn=lambda r: r['channels'],
            label_fn=lambda c: f"{c} channel(s)",
        )
        report.append(
            "Mixed colour channels found. Ensure all images use the same number "
            "of channels (e.g. RGB = 3):\n" + summary
        )

    return report


def validate_dataset(queue, reply):
    try:
        settings = queue.get()
    except Exception as exc:
        _emit_completion(
            reply, is_valid=False,
            errors=[{'filename': '', 'message': f'Failed to read validation settings: {exc}'}],
            consistency=[], total=0, checked=0,
        )
        return

    if isinstance(settings, dict):
        path = settings.get('path')
    else:
        path = settings

    if not path:
        _emit_completion(
            reply, is_valid=False,
            errors=[{'filename': '', 'message': 'No dataset path provided.'}],
            consistency=[], total=0, checked=0,
        )
        return

    if not Path(path).exists():
        _emit_completion(
            reply, is_valid=False,
            errors=[{'filename': '', 'message': f'Dataset path does not exist: {path}'}],
            consistency=[], total=0, checked=0,
        )
        return

    image_files = _discover_image_files(path)
    total = len(image_files)
    if total == 0:
        _emit_completion(
            reply, is_valid=False,
            errors=[{'filename': '', 'message': f'No image files found in {path}.'}],
            consistency=[], total=0, checked=0,
        )
        return

    update_interval = max(1, min(20, total // 200))
    cancelled = False
    capped = False
    checked = 0

    error_entries: list[dict] = []
    loaded: list[dict] = []
    for i, image_path in enumerate(image_files):
        if not queue.empty():
            try:
                if queue.get_nowait() == 'cancel':
                    cancelled = True
                    break
            except Exception:
                pass

        filename = os.path.basename(image_path)
        record = {'filename': filename, 'errors': []}
        try:
            image = _load_image_array(image_path)
        except Exception as exc:
            record['errors'].append(f'Failed to load: {exc}')
        else:
            record['channels'] = int(image.shape[0])
            record['height'] = int(image.shape[1])
            record['width'] = int(image.shape[2])
            loaded.append(record)
            record['errors'].extend(
                _check_absolute(record['channels'], record['height'],
                                record['width'], image.dtype)
            )

        grouped = _group_record_errors(record)
        if grouped is not None:
            error_entries.append(grouped)

        checked = i + 1
        if checked % update_interval == 0 or checked == total:
            _emit_progress(reply, current=checked, total=total, filename=filename)

        if len(error_entries) >= MAX_REPORTED_ERRORS:
            capped = True
            break

    if cancelled:
        _emit_completion(
            reply, is_valid=False,
            errors=[{'filename': '', 'message': 'Validation cancelled.'}],
            consistency=[], total=total, checked=checked,
        )
        return

    if capped:
        error_entries.append({
            'filename': '',
            'message': (f'Stopped after {MAX_REPORTED_ERRORS} invalid images. '
            'Fix and revalidate dataset.'),
        })

    consistency = _build_consistency_report(loaded)
    is_valid = (not error_entries) and (not consistency)
    _emit_completion(
        reply, is_valid=is_valid, errors=error_entries,
        consistency=consistency, total=total, checked=checked,
    )
