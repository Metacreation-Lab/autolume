"""Turning what a controller sends into a model file on disk.

The model path is the one parameter a mapping row drives with something that is
not a number, so it is the one that needs a resolver. Two kinds of value reach
it, and both are what the old app accepted:

    a number    an index into the models folder, in sorted order
    a text      a path, a filename, or a fragment of one

The numeric path is the one that matters for a performance, because a fader, a
button and an encoder all send numbers and almost no controller sends text. It
is also the one the row's expression shapes, which is what turns a 0 to 1 fader
into a selector across the folder.

Indexing a sorted folder is positional: adding or removing a model shifts what
every index means, so a set built around index 3 is a set built around whatever
is third that night. The old app had the same property and a performer's mental
model of it is the file listing, which is why the order here is the plain
sorted one and not, say, recently used first.

Nothing here raises. It runs on the control thread, where a missing folder, an
index off the end of a listing and a name matching nothing are all ordinary
things for a performer to send mid show, and none of them may cost a tick.
"""

import logging
import os
import time
from typing import Callable

logger = logging.getLogger(__name__)

# How long one listing is reused before the folder is read again. The read
# happens on the control thread, so it must not happen once per message: a
# fader sweeping an index sends as fast as its controller does, and resolving
# the data root parses the preferences file on every call. Two seconds is short
# enough that a model dropped into the folder mid show is selectable about as
# fast as the performer can reach the controller, and long enough that a sweep
# reads the folder once rather than a hundred times.
LISTING_SECONDS = 2.0


def _installed_models() -> list[str]:
    """Absolute paths of the models the user has, from the legacy helper."""
    # Imported inside the function because the models folder lives in a flat
    # root module from the old app, and the live runtime's import chain stays
    # free of it. Same reason and same precedent as `presets.preset_dir`.
    from utils.model_dir import list_model_pkls

    return list(list_model_pkls())


class ModelFolder:
    """The models folder, as a controller may address it.

    Holds its listing for `LISTING_SECONDS` so that resolving a stream of
    values costs one directory read rather than one per value.
    """

    def __init__(
        self,
        lister: Callable[[], list[str]] = _installed_models,
        clock: Callable[[], float] = time.monotonic,
        interval: float = LISTING_SECONDS,
    ) -> None:
        self._lister = lister
        self._clock = clock
        self._interval = interval
        self._paths: list[str] = []
        self._real: set[str] = set()
        self._read_at: float | None = None

    def paths(self) -> list[str]:
        """The models in the folder, sorted, from a recent enough listing.

        Sorted here rather than trusted from the lister, because the index a
        performer sends is a position in this list and nothing else may decide
        what that position means. A folder that cannot be read is an empty
        folder: a performer who has not made one yet is a normal state at the
        start of a show, not an error to raise into the control thread.
        """
        now = self._clock()
        if self._read_at is None or now - self._read_at >= self._interval:
            self._read_at = now
            try:
                self._paths = sorted(self._lister())
            except Exception:
                logger.warning("Could not read the models folder", exc_info=True)
                self._paths = []
            # Resolved once per listing, not once per message: `named` checks
            # a typed path against this set, and resolving the whole folder on
            # every value would put a stat storm on the control thread.
            self._real = {os.path.realpath(path) for path in self._paths}
        return self._paths

    def at_index(self, index: float) -> str | None:
        """The model at `index` in the listing, rounded, or None.

        Out of range is ignored rather than clamped. Clamping would make the
        top of a fader load the last model and hold it there, which reads as
        the mapping working, so the performer never learns their range is
        wrong.
        """
        try:
            position = round(float(index))
        except (TypeError, ValueError, OverflowError):
            logger.warning("Ignoring unusable model index %r", index)
            return None
        paths = self.paths()
        if not paths:
            logger.warning("Ignoring model index %d: no models in the folder", position)
            return None
        if not 0 <= position < len(paths):
            logger.warning(
                "Ignoring model index %d: the folder holds %d models",
                position,
                len(paths),
            )
            return None
        return paths[position]

    def named(self, reference: str) -> str | None:
        """The model `reference` names, or None.

        A path into the models folder wins, then a filename, then a fragment
        of one. The fragment is what makes this usable from a controller:
        nobody types `wikiart-1024.pkl` mid set, they type `wikiart`.

        A path is only taken when it resolves to a model in the listing. This
        value arrives from the network, and without the containment check any
        sender on the OSC port could name any existing file on disk as the
        pickle to load. A path elsewhere falls through to the matching below,
        which compares the whole reference and so refuses it like any other
        name the folder does not hold.

        A fragment can match several models, so the first match in the same
        sorted order the index uses wins. That way one listing explains both
        paths and a performer can predict either from it. A whole filename is
        tried before any fragment, so naming a model exactly always selects
        that model even when another filename contains its name.

        Matching ignores case. The old app compared exactly, but the thing
        being compared is a fragment a performer typed into a controller, and
        the only case where the difference selects a different model is a
        folder holding two names that differ in case alone.
        """
        reference = reference.strip()
        if not reference:
            logger.warning("Ignoring empty model reference")
            return None
        paths = self.paths()
        if os.path.isfile(reference) and os.path.realpath(reference) in self._real:
            return reference
        wanted = reference.casefold()
        names = [(path, os.path.basename(path).casefold()) for path in paths]
        for path, name in names:
            if name == wanted:
                return path
        for path, name in names:
            if wanted in name:
                return path
        logger.warning(
            "Ignoring model reference %r: nothing in the folder matches", reference
        )
        return None
