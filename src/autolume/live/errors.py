"""What a failure says to the performer.

Shared by every surface that shows an exception, so that a device error, a
preset that would not save and a folder that would not list all read the same
way.
"""


_MAX_TEXT = 200


def describe(exc: BaseException) -> str:
    """The message an exception carries, or its type when it carries none.

    Bare `str()` on something like a `KeyError` with no message yields an empty
    string, which would leave the performer with a sentence that stops at "Could
    not save evening." and no reason.
    """
    return str(exc) or type(exc).__name__


def safe_describe(exc: BaseException) -> str:
    """`describe(exc)`, defensively: for text that feeds a log line or a
    dedup key.

    Neither may raise (an exception whose ``__str__`` is broken would
    otherwise escape the caller's own handler) and neither may be unbounded
    (a pathological or enormous message would be its own problem in both
    places).
    """
    try:
        text = describe(exc)
    except Exception:
        text = type(exc).__name__
    if len(text) > _MAX_TEXT:
        text = text[:_MAX_TEXT] + "...(truncated)"
    return text
