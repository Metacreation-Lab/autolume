"""What a failure says to the performer.

Shared by every surface that shows an exception, so that a device error, a
preset that would not save and a folder that would not list all read the same
way.
"""


def describe(exc: BaseException) -> str:
    """The message an exception carries, or its type when it carries none.

    Bare `str()` on something like a `KeyError` with no message yields an empty
    string, which would leave the performer with a sentence that stops at "Could
    not save evening." and no reason.
    """
    return str(exc) or type(exc).__name__
