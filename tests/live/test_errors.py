"""The canonical defensive error text, and the surfaces that share it.

`safe_describe` is `describe` hardened for log lines and dedup keys: it
existed as three drifted private copies (`superres.py`, `recorder.py`,
`ndi.py`) before being consolidated into `errors.py`.
"""

from autolume.live.errors import safe_describe


class _BrokenStr(Exception):
    """An exception whose own ``__str__`` raises, like a badly written one might."""

    def __str__(self):
        raise RuntimeError("broken __str__")


def test_the_text_is_the_message_the_exception_carries():
    assert safe_describe(ValueError("no such layer")) == "no such layer"


def test_an_empty_message_falls_back_to_the_type_name():
    assert safe_describe(KeyError()) == "KeyError"


def test_a_broken_str_falls_back_to_the_type_name_instead_of_raising():
    assert safe_describe(_BrokenStr("ignored")) == "_BrokenStr"


def test_an_enormous_message_is_bounded():
    text = safe_describe(ValueError("x" * 5000))
    assert text.endswith("...(truncated)")
    assert len(text) == 200 + len("...(truncated)")


def test_every_surface_that_had_its_own_copy_now_uses_the_one_helper():
    """The three drifted private copies are gone, not merely bypassed.

    Identity, not behaviour: each module must hold the canonical function,
    so a fix to it reaches every log line and dedup key at once.
    """
    from autolume.live.core import superres
    from autolume.live.io import ndi, recorder

    assert superres.safe_describe is safe_describe
    assert recorder.safe_describe is safe_describe
    assert ndi.safe_describe is safe_describe
    assert not hasattr(superres, "_safe_error_text")
    assert not hasattr(recorder, "_safe_error_text")
    assert not hasattr(ndi, "_safe_error_text")
