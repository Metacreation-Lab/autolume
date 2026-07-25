"""Ephemeral record of the inbound control addresses seen so far.

Every address carrying a scalar value is recorded here with its latest
value and timestamp. The address picker lists what arrived recently and
bindings read the last value of their source. Never persisted, never
mutated: observing returns a new table, published by the control loop.
"""

from dataclasses import dataclass, field
from typing import Mapping

MAX_SOURCES = 128
DEFAULT_WINDOW = 60.0
# How recently an address must have been seen to count as writing right now.
# `DEFAULT_WINDOW` answers a different question, "what has this show seen", and
# a picker may safely keep offering an address a minute after the last message.
# A marker claiming a parameter is being driven has to go dark once the sender
# stops, or it is a lie the performer will read while deciding what to touch.
# Two seconds is long enough to stay lit through a controller that only sends
# on change at human speed, a button or a fader let go, and short enough that
# it clears about as fast as the performer can look away and back.
LIVE_WINDOW = 2.0


@dataclass(frozen=True)
class SourceValue:
    value: float
    timestamp: float


def canonical_address(address: str) -> str:
    """The one spelling of an address, shared by everything that compares them.

    The table, the control loop and the picker all normalize through here, so a
    binding on a picked address cannot silently miss the events that filled it.
    """
    address = address.strip()
    if not address:
        return ""
    return address if address.startswith("/") else "/" + address


def as_float(value: object) -> float | None:
    """Return `value` as a float, or None if it does not carry a number."""
    if isinstance(value, (str, bytes, bytearray)):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None


@dataclass(frozen=True)
class SourceTable:
    entries: Mapping[str, SourceValue] = field(default_factory=dict)

    def observe(self, address: str, value: object, timestamp: float) -> "SourceTable":
        number = as_float(value)
        if number is None:
            return self

        key = canonical_address(address)
        entries = dict(self.entries)
        # An open OSC port is an unbounded input, so a new address on a full
        # table displaces the least recently seen one.
        if key not in entries and len(entries) >= MAX_SOURCES:
            oldest = min(entries, key=lambda k: entries[k].timestamp)
            del entries[oldest]
        entries[key] = SourceValue(number, timestamp)
        return SourceTable(entries)

    def recent(self, now: float, window: float = DEFAULT_WINDOW) -> list[str]:
        return sorted(
            address
            for address, entry in self.entries.items()
            if now - entry.timestamp <= window
            and not address.rsplit("/", 1)[-1].startswith("_")
        )

    def get(self, address: str) -> SourceValue | None:
        return self.entries.get(canonical_address(address))

    def active(self, address: str, now: float, window: float = LIVE_WINDOW) -> bool:
        """Whether `address` carried a message recently enough to count as live.

        The table only ever records what arrived from outside the UI, so this
        answers "is something remote writing this right now" and never echoes
        the app's own output back at the performer.
        """
        entry = self.entries.get(canonical_address(address))
        return entry is not None and now - entry.timestamp <= window
