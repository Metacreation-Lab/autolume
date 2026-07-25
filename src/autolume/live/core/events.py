from dataclasses import dataclass


@dataclass(frozen=True)
class ControlEvent:
    """One transport-agnostic control message on its way to the control loop.

    `source` names the producer and is what the control loop gates touch on, so
    it defaults to the unprivileged value. A producer that means the UI says so,
    and one that forgets stays harmless instead of inheriting the privilege.
    """

    address: str
    value: object
    source: str = "unknown"
    timestamp: float | None = None
