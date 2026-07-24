from dataclasses import dataclass


@dataclass(frozen=True)
class ControlEvent:
    address: str
    value: object
    source: str = "ui"
    timestamp: float | None = None
