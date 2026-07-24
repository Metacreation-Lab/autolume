"""OSC input transport: UDP messages become control events."""

import logging
import threading
from typing import Callable

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

from autolume.live.core.events import ControlEvent

logger = logging.getLogger(__name__)

_PORT_ATTEMPTS = 20


class OscInput:
    def __init__(
        self,
        submit: Callable[[ControlEvent], None],
        host: str = "0.0.0.0",
        port: int = 1338,
    ) -> None:
        self._submit = submit
        self._host = host
        self._requested_port = port
        self.port: int | None = None
        self._server: BlockingOSCUDPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> int:
        dispatcher = Dispatcher()
        dispatcher.set_default_handler(self._on_message)
        if self._requested_port == 0:
            self._server = BlockingOSCUDPServer((self._host, 0), dispatcher)
        else:
            for attempt in range(_PORT_ATTEMPTS):
                candidate = self._requested_port + attempt
                try:
                    self._server = BlockingOSCUDPServer(
                        (self._host, candidate), dispatcher
                    )
                    break
                except OSError as exc:
                    logger.debug("OSC port %s unavailable: %s", candidate, exc)
            else:
                raise OSError(
                    f"No OSC port available in "
                    f"{self._requested_port}-{self._requested_port + _PORT_ATTEMPTS - 1}"
                )
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="osc", daemon=True
        )
        self._thread.start()
        logger.info("OSC server listening on %s:%s", self._host, self.port)
        return self.port

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _on_message(self, address: str, *args: object) -> None:
        value = args[0] if args else 1.0
        self._submit(ControlEvent(address=address, value=value, source="osc"))
