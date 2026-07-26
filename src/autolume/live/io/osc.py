"""OSC transport: inbound messages become control events, outbound pulses go out.

`OscInput` is the inbound side, running its own thread. `OscEmitter` is the
outbound side, called directly from the control thread (Task 7): it never
raises, so a dead receiver or a bad address is a log line, not a stalled show.
"""

import ipaddress
import logging
import threading
from typing import Callable

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient

from autolume.live.core.events import ControlEvent

logger = logging.getLogger(__name__)

_PORT_ATTEMPTS = 20
# Distinct failure strings are rare (a handful of socket/DNS errors); this
# only guards against a pathological error message that changes every call.
_ERROR_LOG_LIMIT = 32


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


class OscEmitter:
    """Sends one OSC message per call. Never raises, never blocks on DNS.

    The client is lazily created and recreated only when (ip, port) changes,
    so a steady destination reuses one socket across the show. `ip` must be an
    IP literal: `pythonosc`'s client resolves whatever string it is given via
    `socket.getaddrinfo` at construction time, and a hostname there can block
    on a DNS lookup for seconds, which the control thread cannot afford for a
    field a performer can type freely. A non-literal is rejected before any
    client is built. Every exception, construction or send, is swallowed and
    logged once per distinct error string: this is called only on a
    `started`/`wrapped` edge, not every tick, but a receiver that stays gone
    for the rest of the show must still produce one line, not one per edge.
    """

    def __init__(
        self, client_factory: Callable[[str, int], object] = SimpleUDPClient
    ) -> None:
        self._client_factory = client_factory
        self._client: object | None = None
        self._client_key: tuple[str, int] | None = None
        self._logged_errors: set[str] = set()

    def send(self, ip: str, port: int, address: str, value: float) -> None:
        if not _is_ip_literal(ip):
            self._log_once(f"pulse ip {ip!r} is not a literal address")
            return
        try:
            key = (ip, port)
            if self._client is None or key != self._client_key:
                self._client = self._client_factory(ip, port)
                self._client_key = key
            self._client.send_message(address, value)
        except Exception as exc:
            self._log_once(str(exc))

    def _log_once(self, error: str) -> None:
        if error in self._logged_errors:
            return
        if len(self._logged_errors) >= _ERROR_LOG_LIMIT:
            self._logged_errors.clear()
        self._logged_errors.add(error)
        logger.warning("OSC pulse send failed: %s", error)


def _is_ip_literal(ip: str) -> bool:
    try:
        ipaddress.ip_address(ip)
        return True
    except ValueError:
        return False
