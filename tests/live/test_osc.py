import logging
import time

from pythonosc.udp_client import SimpleUDPClient

from autolume.live.io.osc import OscEmitter, OscInput


def test_message_becomes_control_event():
    received = []
    osc = OscInput(received.append, host="127.0.0.1", port=0)
    port = osc.start()
    try:
        client = SimpleUDPClient("127.0.0.1", port)
        client.send_message("/latent/x", 2.5)
        deadline = time.monotonic() + 2.0
        while not received and time.monotonic() < deadline:
            time.sleep(0.005)
    finally:
        osc.stop()
    assert received
    event = received[0]
    assert event.address == "/latent/x"
    assert event.value == 2.5
    assert event.source == "osc"


def test_argless_message_defaults_to_one():
    received = []
    osc = OscInput(received.append, host="127.0.0.1", port=0)
    port = osc.start()
    try:
        client = SimpleUDPClient("127.0.0.1", port)
        client.send_message("/anim/playing", [])
        deadline = time.monotonic() + 2.0
        while not received and time.monotonic() < deadline:
            time.sleep(0.005)
    finally:
        osc.stop()
    assert received and received[0].value == 1.0


def test_stop_is_idempotent():
    osc = OscInput(lambda e: None, host="127.0.0.1", port=0)
    osc.start()
    osc.stop()
    osc.stop()


# --- OscEmitter (Task 7) -----------------------------------------------------
#
# The emitter is called from the control thread, so none of these tests may
# let a real client open a socket. `FakeClient` and its factory stand in.


class FakeClient:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.sent = []

    def send_message(self, address, value):
        self.sent.append((address, value))


def make_factory():
    clients = []

    def factory(ip, port):
        client = FakeClient(ip, port)
        clients.append(client)
        return client

    return factory, clients


def test_send_reuses_the_client_when_ip_and_port_are_unchanged():
    factory, clients = make_factory()
    emitter = OscEmitter(client_factory=factory)
    emitter.send("127.0.0.1", 5005, "/pulse", 2.0)
    emitter.send("127.0.0.1", 5005, "/pulse", 1.0)
    assert len(clients) == 1
    assert clients[0].sent == [("/pulse", 2.0), ("/pulse", 1.0)]


def test_send_recreates_the_client_when_ip_or_port_changes():
    factory, clients = make_factory()
    emitter = OscEmitter(client_factory=factory)
    emitter.send("127.0.0.1", 5005, "/pulse", 2.0)
    emitter.send("127.0.0.1", 5006, "/pulse", 2.0)
    assert len(clients) == 2
    emitter.send("10.0.0.5", 5006, "/pulse", 2.0)
    assert len(clients) == 3
    emitter.send("10.0.0.5", 5006, "/pulse", 2.0)
    assert len(clients) == 3


class RaisingClient:
    def __init__(self, ip, port):
        pass

    def send_message(self, address, value):
        raise RuntimeError("boom")


def test_a_raising_client_never_propagates_out_of_send():
    emitter = OscEmitter(client_factory=RaisingClient)
    emitter.send("127.0.0.1", 5005, "/pulse", 2.0)  # must not raise


def test_a_failing_client_factory_never_propagates_and_keeps_retrying():
    attempts = []

    def factory(ip, port):
        attempts.append((ip, port))
        raise OSError("no route to host")

    emitter = OscEmitter(client_factory=factory)
    emitter.send("bad-host", 5005, "/pulse", 2.0)
    emitter.send("bad-host", 5005, "/pulse", 2.0)
    assert len(attempts) == 2


def test_send_logs_one_line_per_distinct_error_string(caplog):
    emitter = OscEmitter(client_factory=RaisingClient)
    with caplog.at_level(logging.WARNING):
        emitter.send("127.0.0.1", 5005, "/pulse", 2.0)
        emitter.send("127.0.0.1", 5005, "/pulse", 1.0)
        emitter.send("127.0.0.1", 5005, "/pulse", 1.0)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
