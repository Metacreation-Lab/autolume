import time

from pythonosc.udp_client import SimpleUDPClient

from autolume.live.io.osc import OscInput


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
