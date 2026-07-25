from pythonosc.osc_message import OscMessage

from autolume.audio.publisher import OSC_PREFIX, FeaturePublisher


class FakeDispatcher:
    """Records the OSC messages the publisher loops back."""

    def __init__(self, fail_on=()):
        self.messages = []
        self.fail_on = set(fail_on)

    def call_handlers_for_packet(self, dgram, addr):
        msg = OscMessage(dgram)
        if msg.address in self.fail_on:
            raise RuntimeError("handler blew up")
        self.messages.append((msg.address, msg.params[0], addr))


def test_prefix_is_audio():
    assert OSC_PREFIX == "/audio/"


def test_publishes_one_packet_per_feature():
    disp = FakeDispatcher()
    FeaturePublisher(disp).publish({"level": 0.25, "onset": 1.0})
    assert [(addr, value) for addr, value, _ in disp.messages] == [
        ("/audio/level", 0.25),
        ("/audio/onset", 1.0),
    ]


def test_values_are_floats_from_a_loopback_address():
    disp = FakeDispatcher()
    FeaturePublisher(disp).publish({"bass": 0})
    address, value, addr = disp.messages[0]
    assert address == "/audio/bass"
    assert isinstance(value, float) and value == 0.0
    assert addr == ("127.0.0.1", 0)


def test_custom_prefix():
    disp = FakeDispatcher()
    FeaturePublisher(disp, prefix="/x/").publish({"mid": 0.5})
    assert disp.messages[0][0] == "/x/mid"


def test_handler_failure_is_swallowed_and_publishing_continues():
    disp = FakeDispatcher(fail_on=["/audio/level"])
    FeaturePublisher(disp).publish({"level": 0.1, "high": 0.2})
    assert [addr for addr, _, _ in disp.messages] == ["/audio/high"]
