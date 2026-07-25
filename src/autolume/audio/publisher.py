"""Publish scalar audio features onto the app's in-process OSC dispatcher.

No socket is opened. Each feature becomes a real OSC packet fed directly into
the existing dispatcher via loopback, so /audio/<name> addresses appear in
every widget's OSC address picker and can drive any mappable parameter.
"""

import logging

from pythonosc.osc_message_builder import OscMessageBuilder

logger = logging.getLogger(__name__)

OSC_PREFIX = "/audio/"


class FeaturePublisher:
    """Dispatch normalized features into an OSC dispatcher as /audio/<name>."""

    def __init__(self, dispatcher, prefix=OSC_PREFIX):
        self.dispatcher = dispatcher
        self.prefix = prefix

    def publish(self, features):
        for name, value in features.items():
            self._dispatch(self.prefix + name, value)

    def _dispatch(self, address, value):
        builder = OscMessageBuilder(address)
        builder.add_arg(float(value))
        try:
            self.dispatcher.call_handlers_for_packet(
                builder.build().dgram, ("127.0.0.1", 0))
        except Exception:
            logger.exception("Audio feature dispatch failed for %s", address)
