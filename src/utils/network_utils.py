import socket

import psutil


def list_bind_addresses():
    """Bindable OSC input addresses as (label, ip) pairs, most general first."""
    addresses = [("Any network (0.0.0.0)", "0.0.0.0"), ("This machine (127.0.0.1)", "127.0.0.1")]
    stats = psutil.net_if_stats()
    for name, if_addrs in psutil.net_if_addrs().items():
        stat = stats.get(name)
        if stat is None or not stat.isup:
            continue
        for addr in if_addrs:
            if addr.family == socket.AF_INET and addr.address != "127.0.0.1":
                addresses.append((f"{name} ({addr.address})", addr.address))
    return addresses
