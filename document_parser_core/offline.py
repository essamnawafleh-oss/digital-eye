"""Offline enforcement helpers.

`network_guard()` patches socket connection functions so any connection to a
non-loopback address raises OfflineViolationError. Loopback stays allowed because
the optional subprocess workers and some libraries use local IPC. This is a
process-local guard (it cannot see child processes); for a system-level check use
`scripts/verify_offline.sh` (Linux network namespace) or disconnect the machine.
"""

from __future__ import annotations

import ipaddress
import socket
import threading
from contextlib import contextmanager

from .exceptions import OfflineViolationError

_lock = threading.RLock()
_depth = 0
_originals: dict = {}


def _is_local(address) -> bool:
    if isinstance(address, (str, bytes)):  # AF_UNIX path
        return True
    host = address[0] if isinstance(address, tuple) and address else address
    if host in ("localhost", "", None):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False  # a hostname would need DNS -> treat as network


def _guarded_connect(orig):
    def connect(self, address, *args, **kwargs):
        if not _is_local(address):
            raise OfflineViolationError(f"Network access attempted while offline: {address!r}")
        return orig(self, address, *args, **kwargs)
    return connect


def _guarded_getaddrinfo(orig):
    def getaddrinfo(host, *args, **kwargs):
        if host not in (None, "localhost", "127.0.0.1", "::1") and not _is_local((host,)):
            raise OfflineViolationError(f"DNS lookup attempted while offline: {host!r}")
        return orig(host, *args, **kwargs)
    return getaddrinfo


@contextmanager
def network_guard():
    global _depth
    with _lock:
        if _depth == 0:
            _originals["connect"] = socket.socket.connect
            _originals["connect_ex"] = socket.socket.connect_ex
            _originals["getaddrinfo"] = socket.getaddrinfo
            socket.socket.connect = _guarded_connect(_originals["connect"])
            socket.socket.connect_ex = _guarded_connect(_originals["connect_ex"])
            socket.getaddrinfo = _guarded_getaddrinfo(_originals["getaddrinfo"])
        _depth += 1
    try:
        yield
    finally:
        with _lock:
            _depth -= 1
            if _depth == 0:
                socket.socket.connect = _originals["connect"]
                socket.socket.connect_ex = _originals["connect_ex"]
                socket.getaddrinfo = _originals["getaddrinfo"]
