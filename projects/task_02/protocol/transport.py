"""
TCP framing layer for FL/1 protocol.

Each message is framed as:
  [magic: 4B][msg_type: 1B][payload_len: 4B big-endian][payload: payload_len bytes]

The payload is a pickle-serialised Python dict.
pickle is used for simplicity (handles both plain dicts and bytes objects).
In a production system, consider a safer serialisation format.
"""
import pickle
import socket
import struct
import logging
from typing import Any

from .messages import MAGIC, HEADER_FORMAT, HEADER_SIZE, MsgType

log = logging.getLogger(__name__)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly n bytes from a TCP socket. Raises ConnectionError on EOF."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("connection closed unexpectedly")
        buf.extend(chunk)
    return bytes(buf)


def send_msg(sock: socket.socket, msg_type: MsgType, data: Any) -> None:
    """Serialise data and send it as a framed FL/1 message."""
    payload = pickle.dumps(data, protocol=4)
    header = struct.pack(HEADER_FORMAT, MAGIC, int(msg_type), len(payload))
    sock.sendall(header + payload)
    log.debug("sent %s (%d bytes payload)", msg_type.name, len(payload))


def recv_msg(sock: socket.socket) -> tuple[MsgType, Any]:
    """Receive and deserialise one FL/1 message. Raises on protocol error."""
    raw = _recv_exact(sock, HEADER_SIZE)
    magic, type_byte, payload_len = struct.unpack(HEADER_FORMAT, raw)

    if magic != MAGIC:
        raise ValueError(f"bad magic bytes: {magic.hex()!r} (expected {MAGIC.hex()!r})")

    payload_raw = _recv_exact(sock, payload_len)
    data = pickle.loads(payload_raw)

    msg_type = MsgType(type_byte)
    log.debug("recv %s (%d bytes payload)", msg_type.name, payload_len)
    return msg_type, data
