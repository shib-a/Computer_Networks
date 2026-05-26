import struct
from enum import IntEnum

# Protocol: FL/1 — Federated Learning Protocol version 1
# Frame format: [magic: 4B][msg_type: 1B][payload_len: 4B][payload: N bytes]
MAGIC = b'\xFE\xDE\x00\x01'
HEADER_FORMAT = '!4sBI'   # network order: 4-byte magic, 1-byte type, 4-byte uint payload length
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # = 9 bytes


class MsgType(IntEnum):
    # Client → Server
    JOIN        = 0x01  # join a task (or list tasks)
    SUBMIT      = 0x02  # submit locally trained weights
    STATUS      = 0x03  # poll round status

    # Server → Client
    TASK_INFO   = 0x10  # task assignment: arch_bytes + weights_bytes + metadata
    WEIGHTS     = 0x11  # new global weights after aggregation
    PENDING     = 0x12  # round not complete yet, keep polling
    DONE        = 0x13  # all rounds complete

    # Server → Server (replication)
    HEARTBEAT   = 0x20  # primary → backup alive signal
    REPL_STATE  = 0x21  # primary → backup full state snapshot

    # General
    ACK         = 0xF0
    ERROR       = 0xFF
