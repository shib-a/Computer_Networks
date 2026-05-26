"""
High-availability replication (Complexity 3).

Architecture:
  - Primary: connects to backup's replication port, sends heartbeats every
    HEARTBEAT_INTERVAL seconds and pushes a full state snapshot after each
    aggregation round.
  - Backup: listens on its replication port for the primary's connection.
    A watchdog thread promotes the backup to primary if no heartbeat or state
    message arrives within HEARTBEAT_TIMEOUT seconds.

Failover sequence:
  1. Primary crashes (or network partition).
  2. Backup's watchdog fires after HEARTBEAT_TIMEOUT seconds.
  3. Backup calls on_promote(), which starts the FL client-facing server.
  4. Clients retry their server list and find the backup (now acting as primary).

The backup starts with the last replicated state (weights + round counter),
so it can seamlessly continue from where the primary left off.
"""
import logging
import socket
import threading
import time
from typing import Callable, Optional

from protocol.transport import send_msg, recv_msg
from protocol.messages import MsgType

log = logging.getLogger(__name__)

HEARTBEAT_INTERVAL = 5.0   # seconds between heartbeats sent by primary
HEARTBEAT_TIMEOUT  = 15.0  # seconds of silence before backup promotes


# ---------------------------------------------------------------------------
# Primary side
# ---------------------------------------------------------------------------

class PrimaryReplicator:
    """
    Runs on the primary.  Maintains a persistent TCP connection to the
    backup's replication listener and:
      - sends a heartbeat every HEARTBEAT_INTERVAL seconds
      - sends a full state snapshot whenever notify_state_changed() is called
    """

    def __init__(self, task_manager, backup_repl_addr: tuple) -> None:
        self._tm = task_manager
        self._addr = backup_repl_addr
        self._state_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="primary-repl")

    def start(self) -> None:
        self._thread.start()
        log.info("PrimaryReplicator started → backup at %s:%d", *self._addr)

    def stop(self) -> None:
        self._stop_event.set()

    def notify_state_changed(self) -> None:
        """Called by TaskManager after each aggregation round."""
        self._state_event.set()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            sock = self._connect()
            if sock is None:
                time.sleep(5.0)
                continue
            try:
                # Push initial snapshot so backup is up to date immediately
                self._push_snapshot(sock)

                while not self._stop_event.is_set():
                    # Wait up to HEARTBEAT_INTERVAL for a state-change notification
                    got_change = self._state_event.wait(timeout=HEARTBEAT_INTERVAL)
                    if got_change:
                        self._state_event.clear()
                        self._push_snapshot(sock)
                    else:
                        send_msg(sock, MsgType.HEARTBEAT, {'ts': time.time()})
            except Exception as exc:
                log.warning("replication connection to backup lost: %s — reconnecting", exc)
            finally:
                try:
                    sock.close()
                except Exception:
                    pass

    def _connect(self) -> Optional[socket.socket]:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect(self._addr)
            sock.settimeout(None)
            log.info("replication: connected to backup %s:%d", *self._addr)
            return sock
        except Exception as exc:
            log.debug("replication: cannot reach backup %s:%d — %s", *self._addr, exc)
            return None

    def _push_snapshot(self, sock: socket.socket) -> None:
        snap = self._tm.snapshot()
        send_msg(sock, MsgType.REPL_STATE, {'snapshot': snap})
        log.debug("replication: pushed snapshot (%d tasks)", len(snap))


# ---------------------------------------------------------------------------
# Backup side
# ---------------------------------------------------------------------------

class BackupReplicationListener:
    """
    Runs on the backup.  Listens for the primary's replication connection,
    applies incoming snapshots to the local TaskManager, and promotes
    the backup to primary after a heartbeat timeout.
    """

    def __init__(
        self,
        task_manager,
        repl_port: int,
        on_promote: Callable[[], None],
    ) -> None:
        self._tm = task_manager
        self._repl_port = repl_port
        self._on_promote = on_promote
        self._last_contact = time.time()
        self._promoted = False
        self._stop_event = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._listen_loop, daemon=True, name="backup-repl-listener").start()
        threading.Thread(target=self._watchdog_loop, daemon=True, name="backup-watchdog").start()
        log.info("BackupReplicationListener on port %d (timeout=%.0fs)", self._repl_port, HEARTBEAT_TIMEOUT)

    def stop(self) -> None:
        self._stop_event.set()

    def _listen_loop(self) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(('0.0.0.0', self._repl_port))
        srv.listen(5)
        srv.settimeout(1.0)
        log.info("backup: replication listener bound on port %d", self._repl_port)

        while not self._stop_event.is_set():
            try:
                conn, addr = srv.accept()
                log.info("backup: replication connection from primary %s:%d", *addr)
                threading.Thread(
                    target=self._handle_primary_conn,
                    args=(conn,),
                    daemon=True,
                    name="backup-repl-handler",
                ).start()
            except socket.timeout:
                continue
            except Exception as exc:
                if not self._stop_event.is_set():
                    log.error("backup: replication listener error: %s", exc)
                break

        srv.close()

    def _handle_primary_conn(self, conn: socket.socket) -> None:
        try:
            while not self._stop_event.is_set():
                msg_type, data = recv_msg(conn)
                self._last_contact = time.time()

                if msg_type == MsgType.HEARTBEAT:
                    log.debug("backup: heartbeat from primary (ts=%.1f)", data['ts'])

                elif msg_type == MsgType.REPL_STATE:
                    self._tm.restore_snapshot(data['snapshot'])
                    log.info("backup: state snapshot applied")

        except Exception:
            pass
        finally:
            conn.close()
            log.info("backup: replication connection closed")

    def _watchdog_loop(self) -> None:
        """Promote to primary if primary has been silent too long."""
        # Give the primary time to establish a connection on startup
        time.sleep(HEARTBEAT_TIMEOUT)

        while not self._stop_event.is_set() and not self._promoted:
            elapsed = time.time() - self._last_contact
            if elapsed > HEARTBEAT_TIMEOUT:
                log.warning(
                    "backup: no contact from primary for %.1fs — promoting to primary",
                    elapsed,
                )
                self._promoted = True
                self._on_promote()
                return
            time.sleep(1.0)
