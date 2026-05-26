"""
FL/1 server: accepts TCP client connections and drives the training protocol.

Per-client handler state machine (one persistent TCP connection per client):

  JOIN  →  TASK_INFO  (arch_bytes + weights_bytes for current round)
         ↓
  SUBMIT  →  WEIGHTS   (if this client completes the quorum → aggregation)
           →  ACK       (quorum not yet reached → client will poll)
         ↓
  STATUS  →  WEIGHTS   (aggregation complete since last SUBMIT)
           →  PENDING   (still waiting for other clients)
           →  DONE      (all rounds finished)

The connection stays open across multiple rounds.  When all rounds are done
the server sends DONE and closes its end.
"""
import io
import logging
import socket
import threading
from typing import Optional

import torch

from protocol.transport import send_msg, recv_msg
from protocol.messages import MsgType
from .task_manager import TaskManager

log = logging.getLogger(__name__)


class FLServer:
    def __init__(self, host: str, port: int, task_manager: TaskManager) -> None:
        self.host = host
        self.port = port
        self._tm = task_manager
        self._sock: Optional[socket.socket] = None
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Accept-loop. Blocks until stop() is called."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.listen(256)
        self._sock.settimeout(1.0)
        self._running = True
        log.info("FL server listening on %s:%d", self.host, self.port)

        while self._running:
            try:
                conn, addr = self._sock.accept()
                log.info("new client from %s:%d", *addr)
                threading.Thread(
                    target=self._handle_client,
                    args=(conn, addr),
                    daemon=True,
                    name=f"client-{addr[0]}:{addr[1]}",
                ).start()
            except socket.timeout:
                continue
            except OSError:
                if self._running:
                    raise

        self._sock.close()

    def start_background(self) -> threading.Thread:
        """Start the accept-loop in a background daemon thread."""
        t = threading.Thread(target=self.start, daemon=True, name="fl-server")
        t.start()
        return t

    def stop(self) -> None:
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Per-client handler
    # ------------------------------------------------------------------

    def _handle_client(self, conn: socket.socket, addr) -> None:
        conn.settimeout(600.0)  # 10 min max idle before timeout
        task_id: Optional[str] = None
        current_round: int = -1

        try:
            while True:
                msg_type, data = recv_msg(conn)

                # ── LIST / JOIN ────────────────────────────────────────
                if msg_type == MsgType.JOIN:
                    requested = data.get('task_id')

                    if requested is None:
                        # Client wants a list of available tasks
                        send_msg(conn, MsgType.TASK_INFO, {'tasks': self._tm.list_tasks()})
                        continue

                    task = self._tm.get_task(requested)
                    if task is None:
                        send_msg(conn, MsgType.ERROR, {'error': f'unknown task {requested!r}'})
                        continue

                    task_id = requested
                    with task.lock:
                        current_round = task.current_round
                        is_done = task.is_done
                        arch_bytes = task.arch_bytes
                        weights_bytes = task.weights_bytes

                    if is_done:
                        send_msg(conn, MsgType.DONE, {
                            'task_id': task_id,
                            'rounds_completed': current_round,
                            'weights_bytes': weights_bytes,
                        })
                        break

                    send_msg(conn, MsgType.TASK_INFO, {
                        'task_id': task_id,
                        'current_round': current_round,
                        'max_rounds': task.max_rounds,
                        'arch_bytes': arch_bytes,
                        'weights_bytes': weights_bytes,
                    })

                # ── SUBMIT weights ─────────────────────────────────────
                elif msg_type == MsgType.SUBMIT:
                    if task_id is None:
                        send_msg(conn, MsgType.ERROR, {'error': 'send JOIN first'})
                        continue

                    client_round = data.get('round_num', current_round)
                    weights_bytes = data['weights_bytes']
                    num_samples = data.get('num_samples', 1)
                    client_loss = data.get('loss')
                    client_id_str = data.get('client_id', f'client_{id(conn) % 10000}')

                    state_dict = torch.load(
                        io.BytesIO(weights_bytes), weights_only=False
                    )
                    aggregated = self._tm.submit_update(
                        task_id, state_dict, num_samples, client_round,
                        loss=client_loss, client_id=client_id_str,
                    )

                    task = self._tm.get_task(task_id)
                    with task.lock:
                        is_done = task.is_done
                        new_round = task.current_round
                        new_weights = task.weights_bytes

                    if is_done:
                        send_msg(conn, MsgType.DONE, {
                            'task_id': task_id,
                            'rounds_completed': new_round,
                            'weights_bytes': new_weights,
                        })
                        break
                    elif aggregated:
                        current_round = new_round
                        send_msg(conn, MsgType.WEIGHTS, {
                            'task_id': task_id,
                            'round_num': new_round,
                            'weights_bytes': new_weights,
                        })
                    else:
                        send_msg(conn, MsgType.ACK, {
                            'task_id': task_id,
                            'message': 'weights received; waiting for other clients',
                        })

                # ── STATUS poll ────────────────────────────────────────
                elif msg_type == MsgType.STATUS:
                    if task_id is None:
                        send_msg(conn, MsgType.ERROR, {'error': 'send JOIN first'})
                        continue

                    task = self._tm.get_task(task_id)
                    with task.lock:
                        is_done = task.is_done
                        server_round = task.current_round
                        weights_bytes = task.weights_bytes

                    polled_round = data.get('round_num', current_round)

                    if is_done:
                        send_msg(conn, MsgType.DONE, {
                            'task_id': task_id,
                            'rounds_completed': server_round,
                            'weights_bytes': weights_bytes,
                        })
                        break
                    elif server_round > polled_round:
                        current_round = server_round
                        send_msg(conn, MsgType.WEIGHTS, {
                            'task_id': task_id,
                            'round_num': server_round,
                            'weights_bytes': weights_bytes,
                        })
                    else:
                        send_msg(conn, MsgType.PENDING, {
                            'task_id': task_id,
                            'current_round': server_round,
                        })

                else:
                    send_msg(conn, MsgType.ERROR, {
                        'error': f'unexpected message type 0x{int(msg_type):02X}',
                    })

        except ConnectionError:
            log.info("client %s:%d disconnected", *addr)
        except Exception as exc:
            log.error("client %s:%d handler error: %s", *addr, exc, exc_info=True)
        finally:
            conn.close()
            log.debug("client %s:%d connection closed", *addr)
