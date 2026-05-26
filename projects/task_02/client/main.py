"""
Federated Learning Client entry point.

The client:
  1. Connects to the first available server (tries each address in order —
     primary then backup, implementing transparent failover for Complexity 3).
  2. Joins the specified task (or lists available tasks).
  3. For each round: receives model arch + global weights, trains locally,
     submits updated weights, waits for the round to complete.
  4. Exits after participating in the requested number of rounds (or when
     the task is marked done by the server).

Usage examples
--------------
List available tasks:
    python -m client.main --servers localhost:9000 localhost:9100

Join a task and participate in all rounds:
    python -m client.main --servers localhost:9000 localhost:9100 \
        --task-id mnist_mlp_abc123 --client-id 0 --num-clients 2

Participate in a fixed number of rounds:
    python -m client.main ... --rounds 3
"""
import argparse
import logging
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocol.transport import send_msg, recv_msg
from protocol.messages import MsgType
from client.trainer import LocalTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
log = logging.getLogger(__name__)

POLL_INTERVAL = 2.0  # seconds between STATUS polls when waiting for quorum


def connect(server_addrs: list) -> socket.socket:
    """Try each (host, port) pair in order. Return the first that connects."""
    last_err = None
    for host, port in server_addrs:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((host, port))
            sock.settimeout(600.0)
            log.info("connected to server %s:%d", host, port)
            return sock
        except Exception as exc:
            log.warning("cannot reach %s:%d — %s", host, port, exc)
            last_err = exc
    raise ConnectionError(f"no server available; last error: {last_err}")


def parse_args():
    p = argparse.ArgumentParser(description='FL/1 client')
    p.add_argument('--servers', nargs='+', default=['localhost:9000'],
                   metavar='HOST:PORT',
                   help='server addresses to try in order (default: localhost:9000)')
    p.add_argument('--task-id', default=None,
                   help='task to join (omit to list available tasks)')
    p.add_argument('--client-id', type=int, default=0,
                   help='client index for data partitioning (default 0)')
    p.add_argument('--num-clients', type=int, default=2,
                   help='total number of clients for data partitioning (default 2)')
    p.add_argument('--epochs', type=int, default=1,
                   help='local training epochs per round (default 1)')
    p.add_argument('--lr', type=float, default=0.01,
                   help='SGD learning rate (default 0.01)')
    p.add_argument('--rounds', type=int, default=None,
                   help='max rounds to participate in (default: until task done)')
    p.add_argument('--auto-join', action='store_true',
                   help='poll server until a task is available, then join automatically')
    return p.parse_args()


def parse_addr(s: str):
    host, port_str = s.rsplit(':', 1)
    return host, int(port_str)


def main():
    args = parse_args()
    server_addrs = [parse_addr(s) for s in args.servers]
    trainer = LocalTrainer(
        client_id=args.client_id,
        num_clients=args.num_clients,
        epochs=args.epochs,
        lr=args.lr,
    )

    try:
        sock = connect(server_addrs)
    except ConnectionError as exc:
        log.error("%s", exc)
        sys.exit(1)

    try:
        # ── Auto-join: poll until an active task appears ─────────────────
        if args.task_id is None and args.auto_join:
            log.info("--auto-join: polling server for an available task…")
            task_id = None
            while task_id is None:
                send_msg(sock, MsgType.JOIN, {'task_id': None})
                msg_type, data = recv_msg(sock)
                if msg_type == MsgType.TASK_INFO and 'tasks' in data:
                    active = [t for t in data['tasks'] if not t['is_done']]
                    if active:
                        task_id = active[0]['task_id']
                        log.info("auto-join: found task %s", task_id)
                    else:
                        log.info("auto-join: no active tasks yet, retrying in 5s…")
                        time.sleep(5.0)
                else:
                    log.warning("auto-join: unexpected response %s, retrying in 5s…", msg_type.name)
                    time.sleep(5.0)

        # ── List tasks if no task-id given ───────────────────────────────
        elif args.task_id is None:
            send_msg(sock, MsgType.JOIN, {'task_id': None})
            msg_type, data = recv_msg(sock)
            if msg_type == MsgType.TASK_INFO and 'tasks' in data:
                tasks = data['tasks']
                if not tasks:
                    log.info("no tasks available on server")
                else:
                    log.info("available tasks:")
                    for t in tasks:
                        status = "done" if t['is_done'] else f"round {t['current_round']}/{t['max_rounds']}"
                        log.info("  %s  [%s, min_clients=%d]", t['task_id'], status, t['min_clients'])
                    log.info("re-run with --task-id <id> to participate")
            return

        else:
            task_id = args.task_id

        rounds_done = 0
        max_rounds = args.rounds  # None means "until task done"
        current_round = -1

        # ── Training loop ────────────────────────────────────────────────
        while max_rounds is None or rounds_done < max_rounds:

            # ── Re-join to get latest arch + weights ─────────────────────
            send_msg(sock, MsgType.JOIN, {'task_id': task_id})
            msg_type, data = recv_msg(sock)

            if msg_type == MsgType.ERROR:
                log.error("server error: %s", data['error'])
                sys.exit(1)

            if msg_type == MsgType.DONE:
                log.info("task %s complete (%d rounds). Exiting.", task_id, data['rounds_completed'])
                break

            if msg_type != MsgType.TASK_INFO:
                log.error("unexpected message %s after JOIN", msg_type.name)
                break

            current_round = data['current_round']
            arch_bytes = data['arch_bytes']
            weights_bytes = data['weights_bytes']

            log.info("=== round %d/%d ===", current_round + 1, data['max_rounds'])

            # ── Local training ───────────────────────────────────────────
            updated_weights, n_samples, train_loss = trainer.train(arch_bytes, weights_bytes)

            # ── Submit weights ───────────────────────────────────────────
            send_msg(sock, MsgType.SUBMIT, {
                'task_id':       task_id,
                'round_num':     current_round,
                'weights_bytes': updated_weights,
                'num_samples':   n_samples,
                'loss':          train_loss,
                'client_id':     str(args.client_id),
            })

            msg_type, data = recv_msg(sock)

            if msg_type == MsgType.DONE:
                log.info("training done after round %d. Exiting.", data['rounds_completed'])
                break

            if msg_type == MsgType.WEIGHTS:
                log.info("round %d aggregated. New global round: %d", current_round + 1, data['round_num'])
                rounds_done += 1
                continue  # next iteration will JOIN for the new round

            if msg_type == MsgType.ACK:
                log.info("weights accepted; waiting for other clients…")
                # Poll until this round completes
                while True:
                    time.sleep(POLL_INTERVAL)
                    send_msg(sock, MsgType.STATUS, {
                        'task_id': task_id,
                        'round_num': current_round,
                    })
                    msg_type2, data2 = recv_msg(sock)

                    if msg_type2 == MsgType.DONE:
                        log.info("training done (detected during poll). Exiting.")
                        rounds_done += 1
                        # Signal outer loop to break
                        max_rounds = 0
                        break

                    if msg_type2 == MsgType.WEIGHTS:
                        log.info("round %d complete (detected during poll).", current_round + 1)
                        rounds_done += 1
                        break

                    if msg_type2 == MsgType.PENDING:
                        log.debug("round %d still pending…", current_round + 1)
                        continue

                    log.warning("unexpected message during poll: %s", msg_type2.name)

            else:
                log.error("unexpected response to SUBMIT: %s", msg_type.name)
                break

        log.info("client %d finished; participated in %d round(s)", args.client_id, rounds_done)

    except KeyboardInterrupt:
        log.info("interrupted")
    except ConnectionError as exc:
        log.error("connection lost: %s", exc)
        sys.exit(1)
    except Exception as exc:
        log.error("fatal error: %s", exc, exc_info=True)
        sys.exit(1)
    finally:
        sock.close()


if __name__ == '__main__':
    main()
