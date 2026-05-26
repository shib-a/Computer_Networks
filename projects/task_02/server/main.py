"""
Federated Learning Server entry point.

Usage examples
--------------
Standalone (no HA):
    python -m server.main

Primary (with HA backup):
    python -m server.main --role primary --port 9000 \
        --backup-repl-addr localhost:10100

Backup (waits for promotion):
    python -m server.main --role backup --port 9100 --repl-port 10100

Then start clients:
    python -m client.main --servers localhost:9000 localhost:9100
"""
import argparse
import logging
import os
import sys

# Make the package root importable regardless of where the script is launched
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mnist_mlp import MNISTMLP
from server.task_manager import TaskManager
from server.server import FLServer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description='FL/1 server')
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--port', type=int, default=9000,
                   help='port for client connections (default 9000)')
    p.add_argument('--role', choices=['standalone', 'primary', 'backup'],
                   default='standalone')

    # Primary-only
    p.add_argument('--backup-repl-addr', default=None, metavar='HOST:PORT',
                   help='backup replication address (primary only)')

    # Backup-only
    p.add_argument('--repl-port', type=int, default=None,
                   help='port for incoming replication connection (backup only, '
                        'default: port+1000)')

    # Task defaults
    p.add_argument('--min-clients', type=int, default=2,
                   help='minimum clients per round (default 2)')
    p.add_argument('--rounds', type=int, default=10,
                   help='total training rounds (default 10)')
    p.add_argument('--hidden', type=int, default=256,
                   help='MLP hidden layer size (default 256)')

    # Monitor
    p.add_argument('--monitor-port', type=int, default=8080,
                   help='port for the web dashboard (default 8080, 0 to disable)')
    return p.parse_args()


def main():
    args = parse_args()
    repl_port = args.repl_port or (args.port + 1000)

    task_manager = TaskManager()

    # ── Replication setup ────────────────────────────────────────────────
    if args.role == 'primary':
        from server.replication import PrimaryReplicator
        if args.backup_repl_addr is None:
            log.warning("--role primary but --backup-repl-addr not set; running without replication")
            replicator = None
        else:
            host, port_str = args.backup_repl_addr.rsplit(':', 1)
            backup_addr = (host, int(port_str))
            replicator = PrimaryReplicator(task_manager, backup_addr)
            task_manager.set_replication_callback(
                lambda _task: replicator.notify_state_changed()
            )
            replicator.start()

    elif args.role == 'backup':
        from server.replication import BackupReplicationListener

        fl_server_holder: list = []  # mutable container for the lambda

        def on_promote():
            log.info("*** BACKUP PROMOTED TO PRIMARY — starting client server on port %d ***", args.port)
            srv = FLServer(args.host, args.port, task_manager)
            fl_server_holder.append(srv)
            srv.start_background()

        listener = BackupReplicationListener(task_manager, repl_port, on_promote)
        listener.start()

        log.info("backup mode: waiting for primary at repl-port %d (client port %d will open after promotion)",
                 repl_port, args.port)

        # Block the main thread — the FL server starts in on_promote()
        try:
            import time
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            log.info("backup shutting down")
            listener.stop()
            return

    # ── Create default MNIST task ────────────────────────────────────────
    if args.role in ('standalone', 'primary'):
        model = MNISTMLP(hidden=args.hidden)
        task = task_manager.create_task(
            model,
            min_clients=args.min_clients,
            max_rounds=args.rounds,
            name='mnist_mlp',
        )
        log.info("default task created: %s", task.task_id)

    # ── Start monitor dashboard ──────────────────────────────────────────
    if args.monitor_port and args.role in ('standalone', 'primary'):
        import threading
        from monitor.server import run as run_monitor
        mon_thread = threading.Thread(
            target=run_monitor,
            kwargs={'host': args.host, 'port': args.monitor_port, 'task_manager': task_manager},
            daemon=True,
            name='monitor',
        )
        mon_thread.start()
        log.info("dashboard available at http://localhost:%d", args.monitor_port)

    # ── Start FL server ──────────────────────────────────────────────────
    if args.role in ('standalone', 'primary'):
        server = FLServer(args.host, args.port, task_manager)
        try:
            server.start()
        except KeyboardInterrupt:
            log.info("server shutting down")
            server.stop()


if __name__ == '__main__':
    main()
