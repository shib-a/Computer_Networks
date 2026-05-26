"""
Training task lifecycle management.

Changes vs original:
  - submit_update() accepts optional loss/client_id (forwarded from SUBMIT payload)
  - After each FedAvg aggregation the model is evaluated on the MNIST test set
  - Events (task_created, weights_received, round_complete, training_done) are
    published to monitor.bus.event_bus so the dashboard receives them live

Threading model (unchanged):
  - TaskManager._lock  : guards the _tasks dict
  - TrainingTask.lock  : guards per-task mutation
  Lock order: always manager lock first, then task lock.

TrainingTask.__getstate__/__setstate__ exclude threading.Lock from pickle
(used when the backup server replicates state).
"""
import io
import logging
import threading
import uuid
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .aggregator import fedavg

log = logging.getLogger(__name__)


class TrainingTask:
    def __init__(
        self,
        task_id: str,
        model: nn.Module,
        min_clients_per_round: int,
        max_rounds: int,
    ) -> None:
        self.task_id = task_id
        self.model = model
        self.min_clients_per_round = min_clients_per_round
        self.max_rounds = max_rounds
        self.current_round = 0
        self.is_done = False

        # TorchScript bytes — serialised once, never changes
        scripted = torch.jit.script(model)
        buf = io.BytesIO()
        torch.jit.save(scripted, buf)
        self.arch_bytes: bytes = buf.getvalue()

        # Serialised state_dict — updated after every FedAvg round
        self.weights_bytes: bytes = self._model_weights_bytes()

        # Per-round accumulation
        self._pending: List[Tuple[OrderedDict, int]] = []
        self._pending_meta: List[dict] = []   # {client_id, loss, num_samples}

        self.lock = threading.Lock()

    # ── Pickle support (threading.Lock is not picklable) ──────────────────────
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['lock']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.lock = threading.Lock()
        if '_pending_meta' not in self.__dict__:
            self._pending_meta = []

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _model_weights_bytes(self) -> bytes:
        buf = io.BytesIO()
        torch.save(self.model.state_dict(), buf)
        return buf.getvalue()

    def to_snapshot(self) -> dict:
        with self.lock:
            return {
                'task_id':               self.task_id,
                'min_clients_per_round': self.min_clients_per_round,
                'max_rounds':            self.max_rounds,
                'current_round':         self.current_round,
                'is_done':               self.is_done,
                'arch_bytes':            self.arch_bytes,
                'weights_bytes':         self.weights_bytes,
            }

    @classmethod
    def from_snapshot(cls, snap: dict) -> 'TrainingTask':
        buf = io.BytesIO(snap['arch_bytes'])
        model = torch.jit.load(buf)
        sd = torch.load(io.BytesIO(snap['weights_bytes']), weights_only=False)
        model.load_state_dict(sd)

        task = object.__new__(cls)
        task.task_id               = snap['task_id']
        task.model                 = model
        task.min_clients_per_round = snap['min_clients_per_round']
        task.max_rounds            = snap['max_rounds']
        task.current_round         = snap['current_round']
        task.is_done               = snap['is_done']
        task.arch_bytes            = snap['arch_bytes']
        task.weights_bytes         = snap['weights_bytes']
        task._pending              = []
        task._pending_meta         = []
        task.lock                  = threading.Lock()
        return task


class TaskManager:
    def __init__(self, on_state_changed: Optional[Callable] = None) -> None:
        self._tasks: Dict[str, TrainingTask] = {}
        self._lock = threading.Lock()
        self._on_state_changed: Optional[Callable] = on_state_changed

    def set_replication_callback(self, cb: Callable) -> None:
        self._on_state_changed = cb

    # ── Task CRUD ─────────────────────────────────────────────────────────────
    def create_task(
        self,
        model: nn.Module,
        min_clients: int = 2,
        max_rounds: int = 10,
        name: Optional[str] = None,
    ) -> TrainingTask:
        task_id = (name or '') + '_' + uuid.uuid4().hex[:6]
        task = TrainingTask(task_id, model, min_clients, max_rounds)
        with self._lock:
            self._tasks[task_id] = task
        log.info("created task %s (min_clients=%d, max_rounds=%d)", task_id, min_clients, max_rounds)

        try:
            from monitor.bus import event_bus
            event_bus.publish({
                'type':       'task_created',
                'task_id':    task_id,
                'max_rounds': max_rounds,
                'min_clients': min_clients,
            })
        except Exception:
            pass

        return task

    def get_task(self, task_id: str) -> Optional[TrainingTask]:
        with self._lock:
            return self._tasks.get(task_id)

    def list_tasks(self) -> list:
        with self._lock:
            return [
                {
                    'task_id':      t.task_id,
                    'current_round': t.current_round,
                    'max_rounds':   t.max_rounds,
                    'min_clients':  t.min_clients_per_round,
                    'is_done':      t.is_done,
                }
                for t in self._tasks.values()
            ]

    def first_task_id(self) -> Optional[str]:
        with self._lock:
            ids = list(self._tasks.keys())
        return ids[0] if ids else None

    # ── Update submission & aggregation ──────────────────────────────────────
    def submit_update(
        self,
        task_id: str,
        state_dict: OrderedDict,
        num_samples: int,
        round_num: int,
        loss: Optional[float] = None,
        client_id: Optional[str] = None,
    ) -> bool:
        """
        Record a client update. Returns True if aggregation was triggered.
        FedAvg and model evaluation run outside the lock to keep it brief.
        """
        task = self.get_task(task_id)
        if task is None:
            raise KeyError(f"task {task_id!r} not found")

        should_aggregate = False
        updates_to_aggregate: List[Tuple[OrderedDict, int]] = []
        meta_to_process: List[dict] = []
        submitted_count = 0

        with task.lock:
            if task.is_done:
                return False
            if round_num != task.current_round:
                log.warning("task %s: client submitted for round %d but current is %d — ignored",
                            task_id, round_num, task.current_round)
                return False

            meta = {'client_id': client_id or 'unknown', 'loss': loss, 'num_samples': num_samples}
            task._pending.append((state_dict, num_samples))
            task._pending_meta.append(meta)
            submitted_count = len(task._pending)
            log.info("task %s round %d: %d/%d updates received",
                     task_id, task.current_round, submitted_count, task.min_clients_per_round)

            if submitted_count >= task.min_clients_per_round:
                should_aggregate = True
                updates_to_aggregate = list(task._pending)
                meta_to_process = list(task._pending_meta)
                task._pending = []
                task._pending_meta = []

        # Emit weights_received event (outside lock)
        try:
            from monitor.bus import event_bus
            event_bus.publish({
                'type':        'weights_received',
                'task_id':     task_id,
                'round':       round_num,
                'client_id':   client_id or 'unknown',
                'loss':        loss,
                'num_samples': num_samples,
                'pending_count': submitted_count,
                'quorum':      task.min_clients_per_round,
            })
        except Exception:
            pass

        if not should_aggregate:
            return False

        # ── Aggregate (outside lock) ──────────────────────────────────────────
        log.info("task %s: aggregating round %d (%d clients)",
                 task_id, round_num, len(updates_to_aggregate))
        averaged_sd = fedavg(updates_to_aggregate)

        # Evaluate accuracy on test set (outside lock, ~1–2 s)
        accuracy = self._evaluate(task, averaged_sd)

        # Compute average client loss
        client_losses = [m['loss'] for m in meta_to_process if m['loss'] is not None]
        avg_loss = round(sum(client_losses) / len(client_losses), 6) if client_losses else None

        # ── Commit under lock ─────────────────────────────────────────────────
        with task.lock:
            task.model.load_state_dict(averaged_sd)
            task.weights_bytes = task._model_weights_bytes()
            old_round = task.current_round
            task.current_round += 1
            if task.current_round >= task.max_rounds:
                task.is_done = True
                log.info("task %s: training complete after %d rounds", task_id, task.current_round)
            else:
                log.info("task %s: round %d complete → round %d",
                         task_id, old_round, task.current_round)

        # ── Emit round_complete ───────────────────────────────────────────────
        try:
            from monitor.bus import event_bus
            event_bus.publish({
                'type':        'round_complete',
                'task_id':     task_id,
                'round':       old_round,
                'new_round':   old_round + 1,
                'avg_loss':    avg_loss,
                'accuracy':    accuracy,
                'num_clients': len(updates_to_aggregate),
            })
            if task.is_done:
                event_bus.publish({
                    'type':            'training_done',
                    'task_id':         task_id,
                    'total_rounds':    task.current_round,
                    'final_accuracy':  accuracy,
                })
        except Exception:
            pass

        if self._on_state_changed:
            try:
                self._on_state_changed(task)
            except Exception as exc:
                log.warning("replication callback failed: %s", exc)

        return True

    def _evaluate(self, task: TrainingTask, state_dict: OrderedDict) -> Optional[float]:
        """Run the aggregated model on the MNIST test set. Returns accuracy %."""
        try:
            from models.mnist_mlp import get_data_partition
            model = torch.jit.load(io.BytesIO(task.arch_bytes))
            model.load_state_dict(state_dict)
            model.eval()

            loader = get_data_partition(0, 1, train=False, batch_size=512)
            correct = total = 0
            with torch.no_grad():
                for images, labels in loader:
                    preds = model(images).argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            acc = round(correct / total * 100, 2)
            log.info("task %s: test accuracy after aggregation = %.2f%%", task.task_id, acc)
            return acc
        except Exception as exc:
            log.warning("evaluation failed: %s", exc)
            return None

    # ── Replication ───────────────────────────────────────────────────────────
    def snapshot(self) -> dict:
        with self._lock:
            task_ids = list(self._tasks.keys())
        return {tid: self._tasks[tid].to_snapshot() for tid in task_ids}

    def restore_snapshot(self, snap: dict) -> None:
        with self._lock:
            for task_id, task_snap in snap.items():
                existing = self._tasks.get(task_id)
                if existing is None:
                    self._tasks[task_id] = TrainingTask.from_snapshot(task_snap)
                else:
                    with existing.lock:
                        existing.current_round = task_snap['current_round']
                        existing.is_done       = task_snap['is_done']
                        existing.weights_bytes = task_snap['weights_bytes']
                        sd = torch.load(io.BytesIO(task_snap['weights_bytes']), weights_only=False)
                        existing.model.load_state_dict(sd)
        log.info("snapshot restored: %d tasks", len(snap))
