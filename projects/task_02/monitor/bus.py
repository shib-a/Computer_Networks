"""
Thread-safe event bus used to bridge synchronous FL server code with the
async FastAPI monitor server.

Any part of the server (TaskManager, FLServer) calls event_bus.publish(event).
The FastAPI broadcast loop calls event_bus.drain() every 50 ms and pushes
the batch to all connected WebSocket clients.

History is kept in memory (last MAX_HISTORY events) so new dashboard clients
receive the full picture immediately on connect.
"""
import queue
import threading
import time

MAX_HISTORY = 2000


class EventBus:
    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue()
        self._history: list = []
        self._lock = threading.Lock()

    def publish(self, event: dict) -> None:
        if 'ts' not in event:
            event['ts'] = time.time()
        with self._lock:
            self._history.append(event)
            if len(self._history) > MAX_HISTORY:
                self._history = self._history[-MAX_HISTORY:]
        self._queue.put(event)

    def drain(self) -> list:
        """Return all queued events (non-blocking). Called by the async loop."""
        events = []
        while True:
            try:
                events.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return events

    def get_history(self) -> list:
        with self._lock:
            return list(self._history)


# Global singleton — imported by task_manager and monitor/server
event_bus = EventBus()
