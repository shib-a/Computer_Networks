"""
Monitor HTTP/WebSocket server (runs on port 8080 by default).

Endpoints:
  GET  /           — serves the dashboard HTML
  GET  /api/tasks  — current task list (JSON)
  GET  /api/events — full event history (JSON, for debugging)
  WS   /ws         — real-time event stream

On WebSocket connect the server immediately sends a 'history' message with
all past events so the client can reconstruct the full training state.
Subsequent events are broadcast as they are published to the event bus.
"""
import asyncio
import json
import os
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="FL/1 Monitor", docs_url=None, redoc_url=None)

# ── State shared with the FL server ─────────────────────────────────────────
_task_manager = None          # set by server/main.py before uvicorn starts
_connections: list[WebSocket] = []


def set_task_manager(tm) -> None:
    global _task_manager
    _task_manager = tm


# ── Static files ─────────────────────────────────────────────────────────────
_static = os.path.join(os.path.dirname(__file__), 'static')
app.mount("/static", StaticFiles(directory=_static), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    with open(os.path.join(_static, 'index.html'), encoding='utf-8') as f:
        return f.read()


# ── REST ─────────────────────────────────────────────────────────────────────

@app.get("/api/tasks")
async def api_tasks():
    if _task_manager is None:
        return JSONResponse([])
    return JSONResponse(_task_manager.list_tasks())


@app.get("/api/events")
async def api_events():
    from monitor.bus import event_bus
    return JSONResponse(event_bus.get_history())


class CreateTaskRequest(BaseModel):
    name: str = "mnist_mlp"
    min_clients: int = 2
    max_rounds: int = 10
    hidden: int = 256


@app.post("/api/tasks", status_code=201)
async def api_create_task(req: CreateTaskRequest):
    if _task_manager is None:
        return JSONResponse({"error": "task manager not ready"}, status_code=503)
    import sys, os
    sys.path.insert(0, '/app')
    from models.mnist_mlp import MNISTMLP
    model = MNISTMLP(hidden=req.hidden)
    task = _task_manager.create_task(
        model,
        min_clients=req.min_clients,
        max_rounds=req.max_rounds,
        name=req.name,
    )
    return JSONResponse({"task_id": task.task_id}, status_code=201)


# ── WebSocket ────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    _connections.append(ws)

    # Replay full history so client can reconstruct state from scratch
    from monitor.bus import event_bus
    try:
        history = event_bus.get_history()
        await ws.send_text(json.dumps({'type': 'history', 'events': history}))
    except Exception:
        pass

    try:
        while True:
            # Receive messages only to detect disconnect; clients send pings
            try:
                await asyncio.wait_for(ws.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                await ws.send_text(json.dumps({'type': 'ping'}))
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        if ws in _connections:
            _connections.remove(ws)


# ── Background broadcast loop ────────────────────────────────────────────────

@app.on_event("startup")
async def _startup():
    asyncio.create_task(_broadcast_loop())


async def _broadcast_loop():
    """Drain the event bus every 50 ms and fan-out to all WebSocket clients."""
    from monitor.bus import event_bus
    while True:
        events = event_bus.drain()
        if events and _connections:
            dead: list[WebSocket] = []
            for event in events:
                msg = json.dumps(event)
                for ws in list(_connections):
                    try:
                        await ws.send_text(msg)
                    except Exception:
                        dead.append(ws)
            for ws in dead:
                if ws in _connections:
                    _connections.remove(ws)
        await asyncio.sleep(0.05)


# ── Entry point for running standalone ──────────────────────────────────────

def run(host: str = '0.0.0.0', port: int = 8080, task_manager=None) -> None:
    import uvicorn
    if task_manager is not None:
        set_task_manager(task_manager)
    uvicorn.run(app, host=host, port=port, log_level='warning')
