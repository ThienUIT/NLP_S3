"""FastAPI + WebSocket server: score incoming customer questions against a
trained S3 checkpoint's department axes (see axis_labels.json, precomputed
offline by validate_uts_bank.py -- see that module for how axis->label AUC
was calibrated) and push alerts to WebSocket clients subscribed to the
matching department.

Not a training service -- loads one fixed checkpoint (CHECKPOINT_PATH) and
its precomputed department mapping (AXIS_LABELS_PATH) at startup, encodes
each incoming question with the same encoder the checkpoint was trained
with, and routes with ica.transform() -- the same paper-§3.1 document-
inference formula the rest of this project uses everywhere else, just
wrapped as a live service instead of a batch script.

Run directly:   uvicorn server.main:app --host 0.0.0.0 --port 8000
Run in Docker:  docker compose up   (see docker-compose.yml)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from s3_reproduction.checkpoint import load_model
from s3_reproduction.encoder import ENCODERS

CHECKPOINT_PATH = Path(os.environ.get("CHECKPOINT_PATH", "/data/checkpoint.joblib"))
LABELS_PATH = Path(os.environ.get("AXIS_LABELS_PATH", "/app/axis_labels.json"))

app = FastAPI(title="S3 Department Router")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

state: dict = {"subscribers": {}}


@app.on_event("startup")
def load_resources() -> None:
    checkpoint = load_model(CHECKPOINT_PATH)
    label_info = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    encoder_kind = checkpoint.metadata.get("encoder_kind", "e5")
    encoder = ENCODERS[encoder_kind](
        model_name=checkpoint.metadata.get("encoder"), batch_size=1, device="cpu",
    )
    state["checkpoint"] = checkpoint
    state["encoder"] = encoder
    state["mapping"] = [e for e in label_info["label_axis"] if e.get("axis") is not None]
    print(
        f"Đã nạp {CHECKPOINT_PATH.name} (n_topics={checkpoint.metadata.get('n_topics')}), "
        f"{len(state['mapping'])} phòng ban đã hiệu chỉnh: "
        f"{[e['label'] for e in state['mapping']]}"
    )


class Question(BaseModel):
    text: str
    id: str | None = None


class RouteResult(BaseModel):
    id: str | None
    text: str
    department: str
    score: float
    auc: float


def route_question(text: str) -> RouteResult:
    checkpoint = state["checkpoint"]
    embedding = state["encoder"].encode([text], "question")
    scores = checkpoint.ica.transform(embedding)[0]
    best_label, best_score, best_auc = "(không xác định)", 0.0, 0.0
    for entry in state["mapping"]:
        signed = float(scores[entry["axis"]] * entry["pole"])
        if signed > best_score or best_label == "(không xác định)":
            best_label, best_score, best_auc = entry["label"], signed, entry["auc"]
    return RouteResult(id=None, text=text, department=best_label, score=best_score, auc=best_auc)


@app.post("/questions", response_model=RouteResult)
async def submit_question(question: Question) -> RouteResult:
    result = route_question(question.text)
    result.id = question.id
    targets = state["subscribers"].get(result.department, set()) | state["subscribers"].get("ALL", set())
    payload = result.model_dump_json()
    for websocket in list(targets):
        try:
            await websocket.send_text(payload)
        except Exception:
            pass
    return result


@app.get("/departments")
async def departments() -> list[str]:
    return sorted(entry["label"] for entry in state["mapping"])


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "departments": len(state.get("mapping", []))}


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    subscribed: str | None = None
    try:
        while True:
            message = await websocket.receive_json()
            department = message.get("subscribe")
            if not department:
                continue
            if subscribed:
                state["subscribers"].get(subscribed, set()).discard(websocket)
            subscribed = department
            state["subscribers"].setdefault(department, set()).add(websocket)
            await websocket.send_json({"status": "subscribed", "department": department})
    except WebSocketDisconnect:
        if subscribed:
            state["subscribers"].get(subscribed, set()).discard(websocket)
