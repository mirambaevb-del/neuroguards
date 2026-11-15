"""Local/self-hosted FastAPI backend. Never a medical monitoring service."""
from __future__ import annotations

import os
import asyncio
from contextlib import asynccontextmanager
from typing import Literal
from pathlib import Path
import tempfile

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from inference import NeuroGuardDetector
from neuroguard_core import VERSION


MODEL_FILES = ("seizure_detector.joblib", "config.json", "features.json", "manifest.json")
MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_CONCURRENT_INFERENCES = 2


@asynccontextmanager
async def lifespan(application: FastAPI):
    if os.environ.get("NEUROGUARD_PUBLIC") == "1" and not os.environ.get("NEUROGUARD_API_KEY"):
        raise RuntimeError("Public NeuroGuard deployments require NEUROGUARD_API_KEY")
    application.state.detector = None
    application.state.model_error = None
    application.state.inference_slots = asyncio.Semaphore(MAX_CONCURRENT_INFERENCES)
    try:
        application.state.detector = NeuroGuardDetector(os.environ.get("NEUROGUARD_MODEL_DIR"))
    except (FileNotFoundError, RuntimeError) as exc:
        application.state.model_error = str(exc)
    yield


app = FastAPI(title="NeuroGuard Research API", version=VERSION, lifespan=lifespan,
              description="Prototype seizure detection only. Not clinical advice or emergency alerts.")
origins = [s.strip() for s in os.environ.get("NEUROGUARD_ALLOWED_ORIGINS", "").split(",") if s.strip()]
if origins:
    app.add_middleware(CORSMiddleware, allow_origins=origins,
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type", "X-API-Key"])
MAX_EDF_BYTES = 150 * 1024 * 1024


class EEGRequest(BaseModel):
    channels: list[str] = Field(min_length=8, max_length=64)
    samples: list[list[float]]
    sfreq: float = 256
    start_seconds: float = 0
    unit: Literal["V"]


def check_key(request: Request):
    expected = os.environ.get("NEUROGUARD_API_KEY")
    if expected:
        import secrets
        actual = request.headers.get("X-API-Key", "")
        if not secrets.compare_digest(actual, expected):
            raise HTTPException(status_code=401, detail="Invalid API key")


def detector_or_503():
    detector = getattr(app.state, "detector", None)
    if detector is None:
        raise HTTPException(status_code=503, detail=getattr(app.state, "model_error", "Model not loaded at startup"))
    return detector


async def acquire_slot():
    try:
        await asyncio.wait_for(app.state.inference_slots.acquire(), timeout=0.1)
    except TimeoutError as exc:
        raise HTTPException(status_code=429, detail="Inference service busy") from exc


@app.get("/health")
def health():
    base = Path(os.environ.get("NEUROGUARD_MODEL_DIR") or Path(__file__).parent / "model")
    missing = [name for name in MODEL_FILES if not (base / name).is_file()]
    loaded = getattr(app.state, "detector", None) is not None
    return {"status": "service_running", "model_available": not missing and loaded,
            "missing_model_files": missing, "model_loaded": loaded,
            "version": VERSION, "medical_device": False}


@app.get("/model-info")
def model_info(request: Request):
    check_key(request)
    d = detector_or_503()
    return {"status": "ready", "task": "binary_seizure_detection", "version": d.manifest.get("version"),
            "training_cases": d.manifest.get("training_cases", []),
            "training_subjects": d.manifest.get("training_subjects", []),
            "held_out_cases": d.manifest.get("held_out_cases", []),
            "held_out_subjects": d.manifest.get("held_out_subjects", []),
            "channels": list(d.config.channels), "sample_rate": d.config.sfreq,
            "window_seconds": d.config.window_seconds, "calibrated_probabilities": False,
            "clinical_validation": False}


@app.post("/predict")
async def predict(request: Request):
    check_key(request)
    detector = detector_or_503()
    length = int(request.headers.get("Content-Length", "0") or "0")
    if length > MAX_JSON_BYTES:
        raise HTTPException(status_code=413, detail="EEG JSON upload exceeds 4 MiB")
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_JSON_BYTES:
            raise HTTPException(status_code=413, detail="EEG JSON upload exceeds 4 MiB")
        chunks.append(chunk)
    try:
        data = EEGRequest.model_validate_json(b"".join(chunks))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid EEG JSON: {exc}") from exc
    if len(data.samples) != len(data.channels):
        raise HTTPException(status_code=422, detail="Number of sample rows must match channel names")
    await acquire_slot()
    try:
        return await run_in_threadpool(detector.predict_array, data.samples, data.channels,
                                       data.sfreq, data.start_seconds, unit=data.unit)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        app.state.inference_slots.release()


@app.post("/predict-edf")
async def predict_edf(request: Request):
    """POST raw EDF bytes with Content-Type: application/octet-stream.

    Uploads are handled locally by this server, with a strict size cap and
    temporary file cleanup. Do NOT deploy with real patient data without a
    privacy/security review and appropriate authorization.
    """
    check_key(request)
    detector = detector_or_503()
    await acquire_slot()
    size = 0
    try:
        with tempfile.TemporaryDirectory(prefix="neuroguard_") as directory:
            filename = Path(directory) / "request.edf"
            with filename.open("wb") as out:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_EDF_BYTES:
                        raise HTTPException(status_code=413, detail="EDF upload exceeds 150 MiB")
                    out.write(chunk)
            if size == 0:
                raise HTTPException(status_code=422, detail="Empty EDF upload")
            return await run_in_threadpool(detector.predict_edf, filename)
    except (ValueError, OSError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid EDF or processing error: {exc}") from exc
    finally:
        app.state.inference_slots.release()
