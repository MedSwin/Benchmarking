from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from app.config import settings
from app.models import BenchmarkRequest, DatasetName, MODEL_PROVIDER, TargetModel
from app.runner import BenchmarkManager


logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

manager = BenchmarkManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.output_root.mkdir(parents=True, exist_ok=True)
    settings.data_root.mkdir(parents=True, exist_ok=True)
    yield
    await manager.shutdown()


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

ALL_DATASETS = {dataset.value for dataset in DatasetName}
ALL_MODELS = {model.value: MODEL_PROVIDER[model] for model in TargetModel}


def _enabled_dataset_values() -> set[str]:
    return {dataset.value for dataset in settings.enabled_datasets}


def _enabled_model_ids() -> set[str]:
    return {model["id"] for model in settings.enabled_models}


def _validate_dataset_selection(requested: list[str]) -> None:
    enabled = _enabled_dataset_values()
    disabled = [dataset for dataset in requested if dataset in ALL_DATASETS and dataset not in enabled]
    invalid = [dataset for dataset in requested if dataset not in ALL_DATASETS]
    if disabled or invalid:
        detail_parts = []
        if disabled:
            detail_parts.append(f"disabled: {', '.join(disabled)}")
        if invalid:
            detail_parts.append(f"unknown: {', '.join(invalid)}")
        raise HTTPException(status_code=400, detail="; ".join(detail_parts))


def _normalize_model_id(raw_model: str) -> str:
    try:
        return TargetModel.parse(raw_model).value
    except ValueError:
        return raw_model


# Root Cause vs Logic:
# Root Cause: the API config endpoint returned model ids using `str(TargetModel)` (e.g. TargetModel.gpt_51),
# while request validation only accepted canonical ids (e.g. gpt-5.4).
# Logic: normalize incoming ids through TargetModel.parse so enum-style ids continue working, while preserving
# canonical ids for scheduling and strict unknown/disabled validation.
def _validate_model_selection(requested: list[str]) -> list[str]:
    enabled = _enabled_model_ids()
    normalized = [_normalize_model_id(model) for model in requested]
    disabled = [raw for raw, model in zip(requested, normalized) if model in ALL_MODELS and model not in enabled]
    invalid = [raw for raw, model in zip(requested, normalized) if model not in ALL_MODELS]
    if disabled or invalid:
        detail_parts = []
        if disabled:
            detail_parts.append(f"disabled: {', '.join(disabled)}")
        if invalid:
            detail_parts.append(f"unknown: {', '.join(invalid)}")
        raise HTTPException(status_code=400, detail="; ".join(detail_parts))
    return normalized


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    # Root Cause vs Logic: TemplateResponse expects the request first, so pass it before
    # the template name to ensure Jinja receives a string rather than a dict as the cache key.
    context = {
        "defaults": {
            "workers": settings.default_workers,
        },
    }
    return templates.TemplateResponse(
        request,
        "index.html",
        context=context,
    )


@app.get("/api/config")
async def get_config() -> JSONResponse:
    return JSONResponse(
        {
            "datasets": [dataset.value for dataset in settings.enabled_datasets],
            "models": settings.enabled_models,
            "default_workers": settings.default_workers,
            "refresh_row": settings.refresh_row,
        }
    )


@app.post("/api/jobs")
async def create_job(payload: dict = Body(...)) -> JSONResponse:
    try:
        requested_datasets = payload["datasets"]
        requested_models = payload["models"]
        _validate_dataset_selection(requested_datasets)
        normalized_models = _validate_model_selection(requested_models)
        request = BenchmarkRequest(
            datasets=requested_datasets,
            models=normalized_models,
            max_samples=payload.get("max_samples", 0),
            workers=payload.get("workers", settings.default_workers),
            seed=payload.get("seed", 13),
            output_subdir=payload.get("output_subdir"),
            enable_bert_score=payload.get("enable_bert_score", True),
        )
        job = await manager.start_job(request)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(job.model_dump())


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> JSONResponse:
    try:
        job = manager.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    return JSONResponse(job.model_dump())


@app.delete("/api/jobs/{job_id}")
async def cancel_job(job_id: str) -> JSONResponse:
    try:
        job = await manager.cancel_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    return JSONResponse(job.model_dump())


@app.get("/api/jobs/{job_id}/events")
async def stream_events(job_id: str):
    try:
        queue = manager.event_queues[job_id]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc

    async def generator():
        while True:
            event = await queue.get()
            yield {"event": event.event, "data": json.dumps(event.model_dump())}
            if event.event in {"job_completed", "job_failed", "job_cancelled"}:
                break

    return EventSourceResponse(generator())
