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
from app.models import BenchmarkRequest, DatasetName, TargetModel
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


def _enabled_dataset_values() -> set[str]:
    return {dataset.value for dataset in settings.enabled_datasets}


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


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "defaults": {
                "workers": settings.default_workers,
            },
        },
    )


@app.get("/api/config")
async def get_config() -> JSONResponse:
    return JSONResponse(
        {
            "datasets": [dataset.value for dataset in settings.enabled_datasets],
            "models": [
                {
                    "id": TargetModel.gemini_31_pro_preview.value,
                    "provider": "google",
                    "display_name": TargetModel.gemini_31_pro_preview.display_name,
                },
                {
                    "id": TargetModel.gpt_51.value,
                    "provider": "openai",
                    "display_name": TargetModel.gpt_51.display_name,
                },
                {
                    "id": TargetModel.grok_41_fast_reasoning.value,
                    "provider": "xai",
                    "display_name": TargetModel.grok_41_fast_reasoning.display_name,
                },
                {
                    "id": TargetModel.mistral_large_3.value,
                    "provider": "mistral",
                    "display_name": TargetModel.mistral_large_3.display_name,
                },
            ],
            "default_workers": settings.default_workers,
        }
    )


@app.post("/api/jobs")
async def create_job(payload: dict = Body(...)) -> JSONResponse:
    try:
        requested_datasets = payload["datasets"]
        _validate_dataset_selection(requested_datasets)
        request = BenchmarkRequest(
            datasets=requested_datasets,
            models=payload["models"],
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
