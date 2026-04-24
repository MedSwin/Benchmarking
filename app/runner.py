from __future__ import annotations

import asyncio
import csv
import importlib
import json
import math
import random
import re
import uuid
from collections import defaultdict
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.audit import AuditTrail
from app.config import settings
from app.datasets import extract_messages, load_dataset_rows, normalize_option_text, uniq_sorted_letters
from app.metrics import compute_text_metrics, mean_metric, norm_text
from app.models import (
    BenchmarkRequest,
    DatasetName,
    EventPayload,
    JobInfo,
    JobStatus,
    JobSummary,
    MODEL_PROVIDER,
    TargetModel,
)

_BERT_SCORE_FN = None
_BERT_SCORE_IMPORT_ERROR: Optional[str] = None
_BERT_SCORE_TOKENIZER_PATCHED = False
JOB_STATE_FILENAME = "job.json"
DATASET_NAMES = {dataset.value for dataset in DatasetName}
MODEL_BY_DISPLAY_NAME = {model.display_name: model for model in TargetModel}


def _get_bert_score_fn():
    global _BERT_SCORE_FN, _BERT_SCORE_IMPORT_ERROR
    if _BERT_SCORE_FN is not None:
        return _BERT_SCORE_FN
    if _BERT_SCORE_IMPORT_ERROR is not None:
        return None
    try:
        bert_score_fn = getattr(importlib.import_module("bert_score"), "score")
    except Exception as exc:  # pragma: no cover - runtime optional dependency behavior
        _BERT_SCORE_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
        return None
    _patch_bert_score_tokenizer()
    _BERT_SCORE_FN = bert_score_fn
    return _BERT_SCORE_FN


def _ensure_build_inputs_with_special_tokens(tokenizer: Any) -> None:
    if hasattr(tokenizer, "build_inputs_with_special_tokens"):
        return
    cls_token_id = getattr(tokenizer, "cls_token_id", None)
    sep_token_id = getattr(tokenizer, "sep_token_id", None)

    def build_inputs_with_special_tokens(token_ids: Sequence[int]) -> List[int]:
        tokens = list(token_ids)
        output: List[int] = []
        if cls_token_id is not None:
            output.append(cls_token_id)
        output.extend(tokens)
        if sep_token_id is not None:
            output.append(sep_token_id)
        return output

    tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens


def _patch_bert_score_tokenizer() -> None:
    global _BERT_SCORE_TOKENIZER_PATCHED
    if _BERT_SCORE_TOKENIZER_PATCHED:
        return
    # Motivation vs Logic:
    # Motivation: transformers 5+ ships a new TokenizersBackend that lacks legacy helpers such as
    # `build_inputs_with_special_tokens`, which BERTScore relies on for empty sentences.
    # Logic: patch the imported tokenizer factory so every tokenizer exposes a minimal shim before BERTScore runs.
    import bert_score.utils as bs_utils

    original_get_tokenizer = bs_utils.get_tokenizer

    def patched_get_tokenizer(model_type: str, use_fast: bool = False) -> Any:
        tokenizer = original_get_tokenizer(model_type, use_fast)
        _ensure_build_inputs_with_special_tokens(tokenizer)
        return tokenizer

    bs_utils.get_tokenizer = patched_get_tokenizer
    _BERT_SCORE_TOKENIZER_PATCHED = True


class BenchmarkManager:
    def __init__(self) -> None:
        self.jobs: Dict[str, JobInfo] = {}
        self.event_queues: Dict[str, asyncio.Queue[EventPayload]] = {}
        self.audit = AuditTrail(settings.output_root)
        from app.providers import ProviderPool

        self.provider_pool = ProviderPool()
        self.tasks: Dict[str, asyncio.Task[Any]] = {}
        self.stop_reasons: Dict[str, str] = {}

    def load_persisted_jobs(self) -> None:
        if not settings.output_root.exists():
            return
        for job_dir in sorted(settings.output_root.iterdir()):
            if not job_dir.is_dir():
                continue
            job = self._load_persisted_job(job_dir)
            if job is None:
                continue
            self.jobs[job.job_id] = job

    def list_jobs(self) -> List[JobInfo]:
        return sorted(self.jobs.values(), key=self._job_sort_key, reverse=True)

    async def shutdown(self) -> None:
        for job_id, task in list(self.tasks.items()):
            self.stop_reasons[job_id] = "pause"
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        await self.provider_pool.close()

    async def start_job(self, request: BenchmarkRequest) -> JobInfo:
        if not request.datasets:
            raise ValueError("At least one dataset must be selected.")
        if not request.models:
            raise ValueError("At least one model must be selected.")
        job_id = uuid.uuid4().hex[:12]
        job = JobInfo(job_id=job_id, status=JobStatus.queued, request=request)
        self.jobs[job_id] = job
        self.event_queues[job_id] = asyncio.Queue(maxsize=settings.event_queue_size)
        self.stop_reasons.pop(job_id, None)
        self._save_job_snapshot(job)
        self.tasks[job_id] = asyncio.create_task(self._run_job(job_id))
        await self._emit(job_id, "job_created", f"Queued benchmark job for {len(request.datasets)} datasets.")
        return job

    def get_job(self, job_id: str) -> JobInfo:
        if job_id not in self.jobs:
            raise KeyError(job_id)
        return self.jobs[job_id]

    async def pause_job(self, job_id: str) -> JobInfo:
        job = self.get_job(job_id)
        if job.status == JobStatus.paused:
            return job
        if job.status not in {JobStatus.queued, JobStatus.running}:
            raise ValueError(f"Only queued or running jobs can be paused, found {job.status.value}.")
        task = self.tasks.get(job_id)
        if task:
            self.stop_reasons[job_id] = "pause"
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            return self.get_job(job_id)
        job.status = JobStatus.paused
        job.finished_at = datetime.now(timezone.utc).isoformat()
        job.error = job.error or "Session was paused before the run task could continue."
        self._save_job_snapshot(job)
        await self._emit(job_id, "job_paused", "Job paused.")
        return job

    async def resume_job(self, job_id: str) -> JobInfo:
        job = self.get_job(job_id)
        if job.status in {JobStatus.queued, JobStatus.running}:
            raise ValueError(f"Job {job_id} is already active.")
        if job.status == JobStatus.completed:
            raise ValueError(f"Job {job_id} is already completed.")
        self.event_queues[job_id] = asyncio.Queue(maxsize=settings.event_queue_size)
        self.stop_reasons.pop(job_id, None)
        job.status = JobStatus.queued
        job.finished_at = None
        job.error = None
        self._save_job_snapshot(job)
        self.tasks[job_id] = asyncio.create_task(self._run_job(job_id))
        await self._emit(job_id, "job_resumed", "Resuming benchmark job from saved progress.")
        return job

    async def cancel_job(self, job_id: str) -> JobInfo:
        job = self.get_job(job_id)
        task = self.tasks.get(job_id)
        if task:
            self.stop_reasons[job_id] = "cancel"
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            return self.get_job(job_id)
        job.status = JobStatus.cancelled
        job.finished_at = datetime.now(timezone.utc).isoformat()
        self._save_job_snapshot(job)
        await self._emit(job_id, "job_cancelled", "Job cancellation requested.")
        return job

    def get_job_events(self, job_id: str) -> List[Dict[str, Any]]:
        self.get_job(job_id)
        return self._read_event_records(job_id)

    async def _emit(
        self,
        job_id: str,
        event: str,
        message: str,
        *,
        dataset: Optional[str] = None,
        model: Optional[str] = None,
        data: Optional[dict] = None,
        level: str = "INFO",
    ) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        payload = EventPayload(
            event=event,
            job_id=job_id,
            ts=ts,
            dataset=dataset,
            model=model,
            message=message,
            data=data or {},
        )
        queue = self.event_queues[job_id]
        with suppress(asyncio.QueueFull):
            queue.put_nowait(payload)
        await self.audit.append_event(job_id, payload)
        await self.audit.append_log(job_id, message, level=level, dataset=dataset, model=model, data=data)

    # Motivation vs Logic:
    # Motivation: after toggling provider flags we should not resume work for models that are no longer enabled.
    # Logic: run-time filter removes disabled providers from the request and reports which models were skipped.
    def _filter_disabled_models(self, models: Sequence[TargetModel]) -> tuple[List[TargetModel], List[str]]:
        enabled_providers = {provider for provider, enabled in settings.enabled_providers.items() if enabled}
        allowed: List[TargetModel] = []
        skipped: List[str] = []
        for model in models:
            provider = MODEL_PROVIDER.get(model)
            if not provider or provider not in enabled_providers:
                skipped.append(model.display_name)
                continue
            allowed.append(model)
        return allowed, skipped

    async def _run_job(self, job_id: str) -> None:
        job = self.get_job(job_id)
        job.status = JobStatus.running
        job.started_at = job.started_at or datetime.now(timezone.utc).isoformat()
        job.finished_at = None
        self._save_job_snapshot(job)
        request = job.request
        filtered_models, disabled_models = self._filter_disabled_models(request.models)
        if disabled_models:
            await self._emit(
                job_id,
                "model_skipped",
                f"Skipping disabled model(s): {', '.join(disabled_models)}.",
                level="WARNING",
                data={"models": disabled_models},
            )
        if not filtered_models:
            raise RuntimeError("All requested models are currently disabled in this environment.")
        if filtered_models != request.models:
            request.models = filtered_models
            self._save_job_snapshot(job)
        try:
            for dataset in request.datasets:
                await self._run_dataset(job_id, dataset, request)
            job.status = JobStatus.completed
            job.finished_at = datetime.now(timezone.utc).isoformat()
            self._save_job_snapshot(job)
            await self._emit(job_id, "job_completed", "Benchmark run completed successfully.")
        except asyncio.CancelledError:
            stop_reason = self.stop_reasons.pop(job_id, "")
            job.status = JobStatus.paused if stop_reason == "pause" else JobStatus.cancelled
            job.finished_at = datetime.now(timezone.utc).isoformat()
            if job.status == JobStatus.paused:
                job.error = job.error or "Session paused with partial progress saved."
            else:
                job.error = job.error or "Job cancellation requested."
            self._save_job_snapshot(job)
            await self._emit(
                job_id,
                "job_paused" if job.status == JobStatus.paused else "job_cancelled",
                "Benchmark run paused with resumable progress saved."
                if job.status == JobStatus.paused
                else "Benchmark run cancelled.",
                level="WARNING",
            )
            raise
        except Exception as exc:
            job.status = JobStatus.failed
            job.error = str(exc)
            job.finished_at = datetime.now(timezone.utc).isoformat()
            self._save_job_snapshot(job)
            await self._emit(job_id, "job_failed", str(exc), level="ERROR")
        finally:
            self.tasks.pop(job_id, None)
            self.stop_reasons.pop(job_id, None)

    async def _run_dataset(self, job_id: str, dataset: DatasetName, request: BenchmarkRequest) -> None:
        dataset_rows = load_dataset_rows(settings.data_root, dataset, request.max_samples, request.seed)
        await self._emit(job_id, "dataset_loaded", f"Loaded {len(dataset_rows)} rows.", dataset=dataset.value)
        dataset_dir = self._dataset_dir(job_id, dataset, request.output_subdir)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        workers = max(1, request.workers or settings.default_workers)
        summary_path = dataset_dir / "summary.json"
        existing_summary = self._load_json_file(summary_path) or {}
        summary: Dict[str, Any] = {
            "rows": len(dataset_rows),
            "processed_rows": int(existing_summary.get("processed_rows", 0) or 0),
            "models": dict(existing_summary.get("models") or {}),
        }
        target_rows = min(len(dataset_rows), settings.cap_row) if settings.cap_row else len(dataset_rows)
        error_to_raise: Optional[Exception] = None
        try:
            for model in request.models:
                existing_model = summary["models"].get(model.display_name)
                if existing_model and not existing_model.get("error") and int(existing_model.get("rows", 0) or 0) >= target_rows:
                    summary["processed_rows"] = max(summary["processed_rows"], int(existing_model.get("rows", 0) or 0))
                    continue
                model_rows, model_summary, model_error = await self._evaluate_model(
                    job_id,
                    dataset,
                    model,
                    dataset_rows,
                    dataset_dir,
                    workers,
                    request.enable_bert_score,
                )
                # Root Cause vs Logic:
                # Root Cause: `str(TargetModel)` serializes to enum-style ids
                # (e.g. TargetModel.gemini_31_pro_preview) and leaked into summary/output.
                # Logic: persist canonical model ids via `.value` so artifacts and job overview
                # align with request payload ids.
                model_id = model.value
                summary_csv = dataset_dir / f"{model_id}.csv"
                write_csv(summary_csv, model_rows)
                model_summary.artifacts["detail_csv"] = str(summary_csv)
                display_name = model.display_name
                model_record = model_summary.model_dump()
                model_record.update({"model_id": model_id, "display_name": display_name})
                if model_error:
                    model_record["error"] = str(model_error)
                summary["models"][display_name] = model_record
                summary["processed_rows"] = max(summary["processed_rows"], model_summary.rows)
                event_name = "model_completed" if not model_error else "model_failed"
                event_message = (
                    f"Finished {display_name} on {dataset.value}."
                    if not model_error
                    else f"Stopped {display_name} on {dataset.value} after {model_summary.rows} rows: {model_error}"
                )
                await self._emit(
                    job_id,
                    event_name,
                    event_message,
                    dataset=dataset.value,
                    model=display_name,
                    data={"processed_rows": model_summary.rows, **({"error": str(model_error)} if model_error else {})},
                    level="ERROR" if model_error else "INFO",
                )
                if model_error:
                    error_to_raise = model_error
                    break
        finally:
            summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            job = self.get_job(job_id)
            job.datasets[dataset.value] = summary
            self._save_job_snapshot(job)
        if error_to_raise:
            raise error_to_raise

    async def _evaluate_model(
        self,
        job_id: str,
        dataset: DatasetName,
        model: TargetModel,
        rows: List[Any],
        dataset_dir: Path,
        workers: int,
        enable_bert_score: bool,
    ) -> Tuple[List[Dict[str, Any]], JobSummary, Optional[Exception]]:
        model_id = model.value
        model_label = model.display_name
        refresh_every = max(1, settings.refresh_row)
        rows_to_process = rows
        cap = settings.cap_row
        if cap and len(rows_to_process) > cap:
            # Motivation vs Logic: enforce the env CAP_ROW to bound tokens and duration without altering dataset metadata.
            rows_to_process = rows_to_process[:cap]
        audit_path = dataset_dir / f"{model_id}.audit.jsonl"
        # Motivation vs Logic:
        # Motivation: pause/resume and restart recovery should continue from saved scored rows instead of repeating completed work.
        # Logic: hydrate prior row records from the per-model audit file, queue only missing row ids, then rewrite the audit file
        # from the merged in-memory results once the model finishes or partially completes.
        row_index_by_id = {str(row.id): idx for idx, row in enumerate(rows_to_process)}
        existing_rows = self._read_jsonl(audit_path)
        results: List[Optional[Dict[str, Any]]] = [None] * len(rows_to_process)
        restored_rows = 0
        for record in existing_rows:
            row_id = str(record.get("id") or "").strip()
            idx = row_index_by_id.get(row_id)
            if idx is None or results[idx] is not None:
                continue
            results[idx] = record
            restored_rows += 1

        queue: asyncio.Queue[Tuple[int, Any]] = asyncio.Queue()
        scored_rows_queue: asyncio.Queue[Optional[Tuple[int, Dict[str, Any]]]] = asyncio.Queue()
        for idx, row in enumerate(rows_to_process):
            if results[idx] is None:
                queue.put_nowait((idx, row))
        remaining_rows = queue.qsize()
        max_tokens = {
            DatasetName.medmcqa: settings.medmcqa_max_new_tokens,
            DatasetName.medquad: settings.medquad_max_new_tokens,
            DatasetName.healthbench: settings.healthbench_max_new_tokens,
        }[dataset]
        abort_event = asyncio.Event()
        model_error: Optional[Exception] = None
        bert_score_fn = _get_bert_score_fn()
        if enable_bert_score and remaining_rows and dataset != DatasetName.medmcqa and not bert_score_fn:
            # Root Cause vs Logic:
            # Root Cause: BERTScore availability was validated only after generation completed, so
            # long-running jobs appeared active but never emitted finalized scored rows.
            # Logic: fail before workers start if BERTScore is required but unavailable.
            raise RuntimeError(
                "BERTScore is enabled but unavailable. Install dependencies with "
                "`pip install -e .` and restart the app. "
                f"Import error: {_BERT_SCORE_IMPORT_ERROR or 'unknown'}"
            )

        # Root Cause vs Logic: provider exhaustion stopped the run before partial metrics could be flushed, so we now stop all workers and capture the processed rows before returning the error.
        async def worker(worker_id: int) -> None:
            nonlocal model_error
            while not abort_event.is_set():
                try:
                    idx, row = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    response = await self.provider_pool.generate(model, row.prompt, max_tokens)
                    metrics = self._score_row(dataset, row, response.text)
                    record = {
                        "id": row.id,
                        "prediction": response.text,
                        "reference": row.reference,
                        **metrics,
                        **row.metadata,
                    }
                    results[idx] = record
                    await scored_rows_queue.put((idx, record))
                    await self._emit(
                        job_id,
                        "row_generated",
                        f"worker={worker_id} generated row {row.id}",
                        dataset=dataset.value,
                        model=model_label,
                        data={"row_id": row.id},
                    )
                except Exception as exc:
                    if model_error is None:
                        model_error = exc
                    abort_event.set()
                    await self._emit(
                        job_id,
                        "row_failed",
                        f"worker={worker_id} failed row {row.id}: {exc}",
                        dataset=dataset.value,
                        model=model_label,
                        data={"row_id": row.id, "error": str(exc)},
                        level="ERROR",
                    )
                    return
                finally:
                    queue.task_done()

        async def _flush_scored_batch(batch: List[Tuple[int, Dict[str, Any]]]) -> None:
            if not batch:
                return
            rows_to_emit = [row_dict for _, row_dict in batch]
            if enable_bert_score and bert_score_fn and dataset != DatasetName.medmcqa:
                preds = [item["prediction"] for item in rows_to_emit]
                refs = [item["reference"] for item in rows_to_emit]
                # Root Cause vs Logic:
                # Root Cause: BERTScore still depends on a tokenizer helper that transformers 5+ can omit,
                # which would blow up same as the `build_inputs_with_special_tokens` failure.
                # Logic: shield every scoring call with the patched helper and degrade gracefully on failure.
                try:
                    _, _, f1 = bert_score_fn(
                        preds,
                        refs,
                        lang="en",
                        rescale_with_baseline=False,
                        model_type=settings.bert_score_model_type,
                        use_fast_tokenizer=True,
                    )
                except Exception as exc:
                    await self._emit(
                        job_id,
                        "batch_scoring_error",
                        f"Failed to compute BERTScore for batch: {exc}",
                        dataset=dataset.value,
                        model=model_label,
                        level="WARNING",
                        data={"error": str(exc), "row_ids": [row_dict["id"] for row_dict in rows_to_emit]},
                    )
                    for row_dict in rows_to_emit:
                        row_dict["bert_f"] = 0.0
                else:
                    # Root Cause vs Logic:
                    # Root Cause: baseline rescaling and numerical drift can push raw BERTScore values outside the 0–1 range.
                    # Logic: clamp each per-row F1 score so downstream metrics never claim more than 100%.
                    for row_dict, score in zip(rows_to_emit, f1.tolist()):
                        row_dict["bert_f"] = self._clamp_metric(float(score))
            else:
                for row_dict in rows_to_emit:
                    row_dict.setdefault("bert_f", 0.0)
            self._append_jsonl_records(audit_path, rows_to_emit)
            for _, row_dict in batch:
                row_metrics = {
                    "row_id": row_dict["id"],
                    "rougeL_f": float(row_dict.get("rougeL_f", 0.0)),
                    "tok_f1": float(row_dict.get("tok_f1", 0.0)),
                    "uni_prec": float(row_dict.get("uni_prec", 0.0)),
                    "bi_prec": float(row_dict.get("bi_prec", 0.0)),
                    "bert_f": float(row_dict.get("bert_f", 0.0)),
                }
                await self._emit(
                    job_id,
                    "row_scored",
                    f"Scored row {row_dict['id']}",
                    dataset=dataset.value,
                    model=model_label,
                    data=row_metrics,
                )

        # Motivation vs Logic:
        # Emit `row_scored` continuously in configurable REFRESH_ROW batches so the UI can
        # compute and render rolling averages during execution instead of waiting for model completion.
        async def emit_scored_rows() -> None:
            pending_batch: List[Tuple[int, Dict[str, Any]]] = []
            while True:
                queued = await scored_rows_queue.get()
                if queued is None:
                    await _flush_scored_batch(pending_batch)
                    return
                pending_batch.append(queued)
                if len(pending_batch) >= refresh_every:
                    await _flush_scored_batch(pending_batch)
                    pending_batch = []

        scored_emitter_task = asyncio.create_task(emit_scored_rows())

        await self._emit(
            job_id,
            "model_started",
            (
                f"Resuming {model_label} on {dataset.value} with {workers} workers via {MODEL_PROVIDER[model]} "
                f"({restored_rows} restored, {remaining_rows} remaining)"
                if restored_rows
                else f"Starting {model_label} on {dataset.value} with {workers} workers via {MODEL_PROVIDER[model]}"
            ),
            dataset=dataset.value,
            model=model_label,
        )
        try:
            if remaining_rows:
                await asyncio.gather(*(worker(worker_id) for worker_id in range(1, workers + 1)))
        except asyncio.CancelledError:
            abort_event.set()
            await scored_rows_queue.put(None)
            await asyncio.shield(scored_emitter_task)
            raise
        await scored_rows_queue.put(None)
        await scored_emitter_task

        final_rows = [item for item in results if item is not None]

        metric_keys = ["rougeL_f", "tok_f1", "uni_prec", "bi_prec", "bert_f"]
        means = {key: mean_metric(final_rows, key) for key in metric_keys}
        with audit_path.open("w", encoding="utf-8") as handle:
            for row in final_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary = JobSummary(metric_means=means, rows=len(final_rows), artifacts={"audit_jsonl": str(audit_path)})
        metrics_path = dataset_dir / f"{model_id}.metrics.json"
        metrics_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        summary.artifacts["metrics_json"] = str(metrics_path)
        return final_rows, summary, model_error

    def _score_row(self, dataset: DatasetName, row: Any, prediction: str) -> Dict[str, Any]:
        if dataset == DatasetName.medmcqa:
            return self._score_medmcqa(row, prediction)
        metrics = compute_text_metrics(row.reference, prediction)
        metrics["bert_f"] = 0.0
        return metrics

    def _score_medmcqa(self, row: Any, prediction: str) -> Dict[str, Any]:
        option_map = row.metadata["options"]
        gold_letters = row.metadata["gold_letters"]
        pred_letters, valid, parse_source = parse_medmcqa_prediction(prediction, option_map, row.metadata["choice_type"])
        exact_match = set(pred_letters) == set(gold_letters)
        return {
            "rougeL_f": 1.0 if exact_match else 0.0,
            "tok_f1": 1.0 if exact_match else 0.0,
            "uni_prec": 1.0 if exact_match else 0.0,
            "bi_prec": 1.0 if exact_match else 0.0,
            "bert_f": 1.0 if exact_match else 0.0,
            "pred_letters": ",".join(pred_letters),
            "pred_texts": " | ".join(option_map[letter] for letter in pred_letters if letter in option_map),
            "parse_source": parse_source,
            "is_valid": int(valid),
            "correct": int(exact_match),
            "outcome": "invalid" if not valid else ("correct" if exact_match else "incorrect"),
        }

    @staticmethod
    def _clamp_metric(value: float) -> float:
        if not math.isfinite(value):
            return 0.0
        return max(0.0, min(1.0, value))

    def _dataset_dir(self, job_id: str, dataset: DatasetName, output_subdir: Optional[str]) -> Path:
        base = settings.output_root / job_id
        if output_subdir:
            base = base / output_subdir
        return base / dataset.value

    def _job_dir(self, job_id: str) -> Path:
        return settings.output_root / job_id

    def _job_state_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / JOB_STATE_FILENAME

    def _events_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "events.jsonl"

    def _audit_log_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "audit.log.jsonl"

    def _save_job_snapshot(self, job: JobInfo) -> None:
        state_path = self._job_state_path(job.job_id)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(job.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")

    def _load_json_file(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def _append_jsonl_records(self, path: Path, records: Sequence[Dict[str, Any]]) -> None:
        if not records:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _load_persisted_job(self, job_dir: Path) -> Optional[JobInfo]:
        state_path = job_dir / JOB_STATE_FILENAME
        if state_path.exists():
            try:
                raw = json.loads(state_path.read_text(encoding="utf-8"))
                job = self._job_from_dict(raw)
            except Exception:
                job = None
            else:
                if job.status in {JobStatus.queued, JobStatus.running}:
                    # Root Cause vs Logic:
                    # Root Cause: benchmark execution lives in-process, so an app restart leaves any queued/running
                    # session without workers even though its partial artifacts are still on disk.
                    # Logic: restore the saved dashboard state, but mark the session as paused-interrupted so the UI
                    # can faithfully reload what completed and let the user continue from the saved progress.
                    job.status = JobStatus.paused
                    job.finished_at = job.finished_at or self._last_recorded_ts(job.job_id) or datetime.now(timezone.utc).isoformat()
                    job.error = job.error or "Session was interrupted when the app restarted. Continue will resume from saved progress."
                    self._save_job_snapshot(job)
                return job
        job = self._reconstruct_job_from_artifacts(job_dir)
        if job is not None:
            self._save_job_snapshot(job)
        return job

    def _job_from_dict(self, raw: Dict[str, Any]) -> JobInfo:
        return JobInfo(
            job_id=str(raw["job_id"]),
            status=JobStatus(raw["status"]),
            request=BenchmarkRequest(**raw["request"]),
            started_at=raw.get("started_at"),
            finished_at=raw.get("finished_at"),
            datasets=raw.get("datasets") or {},
            error=raw.get("error"),
        )

    def _reconstruct_job_from_artifacts(self, job_dir: Path) -> Optional[JobInfo]:
        events = self._read_jsonl(self._events_path(job_dir.name))
        audit_logs = self._read_jsonl(self._audit_log_path(job_dir.name))
        summary_files = sorted(job_dir.rglob("summary.json"))
        if not events and not summary_files:
            return None

        datasets_in_order: List[str] = []
        models_in_order: List[str] = []
        workers = settings.default_workers
        output_subdir: Optional[str] = None
        error: Optional[str] = None
        status: Optional[JobStatus] = None
        started_at = audit_logs[0].get("ts") if audit_logs else None
        finished_at = audit_logs[-1].get("ts") if audit_logs else None

        for event in events:
            event_name = str(event.get("event") or "")
            dataset_name = str(event.get("dataset") or "")
            if dataset_name in DATASET_NAMES and dataset_name not in datasets_in_order:
                datasets_in_order.append(dataset_name)
            if event_name == "model_started":
                model_name = str(event.get("model") or "")
                model_id = self._normalize_model_id(model_name)
                if model_id and model_id not in models_in_order:
                    models_in_order.append(model_id)
                match = re.search(r"with\s+(\d+)\s+workers", str(event.get("message") or ""))
                if match:
                    workers = max(1, int(match.group(1)))
            elif event_name == "job_completed":
                status = JobStatus.completed
                finished_at = event.get("ts") or finished_at
            elif event_name == "job_cancelled":
                status = JobStatus.cancelled
                finished_at = event.get("ts") or finished_at
            elif event_name == "job_failed":
                status = JobStatus.failed
                finished_at = event.get("ts") or finished_at
                error = str(event.get("message") or error or "")

        datasets_payload: Dict[str, Dict[str, Any]] = {}
        for summary_path in summary_files:
            relative_parent = summary_path.parent.relative_to(job_dir)
            if not relative_parent.parts:
                continue
            dataset_name = relative_parent.parts[-1]
            if dataset_name not in DATASET_NAMES:
                continue
            if dataset_name not in datasets_in_order:
                datasets_in_order.append(dataset_name)
            prefix_parts = relative_parent.parts[:-1]
            if prefix_parts and output_subdir is None:
                output_subdir = "/".join(prefix_parts)
            try:
                summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            datasets_payload[dataset_name] = summary_payload
            for model_record in summary_payload.get("models", {}).values():
                model_id = self._normalize_model_id(
                    model_record.get("model_id") or model_record.get("display_name") or ""
                )
                if model_id and model_id not in models_in_order:
                    models_in_order.append(model_id)
                if not error and model_record.get("error"):
                    error = str(model_record["error"])

        if not datasets_in_order or not models_in_order:
            return None
        if status is None:
            status = JobStatus.paused
            error = error or "Session was interrupted when the app restarted. Continue will resume from saved progress."

        return JobInfo(
            job_id=job_dir.name,
            status=status,
            request=BenchmarkRequest(
                datasets=datasets_in_order,
                models=models_in_order,
                workers=workers,
                max_samples=0,
                seed=13,
                output_subdir=output_subdir,
                enable_bert_score=True,
            ),
            started_at=started_at,
            finished_at=finished_at,
            datasets=datasets_payload,
            error=error,
        )

    def _read_event_records(self, job_id: str) -> List[Dict[str, Any]]:
        event_records = self._read_jsonl(self._events_path(job_id))
        if not event_records:
            return []
        if any(record.get("ts") for record in event_records):
            return event_records
        audit_records = self._read_jsonl(self._audit_log_path(job_id))
        for idx, record in enumerate(event_records):
            if record.get("ts"):
                continue
            if idx < len(audit_records):
                record["ts"] = audit_records[idx].get("ts")
        return event_records

    def _last_recorded_ts(self, job_id: str) -> Optional[str]:
        audit_records = self._read_jsonl(self._audit_log_path(job_id))
        if audit_records:
            return audit_records[-1].get("ts")
        event_records = self._read_jsonl(self._events_path(job_id))
        if event_records:
            return event_records[-1].get("ts")
        return None

    def _job_sort_key(self, job: JobInfo) -> tuple[str, str]:
        return (job.started_at or job.finished_at or "", job.job_id)

    def _normalize_model_id(self, raw_model: Any) -> str:
        if not raw_model:
            return ""
        text = str(raw_model).strip()
        if not text:
            return ""
        if text in MODEL_BY_DISPLAY_NAME:
            return MODEL_BY_DISPLAY_NAME[text].value
        try:
            return TargetModel.parse(text).value
        except ValueError:
            return ""

    def _read_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    records.append(payload)
        return records


def _extract_answer_segment(text: str) -> str:
    cleaned = norm_text(text)
    if not cleaned:
        return ""
    lower = cleaned.lower()
    import re

    match = re.search(r"(?:^|\b)(?:final answer|answer|ans)\s*[:\-]?\s*(.+)$", lower)
    if match:
        segment = match.group(1).strip()
        if segment:
            return segment
    return cleaned.splitlines()[0].strip() if "\n" in cleaned else cleaned


def _extract_letters_strict(segment: str) -> Optional[List[str]]:
    import re

    cleaned = norm_text(segment).lower()
    if not cleaned:
        return None
    cleaned = cleaned.replace("/", ",").replace(";", ",").replace("|", ",")
    cleaned = re.sub(r"\band\b", ",", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    patterns = [
        r"^(?:option\s+)?([abcd](?:\s*,\s*[abcd])*)$",
        r"^([abcd](?:\s*[,&/]\s*[abcd])*)$",
        r"^\(?([abcd])\)?$",
    ]
    for pattern in patterns:
        match = re.match(pattern, cleaned)
        if match:
            return uniq_sorted_letters(re.findall(r"[abcd]", match.group(1)))
    match = re.match(r"^(?:option\s+)?([1-4](?:\s*,\s*[1-4])*)$", cleaned)
    if match:
        return uniq_sorted_letters(["abcd"[int(digit) - 1] for digit in re.findall(r"[1-4]", match.group(1))])
    match = re.match(r"^(?:option\s+)?([1-4])$", cleaned)
    if match:
        return ["abcd"[int(match.group(1)) - 1]]
    return None


def _map_text_to_options(segment: str, option_map: Dict[str, str]) -> Tuple[List[str], str]:
    normalized = normalize_option_text(segment)
    if not normalized:
        return [], "empty"
    exact: List[str] = []
    contained: List[str] = []
    for letter, text in option_map.items():
        option_text = normalize_option_text(text)
        if not option_text:
            continue
        if normalized == option_text:
            exact.append(letter)
        elif option_text in normalized:
            contained.append(letter)
    if exact:
        return uniq_sorted_letters(exact), "option_text_exact"
    if len(contained) == 1:
        return uniq_sorted_letters(contained), "option_text_contained"
    return [], "unparsed"


def parse_medmcqa_prediction(raw_text: str, option_map: Dict[str, str], choice_type: str) -> Tuple[List[str], bool, str]:
    segment = _extract_answer_segment(raw_text)
    strict = _extract_letters_strict(segment)
    if strict is not None:
        if choice_type == "single":
            return strict[:1], len(strict) == 1, "strict_letters"
        return uniq_sorted_letters(strict), len(strict) >= 1, "strict_letters"
    mapped, source = _map_text_to_options(segment, option_map)
    if mapped:
        if choice_type == "single":
            return mapped[:1], len(mapped) == 1, source
        return uniq_sorted_letters(mapped), len(mapped) >= 1, source
    return [], False, source


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
