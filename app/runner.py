from __future__ import annotations

import asyncio
import csv
import json
import random
import uuid
from collections import defaultdict
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

try:
    from bert_score import score as bert_score
except Exception:  # pragma: no cover - runtime optional dependency behavior
    bert_score = None


class BenchmarkManager:
    def __init__(self) -> None:
        self.jobs: Dict[str, JobInfo] = {}
        self.event_queues: Dict[str, asyncio.Queue[EventPayload]] = {}
        self.audit = AuditTrail(settings.output_root)
        from app.providers import ProviderPool

        self.provider_pool = ProviderPool()
        self.tasks: Dict[str, asyncio.Task[Any]] = {}

    async def shutdown(self) -> None:
        for task in self.tasks.values():
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
        self.tasks[job_id] = asyncio.create_task(self._run_job(job_id))
        await self._emit(job_id, "job_created", f"Queued benchmark job for {len(request.datasets)} datasets.")
        return job

    def get_job(self, job_id: str) -> JobInfo:
        if job_id not in self.jobs:
            raise KeyError(job_id)
        return self.jobs[job_id]

    async def cancel_job(self, job_id: str) -> JobInfo:
        task = self.tasks.get(job_id)
        if task:
            task.cancel()
        job = self.get_job(job_id)
        job.status = JobStatus.cancelled
        job.finished_at = datetime.now(timezone.utc).isoformat()
        await self._emit(job_id, "job_cancelled", "Job cancellation requested.")
        return job

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
        payload = EventPayload(
            event=event,
            job_id=job_id,
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

    async def _run_job(self, job_id: str) -> None:
        job = self.get_job(job_id)
        job.status = JobStatus.running
        job.started_at = datetime.now(timezone.utc).isoformat()
        request = job.request
        try:
            for dataset in request.datasets:
                await self._run_dataset(job_id, dataset, request)
            job.status = JobStatus.completed
            job.finished_at = datetime.now(timezone.utc).isoformat()
            await self._emit(job_id, "job_completed", "Benchmark run completed successfully.")
        except asyncio.CancelledError:
            job.status = JobStatus.cancelled
            job.finished_at = datetime.now(timezone.utc).isoformat()
            await self._emit(job_id, "job_cancelled", "Benchmark run cancelled.", level="WARNING")
            raise
        except Exception as exc:
            job.status = JobStatus.failed
            job.error = str(exc)
            job.finished_at = datetime.now(timezone.utc).isoformat()
            await self._emit(job_id, "job_failed", str(exc), level="ERROR")

    async def _run_dataset(self, job_id: str, dataset: DatasetName, request: BenchmarkRequest) -> None:
        dataset_rows = load_dataset_rows(settings.data_root, dataset, request.max_samples, request.seed)
        await self._emit(job_id, "dataset_loaded", f"Loaded {len(dataset_rows)} rows.", dataset=dataset.value)
        dataset_dir = self._dataset_dir(job_id, dataset, request.output_subdir)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        summary: Dict[str, Any] = {"rows": len(dataset_rows), "models": {}}
        workers = max(1, request.workers or settings.default_workers)
        for model in request.models:
            model_rows, model_summary = await self._evaluate_model(job_id, dataset, model, dataset_rows, dataset_dir, workers, request.enable_bert_score)
            summary["models"][str(model)] = model_summary.model_dump()
            summary_csv = dataset_dir / f"{str(model)}.csv"
            write_csv(summary_csv, model_rows)
            model_summary.artifacts["detail_csv"] = str(summary_csv)
            await self._emit(
                job_id,
                "model_completed",
                f"Finished {model} on {dataset.value}.",
                dataset=dataset.value,
                model=str(model),
                data=model_summary.model_dump(),
            )
        summary_path = dataset_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        job = self.get_job(job_id)
        job.datasets[dataset.value] = summary

    async def _evaluate_model(
        self,
        job_id: str,
        dataset: DatasetName,
        model: TargetModel,
        rows: List[Any],
        dataset_dir: Path,
        workers: int,
        enable_bert_score: bool,
    ) -> Tuple[List[Dict[str, Any]], JobSummary]:
        model_name = str(model)
        queue: asyncio.Queue[Tuple[int, Any]] = asyncio.Queue()
        for idx, row in enumerate(rows):
            queue.put_nowait((idx, row))
        results: List[Optional[Dict[str, Any]]] = [None] * len(rows)
        max_tokens = {
            DatasetName.medmcqa: settings.medmcqa_max_new_tokens,
            DatasetName.medquad: settings.medquad_max_new_tokens,
            DatasetName.healthbench: settings.healthbench_max_new_tokens,
        }[dataset]

        async def worker(worker_id: int) -> None:
            while True:
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
                    await self._emit(
                        job_id,
                        "row_scored",
                        f"worker={worker_id} scored row {row.id}",
                        dataset=dataset.value,
                        model=model_name,
                        data={"row_id": row.id, **metrics},
                    )
                except Exception as exc:
                    await self._emit(
                        job_id,
                        "row_failed",
                        f"worker={worker_id} failed row {row.id}: {exc}",
                        dataset=dataset.value,
                        model=model_name,
                        data={"row_id": row.id},
                        level="ERROR",
                    )
                    raise
                finally:
                    queue.task_done()

        await self._emit(
            job_id,
            "model_started",
            f"Starting {model_name} on {dataset.value} with {workers} workers via {MODEL_PROVIDER[model]}",
            dataset=dataset.value,
            model=model_name,
        )
        await asyncio.gather(*(worker(worker_id) for worker_id in range(1, workers + 1)))

        final_rows = [item for item in results if item is not None]
        if enable_bert_score and bert_score and final_rows and dataset != DatasetName.medmcqa:
            preds = [item["prediction"] for item in final_rows]
            refs = [item["reference"] for item in final_rows]
            _, _, f1 = bert_score(
                preds,
                refs,
                lang="en",
                rescale_with_baseline=False,
                model_type=settings.bert_score_model_type,
            )
            for row_dict, score in zip(final_rows, f1.tolist()):
                row_dict["bert_f"] = float(score)
        else:
            for row_dict in final_rows:
                row_dict.setdefault("bert_f", 0.0)

        metric_keys = ["rougeL_f", "tok_f1", "uni_prec", "bi_prec", "bert_f"]
        means = {key: mean_metric(final_rows, key) for key in metric_keys}
        audit_path = dataset_dir / f"{model_name}.audit.jsonl"
        with audit_path.open("w", encoding="utf-8") as handle:
            for row in final_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary = JobSummary(metric_means=means, rows=len(final_rows), artifacts={"audit_jsonl": str(audit_path)})
        metrics_path = dataset_dir / f"{model_name}.metrics.json"
        metrics_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        summary.artifacts["metrics_json"] = str(metrics_path)
        return final_rows, summary

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

    def _dataset_dir(self, job_id: str, dataset: DatasetName, output_subdir: Optional[str]) -> Path:
        base = settings.output_root / job_id
        if output_subdir:
            base = base / output_subdir
        return base / dataset.value


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
