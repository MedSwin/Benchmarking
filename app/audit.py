from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from app.models import EventPayload, LogRecord


class AuditTrail:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._locks: Dict[str, asyncio.Lock] = {}
        self.logger = logging.getLogger("benchmark.audit")

    def _lock(self, job_id: str) -> asyncio.Lock:
        if job_id not in self._locks:
            self._locks[job_id] = asyncio.Lock()
        return self._locks[job_id]

    async def append_log(
        self,
        job_id: str,
        message: str,
        *,
        level: str = "INFO",
        dataset: Optional[str] = None,
        model: Optional[str] = None,
        data: Optional[dict] = None,
    ) -> LogRecord:
        payload = LogRecord(
            ts=datetime.now(timezone.utc).isoformat(),
            level=level,
            job_id=job_id,
            dataset=dataset,
            model=model,
            message=message,
            data=data or {},
        )
        line = payload.model_dump_json()
        log_path = self.root / job_id / "audit.log.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        async with self._lock(job_id):
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        getattr(self.logger, level.lower(), self.logger.info)(
            "%s job=%s dataset=%s model=%s %s",
            payload.ts,
            job_id,
            dataset,
            model,
            message,
        )
        return payload

    async def append_event(self, job_id: str, event: EventPayload) -> None:
        path = self.root / job_id / "events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        async with self._lock(job_id):
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event.model_dump(), ensure_ascii=False) + "\n")
