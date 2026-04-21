from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AsyncRateLimiter:
    requests_per_minute: int
    backoff_factor: float
    recovery_seconds: float
    timestamps: List[float] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _base_requests_per_minute: int = field(init=False)
    _current_requests_per_minute: int = field(init=False)
    _last_backoff: Optional[float] = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._base_requests_per_minute = self.requests_per_minute
        self._current_requests_per_minute = max(1, self.requests_per_minute)
        self.backoff_factor = max(0.1, min(1.0, self.backoff_factor))

    async def acquire(self) -> None:
        if self._base_requests_per_minute <= 0:
            return
        while True:
            async with self.lock:
                now = time.monotonic()
                self._maybe_restore(now)
                limit = self._current_requests_per_minute
                cutoff = now - 60.0
                self.timestamps = [stamp for stamp in self.timestamps if stamp >= cutoff]
                if len(self.timestamps) < limit:
                    self.timestamps.append(now)
                    return
                sleep_for = max(0.05, 60.0 - (now - self.timestamps[0]))
            await asyncio.sleep(sleep_for)

    # Motivation vs Logic: reduce throughput after repeated 429s so the same key slows down.
    def backoff(self) -> None:
        if self._base_requests_per_minute <= 0:
            return
        now = time.monotonic()
        next_limit = max(1, int(self._current_requests_per_minute * self.backoff_factor))
        if next_limit < self._current_requests_per_minute:
            self._current_requests_per_minute = next_limit
            self._last_backoff = now

    def _maybe_restore(self, now: float) -> None:
        if (
            self._last_backoff is None
            or self.recovery_seconds <= 0
            or self._current_requests_per_minute == self._base_requests_per_minute
        ):
            return
        if now - self._last_backoff >= self.recovery_seconds:
            self._current_requests_per_minute = self._base_requests_per_minute
            self._last_backoff = None
