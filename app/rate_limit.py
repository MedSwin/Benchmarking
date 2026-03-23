from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import List


@dataclass
class AsyncRateLimiter:
    requests_per_minute: int
    timestamps: List[float] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def acquire(self) -> None:
        if self.requests_per_minute <= 0:
            return
        while True:
            async with self.lock:
                now = time.monotonic()
                cutoff = now - 60.0
                self.timestamps = [stamp for stamp in self.timestamps if stamp >= cutoff]
                if len(self.timestamps) < self.requests_per_minute:
                    self.timestamps.append(now)
                    return
                sleep_for = max(0.05, 60.0 - (now - self.timestamps[0]))
            await asyncio.sleep(sleep_for)
