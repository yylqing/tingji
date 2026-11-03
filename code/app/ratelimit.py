"""极简限流：按客户端 IP 的令牌桶（用于对外 OpenAI 兼容接口）。

默认关闭（`openai_api.rate_limit_per_min = 0`）；本地单用户没必要限。
"""
from __future__ import annotations

import threading
import time


class TokenBucket:
    def __init__(self, per_min: int):
        self.per_min = max(0, per_min)
        self.capacity = max(1, self.per_min)
        self.tokens = float(self.capacity)
        self.ts = time.monotonic()
        self.lock = threading.Lock()

    def take(self) -> bool:
        if self.per_min <= 0:
            return True
        with self.lock:
            now = time.monotonic()
            rate = self.per_min / 60.0
            self.tokens = min(self.capacity, self.tokens + (now - self.ts) * rate)
            self.ts = now
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False


class RateLimiter:
    def __init__(self, per_min: int):
        self.per_min = per_min
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        if self.per_min <= 0:
            return True
        with self._lock:
            b = self._buckets.get(key)
            if b is None:
                b = self._buckets[key] = TokenBucket(self.per_min)
        return b.take()
