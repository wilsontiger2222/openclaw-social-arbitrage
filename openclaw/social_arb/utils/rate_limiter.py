"""
Async-compatible rate limiter with token bucket algorithm.

Ensures we stay within API rate limits across all collectors.
Includes retry logic with exponential backoff for transient failures.
"""

from __future__ import annotations

import asyncio
import time
import functools
from typing import Any, Callable, TypeVar

from openclaw.social_arb.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class RateLimiter:
    """
    Token-bucket rate limiter.

    Tokens refill at a constant rate. Each API call consumes one token.
    If no tokens are available, the caller awaits until one is refilled.
    """

    def __init__(self, max_calls: int, period_seconds: float):
        """
        Args:
            max_calls: Maximum calls allowed in the period.
            period_seconds: Length of the rate limit window in seconds.
        """
        self.max_calls = max_calls
        self.period = period_seconds
        self.tokens = max_calls
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a token is available, then consume it."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            # Refill tokens based on elapsed time
            refill = elapsed * (self.max_calls / self.period)
            self.tokens = min(self.max_calls, self.tokens + refill)
            self.last_refill = now

            if self.tokens < 1:
                # Calculate wait time until next token
                wait_time = (1 - self.tokens) * (self.period / self.max_calls)
                logger.info(
                    f"Rate limit reached, waiting {wait_time:.1f}s",
                    extra={"data": {"wait_seconds": wait_time}},
                )
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1


async def retry_with_backoff(
    func: Callable[..., Any],
    *args: Any,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    **kwargs: Any,
) -> Any:
    """
    Retry an async function with exponential backoff.

    Catches all exceptions, logs them, and retries up to max_attempts.
    Delay doubles each attempt: 1s → 2s → 4s → ...
    """
    last_exception: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt == max_attempts:
                logger.error(
                    f"All {max_attempts} attempts failed for {func.__name__}",
                    extra={"data": {"error": str(e), "attempt": attempt}},
                )
                raise
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            logger.warning(
                f"Attempt {attempt}/{max_attempts} failed for {func.__name__}, "
                f"retrying in {delay:.1f}s",
                extra={"data": {"error": str(e), "delay": delay}},
            )
            await asyncio.sleep(delay)
    raise last_exception  # type: ignore[misc]


def rate_limited(limiter: RateLimiter):
    """Decorator that applies rate limiting to an async function."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            await limiter.acquire()
            return await func(*args, **kwargs)
        return wrapper  # type: ignore[return-value]
    return decorator
