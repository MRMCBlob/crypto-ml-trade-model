"""Utility decorators for rate limiting and retries."""
import time
import functools
from collections import deque
from threading import Lock
from typing import Callable, Any

from .logger import get_logger

logger = get_logger(__name__)


def rate_limit(calls: int, period: float):
    """
    Decorator to rate limit function calls.

    Args:
        calls: Maximum number of calls allowed
        period: Time period in seconds
    """
    def decorator(func: Callable) -> Callable:
        timestamps = deque(maxlen=calls)
        lock = Lock()

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            with lock:
                now = time.time()

                # Remove timestamps outside the period
                while timestamps and now - timestamps[0] > period:
                    timestamps.popleft()

                # If at limit, wait
                if len(timestamps) >= calls:
                    sleep_time = period - (now - timestamps[0])
                    if sleep_time > 0:
                        logger.debug(f"Rate limit reached, sleeping {sleep_time:.2f}s")
                        time.sleep(sleep_time)

                timestamps.append(time.time())

            return func(*args, **kwargs)

        return wrapper
    return decorator


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator to retry function on failure with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            current_delay = delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"{func.__name__} failed after {max_retries + 1} attempts: {e}")

            raise last_exception

        return wrapper
    return decorator
