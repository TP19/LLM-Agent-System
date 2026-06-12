"""
Retry utilities with exponential backoff for LLM-Agent-System.

Integrates dev-utils retry decorators for resilient LLM calls and network operations.

Usage:
    from core.retry import retry, retry_on_exception, exponential_backoff

    @retry(max_attempts=3, exceptions=(requests.exceptions.RequestException,))
    def call_llm_api():
        ...

    @retry_on_exception(ConnectionError, max_attempts=5, delay=2.0)
    def ssh_command():
        ...
"""

import functools
import time
import logging
from typing import Callable, Optional, Tuple, Type

logger = logging.getLogger(__name__)


def exponential_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
    """
    Calculate delay using exponential backoff strategy.

    Args:
        attempt: Current attempt number (0-indexed)
        base_delay: Base delay in seconds (default: 1.0)
        max_delay: Maximum delay in seconds (default: 60.0)

    Returns:
        Delay in seconds for this attempt

    Examples:
        >>> exponential_backoff(0)  # First retry
        1.0
        >>> exponential_backoff(1)  # Second retry
        2.0
        >>> exponential_backoff(2)  # Third retry
        4.0
    """
    if attempt is None:
        raise TypeError("attempt cannot be None")

    if attempt < 0:
        raise ValueError("attempt must be non-negative")

    if base_delay <= 0:
        raise ValueError("base_delay must be positive")

    if max_delay <= 0:
        raise ValueError("max_delay must be positive")

    if base_delay > max_delay:
        raise ValueError("base_delay cannot be greater than max_delay")

    # Calculate exponential delay: base_delay * (2 ^ attempt)
    delay = base_delay * (2 ** attempt)

    # Cap at max_delay
    return min(delay, max_delay)


def linear_backoff(attempt: int, increment: float = 1.0, max_delay: float = 60.0) -> float:
    """
    Calculate delay using linear backoff strategy.

    Args:
        attempt: Current attempt number (0-indexed)
        increment: Delay increment per attempt in seconds (default: 1.0)
        max_delay: Maximum delay in seconds (default: 60.0)

    Returns:
        Delay in seconds for this attempt
    """
    if attempt is None:
        raise TypeError("attempt cannot be None")

    if attempt < 0:
        raise ValueError("attempt must be non-negative")

    if increment <= 0:
        raise ValueError("increment must be positive")

    if max_delay <= 0:
        raise ValueError("max_delay must be positive")

    # Calculate linear delay: increment * (attempt + 1)
    delay = increment * (attempt + 1)

    # Cap at max_delay
    return min(delay, max_delay)


def retry(
    max_attempts: int = 3,
    backoff_strategy: str = "exponential",
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None,
) -> Callable:
    """
    Decorator to retry a function with configurable backoff strategy.

    Args:
        max_attempts: Maximum number of attempts (default: 3)
        backoff_strategy: "exponential" or "linear" (default: "exponential")
        base_delay: Base delay for backoff in seconds (default: 1.0)
        max_delay: Maximum delay in seconds (default: 60.0)
        exceptions: Tuple of exception types to catch (default: (Exception,))
        on_retry: Optional callback(attempt, exception) called before each retry

    Returns:
        Decorated function with retry logic

    Examples:
        @retry(max_attempts=3)
        def flaky_function():
            # Function that might fail
            pass

        @retry(max_attempts=5, exceptions=(ConnectionError, TimeoutError))
        def network_call():
            pass
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    if backoff_strategy not in ("exponential", "linear"):
        raise ValueError("backoff_strategy must be 'exponential' or 'linear'")

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    # If this was the last attempt, raise the exception
                    if attempt == max_attempts - 1:
                        logger.warning(
                            f"[retry] {func.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        raise

                    # Calculate delay for next attempt
                    if backoff_strategy == "exponential":
                        delay = exponential_backoff(attempt, base_delay, max_delay)
                    else:  # linear
                        delay = linear_backoff(attempt, base_delay, max_delay)

                    logger.info(
                        f"[retry] {func.__name__} attempt {attempt + 1}/{max_attempts} "
                        f"failed: {e}. Retrying in {delay:.1f}s..."
                    )

                    # Call retry callback if provided
                    if on_retry:
                        on_retry(attempt, e)

                    # Wait before retrying
                    time.sleep(delay)

            # This should never be reached, but just in case
            if last_exception:
                raise last_exception

        return wrapper

    return decorator


def retry_on_exception(
    exception_type: Type[Exception],
    max_attempts: int = 3,
    delay: float = 1.0,
) -> Callable:
    """
    Decorator to retry a function only on specific exception types.

    Args:
        exception_type: Exception type to catch and retry on
        max_attempts: Maximum number of attempts (default: 3)
        delay: Fixed delay between retries in seconds (default: 1.0)

    Returns:
        Decorated function with retry logic

    Examples:
        @retry_on_exception(ValueError, max_attempts=3)
        def parse_value(text):
            return int(text)
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    if delay < 0:
        raise ValueError("delay must be non-negative")

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exception_type as e:
                    last_exception = e

                    # If this was the last attempt, raise the exception
                    if attempt == max_attempts - 1:
                        logger.warning(
                            f"[retry] {func.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        raise

                    logger.info(
                        f"[retry] {func.__name__} attempt {attempt + 1}/{max_attempts} "
                        f"failed with {type(e).__name__}. Retrying in {delay:.1f}s..."
                    )

                    # Wait before retrying (except after last attempt)
                    if delay > 0:
                        time.sleep(delay)

            # This should never be reached, but just in case
            if last_exception:
                raise last_exception

        return wrapper

    return decorator


# Convenience decorators for common use cases

def retry_llm_call(max_attempts: int = 3, base_delay: float = 2.0):
    """
    Decorator specifically for LLM API calls.
    Uses exponential backoff with longer delays suitable for rate limiting.
    """
    import requests
    return retry(
        max_attempts=max_attempts,
        backoff_strategy="exponential",
        base_delay=base_delay,
        max_delay=30.0,
        exceptions=(
            requests.exceptions.RequestException,
            requests.exceptions.Timeout,
            ConnectionError,
            TimeoutError,
        )
    )


def retry_ssh_command(max_attempts: int = 2, delay: float = 1.0):
    """
    Decorator for SSH command execution.
    Uses shorter delays since SSH failures are often network-related.
    """
    import subprocess
    return retry(
        max_attempts=max_attempts,
        backoff_strategy="linear",
        base_delay=delay,
        max_delay=10.0,
        exceptions=(
            subprocess.SubprocessError,
            ConnectionError,
            OSError,
        )
    )
