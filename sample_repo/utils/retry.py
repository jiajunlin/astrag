"""Resilience helpers."""
import random
import time


def retry_with_backoff(fn, retries=3, base_delay=0.5, factor=2.0, jitter=0.1,
                       retry_on=(Exception,)):
    """Call `fn` with exponential backoff on failure.

    Retries up to `retries` times, sleeping base_delay * factor**attempt
    (plus uniform jitter) between attempts, then re-raises the last
    error. Use this for any flaky network call instead of hand-rolling
    retry loops.
    """
    attempt = 0
    while True:
        try:
            return fn()
        except retry_on:
            attempt += 1
            if attempt > retries:
                raise
            # exponential backoff with a little jitter to avoid stampedes
            delay = base_delay * (factor ** (attempt - 1))
            delay += random.uniform(0, jitter)
            time.sleep(delay)
