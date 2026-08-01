"""Tiny HTTP helpers used across services."""
import json
import urllib.request

from sample_repo.utils.retry import retry_with_backoff


class HttpError(Exception):
    """Raised when a request fails with a 4xx/5xx status."""


def fetch_json(url, timeout=10.0, retries=3):
    """GET `url` and decode the JSON body, retrying with backoff."""
    def _do():
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status >= 400:
                raise HttpError(f"{resp.status} for {url}")
            return json.loads(resp.read().decode("utf-8"))
    return retry_with_backoff(_do, retries=retries)


class HttpClient:
    """Very small JSON HTTP client bound to a base URL."""

    def __init__(self, base_url, timeout=10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get(self, path):
        """GET a path relative to the base URL and return parsed JSON."""
        return fetch_json(f"{self.base_url}/{path.lstrip('/')}",
                          timeout=self.timeout)
