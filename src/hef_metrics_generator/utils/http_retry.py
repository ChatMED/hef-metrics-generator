"""
HTTP Retry Utility for hef_metrics_generator.

Provides a generic retry_request() function with exponential backoff
for use across all external API tools (OpenAlex, PubMed, ArXiv, etc.).
"""

import time
import logging
import urllib.error
import urllib.request
from typing import Optional, Dict

default_logger = logging.getLogger("http_retry")


def retry_request(
        url: str,
        headers: Optional[Dict[str, str]] = None,
        retries: int = 2,
        backoff: float = 2.0,
        timeout: int = 15
) -> Optional[str]:
    """
    Perform an HTTP GET request with simple exponential backoff on
    transient errors (429, 5xx). Returns response text or None on failure.

    Args:
        url: Target URL to fetch.
        headers: Optional HTTP headers dict.
        retries: Number of retries before failing.
        backoff: Base seconds for exponential backoff.
        timeout: Socket timeout in seconds.

    Returns:
        Optional[str]: Response body (decoded UTF-8) if successful, else None.
    """
    headers = headers or {}
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                sleep_s = backoff * (2 ** attempt)
                default_logger.warning(
                    f"[retry_request] HTTP {e.code} for {url}; retrying in {sleep_s:.1f}s..."
                )
                time.sleep(sleep_s)
                continue
            default_logger.error(f"[retry_request] HTTP error: {e}")
            return None
        except Exception as e:
            default_logger.error(f"[retry_request] Error fetching {url}: {e}")
            return None
    return None
