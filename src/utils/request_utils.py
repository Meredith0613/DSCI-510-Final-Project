from __future__ import annotations

import time
from typing import Any, Dict, Optional

import requests


def fetch_serpapi_json(
    params: Dict[str, Any],
    max_retries: int = 3,
    backoff_factor: float = 1.5,
    timeout: int = 30,
) -> Optional[Dict[str, Any]]:
    """
    Call the SerpAPI /search endpoint with simple retry + backoff.

    Parameters
    ----------
    params : Dict[str, Any]
        Dictionary of querystring parameters. Must include:
        - engine
        - q
        - api_key
        - hl, gl, location, start, etc. as needed.
    max_retries : int, default=3
        Maximum number of attempts before giving up.
    backoff_factor : float, default=1.5
        Exponential backoff factor: sleep = backoff_factor ** (attempt - 1)
    timeout : int, default=30
        Per-request timeout in seconds.

    Returns
    -------
    Optional[Dict[str, Any]]
        Parsed JSON response if successful, otherwise None.
    """
    base_url = "https://serpapi.com/search"
    session = requests.Session()

    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(base_url, params=params, timeout=timeout)
            status = resp.status_code
            print(f"[DEBUG] Request URL: {resp.url} | Status: {status}")

            # Success: any 2xx
            if 200 <= status < 300:
                return resp.json()

            # Non-retryable 4xx (except 429)
            if 400 <= status < 500 and status not in {429}:
                print(f"[ERROR] Non-retryable client error {status}: {resp.text[:300]}")
                return None

            # Retryable: 5xx or 429
            if attempt < max_retries:
                sleep_sec = backoff_factor ** (attempt - 1)
                print(
                    f"[WARN] Status {status}. Retrying in {sleep_sec:.1f}s "
                    f"(attempt {attempt}/{max_retries})..."
                )
                time.sleep(sleep_sec)
            else:
                print(
                    f"[ERROR] Giving up after {max_retries} attempts. "
                    f"Last status: {status}"
                )
                return None

        except requests.RequestException as e:
            if attempt < max_retries:
                sleep_sec = backoff_factor ** (attempt - 1)
                print(
                    f"[WARN] RequestException on attempt {attempt}/{max_retries}: {e}. "
                    f"Retrying in {sleep_sec:.1f}s..."
                )
                time.sleep(sleep_sec)
            else:
                print(f"[ERROR] Request failed after {max_retries} attempts: {e}")
                return None

    return None



