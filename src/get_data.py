"""
get_data.py

Collects job postings from SerpAPI's Google Jobs engine for multiple
data-related roles and regions, and writes the raw results to:

    data/raw/jobs_raw.csv
    data/raw/jobs_raw.json

This script implements a feasible, API-based data collection strategy
in response to project feedback:

- Uses SerpAPI (Google Jobs) instead of scraping LinkedIn/Glassdoor/Indeed.
- Avoids the deprecated `start` parameter and relies on `next_page_token`
  pagination where available.
- Handles API errors gracefully and will NOT overwrite existing raw data
  if no new jobs are collected (e.g., due to quota or client errors).

Run from the project root:

    export SERPAPI_API_KEY="YOUR_KEY"
    python -m src.get_data
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .utils.request_utils import fetch_serpapi_json

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# Soft target for how many total postings we would like across all roles/regions.
TARGET_POSTINGS: int = 1000

# Maximum pages per (role, region) combo. In practice, pagination is constrained
# by SerpAPI's free tier and the availability of `next_page_token`.
MAX_PAGES_PER_COMBO: int = 5

ROLES: List[str] = [
    "Data Scientist",
    "Machine Learning Engineer",
    "Data Analyst",
    "Data Engineer",
]

REGIONS: List[Dict[str, str]] = [
    {"name": "google_jobs_us", "hl": "en", "gl": "us", "location": "United States"},
    {"name": "google_jobs_ca", "hl": "en", "gl": "ca", "location": "Canada"},
    {"name": "google_jobs_uk", "hl": "en", "gl": "gb", "location": "United Kingdom"},
]


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------


def get_serpapi_key() -> str:
    """
    Retrieve the SerpAPI API key from the environment.

    Returns:
        str: The value of the SERPAPI_API_KEY environment variable.

    Raises:
        RuntimeError: If SERPAPI_API_KEY is not set.
    """
    api_key = os.environ.get("SERPAPI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "SERPAPI_API_KEY is not set. Use:\n"
            "  export SERPAPI_API_KEY='YOUR_KEY_HERE'\n"
            "and then re-run this script."
        )
    return api_key


def project_paths() -> Dict[str, Path]:
    """
    Compute and create (if needed) the main project paths for raw data.

    Returns:
        Dict[str, Path]: A dictionary with keys:
            - 'root': project root directory (Path)
            - 'raw_csv': CSV path for raw jobs (Path)
            - 'raw_json': JSON path for raw jobs (Path)
    """
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    return {
        "root": root,
        "raw_csv": raw_dir / "jobs_raw.csv",
        "raw_json": raw_dir / "jobs_raw.json",
    }


def normalize_job_record(
    job: Dict[str, Any],
    role_query: str,
    region_name: str,
    region_location: str,
    page_index: int,
) -> Dict[str, Any]:
    """
    Flatten a SerpAPI Google Jobs job dict into a normalized record.

    Args:
        job: Raw job dictionary from SerpAPI.
        role_query: The role query used, e.g. "Data Scientist".
        region_name: Internal region name, e.g. "google_jobs_us".
        region_location: Human-readable region/location string.
        page_index: 1-based index of the page from which this job came.

    Returns:
        Dict[str, Any]: A normalized dictionary ready for DataFrame construction.
    """
    job_id = job.get("job_id") or job.get("job_id_original") or job.get(
        "job_id_custom"
    )

    title = job.get("title")

    # Company field can be a string OR a dict, depending on SerpAPI format.
    raw_company = job.get("company_name")
    if isinstance(raw_company, dict):
        company = raw_company.get("name")
    elif isinstance(raw_company, str):
        company = raw_company
    else:
        raw_company = job.get("company")
        if isinstance(raw_company, dict):
            company = raw_company.get("name") or raw_company.get("display_name")
        elif isinstance(raw_company, str):
            company = raw_company
        else:
            company = None

    location = job.get("location")
    description = job.get("description")

    # Posted date can appear in different fields depending on engine version.
    posted_at = job.get("detected_extensions", {}).get("posted_at")

    # Salary fields are extremely inconsistent; store raw string if available.
    salary = job.get("detected_extensions", {}).get("salary")

    normalized = {
        "job_id": job_id,
        "title": title,
        "company": company,
        "location": location,
        "description": description,
        "posted_at": posted_at,
        "salary_raw": salary,
        "query_role": role_query,
        "region_engine": region_name,
        "region_location": region_location,
        "page_index": page_index,
        "via_serpapi_raw": json.dumps(job, ensure_ascii=False),
    }
    return normalized


def parse_next_page_token(data: Dict[str, Any]) -> Optional[str]:
    """
    Extract the next_page_token from a SerpAPI response, if present.

    Args:
        data: JSON response from SerpAPI.

    Returns:
        Optional[str]: The next_page_token string if available; otherwise, None.
    """
    pagination = data.get("serpapi_pagination") or {}
    return pagination.get("next_page_token")


def collect_for_role_and_region(
    role_query: str,
    region: Dict[str, str],
    api_key: str,
    max_pages: int,
) -> List[Dict[str, Any]]:
    """
    Collect job postings for a single (role, region) combination using SerpAPI.

    This implementation:
    - Requests the first page without `start`.
    - Uses `next_page_token` for subsequent pages (up to max_pages).
    - Stops early if the API returns errors or no results.
    - Returns a list of normalized job dictionaries.

    Args:
        role_query: The job role to search for (e.g., "Data Scientist").
        region: A dict with keys 'name', 'hl', 'gl', 'location'.
        api_key: SerpAPI API key.
        max_pages: Maximum number of pages to request for this combo.

    Returns:
        List[Dict[str, Any]]: A list of normalized job records.
    """
    records: List[Dict[str, Any]] = []
    engine_name = region["name"]

    print(
        f"[INFO]   Role='{role_query}' | Region='{engine_name}' | "
        f"Planned pages={max_pages}"
    )

    next_page_token: Optional[str] = None
    page_index = 0

    while page_index < max_pages:
        page_index += 1

        params: Dict[str, Any] = {
            "engine": "google_jobs",
            "q": role_query,
            "api_key": api_key,
            "hl": region["hl"],
            "gl": region["gl"],
            "location": region["location"],
        }
        if next_page_token:
            params["next_page_token"] = next_page_token

        print(
            f"[DEBUG]    Fetching page {page_index}/{max_pages} for "
            f"role='{role_query}' | engine='{engine_name}'"
        )

        # fetch_serpapi_json may return:
        #   - just the JSON dict, or
        #   - a tuple like (data, status_code, url)
        # It returns None on failure.
        result = fetch_serpapi_json(params)

        if result is None:
            print(
                f"[ERROR] Request failed for role='{role_query}' | "
                f"region='{engine_name}'. Stopping this combo."
            )
            break

        if isinstance(result, tuple):
            data = result[0]
        else:
            data = result

        jobs = data.get("jobs_results") or []
        if not jobs:
            print(
                f"[INFO]  No 'jobs_results' found for page {page_index}; "
                f"stopping further pagination for this combo."
            )
            break

        for job in jobs:
            rec = normalize_job_record(
                job=job,
                role_query=role_query,
                region_name=engine_name,
                region_location=region["location"],
                page_index=page_index,
            )
            records.append(rec)

        print(
            f"[INFO]    Collected {len(jobs)} jobs on page {page_index} "
            f"for role='{role_query}' | region='{engine_name}'. "
            f"Total so far for this combo: {len(records)}"
        )

        # Try to get next_page_token; if not present, pagination is over.
        next_page_token = parse_next_page_token(data)
        if not next_page_token:
            print(
                f"[INFO]  No next_page_token present after page {page_index}; "
                f"stopping pagination for this combo."
            )
            break

    print(
        f"[INFO] Finished collection for role='{role_query}' | region='{engine_name}'. "
        f"Total records collected: {len(records)}"
    )
    return records


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------


def main() -> None:
    """
    Orchestrate data collection across all roles and regions and write
    raw outputs to data/raw/.

    Behavior:
    - Attempts to collect up to TARGET_POSTINGS across all (role, region) combos.
    - Caps pages per combo to MAX_PAGES_PER_COMBO.
    - If no new jobs are collected (e.g., due to API errors or quota),
      it does NOT overwrite existing raw data files.
    """
    paths = project_paths()
    api_key = get_serpapi_key()

    num_combos = len(ROLES) * len(REGIONS)
    # Approximate pages per combo, still safety-capped.
    approx_pages_per_combo = max(
        1, min(MAX_PAGES_PER_COMBO, TARGET_POSTINGS // max(1, num_combos * 10))
    )

    print("=" * 80)
    print("[get_data] Collecting job postings via SerpAPI Google Jobs.")
    print(
        f"           Roles: {', '.join(ROLES)}\n"
        f"          Regions: {', '.join(r['name'] for r in REGIONS)}\n"
        f"     Target total: ~{TARGET_POSTINGS} postings "
        f"(API limits may reduce this)."
    )
    print("=" * 80)
    print(
        f"[INFO] Will request up to {approx_pages_per_combo} page(s) per "
        f"(role, region) combo (capped at {MAX_PAGES_PER_COMBO})."
    )

    all_records: List[Dict[str, Any]] = []

    for role in ROLES:
        print("-" * 80)
        print(f"[INFO] Starting collection for role='{role}'")

        for region in REGIONS:
            combo_records = collect_for_role_and_region(
                role_query=role,
                region=region,
                api_key=api_key,
                max_pages=approx_pages_per_combo,
            )
            all_records.extend(combo_records)

    total_collected = len(all_records)
    print("-" * 80)
    print(f"[INFO] Total records collected across all roles/regions: {total_collected}")

    if total_collected == 0:
        print(
            "[WARN] No new jobs collected; keeping existing raw data file (if any).\n"
            "       This can happen if the SerpAPI free tier is exhausted or "
            "if the API returns client errors.\n"
            "       Downstream scripts (clean_data, run_analysis, visualize_results) "
            "can still operate on previously saved data."
        )
        return

    # Build DataFrame and write to data/raw
    df = pd.DataFrame(all_records)
    df.to_csv(paths["raw_csv"], index=False)
    with paths["raw_json"].open("w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    print(
        f"[INFO] Wrote raw CSV to: {paths['raw_csv']}\n"
        f"[INFO] Wrote raw JSON to: {paths['raw_json']}"
    )


if __name__ == "__main__":
    main()





