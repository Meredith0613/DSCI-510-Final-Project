"""
clean_data.py

Load raw job postings collected by get_data.py, perform basic cleaning
and light normalization, and save a standardized CSV to:

    data/processed/jobs_clean.csv

This script is intended to provide a stable, analysis-ready dataset for
run_analysis.py and visualize_results.py by:

- Ensuring core columns (title, company, location, description) exist.
- Dropping rows with missing key fields and removing duplicates.
- Normalizing important text columns.
- Parsing posted dates (if available).
- Extracting simple keyword-based skills from job descriptions.
- Standardizing column names so downstream scripts can rely on them.
"""

import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_PATH = RAW_DIR / "jobs_raw.csv"
CLEAN_PATH = PROCESSED_DIR / "jobs_clean.csv"


# -------------------------------------------------------------------
# Skill dictionary (can be extended later)
# -------------------------------------------------------------------
SKILL_KEYWORDS = [
    # Programming / DS core
    "python", "r", "sql", "java", "scala", "c++", "c#", "matlab", "sas",
    "pyspark", "spark",
    # ML / DS
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "recommendation", "recommender", "time series",
    "statistics", "bayesian", "regression", "classification", "clustering",
    "feature engineering", "a/b testing", "experimentation",
    # Tools / libraries
    "pandas", "numpy", "scikit-learn", "sklearn", "tensorflow", "keras",
    "pytorch", "xgboost", "lightgbm", "sql server", "bigquery",
    "snowflake", "redshift", "databricks",
    # Visualization / BI
    "tableau", "power bi", "looker", "superset", "ggplot", "matplotlib",
    "seaborn",
    # Data engineering / cloud
    "airflow", "dbt", "kafka", "spark", "hadoop", "etl", "elt",
    "data pipeline", "data warehouse", "data lake",
    "aws", "azure", "gcp", "google cloud", "cloud",
    # General DS / analytics
    "excel", "dashboard", "bi tools", "business intelligence",
    "data analysis", "data analytics",
]


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def normalize_text(s: str) -> str:
    """
    Perform basic text normalization for a single string.

    Operations:
        - Return empty string for non-string inputs.
        - Strip leading/trailing whitespace.
        - Collapse internal whitespace to single spaces.
        - Convert to lowercase.

    Args:
        s: The input string to normalize.

    Returns:
        A normalized, lowercase string.
    """
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def extract_skills(text: str) -> List[str]:
    """
    Extract a simple set of skills from free text using keyword matching.

    This is intentionally lightweight and keyword-based:
    it checks whether each entry in SKILL_KEYWORDS appears
    as a substring of the given text (case-insensitive).

    Args:
        text: Free-text job description or similar field.

    Returns:
        A sorted list of unique skill keywords that appear in the text.
    """
    if not isinstance(text, str):
        return []

    text_lower = text.lower()
    found = set()

    for skill in SKILL_KEYWORDS:
        # Simple substring match
        if skill in text_lower:
            found.add(skill)

    # Return a sorted list for consistency
    return sorted(found)


def standardize_columns_for_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names so downstream scripts can rely on a
    consistent schema.

    This function:
        - Ensures a 'title' column exists (from job_title if needed).
        - Ensures a 'role_query' column exists from one of:
            'query_role' (SerpAPI), 'search_query', or 'role'.
        - Ensures a 'description' column exists (from job_description if needed).
        - Unifies skills into a 'skills' column when possible.
        - Attempts to populate 'country' and 'location_normalized'
          from related location columns.
        - Tries to standardize salary-related columns into
          'salary_raw', 'salary_min', and 'salary_max'.

    Args:
        df: DataFrame of (already cleaned) job postings.

    Returns:
        The same DataFrame with additional standardized columns
        added where possible.
    """
    # ---------- ROLE / TITLE ----------
    # Make sure we at least have 'title'
    if "job_title" in df.columns and "title" not in df.columns:
        df["title"] = df["job_title"]

    # Standardize role column into 'role_query'
    # Priority:
    #   1) 'query_role'  (from get_data.py / SerpAPI)
    #   2) 'search_query'
    #   3) 'role'
    if "role_query" not in df.columns:
        if "query_role" in df.columns:
            df["role_query"] = df["query_role"]
        elif "search_query" in df.columns:
            df["role_query"] = df["search_query"]
        elif "role" in df.columns:
            df["role_query"] = df["role"]

    # ---------- DESCRIPTION ----------
    # Make sure 'description' exists
    if "job_description" in df.columns and "description" not in df.columns:
        df["description"] = df["job_description"]

    # ---------- SKILLS ----------
    # We want a single unified 'skills' column (string; comma- or pipe-separated).
    skill_candidates: List[str] = [
        "skills",
        "skill",
        "skills_extracted",
        "required_skills",
        "jd_skills",
        "skills_list",
    ]
    if "skills" not in df.columns:
        for col in skill_candidates:
            if col in df.columns:
                df["skills"] = df[col]
                break

    # ---------- LOCATION ----------
    # Prefer country → location_normalized → location
    if "country" not in df.columns:
        # Sometimes we might have a normalized version under other names
        for col in ["location_country", "job_country"]:
            if col in df.columns:
                df["country"] = df[col]
                break

    if "location_normalized" not in df.columns:
        for col in [
            "normalized_location",
            "location_norm",
            "location_clean",
        ]:
            if col in df.columns:
                df["location_normalized"] = df[col]
                break

    # Base 'location'
    if "location" not in df.columns:
        for col in ["job_location", "city", "city_state", "full_location"]:
            if col in df.columns:
                df["location"] = df[col]
                break

    # ---------- SALARY ----------
    # Try to find or create salary_min / salary_max / salary_raw
    if "salary" in df.columns:
        # If a generic 'salary' exists, treat it as raw string
        df["salary_raw"] = df["salary"]

    # Look for numeric min/max salary columns with different names
    salary_min_candidates = ["salary_min", "min_salary", "salary_lower", "lower_salary"]
    salary_max_candidates = ["salary_max", "max_salary", "salary_upper", "upper_salary"]

    if "salary_min" not in df.columns:
        for col in salary_min_candidates:
            if col in df.columns:
                df["salary_min"] = df[col]
                break

    if "salary_max" not in df.columns:
        for col in salary_max_candidates:
            if col in df.columns:
                df["salary_max"] = df[col]
                break

    return df


# -------------------------------------------------------------------
# Main cleaning logic
# -------------------------------------------------------------------
def clean_jobs(raw_path: Path = RAW_PATH) -> pd.DataFrame:
    """
    Load raw job postings from CSV and perform basic cleaning.

    Steps:
        - Load the raw CSV from raw_path.
        - Ensure core columns exist (title, company, location, description).
        - Drop rows missing title and/or company.
        - Drop duplicate postings based on job_id, title, company, and location.
        - Normalize key text columns (title, company, location, description).
        - Parse posted_at as datetime (UTC) if present.
        - Add a skills_extracted list column using extract_skills().

    Args:
        raw_path: Path to the raw CSV file produced by get_data.py.

    Returns:
        A cleaned pandas DataFrame, ready for column standardization.
    """
    print(f"[INFO] Loading raw data from: {raw_path}")
    df = pd.read_csv(raw_path)

    print(f"[INFO] Loaded {len(df)} rows with {df.shape[1]} columns.")

    # Ensure expected core columns exist (if not, create empty)
    for col in ["title", "company", "location", "description"]:
        if col not in df.columns:
            df[col] = np.nan

    # Drop rows missing core fields
    core_cols = ["title", "company"]
    before = len(df)
    df = df.dropna(subset=core_cols, how="any")
    after = len(df)
    print(f"[INFO] Dropped {before - after} rows with missing core fields {core_cols}.")

    # Drop duplicate postings
    dedup_cols = ["job_id", "title", "company", "location"]
    dedup_cols = [c for c in dedup_cols if c in df.columns]
    if dedup_cols:
        before = len(df)
        df = df.drop_duplicates(subset=dedup_cols)
        after = len(df)
        print(f"[INFO] Dropped {before - after} duplicate rows based on {dedup_cols}.")

    # Normalize key text columns
    text_cols = ["title", "company", "location", "description"]
    existing_text_cols = [c for c in text_cols if c in df.columns]
    for col in existing_text_cols:
        df[col] = df[col].astype(str).map(normalize_text)

    print(f"[INFO] Lightly cleaned text columns: {existing_text_cols}")

    # Parse posted_at if present
    if "posted_at" in df.columns:
        df["posted_at"] = pd.to_datetime(
            df["posted_at"], errors="coerce", utc=True
        )
        print("[INFO] Parsed column 'posted_at' as datetime (UTC, coerce errors).")

    # ----------------------------------------------------------------
    # NEW: skills_extracted column
    # ----------------------------------------------------------------
    if "description" in df.columns:
        print("[INFO] Extracting skills from job descriptions...")
        df["skills_extracted"] = df["description"].apply(extract_skills)
    else:
        print("[WARN] No 'description' column found; creating empty skills_extracted.")
        df["skills_extracted"] = [[] for _ in range(len(df))]

    print(f"[INFO] Finished cleaning. Final row count: {len(df)}")
    return df


def main() -> None:
    """
    Entry point for the cleaning step.

    Behavior:
        - Ensures the processed data directory exists.
        - Runs clean_jobs() on RAW_PATH.
        - Standardizes column names via standardize_columns_for_analysis().
        - Selects a subset of commonly used columns for analysis/visualization.
        - Saves the resulting DataFrame to CLEAN_PATH as a single CSV.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: basic cleaning
    df_clean = clean_jobs(RAW_PATH)

    # Step 2: standardize column names for downstream scripts
    df_clean = standardize_columns_for_analysis(df_clean)

    # Step 3: select columns to keep
    cols_to_keep = [
        # core fields
        "title",
        "company",
        "description",
        "location",

        # analysis helpers
        "role_query",
        "country",
        "location_normalized",

        # skills (either unified or extracted list)
        "skills",
        "skills_extracted",

        # salary
        "salary",
        "salary_raw",
        "salary_min",
        "salary_max",
    ]
    cols_to_keep = [c for c in cols_to_keep if c in df_clean.columns]
    df_clean = df_clean[cols_to_keep]

    # Step 4: save once, in the standard path
    df_clean.to_csv(CLEAN_PATH, index=False)
    print(
        f"[INFO] Saved standardized cleaned data with {len(df_clean)} rows "
        f"to: {CLEAN_PATH}"
    )


if __name__ == "__main__":
    main()





