"""
run_analysis.py

Analysis script for the job postings dataset.

- Loads cleaned jobs from data/processed/jobs_clean.csv.
- Ensures we have a clean text column ('description_clean') to use.
- Computes TF-IDF keywords:
    * by role (prefers 'role_query', falls back to 'role'), or
    * if no role column exists, computes corpus-level TF-IDF.
- Saves outputs to data/analysis/tfidf_by_role.csv (or a global TF-IDF table
  with columns: term, tfidf when no role column is available).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# --------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
ANALYSIS_DIR = DATA_DIR / "analysis"

CLEANED_PATH = PROCESSED_DIR / "jobs_clean.csv"
TFIDF_OUTPUT_PATH = ANALYSIS_DIR / "tfidf_by_role.csv"


# --------------------------------------------------------------------
# Logging helpers
# --------------------------------------------------------------------
def log_info(msg: str) -> None:
    """Print an informational log message."""
    print(f"[INFO] {msg}")


def log_warn(msg: str) -> None:
    """Print a warning log message."""
    print(f"[WARN] {msg}")


def log_error(msg: str) -> None:
    """Print an error log message to stderr."""
    print(f"[ERROR] {msg}", file=sys.stderr)


# --------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------
def load_cleaned_jobs(path: Path = CLEANED_PATH) -> pd.DataFrame:
    """
    Load the cleaned job postings CSV produced by clean_data.py.

    Args:
        path: Path to the cleaned CSV file.

    Returns:
        A pandas DataFrame containing the cleaned job postings.

    Raises:
        FileNotFoundError: If the cleaned CSV file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned jobs file not found at {path}. "
            "Make sure you ran the cleaning step first."
        )

    df = pd.read_csv(path)
    log_info(f"Loaded {len(df)} cleaned rows from: {path}")
    return df


# --------------------------------------------------------------------
# Text preparation
# --------------------------------------------------------------------
def ensure_description_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a 'description_clean' column exists in the DataFrame.

    If 'description_clean' is missing, this function creates it from
    the 'description' column by:
        - filling NaNs with empty strings,
        - casting to string,
        - converting to lowercase,
        - stripping surrounding whitespace.

    Args:
        df: DataFrame containing at least a 'description' column.

    Returns:
        The same DataFrame (or a shallow copy) with a 'description_clean'
        column ensured.

    Raises:
        KeyError: If neither 'description_clean' nor 'description' exists.
    """
    if "description_clean" not in df.columns:
        if "description" not in df.columns:
            raise KeyError(
                "Neither 'description_clean' nor 'description' column exists in the "
                "dataframe. Cannot create clean text for TF-IDF."
            )

        log_warn("'description_clean' missing; creating from 'description'.")
        desc = (
            df["description"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.strip()
        )
        df = df.copy()
        df["description_clean"] = desc

    return df


# --------------------------------------------------------------------
# TF-IDF by role
# --------------------------------------------------------------------
def compute_tfidf_by_role(df: pd.DataFrame, role_col: str = "role_query") -> pd.DataFrame:
    """
    Compute TF-IDF keywords aggregated by role.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of job postings. Needs:
        - a role column (prefer 'role_query', fallback 'role')
        - 'description_clean' (or 'description' to auto-create)

    role_col : str, default "role_query"
        Preferred column to use for grouping roles.

    Returns
    -------
    tfidf_long : pd.DataFrame
        Long-format TF-IDF table with columns:
        - role_col (e.g., 'role_query' or 'role')
        - term
        - tfidf

    Raises
    ------
    KeyError
        If neither 'role_query' nor 'role' is found in the DataFrame.
    ValueError
        If no non-empty descriptions or groups are available.
    """
    # 1. Pick a role column
    if role_col not in df.columns:
        if "role" in df.columns:
            log_warn(
                f"'{role_col}' column missing; using 'role' instead for TF-IDF grouping."
            )
            role_col = "role"
        else:
            raise KeyError(
                "No suitable role column found. Expected 'role_query' or 'role'. "
                "Make sure at least one of these is preserved from the scraping step."
            )

    # 2. Ensure clean text
    df = ensure_description_clean(df)

    # Filter out rows with empty descriptions
    df = df[df["description_clean"].fillna("").str.strip() != ""].copy()
    if df.empty:
        raise ValueError(
            "No non-empty descriptions available after cleaning. "
            "Cannot compute TF-IDF."
        )

    # 3. Aggregate text per role
    grouped = (
        df.groupby(role_col)["description_clean"]
        .apply(lambda x: " ".join(x.astype(str)))
        .reset_index(name="all_text")
    )

    if grouped.empty:
        raise ValueError(
            f"No groups found when grouping by '{role_col}'. "
            "Check that the column exists and has non-null values."
        )

    roles = grouped[role_col].tolist()
    corpus = grouped["all_text"].tolist()

    log_info(
        f"Computing TF-IDF for {len(roles)} role groups using aggregated descriptions."
    )

    # 4. TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        max_features=100,  # keep it modest for readability
        stop_words="english",
        min_df=2,          # ignore very rare terms
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    terms = vectorizer.get_feature_names_out()

    # 5. Convert to long format DataFrame
    tfidf_array = tfidf_matrix.toarray()
    records = []
    for i, role in enumerate(roles):
        row_vals = tfidf_array[i]
        # Only keep non-zero entries
        nz_indices = np.where(row_vals > 0)[0]
        for idx in nz_indices:
            records.append(
                {
                    role_col: role,
                    "term": terms[idx],
                    "tfidf": float(row_vals[idx]),
                }
            )

    tfidf_long = pd.DataFrame.from_records(records)

    # Optional: sort for nicer inspection
    tfidf_long = tfidf_long.sort_values(
        by=[role_col, "tfidf"], ascending=[True, False]
    ).reset_index(drop=True)

    log_info(
        f"Computed TF-IDF table with {len(tfidf_long)} (role, term) pairs "
        f"across {len(terms)} unique terms."
    )

    return tfidf_long


# --------------------------------------------------------------------
# TF-IDF global (no role column available)
# --------------------------------------------------------------------
def compute_tfidf_global(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute corpus-level TF-IDF keywords when no role column exists.

    The function:
        - Ensures 'description_clean' exists.
        - Computes TF-IDF over all descriptions.
        - Aggregates term importance by taking the average TF-IDF
          score across documents.

    Args:
        df: DataFrame of job postings.

    Returns:
        A DataFrame with columns:
            - term
            - tfidf (average TF-IDF across all documents)

    Raises:
        ValueError: If no non-empty descriptions are available.
    """
    df = ensure_description_clean(df)

    texts = (
        df["description_clean"]
        .fillna("")
        .astype(str)
    )
    texts = texts[texts.str.strip() != ""]

    if texts.empty:
        raise ValueError(
            "No non-empty descriptions available after cleaning. "
            "Cannot compute global TF-IDF."
        )

    log_info(f"Computing global TF-IDF over {len(texts)} descriptions (no role column).")

    vectorizer = TfidfVectorizer(
        max_features=200,
        stop_words="english",
        min_df=2,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out()

    # Average TF-IDF score per term across all documents
    avg_scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()

    result = pd.DataFrame(
        {
            "term": terms,
            "tfidf": avg_scores,
        }
    ).sort_values("tfidf", ascending=False).reset_index(drop=True)

    log_info(
        f"Computed global TF-IDF with {len(result)} terms "
        f"from {tfidf_matrix.shape[0]} documents."
    )
    return result


# --------------------------------------------------------------------
# (Optional) Basic stats
# --------------------------------------------------------------------
def compute_basic_stats(df: pd.DataFrame, role_col_preferred: str = "role_query") -> None:
    """
    Print a few basic sanity-check stats to the console.

    This includes:
        - total row count,
        - top role counts (using role_query or role),
        - a very simple salary range check if numeric salary columns exist.

    Args:
        df: DataFrame of cleaned job postings.
        role_col_preferred: Preferred role column name to use when summarizing roles.
    """
    n_rows = len(df)
    log_info(f"Total rows in cleaned dataset: {n_rows}")

    # Count by role (using same fallback logic)
    if role_col_preferred in df.columns:
        role_col = role_col_preferred
    elif "role" in df.columns:
        role_col = "role"
    else:
        role_col = None

    if role_col is not None:
        counts = df[role_col].value_counts(dropna=False)
        log_info(f"Top roles by '{role_col}':")
        for role, cnt in counts.head(10).items():
            print(f"  - {role}: {cnt} postings")
    else:
        log_warn("No 'role_query' or 'role' column found for basic role stats.")

    # Simple salary sanity check if available
    if {"salary_min", "salary_max"}.issubset(df.columns):
        valid = df[["salary_min", "salary_max"]].dropna()
        if not valid.empty:
            avg_min = valid["salary_min"].mean()
            avg_max = valid["salary_max"].mean()
            log_info(
                f"Average salary range (on rows with non-null salaries): "
                f"{avg_min:,.0f} – {avg_max:,.0f}"
            )
        else:
            log_warn("No non-null salary_min/salary_max rows for salary stats.")
    else:
        log_warn("Missing 'salary_min' or 'salary_max'; skipping salary stats.")


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main() -> None:
    """
    Main entry point for the analysis step.

    Behavior:
        - Ensures the analysis directory exists.
        - Loads cleaned job postings.
        - Prints basic sanity stats.
        - Attempts to compute TF-IDF by role and save to TFIDF_OUTPUT_PATH.
        - If no suitable role column exists, falls back to global TF-IDF
          (no role grouping) and still saves to TFIDF_OUTPUT_PATH.
    """
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    df = load_cleaned_jobs()

    # 2. Optional sanity stats
    compute_basic_stats(df)

    # 3. TF-IDF
    try:
        # First, try per-role TF-IDF
        tfidf_result = compute_tfidf_by_role(df)
        tfidf_result.to_csv(TFIDF_OUTPUT_PATH, index=False)
        log_info(f"Saved TF-IDF-by-role table to: {TFIDF_OUTPUT_PATH}")
    except KeyError as e:
        # If the problem is missing role columns, fall back to global TF-IDF
        msg = str(e)
        if "No suitable role column found" in msg:
            log_warn(
                "No 'role_query' or 'role' column found; "
                "falling back to corpus-level TF-IDF (no role grouping)."
            )
            tfidf_global = compute_tfidf_global(df)
            # Save under the same filename so downstream code still finds something
            tfidf_global.to_csv(TFIDF_OUTPUT_PATH, index=False)
            log_info(
                "Saved corpus-level TF-IDF (columns: term, tfidf) "
                f"to: {TFIDF_OUTPUT_PATH}"
            )
        else:
            log_error(f"Failed to compute TF-IDF by role: {e}")
            raise
    except Exception as e:
        log_error(f"Unexpected error during TF-IDF computation: {e}")
        raise


if __name__ == "__main__":
    main()




