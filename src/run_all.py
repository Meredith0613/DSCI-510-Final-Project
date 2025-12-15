"""
run_all.py

Convenience script to run the full project pipeline end-to-end:

    1. Collect raw job postings via SerpAPI (src/get_data.py).
    2. Clean and normalize the data (src/clean_data.py).
    3. Run TF-IDF-based analysis (src/run_analysis.py).
    4. Generate summary visualizations (src/visualize_results.py).

Usage (from the project root):

    export SERPAPI_API_KEY="YOUR_KEY"
    python -m src.run_all
"""

from __future__ import annotations

from src import get_data, clean_data, run_analysis, visualize_results


def header(title: str) -> None:
    """
    Print a formatted pipeline section header to stdout.

    Args:
        title: Short description of the pipeline step.
    """
    print("=" * 80)
    print(f"[run_all] {title}")
    print("=" * 80)


def main() -> None:
    """
    Run the full project pipeline in sequence.

    Steps:
        1. Data collection
           - Uses SerpAPI's Google Jobs engine via get_data.main().
           - Attempts to collect up to ~TARGET_POSTINGS job postings across
             multiple roles and regions.
           - If no new jobs are collected (e.g., due to API quota/client
             errors), previously saved raw data in data/raw/ is preserved.

        2. Data cleaning
           - Cleans and standardizes the raw CSV via clean_data.main().
           - Writes a single analysis-ready CSV to data/processed/jobs_clean.csv.

        3. Analysis
           - Computes TF-IDF keywords (by role when possible) via
             run_analysis.main().
           - Writes a TF-IDF table to data/analysis/tfidf_by_role.csv.

        4. Visualization
           - Generates summary figures (skills, locations, salary vs skills)
             via visualize_results.main().
           - Saves PNGs under results/figures/.
    """
    # ------------------------------------------------------------------
    # Step 1: Data collection
    # ------------------------------------------------------------------
    header("Step 1/4: Fetching raw job postings...")
    print(
        "[INFO] Targeting up to ~1000 postings across multiple roles and "
        "regions using the SerpAPI Google Jobs engine.\n"
        "       If the API returns 4xx errors or hits quota limits, the "
        "script will NOT overwrite any existing raw CSV/JSON and "
        "downstream steps can still run on previously saved data."
    )
    get_data.main()

    # ------------------------------------------------------------------
    # Step 2: Cleaning
    # ------------------------------------------------------------------
    header("Step 2/4: Cleaning data...")
    clean_data.main()

    # ------------------------------------------------------------------
    # Step 3: Analysis
    # ------------------------------------------------------------------
    header("Step 3/4: Running analysis...")
    run_analysis.main()

    # ------------------------------------------------------------------
    # Step 4: Visualization
    # ------------------------------------------------------------------
    header("Step 4/4: Generating visualizations...")
    visualize_results.main()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    header("Pipeline completed successfully.")
    print("Outputs:")
    print("  • Raw data:       data/raw/jobs_raw.csv / jobs_raw.json")
    print("  • Cleaned data:   data/processed/jobs_clean.csv")
    print("  • Analysis:       data/analysis/tfidf_by_role.csv")
    print("  • Figures:        results/figures/*.png")
    print("=" * 80)


if __name__ == "__main__":
    main()




