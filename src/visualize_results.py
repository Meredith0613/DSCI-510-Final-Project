"""
visualize_results.py

Visualization script for the job postings dataset.

Reads the cleaned jobs from:
    data/processed/jobs_clean.csv

and produces several figures in:
    results/figures/

Currently generates:
    - top_skills_bar.png:       Bar chart of the most common listed skills.
    - skills_wordcloud.png:     Word cloud of skills (if wordcloud is installed).
    - skill_cooccurrence_network.png: Network of skill co-occurrence
                                      (if networkx is installed).
    - salary_vs_skill_count.png:Scatter plot of approximate salary vs number
                                of listed skills.
    - location_distribution.png:Bar chart of top locations by posting count.
"""

import logging
import re
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

try:
    from wordcloud import WordCloud
except ImportError:
    WordCloud = None

try:
    import networkx as nx
except ImportError:
    nx = None

# ---------------------------------------------------------------------
# Paths & logging
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

CLEANED_PATH = DATA_DIR / "jobs_clean.csv"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def load_cleaned_data(path: Path) -> pd.DataFrame:
    """
    Load the cleaned job postings dataset.

    Args:
        path: Path to the cleaned CSV file (typically jobs_clean.csv).

    Returns:
        A pandas DataFrame containing the cleaned postings.

    Raises:
        FileNotFoundError: If the cleaned file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Cleaned data not found at: {path}")
    df = pd.read_csv(path)
    logger.info(
        "Loaded cleaned data with %d rows from: %s",
        len(df),
        path,
    )
    return df


def _parse_skills_cell(cell: str | float | None) -> List[str]:
    """
    Parse a single 'skills' cell into a list of skill strings.

    Expected formats:
        - pipe-separated:  "Python|SQL|Pandas"
        - comma-separated: "Python, SQL, Pandas"

    Non-string values yield an empty list.

    Args:
        cell: Raw cell value from the 'skills' column.

    Returns:
        A list of individual skill strings (stripped of whitespace).
    """
    if not isinstance(cell, str):
        return []

    if "|" in cell:
        parts = cell.split("|")
    else:
        parts = cell.split(",")

    skills = [p.strip() for p in parts if p.strip()]
    return skills


# ---------------------------------------------------------------------
# Top skills bar chart
# ---------------------------------------------------------------------


def plot_top_skills_bar(df: pd.DataFrame, output_path: Path, top_n: int = 20) -> None:
    """
    Plot a horizontal bar chart of the top N most common skills.

    Args:
        df: DataFrame containing at least a 'skills' column.
        output_path: File path where the PNG will be saved.
        top_n: Number of skills to display.
    """
    if "skills" not in df.columns:
        logger.warning("No 'skills' column found; skipping top skills bar chart.")
        return

    all_skills: List[str] = []
    for s in df["skills"].fillna(""):
        all_skills.extend(_parse_skills_cell(s))

    if not all_skills:
        logger.warning("No skills extracted; skipping top skills bar chart.")
        return

    counter = Counter(all_skills)
    top = counter.most_common(top_n)
    skills, counts = zip(*top)

    plt.figure(figsize=(10, 6))
    plt.barh(skills[::-1], counts[::-1])
    plt.xlabel("Count")
    plt.ylabel("Skill")
    plt.title(f"Top {top_n} Skills in Job Postings")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Saved top skills bar chart to: %s", output_path)


# ---------------------------------------------------------------------
# Skills word cloud
# ---------------------------------------------------------------------


def plot_skills_wordcloud(df: pd.DataFrame, output_path: Path) -> None:
    """
    Plot a word cloud of skills if the wordcloud package is available.

    Args:
        df: DataFrame containing a 'skills' column.
        output_path: File path where the PNG will be saved.
    """
    if WordCloud is None:
        logger.warning(
            "wordcloud package not installed; skipping skills word cloud."
        )
        return

    if "skills" not in df.columns:
        logger.warning("No 'skills' column found; skipping skills word cloud.")
        return

    all_skills: List[str] = []
    for s in df["skills"].fillna(""):
        all_skills.extend(_parse_skills_cell(s))

    if not all_skills:
        logger.warning("No skills extracted; skipping skills word cloud.")
        return

    text = " ".join(all_skills)
    wc = WordCloud(width=1000, height=600, background_color="white")
    wc.generate(text)

    plt.figure(figsize=(10, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Saved skills word cloud to: %s", output_path)


# ---------------------------------------------------------------------
# Skill co-occurrence network
# ---------------------------------------------------------------------


def plot_skill_cooccurrence(
    df: pd.DataFrame,
    output_path: Path,
    min_cooccurrence: int = 1,
) -> None:
    """
    Plot a skill co-occurrence network using NetworkX, if available.

    Each node is a skill. An undirected edge connects two skills if they
    appear together in at least `min_cooccurrence` postings. Edge width
    is scaled by co-occurrence frequency.

    Args:
        df: DataFrame containing a 'skills' column.
        output_path: File path where the PNG will be saved.
        min_cooccurrence: Minimum co-occurrence count needed to draw an edge.
    """
    if nx is None:
        logger.warning(
            "networkx package not installed; skipping skill co-occurrence network."
        )
        return

    if "skills" not in df.columns:
        logger.warning("No 'skills' column found; skipping skill co-occurrence network.")
        return

    pair_counts: Counter[tuple[str, str]] = Counter()

    for skills_cell in df["skills"].fillna(""):
        skills = sorted(set(_parse_skills_cell(skills_cell)))
        if len(skills) < 2:
            continue
        for a, b in combinations(skills, 2):
            pair_counts[(a, b)] += 1

    # Filter edges by min_cooccurrence
    edges = [
        (a, b, w)
        for (a, b), w in pair_counts.items()
        if w >= min_cooccurrence
    ]

    if not edges:
        logger.warning(
            "No skill pairs with co-occurrence >= %d.",
            min_cooccurrence,
        )
        return

    G = nx.Graph()
    for a, b, w in edges:
        G.add_edge(a, b, weight=w)

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=0.5, seed=42)

    weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(weights)
    min_w = min(weights)

    # Normalize edge width between 0.5 and 4.0
    if max_w == min_w:
        widths = [2.0 for _ in weights]
    else:
        widths = [
            0.5 + 3.5 * (w - min_w) / (max_w - min_w) for w in weights
        ]

    nx.draw_networkx_nodes(G, pos, node_size=400)
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title(f"Skill Co-occurrence Network (min co-occurrence = {min_cooccurrence})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Saved skill co-occurrence network to: %s", output_path)


# ---------------------------------------------------------------------
# Salary vs skill-count plot
# ---------------------------------------------------------------------


def _ensure_salary_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to construct a numeric 'salary' column if it doesn't exist.

    Priority:
        1) Use existing 'salary' if present.
        2) Use the average of 'salary_min' and 'salary_max'.
        3) Roughly parse 'salary_raw' as a free-text salary and convert
           to an approximate annual figure.

    Args:
        df: DataFrame possibly containing salary-related columns.

    Returns:
        A DataFrame that has a 'salary' column if it could be derived;
        otherwise the original DataFrame.
    """
    if "salary" in df.columns:
        return df

    # 2) salary_min / salary_max
    if {"salary_min", "salary_max"}.issubset(df.columns):
        tmp = df.copy()
        tmp["salary"] = (tmp["salary_min"] + tmp["salary_max"]) / 2.0
        return tmp

    # 3) very rough parsing from 'salary_raw'
    if "salary_raw" in df.columns:
        tmp = df.copy()

        def parse_salary(text: str) -> float | None:
            if not isinstance(text, str):
                return None

            # Find all integer-ish numbers (e.g. "120,000", "80")
            nums = re.findall(r"\d[\d,]*", text)
            if not nums:
                return None

            values = [float(n.replace(",", "")) for n in nums]

            if len(values) >= 2:
                base = sum(values[:2]) / 2.0
            else:
                base = values[0]

            text_lower = text.lower()

            # Very rough guess for time basis
            if "hour" in text_lower or "hr" in text_lower:
                # approx 40 hours/week * 52 weeks/year
                base = base * 40 * 52
            elif "month" in text_lower or "mo" in text_lower:
                base = base * 12
            # Otherwise assume already annual-ish

            return base

        tmp["salary"] = tmp["salary_raw"].apply(parse_salary)
        tmp = tmp[~tmp["salary"].isna()]
        return tmp

    # If nothing works, just return original df
    return df


def plot_salary_vs_skill_count(df: pd.DataFrame, output_path: Path) -> None:
    """
    Plot a scatter plot of (approximate) salary vs number of listed skills.

    Uses _ensure_salary_column() to derive a numeric 'salary' column
    where possible.

    Args:
        df: DataFrame with 'skills' and some form of salary information.
        output_path: File path where the PNG will be saved.
    """
    df = _ensure_salary_column(df)

    if "salary" not in df.columns:
        logger.warning(
            "No usable salary information found (salary / salary_min+max / salary_raw). "
            "Skipping salary vs skill count plot."
        )
        return

    if "skills" not in df.columns:
        logger.warning("No 'skills' column found; skipping salary vs skill count plot.")
        return

    df_plot = df.copy()
    df_plot["skill_count"] = df_plot["skills"].fillna("").apply(
        lambda s: len([x for x in _parse_skills_cell(s)])
    )

    df_plot = df_plot[
        df_plot["skill_count"].notna() & df_plot["salary"].notna()
    ]

    if df_plot.empty:
        logger.warning(
            "No rows with both salary and skills; skipping salary vs skill count plot."
        )
        return

    plt.figure(figsize=(8, 6))
    plt.scatter(df_plot["skill_count"], df_plot["salary"], alpha=0.4)
    plt.xlabel("Number of listed skills")
    plt.ylabel("Salary (approx. annual)")
    plt.title("Salary vs Skill Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Saved salary vs skill count plot to: %s", output_path)


# ---------------------------------------------------------------------
# Location distribution plot
# ---------------------------------------------------------------------


def plot_location_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """
    Plot top locations by posting count.

    Location preference order:
        1) 'country'
        2) 'location_normalized'
        3) 'location'

    Args:
        df: DataFrame with at least one of the above location columns.
        output_path: File path where the PNG will be saved.
    """
    loc_col = None
    if "country" in df.columns:
        loc_col = "country"
    elif "location_normalized" in df.columns:
        loc_col = "location_normalized"
    elif "location" in df.columns:
        loc_col = "location"

    if loc_col is None:
        logger.warning(
            "No country, location_normalized, or location column; skipping location chart."
        )
        return

    df_plot = df.copy()
    df_plot = df_plot[df_plot[loc_col].notna()]

    if df_plot.empty:
        logger.warning("Location column is empty; skipping location chart.")
        return

    loc_counts = (
        df_plot[loc_col]
        .astype(str)
        .str.strip()
        .value_counts()
        .head(20)  # top 20 locations
    )

    if loc_counts.empty:
        logger.warning(
            "No non-empty values in '%s'; skipping location chart.",
            loc_col,
        )
        return

    plt.figure(figsize=(10, 6))
    loc_counts.sort_values().plot(kind="barh")
    plt.xlabel("Number of postings")
    plt.ylabel("Location")
    plt.title(f"Top locations by job postings ({loc_col})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Saved location distribution chart to: %s", output_path)


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------


def main() -> None:
    """
    Main entry point for generating all visualizations.

    Steps:
        - Load the cleaned jobs dataset.
        - Generate:
            * top skills bar chart,
            * skills word cloud (if available),
            * skill co-occurrence network (if available),
            * salary vs skill count scatter plot,
            * location distribution bar chart.
        - Save all figures under results/figures/.
    """
    df = load_cleaned_data(CLEANED_PATH)

    # Top skills bar chart
    plot_top_skills_bar(
        df,
        FIGURES_DIR / "top_skills_bar.png",
    )

    # Skills word cloud
    plot_skills_wordcloud(
        df,
        FIGURES_DIR / "skills_wordcloud.png",
    )

    # Skill co-occurrence network
    plot_skill_cooccurrence(
        df,
        FIGURES_DIR / "skill_cooccurrence_network.png",
        min_cooccurrence=1,  # relaxed threshold
    )

    # Salary vs skill count
    plot_salary_vs_skill_count(
        df,
        FIGURES_DIR / "salary_vs_skill_count.png",
    )

    # Location distribution
    plot_location_distribution(
        df,
        FIGURES_DIR / "location_distribution.png",
    )


if __name__ == "__main__":
    main()


