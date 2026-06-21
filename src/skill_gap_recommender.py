"""Market-informed skill-gap recommendations for data job candidates."""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "jobs_clean.csv"


def normalize_skill(skill: object) -> str:
    """Normalize a user- or dataset-provided skill for reliable comparison."""
    return " ".join(str(skill).strip().casefold().split()) if skill is not None else ""


def parse_skills(value: object) -> list[str]:
    """Read skills from Python-list, pipe-separated, or comma-separated cells."""
    if isinstance(value, (list, tuple, set)):
        raw_skills = value
    elif not isinstance(value, str) or not value.strip():
        return []
    else:
        try:
            parsed = ast.literal_eval(value)
            raw_skills = parsed if isinstance(parsed, (list, tuple, set)) else []
        except (ValueError, SyntaxError):
            raw_skills = value.replace("|", ",").split(",")
    return sorted({skill for item in raw_skills if (skill := normalize_skill(item))})


def top_skills_for_role(
    target_role: str, top_n: int = 10, data_path: Path = DEFAULT_DATA_PATH
) -> list[str]:
    """Return the most frequent extracted skills for one role in cleaned data."""
    if not isinstance(target_role, str) or not target_role.strip():
        raise ValueError("target_role must be a non-empty role name.")
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")
    if not data_path.exists():
        raise FileNotFoundError(f"Cleaned jobs data was not found at: {data_path}")

    jobs = pd.read_csv(data_path)
    if "role_query" not in jobs or "skills" not in jobs:
        raise ValueError("Cleaned jobs data must include 'role_query' and 'skills' columns.")

    role = normalize_skill(target_role)
    role_jobs = jobs[jobs["role_query"].map(normalize_skill) == role]
    if role_jobs.empty:
        available = sorted(jobs["role_query"].dropna().unique())
        raise ValueError(f"Unknown role '{target_role}'. Available roles: {', '.join(available)}")

    counts = Counter(
        skill for skills in role_jobs["skills"] for skill in parse_skills(skills)
    )
    return [skill for skill, _ in counts.most_common(top_n)]


def recommend_skills(
    target_role: str, candidate_skills: Iterable[str] | None, top_n: int = 10,
    data_path: Path = DEFAULT_DATA_PATH,
) -> dict[str, object]:
    """Compare a candidate's skills with the role's most common job-market skills.

    The readiness score is the percentage of the requested top skills the
    candidate already has. It is a transparent market-alignment indicator, not
    an assessment of hiring suitability.
    """
    if candidate_skills is None:
        candidate_skills = []
    if isinstance(candidate_skills, str):
        candidate_skills = candidate_skills.split(",")

    candidate = {normalize_skill(skill) for skill in candidate_skills}
    candidate.discard("")
    market_skills = top_skills_for_role(target_role, top_n, data_path)
    matched = [skill for skill in market_skills if skill in candidate]
    missing = [skill for skill in market_skills if skill not in candidate]
    score = round(100 * len(matched) / len(market_skills)) if market_skills else 0

    return {
        "target_role": target_role.strip(),
        "market_skills": market_skills,
        "matched_skills": matched,
        "missing_high_priority_skills": missing,
        "recommended_skills": missing,
        "readiness_score": score,
    }
