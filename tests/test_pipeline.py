from pathlib import Path

import pandas as pd

from src.clean_data import clean_jobs, extract_skills
from src.run_analysis import compute_tfidf_by_role
from src.skill_gap_recommender import recommend_skills


def test_skill_extraction_matches_complete_terms():
    skills = extract_skills("Required: Python, SQL, and machine learning experience.")
    assert {"python", "sql", "machine learning"}.issubset(skills)
    assert "r" not in extract_skills("Required Python experience")


def test_missing_salary_does_not_break_cleaning(tmp_path: Path):
    raw = tmp_path / "jobs.csv"
    pd.DataFrame([{"title": "Analyst", "company": "Acme", "location": "LA", "description": "Python and SQL"}]).to_csv(raw, index=False)
    cleaned = clean_jobs(raw)
    assert len(cleaned) == 1
    assert "python" in cleaned.loc[0, "skills_extracted"]


def test_duplicate_jobs_are_removed(tmp_path: Path):
    raw = tmp_path / "jobs.csv"
    row = {"job_id": "1", "title": "Analyst", "company": "Acme", "location": "LA", "description": "SQL"}
    pd.DataFrame([row, row]).to_csv(raw, index=False)
    assert len(clean_jobs(raw)) == 1


def test_cleaning_output_has_expected_columns(tmp_path: Path):
    raw = tmp_path / "jobs.csv"
    pd.DataFrame([{"title": "Analyst", "company": "Acme", "location": "LA", "description": "Python"}]).to_csv(raw, index=False)
    cleaned = clean_jobs(raw)
    assert {"title", "company", "location", "description", "skills_extracted"}.issubset(cleaned.columns)


def test_tfidf_output_is_generated():
    jobs = pd.DataFrame({
        "role_query": ["Data Analyst", "Data Analyst", "Data Engineer", "Data Engineer"],
        "description": ["sql dashboard reporting", "sql excel reporting", "python spark pipeline", "python cloud pipeline"],
    })
    result = compute_tfidf_by_role(jobs)
    assert {"role_query", "term", "tfidf"}.issubset(result.columns)
    assert not result.empty


def _recommender_data(tmp_path: Path) -> Path:
    path = tmp_path / "jobs_clean.csv"
    pd.DataFrame({
        "role_query": ["Data Scientist", "Data Scientist", "Data Engineer"],
        "skills": ["['Python', 'SQL', 'machine learning']", "['Python', 'SQL']", "['SQL', 'Spark']"],
    }).to_csv(path, index=False)
    return path


def test_recommender_returns_matches_and_missing_skills(tmp_path: Path):
    result = recommend_skills(" data scientist ", ["PYTHON"], top_n=3, data_path=_recommender_data(tmp_path))
    assert result["matched_skills"] == ["python"]
    assert {"sql", "machine learning"}.issubset(result["missing_high_priority_skills"])


def test_recommender_handles_empty_candidate_skills(tmp_path: Path):
    result = recommend_skills("Data Scientist", [], top_n=3, data_path=_recommender_data(tmp_path))
    assert result["matched_skills"] == []
    assert result["readiness_score"] == 0
