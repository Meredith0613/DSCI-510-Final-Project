# Data Science Job Market Intelligence Pipeline

An end-to-end Python pipeline that collects live job postings, cleans and standardizes job market data, extracts skill requirements, and analyzes role-specific demand across Data Scientist, Machine Learning Engineer, Data Analyst, and Data Engineer roles.

## Project Overview

The project turns job-posting text into a reproducible market-intelligence dataset. It collects postings through SerpAPI's Google Jobs engine, extracts a consistent set of technical skills, calculates TF-IDF terms by role, and produces charts for skills, locations, salary context, and skill co-occurrence. A Streamlit demo adds a practical candidate skill-gap checker.

## Motivation

Job descriptions contain useful but unstructured signals about what employers value. This project makes those signals easier to compare across adjacent data roles, helping candidates prioritize learning and helping teams understand role-specific demand.

## Data Source

Live postings are retrieved from the [SerpAPI Google Jobs engine](https://serpapi.com/google-jobs-api). The repository includes a saved snapshot of 578 cleaned postings so analysis, tests, and the dashboard work without API access. Live results change as postings and API availability change.

## Pipeline Workflow

```text
SerpAPI Google Jobs → raw CSV/JSON → cleaning + skill extraction
→ role-level TF-IDF + summary analysis → figures + Streamlit dashboard
```

1. `src.get_data` retrieves postings for four target roles.
2. `src.clean_data` normalizes core fields, removes duplicates, and extracts skills.
3. `src.run_analysis` creates role-level TF-IDF output.
4. `src.visualize_results` creates reusable PNG charts.
5. `src.skill_gap_recommender` compares candidate skills with role demand.

## Key Features

- Environment-based API configuration—no API key is stored in the repository.
- Repeatable data-cleaning and duplicate-removal steps.
- Lightweight, transparent keyword-based skill extraction.
- Role-level TF-IDF analysis and skill co-occurrence visualization.
- Interactive dashboard with a candidate skill-gap recommendation system.

## Business Impact / Key Insights

Insights below come from the included 578-posting snapshot; they are directional and will evolve with a new collection run.

- Python (412 postings) and SQL (334) are the strongest broadly requested technical skills after cleaning; both appear across every target role.
- Data Scientist postings emphasize machine learning (88) and statistics (60), while Data Engineer postings more often cite cloud (103) and ETL (80).
- Machine Learning Engineer postings show the strongest machine-learning signal (136) and frequently request PyTorch (87) and deep learning (71).
- Canada has the largest named location concentration in this snapshot (142 postings); remote/`anywhere` postings are also prominent (40).
- Python and SQL co-occur in 271 postings, a useful foundational pairing for entry-level candidates. Prioritize Python, SQL, one visualization tool, and then role-specific skills such as machine learning or cloud/data-pipeline tooling.

## Visualizations

Generated charts are saved in `results/figures/`:

- `top_skills_bar.png` and `skills_wordcloud.png` summarize technical demand.
- `skill_cooccurrence_network.png` shows skills frequently named together.
- `location_distribution.png` maps hiring concentration.
- `salary_vs_skill_count.png` provides exploratory salary context where salary text is available.

Regenerate them with `python -m src.visualize_results`.

## Skill Gap Recommendation System

`recommend_skills` uses the top extracted skills in the cleaned data for a selected role. It accepts mixed capitalization and extra whitespace, then returns matched skills, high-priority gaps, recommended next skills, and a transparent market-alignment score.

```python
from src.skill_gap_recommender import recommend_skills

recommend_skills("Data Scientist", ["Python", "SQL", "Tableau"])
```

The score is the percentage of the selected role's top skills a candidate already has; it is a learning-priority aid, not a hiring prediction.

## How to Run

```bash
git clone https://github.com/Meredith0613/DSCI-510-Final-Project.git
cd DSCI-510-Final-Project
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

To collect a fresh live dataset, set the key in your shell (never commit it):

```bash
export SERPAPI_API_KEY="your_api_key_here"
python -m src.run_all
```

To work from the included snapshot without making API calls:

```bash
python -m src.run_analysis
python -m src.visualize_results
```

Launch the demo with:

```bash
streamlit run app.py
```

## Testing

```bash
pip install -r requirements.txt
pytest
```

The test suite covers skill extraction, missing salaries, duplicate handling, expected cleaned columns, TF-IDF output, and the core recommender behaviors.

## Future Improvements

- Add a scheduled collection job and historical trend tracking.
- Replace keyword matching with a maintained skills taxonomy or entity extraction.
- Add seniority, employer, and compensation normalization.
- Persist results in a database and add filtering by geography or time period.
