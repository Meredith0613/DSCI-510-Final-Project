"""Streamlit demo for the Data Science Job Market Intelligence Pipeline."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.skill_gap_recommender import recommend_skills

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "jobs_clean.csv"
FIGURES = ROOT / "results" / "figures"
ROLES = ["Data Scientist", "Machine Learning Engineer", "Data Analyst", "Data Engineer"]

st.set_page_config(page_title="Job Market Intelligence", page_icon="📊", layout="wide")
st.title("Data Science Job Market Intelligence")
st.caption("Explore role demand in the included job-posting snapshot.")


@st.cache_data
def load_jobs() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def show_figure(filename: str, caption: str) -> None:
    path = FIGURES / filename
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info("Run `python -m src.visualize_results` to generate this figure.")


page = st.sidebar.radio(
    "Explore", ["Role Comparison", "Top Skills by Role", "Skill Co-occurrence Network", "Location Distribution", "Candidate Skill Gap Checker"]
)

if page == "Role Comparison":
    jobs = load_jobs()
    counts = jobs["role_query"].value_counts().rename_axis("Role").reset_index(name="Postings")
    st.subheader("Postings by target role")
    st.bar_chart(counts.set_index("Role"))
    st.dataframe(counts, use_container_width=True, hide_index=True)
elif page == "Top Skills by Role":
    role = st.selectbox("Target role", ROLES)
    result = recommend_skills(role, [], top_n=10)
    st.subheader(f"Most requested skills: {role}")
    st.bar_chart(pd.DataFrame({"Skill": result["market_skills"], "Rank": range(10, 0, -1)}).set_index("Skill"))
    show_figure("top_skills_bar.png", "Overall top skills")
elif page == "Skill Co-occurrence Network":
    st.subheader("Skills that appear together in postings")
    show_figure("skill_cooccurrence_network.png", "Skill co-occurrence network")
elif page == "Location Distribution":
    st.subheader("Where postings are concentrated")
    show_figure("location_distribution.png", "Top posting locations")
else:
    st.subheader("Candidate Skill Gap Checker")
    role = st.selectbox("Target role", ROLES)
    entered_skills = st.text_input("Your current skills (comma-separated)", placeholder="Python, SQL, Tableau")
    if st.button("Check my alignment", type="primary"):
        result = recommend_skills(role, entered_skills.split(","))
        st.metric("Market-alignment score", f"{result['readiness_score']}%")
        st.write("**Matched skills:**", ", ".join(result["matched_skills"]) or "None yet")
        st.write("**High-priority gaps:**", ", ".join(result["missing_high_priority_skills"]) or "None")
        st.write("**Suggested next skills:**", ", ".join(result["recommended_skills"]) or "Keep deepening your current strengths")
