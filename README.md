# DSCI 510 Final Project — Data Science Job Market Data Mining: Identifying the Most In-Demand Skills for Data Scientists

## 1. Project Overview
This project implements a complete data pipeline to examine job postings for four major data-related roles using the SerpAPI Google Jobs engine:
- Data Scientist  
- Machine Learning Engineer  
- Data Analyst  
- Data Engineer  

The objective is to identify:
- The most in-demand technical skills  
- Differences in skill requirements across roles  
- Co-occurrence patterns among skills  
- Geographic trends across postings  
- Market signals captured through TF-IDF keyword analysis  

---

## 2. Author Information
**Jui-Ching Yu**  
Email: **juiching@usc.edu**  
GitHub: **https://github.com/Meredith0613**  
USC ID: **5507402044**  
Course: **DSCI 510 – Principles of Programming for Data Science**  
Semester: **Fall 2025**

---

## 3. Repository Structure

DSCI-510-Final-Project/<br>
├── README.md<br>
├── requirements.txt<br>
│<br>
├── data/<br>
│&nbsp;&nbsp; ├── raw/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # Raw API results (generated during get_data.py)<br>
│&nbsp;&nbsp; ├── processed/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # Cleaned dataset (jobs_clean.csv)<br>
│&nbsp;&nbsp; └── analysis/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # TF-IDF outputs and analysis artifacts<br>
│<br>
├── results/<br>
│&nbsp;&nbsp; └── figures/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # All generated visualizations<br>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ├── top_skills_bar.png<br>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ├── skills_wordcloud.png<br>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ├── skill_cooccurrence_network.png<br>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ├── salary_vs_skill_count.png<br>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; └── location_distribution.png<br>
│<br>
├── src/<br>
│&nbsp;&nbsp; ├── get_data.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # Collects job postings from SerpAPI<br>
│&nbsp;&nbsp; ├── clean_data.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # Cleans, standardizes, and extracts skills<br>
│&nbsp;&nbsp; ├── run_analysis.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # TF-IDF computation and summary analysis<br>
│&nbsp;&nbsp; ├── visualize_results.py &nbsp; # Generates plots and visual outputs<br>
│&nbsp;&nbsp; ├── run_all.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # Orchestrates the full pipeline end-to-end<br>
│&nbsp;&nbsp; └── utils/<br>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; └── request_utils.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # Helper functions for SerpAPI requests<br>

---

## 4. Environment Setup

### Clone the Repository
git clone https://github.com/Meredith0613/DSCI-510-Final-Project.git <br>
cd DSCI-510-Final-Project <br>

### Create and Activate a Python Environment
conda create -n dsc510_final python=3.11 -y <br>
conda activate dsc510_final <br>

### Install Dependencies
pip install -r requirements.txt <br>

### Configure SerpAPI Key
export SERPAPI_API_KEY="974c70e1f158162f9714c12350c8756800eb48b2c5be39edf6adf36c8b07dc7f" <br>

---

## 5. Running the Pipeline
Step-by-step execution <br>
python -m src.get_data          # Data collection <br>
python -m src.clean_data        # Cleaning and standardization <br>
python -m src.run_analysis      # TF-IDF and summary analysis <br>
python -m src.visualize_results # Generate all visualizations <br>

Run the full pipeline (optional) <br>
python -m src.run_all <br>

---

## 6. Reproducibility Notes
- The full workflow is automated through src/run_all.py.
- Scripts are idempotent and safe to rerun.
- Raw data is not stored permanently; it can be regenerated with a valid API key.
- Live API responses vary, meaning results (e.g., dataset size, TF-IDF scores, skill frequencies) will differ across runs.

---

## 7. Variability Notice
Because the project retrieves live job postings through SerpAPI, outputs depend on current job availability, pagination behavior, and API quota conditions. The results shown in the report reflect the dataset produced during the documented test run and may not match results generated at a later date.





