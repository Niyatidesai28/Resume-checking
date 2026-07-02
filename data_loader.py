import pandas as pd


def load_resumes(resume_path: str) -> pd.DataFrame:
    return pd.read_csv(resume_path)


def load_jds(jd_path: str) -> pd.DataFrame:
    return pd.read_csv(jd_path)


def create_sample_jd_dataframe() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "jd_id": 1,
            "target_role": "Software Engineer",
            "jd_text": "Python, SQL, APIs, backend development, cloud, debugging, testing",
        },
        {
            "jd_id": 2,
            "target_role": "Data Analyst",
            "jd_text": "SQL, Excel, Power BI, Tableau, data cleaning, dashboards, reporting",
        },
        {
            "jd_id": 3,
            "target_role": "AI Engineer",
            "jd_text": "Python, machine learning, NLP, embeddings, LLMs, vector databases",
        },
    ])
