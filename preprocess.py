import re
import pandas as pd


SKILL_KEYWORDS = [
    "python", "java", "c++", "javascript", "typescript", "react", "node.js",
    "node", "fastapi", "flask", "django", "sql", "mysql", "postgresql",
    "mongodb", "aws", "azure", "gcp", "docker", "kubernetes", "jenkins",
    "git", "machine learning", "deep learning", "nlp", "llm", "rag",
    "langchain", "langgraph", "pandas", "numpy", "scikit-learn", "tensorflow",
    "pytorch", "power bi", "tableau", "airflow", "spark", "hadoop"
]


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9+.#/ -]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_skills(text: str, skill_keywords=None) -> list:
    if skill_keywords is None:
        skill_keywords = SKILL_KEYWORDS

    cleaned = clean_text(text)
    found_skills = []

    for skill in skill_keywords:
        if skill.lower() in cleaned:
            found_skills.append(skill.lower())

    return sorted(list(set(found_skills)))


def preprocess_resume_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    possible_text_cols = [
        "resume_text", "Resume", "resume", "text", "cleaned_resume",
        "Resume_str", "resume_str", "Text", "content"
    ]

    text_col = None
    for col in possible_text_cols:
        if col in df.columns:
            text_col = col
            break

    if text_col is None:
        raise ValueError(
            f"Could not find resume text column. Available columns: {list(df.columns)}"
        )

    df["resume_text"] = df[text_col].astype(str)
    df["cleaned_resume_text"] = df["resume_text"].apply(clean_text)
    df["skills_extracted"] = df["resume_text"].apply(extract_skills)

    if "candidate_id" not in df.columns:
        df["candidate_id"] = range(1, len(df) + 1)

    if "category" not in df.columns:
        df["category"] = "unknown"

    return df


def preprocess_jd_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "jd_text" not in df.columns:
        raise ValueError("JD dataframe must contain 'jd_text' column.")

    if "jd_id" not in df.columns:
        df["jd_id"] = range(1, len(df) + 1)

    if "target_role" not in df.columns:
        df["target_role"] = "unknown_role"

    df["cleaned_jd_text"] = df["jd_text"].apply(clean_text)
    df["jd_skills"] = df["jd_text"].apply(extract_skills)

    return df