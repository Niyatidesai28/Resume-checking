import pdfplumber
from docx import Document
import streamlit as st
import pandas as pd
from datetime import datetime

from data_loader import create_sample_jd_dataframe
from preprocess import (
    preprocess_resume_df,
    preprocess_jd_df,
    clean_text,
    extract_skills
)
from matcher import ResumeMatcher

st.set_page_config(page_title="AI Resume Matcher", layout="wide")
st.title("AI Resume Screening & Matching Platform")
st.markdown("Choose a mode based on what you want to do with resumes and job descriptions.")


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def read_uploaded_table(uploaded_file):
    if uploaded_file is None:
        return None

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif file_name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Only CSV and XLSX files are supported for table uploads.")


def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


def extract_text_from_docx(uploaded_file):
    doc = Document(uploaded_file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text.strip()


def extract_text_from_uploaded_resume(uploaded_file):
    if uploaded_file is None:
        return ""

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore").strip()
    elif file_name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif file_name.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)
    else:
        raise ValueError("Supported resume file types: txt, pdf, docx")


def build_single_resume_df(resume_text: str, category: str = "uploaded_resume"):
    df = pd.DataFrame({
        "candidate_id": [1],
        "category": [category],
        "resume_text": [resume_text]
    })
    return preprocess_resume_df(df)


def build_single_jd_df(jd_text: str, target_role: str = "custom_role"):
    df = pd.DataFrame({
        "jd_id": [1],
        "target_role": [target_role],
        "jd_text": [jd_text]
    })
    return preprocess_jd_df(df)


def remember_resume_version(label: str, resume_text: str):
    if not resume_text.strip():
        return

    if "resume_versions" not in st.session_state:
        st.session_state.resume_versions = []

    st.session_state.resume_versions.append({
        "label": label,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "word_count": len(clean_text(resume_text).split()),
        "skills": extract_skills(resume_text),
    })


def render_resume_retention_controls():
    versions = st.session_state.get("resume_versions", [])

    if st.button("Forget retained resume versions", key="forget_resume_versions"):
        st.session_state.resume_versions = []
        st.success("Retained resume versions cleared for this session.")
        return

    if versions:
        with st.expander("Retained resume versions for this session"):
            st.dataframe(pd.DataFrame(versions))


def resume_health_score(resume_text: str):
    cleaned = clean_text(resume_text)
    skills = extract_skills(cleaned)

    score = 0
    reasons = []

    word_count = len(cleaned.split())
    if word_count >= 250:
        score += 20
    else:
        reasons.append("Resume content is too short.")

    if len(skills) >= 5:
        score += 25
    elif len(skills) >= 3:
        score += 15
    else:
        reasons.append("Very few recognizable technical skills found.")

    section_keywords = ["experience", "project", "education", "skill", "summary"]
    found_sections = sum(1 for kw in section_keywords if kw in cleaned)
    score += found_sections * 8

    if found_sections < 3:
        reasons.append("Resume seems to miss important sections like Experience, Projects, Skills, or Education.")

    action_words = ["built", "developed", "designed", "implemented", "optimized", "led", "created"]
    found_actions = sum(1 for w in action_words if w in cleaned)
    if found_actions >= 3:
        score += 15
    else:
        reasons.append("Resume bullets may need stronger action-oriented language.")

    has_metrics = any(ch.isdigit() for ch in cleaned)
    if has_metrics:
        score += 15
    else:
        reasons.append("Resume does not clearly show measurable impact or metrics.")

    score = min(score, 100)

    if score >= 80:
        verdict = "Strong Resume"
    elif score >= 60:
        verdict = "Average Resume"
    else:
        verdict = "Needs Improvement"

    return score, verdict, skills, reasons


# -------------------------------------------------
# Mode selection
# -------------------------------------------------
mode = st.radio(
    "Select what you want to do:",
    [
        "Resume vs JD Score",
        "Multiple Resumes vs One JD",
        "Compare Two Resumes",
        "Personal Resume Health Score"
    ],
    horizontal=True
)

matcher = ResumeMatcher()
if matcher.model is None:
    st.warning(
        "Embedding model could not be loaded. Semantic similarity will use a 0 fallback, "
        "but skill matching and resume health scoring still work."
    )
    if matcher.model_load_error:
        with st.expander("Model loading details"):
            st.write(matcher.model_load_error)


# =========================================================
# MODE 1: Resume vs JD Score
# =========================================================
if mode == "Resume vs JD Score":
    st.subheader("Resume vs Job Description")

    resume_input_method = st.radio(
        "Choose resume input method",
        [
            "Paste Resume Text",
            "Upload Single Resume File (PDF/DOCX/TXT)",
            "Upload Resume Dataset (CSV/XLSX)"
        ],
        horizontal=True,
        key="mode1_resume_method"
    )

    resume_df = None

    if resume_input_method == "Paste Resume Text":
        resume_text = st.text_area("Paste your resume text", height=250, key="mode1_resume_text")
        if resume_text.strip():
            resume_df = build_single_resume_df(resume_text, category="pasted_resume")

    elif resume_input_method == "Upload Single Resume File (PDF/DOCX/TXT)":
        uploaded_resume_file = st.file_uploader(
            "Upload single resume file",
            type=["pdf", "docx", "txt"],
            key="mode1_single_resume_file"
        )

        if uploaded_resume_file is not None:
            resume_text = extract_text_from_uploaded_resume(uploaded_resume_file)
            if resume_text.strip():
                resume_df = build_single_resume_df(resume_text, category="uploaded_single_resume")
                with st.expander("Preview extracted resume text"):
                    st.text(resume_text[:3000])

    else:
        uploaded_resume_dataset = st.file_uploader(
            "Upload resume dataset file",
            type=["csv", "xlsx"],
            key="mode1_resume_dataset"
        )

        if uploaded_resume_dataset is not None:
            raw_resume_df = read_uploaded_table(uploaded_resume_dataset)
            resume_df = preprocess_resume_df(raw_resume_df)
            st.dataframe(resume_df.head())

    jd_input_method = st.radio(
        "Choose JD input method",
        ["Paste JD Text", "Use Sample JDs", "Upload JD File (CSV/XLSX)"],
        horizontal=True,
        key="mode1_jd_method"
    )

    jd_df = None

    if jd_input_method == "Paste JD Text":
        jd_text = st.text_area("Paste job description text", height=220, key="mode1_jd_text")
        if jd_text.strip():
            jd_df = build_single_jd_df(jd_text, target_role="custom_jd")

    elif jd_input_method == "Use Sample JDs":
        jd_df = preprocess_jd_df(create_sample_jd_dataframe())

    else:
        uploaded_jd = st.file_uploader(
            "Upload JD file",
            type=["csv", "xlsx"],
            key="mode1_jd_upload"
        )
        if uploaded_jd is not None:
            raw_jd_df = read_uploaded_table(uploaded_jd)
            jd_df = preprocess_jd_df(raw_jd_df)

    if resume_df is not None and jd_df is not None:
        selected_role = st.selectbox("Select target role", jd_df["target_role"].tolist(), key="mode1_role")
        selected_jd_row = jd_df[jd_df["target_role"] == selected_role].iloc[0]

        if st.button("Run Resume vs JD Matching", key="mode1_run"):
            result_df = matcher.match_resumes_to_single_jd(resume_df, selected_jd_row)

            st.subheader("Match Result")
            cols = [
                "candidate_id", "category", "semantic_score",
                "skill_match_score", "final_score", "recommendation",
                "matched_skills", "missing_skills"
            ]
            cols = [c for c in cols if c in result_df.columns]
            st.dataframe(result_df[cols])


# =========================================================
# MODE 2: Multiple Resumes vs One JD
# =========================================================
elif mode == "Multiple Resumes vs One JD":
    st.subheader("Rank Multiple Resumes Against One JD")

    uploaded_resume = st.file_uploader(
        "Upload multiple resumes dataset (CSV/XLSX)",
        type=["csv", "xlsx"],
        key="mode2_resume_upload"
    )

    jd_option = st.radio(
        "Choose JD source",
        ["Paste JD Text", "Use Sample JDs", "Upload JD File (CSV/XLSX)"],
        horizontal=True,
        key="mode2_jd_option"
    )

    jd_df = None

    if jd_option == "Paste JD Text":
        jd_text = st.text_area("Paste JD text", height=220, key="mode2_jd_text")
        if jd_text.strip():
            jd_df = build_single_jd_df(jd_text, target_role="custom_jd")

    elif jd_option == "Use Sample JDs":
        jd_df = preprocess_jd_df(create_sample_jd_dataframe())

    else:
        uploaded_jd = st.file_uploader(
            "Upload JD file (CSV/XLSX)",
            type=["csv", "xlsx"],
            key="mode2_jd_upload"
        )
        if uploaded_jd is not None:
            raw_jd_df = read_uploaded_table(uploaded_jd)
            jd_df = preprocess_jd_df(raw_jd_df)

    if uploaded_resume is not None and jd_df is not None:
        raw_resume_df = read_uploaded_table(uploaded_resume)
        resume_df = preprocess_resume_df(raw_resume_df)

        selected_role = st.selectbox("Select target role", jd_df["target_role"].tolist(), key="mode2_role")
        selected_jd_row = jd_df[jd_df["target_role"] == selected_role].iloc[0]

        if st.button("Rank All Resumes", key="mode2_run"):
            result_df = matcher.match_resumes_to_single_jd(resume_df, selected_jd_row)

            st.subheader("Top Matches")
            cols = [
                "candidate_id", "category", "semantic_score",
                "skill_match_score", "final_score", "recommendation",
                "matched_skills", "missing_skills"
            ]
            cols = [c for c in cols if c in result_df.columns]
            st.dataframe(result_df[cols].head(50))

            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Ranked Results",
                csv,
                "ranked_resume_results.csv",
                "text/csv"
            )


# =========================================================
# MODE 3: Compare Two Resumes
# =========================================================
elif mode == "Compare Two Resumes":
    st.subheader("Compare Two Resumes")

    compare_method = st.radio(
        "Choose compare method",
        ["Paste Both Resumes", "Upload Both Resume Files (PDF/DOCX/TXT)"],
        horizontal=True,
        key="mode3_compare_method"
    )

    resume_1 = ""
    resume_2 = ""

    if compare_method == "Paste Both Resumes":
        resume_1 = st.text_area("Paste Resume 1", height=220, key="resume1")
        resume_2 = st.text_area("Paste Resume 2", height=220, key="resume2")

    else:
        uploaded_resume_1 = st.file_uploader(
            "Upload Resume 1",
            type=["pdf", "docx", "txt"],
            key="mode3_file1"
        )
        uploaded_resume_2 = st.file_uploader(
            "Upload Resume 2",
            type=["pdf", "docx", "txt"],
            key="mode3_file2"
        )

        if uploaded_resume_1 is not None:
            resume_1 = extract_text_from_uploaded_resume(uploaded_resume_1)

        if uploaded_resume_2 is not None:
            resume_2 = extract_text_from_uploaded_resume(uploaded_resume_2)

    if st.button("Compare Resumes", key="mode3_run"):
        if resume_1.strip() and resume_2.strip():
            if matcher.model is None:
                similarity = 0.0
                st.warning("Embedding model is unavailable, so semantic similarity is shown as 0.00.")
            else:
                emb = matcher.encode_texts([clean_text(resume_1), clean_text(resume_2)])
                similarity = float((emb[0] @ emb[1]) / ((emb[0] @ emb[0]) ** 0.5 * (emb[1] @ emb[1]) ** 0.5))

            skills_1 = set(extract_skills(resume_1))
            skills_2 = set(extract_skills(resume_2))

            st.metric("Resume Similarity Score", f"{similarity:.2f}")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Resume 1 Skills")
                st.write(sorted(skills_1))

            with col2:
                st.markdown("### Resume 2 Skills")
                st.write(sorted(skills_2))

            st.markdown("### Common Skills")
            st.write(sorted(skills_1.intersection(skills_2)))

            st.markdown("### Skills only in Resume 1")
            st.write(sorted(skills_1 - skills_2))

            st.markdown("### Skills only in Resume 2")
            st.write(sorted(skills_2 - skills_1))
        else:
            st.warning("Provide both resumes first.")


# =========================================================
# MODE 4: Personal Resume Health Score
# =========================================================
elif mode == "Personal Resume Health Score":
    st.subheader("Personal Resume Health Score")

    health_input_method = st.radio(
        "Choose resume input method",
        ["Paste Resume Text", "Upload Resume File (PDF/DOCX/TXT)"],
        horizontal=True,
        key="health_resume_method"
    )

    resume_text = ""

    if health_input_method == "Paste Resume Text":
        resume_text = st.text_area("Paste your resume text", height=280, key="health_resume_text")
    else:
        uploaded_health_resume = st.file_uploader(
            "Upload resume file",
            type=["pdf", "docx", "txt"],
            key="health_resume_file"
        )
        if uploaded_health_resume is not None:
            resume_text = extract_text_from_uploaded_resume(uploaded_health_resume)
            with st.expander("Preview extracted resume text"):
                st.text(resume_text[:3000])

    render_resume_retention_controls()

    if st.button("Analyze Resume Health", key="health_run"):
        if resume_text.strip():
            score, verdict, skills, reasons = resume_health_score(resume_text)
            remember_resume_version("health_score_resume", resume_text)

            st.metric("Resume Health Score", f"{score}/100")
            st.markdown(f"### Verdict: {verdict}")

            st.markdown("## Recognized Skills")
            st.write(skills)

            if reasons:
                st.markdown("## Improvement Suggestions")
                for reason in reasons:
                    st.write(f"- {reason}")
        else:
            st.warning("Please paste or upload a readable resume first.")
