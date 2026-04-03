import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ===================== NLTK SETUP =====================
nltk.download("punkt")
nltk.download("stopwords")

# ===================== PAGE SETUP =====================
st.set_page_config(
    page_title="Resume Job Match Scorer",
    page_icon="📄",
    layout="wide"
)

st.markdown("""
Upload your resume (PDF) and paste a job description to see how well they match.  
This tool uses **TF-IDF + Cosine Similarity** like real ATS systems.
""")

# ===================== SIDEBAR =====================
with st.sidebar:
    st.header("Resume Screener")
    st.info("""
    - ATS-style resume screening  
    - Keyword gap detection  
    - Resume improvement insights  
    """)

    st.header("How It Works")
    st.write("""
    1. Upload resume (PDF)  
    2. Paste job description  
    3. Click **Analyze Match**
    """)

# ===================== SESSION STATE =====================
if "job_description" not in st.session_state:
    st.session_state.job_description = ""

# ===================== HELPERS =====================
def extract_text_from_pdf(uploaded_file):
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"PDF Error: {e}")
        return ""

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text)
    return " ".join(w for w in words if w not in stop_words)

def calculate_similarity(resume_text, job_text):
    resume_clean = remove_stopwords(clean_text(resume_text))
    job_clean = remove_stopwords(clean_text(job_text))

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([resume_clean, job_clean])

    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100
    return round(score, 2), resume_clean, job_clean

def extract_keywords(text, top_n=30):
    vectorizer = TfidfVectorizer(
        max_features=top_n,
        ngram_range=(1, 2)
    )
    vectorizer.fit([text])
    return set(vectorizer.get_feature_names_out())

def clear_job_description():
    st.session_state.job_description = ""

# ===================== MAIN APP =====================
def main():
    st.subheader("Upload & Analyze")

    uploaded_file = st.file_uploader(
        "Upload your resume (PDF)",
        type=["pdf"]
    )

    st.text_area(
        "Paste the job description",
        height=180,
        key="job_description"
    )

    # 🔘 Buttons with NO GAP
    btn_col1, btn_col2 = st.columns(2)

    with btn_col1:
        analyze = st.button("Analyze Match", use_container_width=True)

    with btn_col2:
        st.button(
            "Clear Job Description",
            on_click=clear_job_description,
            use_container_width=True
        )

    # ===================== ANALYSIS =====================
    if analyze:
        if not uploaded_file:
            st.warning("Please upload your resume.")
            return

        if not st.session_state.job_description.strip():
            st.warning("Please paste the job description.")
            return

        with st.spinner("Analyzing resume against job description..."):
            resume_text = extract_text_from_pdf(uploaded_file)

            if not resume_text.strip():
                st.error("Unable to extract text from PDF.")
                return

            score, resume_clean, job_clean = calculate_similarity(
                resume_text,
                st.session_state.job_description
            )

            resume_keywords = extract_keywords(resume_clean)
            job_keywords = extract_keywords(job_clean)

            matched_keywords = resume_keywords & job_keywords
            missing_keywords = job_keywords - resume_keywords

        # ===================== RESULTS =====================
        st.subheader("Match Results")
        st.metric("ATS Match Score", f"{score:.2f}%")

        fig, ax = plt.subplots(figsize=(6, 0.6))
        colors = ["#ff4b4b", "#ffa726", "#0f9d58"]
        ax.barh([0], [score], color=colors[min(int(score // 33), 2)])
        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.set_xlabel("Match Percentage")
        ax.set_title("Resume vs Job Match")

        st.pyplot(fig)

        if score < 40:
            st.warning("Low match. Tailor your resume.")
        elif score < 70:
            st.info("Good match. Improvements possible.")
        else:
            st.success("Excellent match!")

        # ===================== KEYWORD ANALYSIS =====================
        st.subheader("Keyword Gap Analysis (ATS Style)")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### ✅ Matched Keywords")
            if matched_keywords:
                st.write(", ".join(sorted(matched_keywords)))
            else:
                st.info("No strong keyword matches found.")

        with col2:
            st.markdown("### ❌ Missing Keywords (Add These)")
            if missing_keywords:
                st.write(", ".join(sorted(missing_keywords)))
            else:
                st.success("Great! No important keywords missing.")

# ===================== RUN APP =====================
if __name__ == "__main__":
    main()
