# AI Resume Screening & Matching Platform

Streamlit app for comparing resumes against job descriptions, ranking multiple resumes, comparing two resumes, and checking personal resume health.

## Features

- Resume vs job description scoring
- Multiple resume ranking against one job description
- Two-resume comparison
- Personal resume health score
- PDF, DOCX, TXT, CSV, and XLSX upload support
- Session-only resume version retention with a forget button
- Safe fallback when the embedding model cannot be loaded

## Setup

```bash
cd "/Users/niyatidesai/Desktop/Resume checking"
python3 -m pip install -r requirements.txt
```

## Run

```bash
python3 -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Open:

```text
http://127.0.0.1:8501
```

## Optional Model Cache Step

The app uses `sentence-transformers/all-MiniLM-L6-v2` for semantic similarity. If Hugging Face is unreachable, the app still opens and falls back to skill-based scoring with semantic scores set to `0`.

To cache the model before running:

```bash
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

## Notes

- No environment variables are required.
- Resume retention is only stored in the active Streamlit session and can be cleared with the forget button.
- Uploaded files are processed in memory; this project does not use a database.
