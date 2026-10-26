# Resume-checking baseline — 26 October 2026

Repository: https://github.com/Niyatidesai28/Resume-checking
Default branch verified by GitHub API and origin/HEAD: main.
Original HEAD: 1b87517c3e6d20b38211dd0c5a612ba55c75a48b.
Fresh clean clone, no user changes; isolated branch codex/resume-audit.
CLI owner account has admin/push permission. Connector account is read-only.

2,502 original tracked files, including 2,484 resume PDFs, four images, a video,
a job JSON dataset, extensionless skill data, an exploratory notebook and five Python files.
Main application: app.py (Streamlit), matcher.py, preprocess.py, data_loader.py.
Separate legacy demo: archive (13)/screening.py. Notebook references missing CSV inputs.
No SQL. Main semantic integration loads all-MiniLM-L6-v2; rule-based health/skill scoring
is separate. No original tests or CI. Use only synthetic resumes for tests; do not print
or republish corpus personal data. No production service or paid inference will be used.
