# matcher.py

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class ResumeMatcher:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Load embedding model.
        """
        self.model_name = model_name
        self.model = None
        self.model_load_error = None

        try:
            self.model = SentenceTransformer(model_name)
        except Exception as exc:
            self.model_load_error = str(exc)

    def encode_texts(self, texts: list) -> np.ndarray:
        """
        Convert list of texts into embeddings.
        """
        if self.model is None:
            raise RuntimeError(
                f"Embedding model '{self.model_name}' is unavailable: {self.model_load_error}"
            )

        return self.model.encode(texts, convert_to_numpy=True)

    def compute_semantic_scores(
        self,
        resume_texts: list,
        jd_text: str
    ) -> np.ndarray:
        """
        Compute cosine similarity between each resume and one JD.
        """
        if self.model is None:
            return np.zeros(len(resume_texts))

        resume_embeddings = self.encode_texts(resume_texts)
        jd_embedding = self.encode_texts([jd_text])

        scores = cosine_similarity(resume_embeddings, jd_embedding).flatten()
        return scores

    @staticmethod
    def skill_overlap_score(resume_skills: list, jd_skills: list) -> float:
        """
        Simple overlap-based score.
        """
        if not jd_skills:
            return 0.0

        resume_set = set(resume_skills)
        jd_set = set(jd_skills)

        overlap = len(resume_set.intersection(jd_set))
        return overlap / max(len(jd_set), 1)

    def match_resumes_to_single_jd(
        self,
        resume_df: pd.DataFrame,
        jd_row: pd.Series,
        semantic_weight: float = 0.7,
        skill_weight: float = 0.3
    ) -> pd.DataFrame:
        """
        Score all resumes against one selected JD.
        """
        result_df = resume_df.copy()

        semantic_scores = self.compute_semantic_scores(
            result_df["cleaned_resume_text"].tolist(),
            jd_row["cleaned_jd_text"]
        )

        result_df["semantic_score"] = semantic_scores

        result_df["skill_match_score"] = result_df["skills_extracted"].apply(
            lambda skills: self.skill_overlap_score(skills, jd_row["jd_skills"])
        )

        result_df["final_score"] = (
            semantic_weight * result_df["semantic_score"] +
            skill_weight * result_df["skill_match_score"]
        )

        result_df["matched_skills"] = result_df["skills_extracted"].apply(
            lambda skills: sorted(list(set(skills).intersection(set(jd_row["jd_skills"]))))
        )

        result_df["missing_skills"] = result_df["skills_extracted"].apply(
            lambda skills: sorted(list(set(jd_row["jd_skills"]) - set(skills)))
        )

        result_df["jd_id"] = jd_row["jd_id"]
        result_df["target_role"] = jd_row["target_role"]

        result_df["recommendation"] = result_df["final_score"].apply(self.get_recommendation)

        result_df = result_df.sort_values("final_score", ascending=False).reset_index(drop=True)
        return result_df

    @staticmethod
    def get_recommendation(score: float) -> str:
        """
        Map numeric score to recruiter-friendly label.
        """
        if score >= 0.80:
            return "Strong Match"
        elif score >= 0.65:
            return "Consider"
        else:
            return "Weak Match"
