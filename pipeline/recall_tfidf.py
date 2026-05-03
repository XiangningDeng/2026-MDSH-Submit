from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_news_text(news: pd.DataFrame) -> pd.Series:
    title = news["title"].fillna("").astype(str)
    abstract = news["abstract"].fillna("").astype(str)
    return title + " " + title + " " + abstract


class TfidfRecallScorer:
    def __init__(self, tfidf_params: dict):
        self.vectorizer = TfidfVectorizer(**tfidf_params)
        self.tfidf_matrix: sparse.csr_matrix | None = None
        self.news_id_to_idx: dict[str, int] = {}
        self._history_vector_cache: dict[str, np.ndarray | None] = {}

    def fit(self, news: pd.DataFrame) -> "TfidfRecallScorer":
        news = news.copy()
        news["news_id"] = news["news_id"].astype(str)
        text = build_news_text(news)
        self.tfidf_matrix = self.vectorizer.fit_transform(text)
        self.news_id_to_idx = {news_id: i for i, news_id in enumerate(news["news_id"])}
        return self

    @classmethod
    def load_artifacts(cls, cache_dir: Path, tfidf_params: dict) -> "TfidfRecallScorer":
        scorer = cls(tfidf_params)
        with (cache_dir / "tfidf_vectorizer.pkl").open("rb") as f:
            scorer.vectorizer = pickle.load(f)
        scorer.tfidf_matrix = sparse.load_npz(cache_dir / "tfidf_matrix.npz")
        with (cache_dir / "news_id_to_idx.json").open("r", encoding="utf-8") as f:
            scorer.news_id_to_idx = json.load(f)
        return scorer

    @staticmethod
    def artifacts_exist(cache_dir: Path) -> bool:
        return all(
            (cache_dir / name).exists()
            for name in ["tfidf_vectorizer.pkl", "tfidf_matrix.npz", "news_id_to_idx.json"]
        )

    def save_artifacts(self, cache_dir: Path) -> None:
        if self.tfidf_matrix is None:
            raise RuntimeError("TF-IDF scorer must be fit before save_artifacts.")
        cache_dir.mkdir(parents=True, exist_ok=True)
        with (cache_dir / "tfidf_vectorizer.pkl").open("wb") as f:
            pickle.dump(self.vectorizer, f)
        sparse.save_npz(cache_dir / "tfidf_matrix.npz", self.tfidf_matrix)
        with (cache_dir / "news_id_to_idx.json").open("w", encoding="utf-8") as f:
            json.dump(self.news_id_to_idx, f, ensure_ascii=False)

    def _user_vector(self, history: str):
        if self.tfidf_matrix is None:
            raise RuntimeError("TF-IDF scorer must be fit before scoring.")
        if not isinstance(history, str) or not history.strip():
            return None
        if history in self._history_vector_cache:
            return self._history_vector_cache[history]
        indices = [self.news_id_to_idx[nid] for nid in history.split() if nid in self.news_id_to_idx]
        if not indices:
            self._history_vector_cache[history] = None
            return None
        user_vec = np.asarray(self.tfidf_matrix[indices].mean(axis=0))
        self._history_vector_cache[history] = user_vec
        return user_vec

    def score_candidates(self, candidates: pd.DataFrame) -> pd.DataFrame:
        if self.tfidf_matrix is None:
            raise RuntimeError("TF-IDF scorer must be fit before scoring.")

        scored = candidates[["impression_id", "candidate_news_id"]].copy()
        scored["impression_id"] = scored["impression_id"].astype(str)
        scored["candidate_news_id"] = scored["candidate_news_id"].astype(str)
        scored["tfidf_score"] = 0.0
        if "label" in candidates.columns:
            scored["label"] = candidates["label"].astype(int).to_numpy()

        for _, group in candidates.groupby("impression_id", sort=False):
            user_vec = self._user_vector(str(group["history"].iloc[0]))
            if user_vec is None:
                continue

            candidate_news_ids = group["candidate_news_id"].astype(str)
            known_mask = candidate_news_ids.isin(self.news_id_to_idx)
            if not known_mask.any():
                continue

            known_index = group.index[known_mask]
            matrix_rows = [self.news_id_to_idx[nid] for nid in candidate_news_ids[known_mask]]
            candidate_vecs = self.tfidf_matrix[matrix_rows]
            scores = cosine_similarity(user_vec, candidate_vecs).ravel()
            scored.loc[known_index, "tfidf_score"] = scores

        return scored
