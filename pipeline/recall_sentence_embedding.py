from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class SentenceEmbeddingRecallScorer:
    """Score candidates by cosine similarity over cached sentence-transformer news embeddings."""

    def __init__(self, embedding_path: Path):
        self.embedding_path = Path(embedding_path)
        self.news_id_to_idx: dict[str, int] = {}
        self.news_vectors: np.ndarray | None = None
        self._history_vector_cache: dict[str, np.ndarray | None] = {}

    def fit(self, news: pd.DataFrame | None = None) -> "SentenceEmbeddingRecallScorer":
        if not self.embedding_path.exists():
            raise FileNotFoundError(
                f"Missing sentence embedding cache: {self.embedding_path}. "
                "Run pipeline/generate_sentence_embeddings.py first."
            )

        data = np.load(self.embedding_path, allow_pickle=True)
        news_ids = data["news_ids"].astype(str)
        embeddings = data["embeddings"].astype(np.float32)

        if embeddings.ndim != 2 or len(news_ids) != embeddings.shape[0]:
            raise ValueError(
                "Invalid embedding cache: expected news_ids length to match embeddings rows."
            )

        embeddings = self._normalize_rows(embeddings)
        self.news_id_to_idx = {news_id: i for i, news_id in enumerate(news_ids)}
        self.news_vectors = embeddings
        self._history_vector_cache = {}
        return self

    def score_candidates(self, candidates: pd.DataFrame) -> pd.DataFrame:
        if self.news_vectors is None:
            raise RuntimeError("SentenceEmbeddingRecallScorer must be fit before scoring.")

        scored = candidates[["impression_id", "candidate_news_id"]].copy()
        scored["impression_id"] = scored["impression_id"].astype(str)
        scored["candidate_news_id"] = scored["candidate_news_id"].astype(str)
        score_values = np.zeros(len(scored), dtype=np.float32)
        candidate_ids = scored["candidate_news_id"].to_numpy()
        histories = candidates["history"].fillna("").astype(str).to_numpy()

        for positions in scored.groupby("impression_id", sort=False).indices.values():
            user_vec = self._user_vector(str(histories[positions[0]]))
            if user_vec is None:
                continue

            known_positions = []
            matrix_rows = []
            for pos in positions:
                news_idx = self.news_id_to_idx.get(candidate_ids[pos])
                if news_idx is not None:
                    known_positions.append(pos)
                    matrix_rows.append(news_idx)
            if not known_positions:
                continue

            scores = self.news_vectors[np.asarray(matrix_rows, dtype=np.int64)] @ user_vec
            score_values[np.asarray(known_positions, dtype=np.int64)] = scores

        scored["sentence_embedding_score"] = score_values
        scored["recalled_by_sentence_embedding"] = (
            scored["sentence_embedding_score"] > 0
        ).astype(np.int8)
        scored["sentence_embedding_rank"] = self._rank_within_impression(
            scored,
            "sentence_embedding_score",
        )
        keep_cols = [
            "impression_id",
            "candidate_news_id",
            "sentence_embedding_score",
            "sentence_embedding_rank",
            "recalled_by_sentence_embedding",
        ]
        if "label" in candidates.columns:
            scored["label"] = candidates["label"].astype(int).to_numpy()
            keep_cols.append("label")
        return scored[keep_cols]

    def _user_vector(self, history: str) -> np.ndarray | None:
        if self.news_vectors is None:
            raise RuntimeError("SentenceEmbeddingRecallScorer must be fit before scoring.")
        if not isinstance(history, str) or not history.strip():
            return None
        if history in self._history_vector_cache:
            return self._history_vector_cache[history]

        indices = [self.news_id_to_idx[nid] for nid in history.split() if nid in self.news_id_to_idx]
        if not indices:
            self._history_vector_cache[history] = None
            return None

        user_vec = self.news_vectors[np.asarray(indices, dtype=np.int64)].mean(axis=0)
        norm = np.linalg.norm(user_vec)
        if norm <= 0:
            self._history_vector_cache[history] = None
            return None

        user_vec = (user_vec / norm).astype(np.float32)
        self._history_vector_cache[history] = user_vec
        return user_vec

    @staticmethod
    def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
        if matrix.size == 0:
            return matrix
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (matrix / norms).astype(np.float32)

    @staticmethod
    def _rank_within_impression(scored: pd.DataFrame, score_col: str) -> pd.Series:
        order = scored[["impression_id", score_col]].copy()
        order["_original_order"] = np.arange(len(order))
        order["_rank"] = (
            order.sort_values(
                ["impression_id", score_col, "_original_order"],
                ascending=[True, False, True],
            )
            .groupby("impression_id", sort=False)
            .cumcount()
            + 1
        )
        return order.sort_values("_original_order")["_rank"].astype(np.int32).to_numpy()
