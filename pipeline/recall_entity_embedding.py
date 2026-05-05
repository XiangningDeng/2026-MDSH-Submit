from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


class EntityEmbeddingRecallScorer:
    """Score candidates by cosine similarity between user and news entity vectors."""

    def __init__(self, entity_embedding_paths: list[Path] | None = None):
        self.entity_embedding_paths = entity_embedding_paths or []
        self.entity_vectors: dict[str, np.ndarray] = {}
        self.news_id_to_idx: dict[str, int] = {}
        self.news_vectors: np.ndarray | None = None
        self._history_vector_cache: dict[str, np.ndarray | None] = {}

    def fit(self, news: pd.DataFrame) -> "EntityEmbeddingRecallScorer":
        self.entity_vectors = self._load_entity_embeddings(self.entity_embedding_paths)
        news_vectors = []
        news_ids = []

        prepared = news.copy()
        prepared["news_id"] = prepared["news_id"].astype(str)
        for _, row in prepared.iterrows():
            vector = self._news_entity_vector(row)
            if vector is None:
                continue
            news_ids.append(row["news_id"])
            news_vectors.append(vector)

        if news_vectors:
            matrix = np.vstack(news_vectors).astype(np.float32)
        else:
            dim = self._embedding_dim()
            matrix = np.zeros((0, dim), dtype=np.float32)

        self.news_id_to_idx = {news_id: i for i, news_id in enumerate(news_ids)}
        self.news_vectors = self._normalize_rows(matrix)
        return self

    def score_candidates(self, candidates: pd.DataFrame) -> pd.DataFrame:
        if self.news_vectors is None:
            raise RuntimeError("EntityEmbeddingRecallScorer must be fit before scoring.")

        scored = candidates[["impression_id", "candidate_news_id", "history"]].copy()
        scored["impression_id"] = scored["impression_id"].astype(str)
        scored["candidate_news_id"] = scored["candidate_news_id"].astype(str)
        scored["history"] = scored["history"].fillna("").astype(str)
        scored["entity_embedding_score"] = 0.0

        for _, group in scored.groupby("impression_id", sort=False):
            user_vec = self._user_vector(str(group["history"].iloc[0]))
            if user_vec is None:
                continue

            candidate_ids = group["candidate_news_id"].astype(str)
            known_mask = candidate_ids.isin(self.news_id_to_idx)
            if not known_mask.any():
                continue

            matrix_rows = [self.news_id_to_idx[nid] for nid in candidate_ids[known_mask]]
            scores = self.news_vectors[matrix_rows] @ user_vec
            scored.loc[group.index[known_mask], "entity_embedding_score"] = scores

        scored["recalled_by_entity_embedding"] = (
            scored["entity_embedding_score"] > 0
        ).astype(np.int8)
        scored["entity_embedding_rank"] = self._rank_within_impression(
            scored, "entity_embedding_score"
        )
        keep_cols = [
            "impression_id",
            "candidate_news_id",
            "entity_embedding_score",
            "entity_embedding_rank",
            "recalled_by_entity_embedding",
        ]
        if "label" in candidates.columns:
            scored["label"] = candidates["label"].astype(int).to_numpy()
            keep_cols.append("label")
        return scored[keep_cols]

    def select_top_n(self, candidates: pd.DataFrame, top_n: int) -> pd.DataFrame:
        scored = self.score_candidates(candidates)
        scored["_original_order"] = np.arange(len(scored))
        selected = (
            scored.sort_values(
                ["impression_id", "entity_embedding_score", "_original_order"],
                ascending=[True, False, True],
            )
            .groupby("impression_id", sort=False)
            .head(top_n)
        )
        return selected.drop(columns=["_original_order"]).reset_index(drop=True)

    def _user_vector(self, history: str) -> np.ndarray | None:
        if self.news_vectors is None:
            raise RuntimeError("EntityEmbeddingRecallScorer must be fit before scoring.")
        if not isinstance(history, str) or not history.strip():
            return None
        if history in self._history_vector_cache:
            return self._history_vector_cache[history]

        indices = [self.news_id_to_idx[nid] for nid in history.split() if nid in self.news_id_to_idx]
        if not indices:
            self._history_vector_cache[history] = None
            return None

        user_vec = self.news_vectors[indices].mean(axis=0)
        norm = np.linalg.norm(user_vec)
        if norm <= 0:
            self._history_vector_cache[history] = None
            return None

        user_vec = (user_vec / norm).astype(np.float32)
        self._history_vector_cache[history] = user_vec
        return user_vec

    def _news_entity_vector(self, row: pd.Series) -> np.ndarray | None:
        entity_ids = []
        for column in ["title_entities", "abstract_entities"]:
            entity_ids.extend(self._parse_entity_ids(row.get(column, "")))

        vectors = []
        seen = set()
        for entity_id in entity_ids:
            if entity_id in seen:
                continue
            seen.add(entity_id)
            vector = self.entity_vectors.get(entity_id)
            if vector is not None:
                vectors.append(vector)

        if not vectors:
            return None
        return np.mean(vectors, axis=0)

    @staticmethod
    def _parse_entity_ids(raw_entities: str) -> list[str]:
        if not isinstance(raw_entities, str) or not raw_entities.strip():
            return []
        try:
            entities = json.loads(raw_entities)
        except json.JSONDecodeError:
            return []
        return [
            entity.get("WikidataId")
            for entity in entities
            if isinstance(entity, dict) and entity.get("WikidataId")
        ]

    @staticmethod
    def _load_entity_embeddings(paths: list[Path]) -> dict[str, np.ndarray]:
        vectors: dict[str, np.ndarray] = {}
        for path in paths:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 2 or parts[0] in vectors:
                        continue
                    values = [value for value in parts[1:] if value != ""]
                    vectors[parts[0]] = np.asarray(values, dtype=np.float32)
        return vectors

    def _embedding_dim(self) -> int:
        if not self.entity_vectors:
            return 0
        return len(next(iter(self.entity_vectors.values())))

    @staticmethod
    def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
        if matrix.size == 0:
            return matrix
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

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
