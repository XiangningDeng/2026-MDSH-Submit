from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer


def build_news_text(news: pd.DataFrame) -> pd.Series:
    title = news["title"].fillna("").astype(str)
    abstract = news["abstract"].fillna("").astype(str)
    return title + " " + title + " " + abstract


class BM25RecallScorer:
    """Score candidates with BM25 lexical retrieval over title and abstract text."""

    def __init__(self, bm25_params: dict):
        self.vectorizer = CountVectorizer(**bm25_params["vectorizer"])
        self.k1 = float(bm25_params.get("k1", 1.5))
        self.b = float(bm25_params.get("b", 0.75))
        self.news_id_to_idx: dict[str, int] = {}
        self.count_matrix: sparse.csr_matrix | None = None
        self.bm25_matrix: sparse.csr_matrix | None = None
        self._history_query_cache: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}

    def fit(self, news: pd.DataFrame) -> "BM25RecallScorer":
        news = news.copy()
        news["news_id"] = news["news_id"].astype(str)
        counts = self.vectorizer.fit_transform(build_news_text(news)).astype(np.float32).tocsr()
        self.count_matrix = counts
        self.news_id_to_idx = {news_id: i for i, news_id in enumerate(news["news_id"])}

        doc_freq = np.asarray((counts > 0).sum(axis=0)).ravel().astype(np.float32)
        num_docs = counts.shape[0]
        idf = np.log1p((num_docs - doc_freq + 0.5) / (doc_freq + 0.5)).astype(np.float32)

        doc_len = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
        avg_doc_len = float(doc_len.mean()) if len(doc_len) else 0.0
        if avg_doc_len <= 0:
            avg_doc_len = 1.0

        coo = counts.tocoo()
        rows = coo.row
        cols = coo.col
        term_freq = coo.data.astype(np.float32)
        denom = term_freq + self.k1 * (1.0 - self.b + self.b * doc_len[rows] / avg_doc_len)
        weights = idf[cols] * (term_freq * (self.k1 + 1.0) / denom)
        self.bm25_matrix = sparse.csr_matrix(
            (weights.astype(np.float32), (rows, cols)),
            shape=counts.shape,
        )
        return self

    def score_candidates(self, candidates: pd.DataFrame) -> pd.DataFrame:
        if self.bm25_matrix is None:
            raise RuntimeError("BM25RecallScorer must be fit before scoring.")

        scored = candidates[["impression_id", "candidate_news_id"]].copy()
        scored["impression_id"] = scored["impression_id"].astype(str)
        scored["candidate_news_id"] = scored["candidate_news_id"].astype(str)
        score_values = np.zeros(len(scored), dtype=np.float32)
        candidate_ids = scored["candidate_news_id"].to_numpy()
        histories = candidates["history"].fillna("").astype(str).to_numpy()

        for positions in scored.groupby("impression_id", sort=False).indices.values():
            query = self._query_terms(str(histories[positions[0]]))
            if query is None:
                continue
            query_cols, query_weights = query

            known_positions = []
            matrix_rows = []
            for pos in positions:
                news_idx = self.news_id_to_idx.get(candidate_ids[pos])
                if news_idx is not None:
                    known_positions.append(pos)
                    matrix_rows.append(news_idx)
            if not known_positions:
                continue

            scores = self.bm25_matrix[np.asarray(matrix_rows, dtype=np.int64)][:, query_cols].dot(
                query_weights
            )
            score_values[np.asarray(known_positions, dtype=np.int64)] = np.asarray(scores).ravel()

        scored["bm25_score"] = score_values
        scored["recalled_by_bm25"] = (scored["bm25_score"] > 0).astype(np.int8)
        scored["bm25_rank"] = self._rank_within_impression(scored, "bm25_score")
        keep_cols = [
            "impression_id",
            "candidate_news_id",
            "bm25_score",
            "bm25_rank",
            "recalled_by_bm25",
        ]
        if "label" in candidates.columns:
            scored["label"] = candidates["label"].astype(int).to_numpy()
            keep_cols.append("label")
        return scored[keep_cols]

    def _query_terms(self, history: str) -> tuple[np.ndarray, np.ndarray] | None:
        if self.count_matrix is None:
            raise RuntimeError("BM25RecallScorer must be fit before scoring.")
        if not isinstance(history, str) or not history.strip():
            return None
        if history in self._history_query_cache:
            return self._history_query_cache[history]

        indices = [self.news_id_to_idx[nid] for nid in history.split() if nid in self.news_id_to_idx]
        if not indices:
            self._history_query_cache[history] = None
            return None

        query_counts = np.asarray(self.count_matrix[indices].sum(axis=0)).ravel()
        query_cols = np.flatnonzero(query_counts)
        if len(query_cols) == 0:
            self._history_query_cache[history] = None
            return None

        query_weights = np.log1p(query_counts[query_cols]).astype(np.float32)
        out = (query_cols.astype(np.int32), query_weights)
        self._history_query_cache[history] = out
        return out

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
