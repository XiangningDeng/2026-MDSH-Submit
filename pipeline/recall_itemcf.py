from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd


class ItemCFRecallScorer:
    """Score candidates with item-item co-click similarity from train behaviors."""

    def __init__(self, max_history_items: int = 50, top_neighbors: int = 100):
        self.max_history_items = max_history_items
        self.top_neighbors = top_neighbors
        self.item_neighbors: dict[str, dict[str, float]] = {}
        self._history_score_cache: dict[str, dict[str, float]] = {}

    def fit(self, train_candidates: pd.DataFrame) -> "ItemCFRecallScorer":
        df = train_candidates.copy()
        df["impression_id"] = df["impression_id"].astype(str)
        df["candidate_news_id"] = df["candidate_news_id"].astype(str)
        df["history"] = df["history"].fillna("").astype(str)
        df["label"] = df["label"].fillna(0).astype(int)

        item_counts: Counter[str] = Counter()
        co_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)

        impression_base = df[["impression_id", "history"]].drop_duplicates("impression_id")
        clicked = (
            df[df["label"] == 1]
            .groupby("impression_id")["candidate_news_id"]
            .apply(list)
            .reset_index(name="clicked_news_ids")
        )
        baskets = impression_base.merge(clicked, on="impression_id", how="left")

        for _, row in baskets.iterrows():
            history_ids = self._parse_history(row["history"])
            clicked_ids = row["clicked_news_ids"]
            if not isinstance(clicked_ids, list):
                clicked_ids = []
            if not history_ids or not clicked_ids:
                continue

            history_ids = history_ids[-self.max_history_items :]
            unique_items = list(dict.fromkeys(history_ids + clicked_ids))
            item_counts.update(unique_items)

            for clicked_id in clicked_ids:
                for history_id in history_ids:
                    if clicked_id == history_id:
                        continue
                    co_counts[clicked_id][history_id] += 1
                    co_counts[history_id][clicked_id] += 1

        self.item_neighbors = {}
        for item_id, neighbors in co_counts.items():
            scored_neighbors = {}
            item_count = item_counts[item_id]
            for neighbor_id, co_count in neighbors.items():
                denom = np.sqrt(item_count * item_counts[neighbor_id])
                scored_neighbors[neighbor_id] = float(co_count / denom) if denom > 0 else 0.0
            top = sorted(scored_neighbors.items(), key=lambda x: x[1], reverse=True)[
                : self.top_neighbors
            ]
            self.item_neighbors[item_id] = dict(top)
        return self

    def score_candidates(self, candidates: pd.DataFrame) -> pd.DataFrame:
        scored = candidates[["impression_id", "candidate_news_id", "history"]].copy()
        scored["impression_id"] = scored["impression_id"].astype(str)
        scored["candidate_news_id"] = scored["candidate_news_id"].astype(str)
        scored["history"] = scored["history"].fillna("").astype(str)
        scored["itemcf_score"] = 0.0

        for _, group in scored.groupby("impression_id", sort=False):
            history = str(group["history"].iloc[0])
            candidate_ids = group["candidate_news_id"].astype(str)
            score_map = self._scores_for_history(history)
            if not score_map:
                continue
            scored.loc[group.index, "itemcf_score"] = candidate_ids.map(score_map).fillna(0.0).to_numpy()

        scored["recalled_by_itemcf"] = (scored["itemcf_score"] > 0).astype(np.int8)
        scored["itemcf_rank"] = self._rank_within_impression(scored, "itemcf_score")
        keep_cols = [
            "impression_id",
            "candidate_news_id",
            "itemcf_score",
            "itemcf_rank",
            "recalled_by_itemcf",
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
                ["impression_id", "itemcf_score", "_original_order"],
                ascending=[True, False, True],
            )
            .groupby("impression_id", sort=False)
            .head(top_n)
        )
        return selected.drop(columns=["_original_order"]).reset_index(drop=True)

    def _scores_for_history(self, history: str) -> dict[str, float]:
        if history in self._history_score_cache:
            return self._history_score_cache[history]

        history_ids = self._parse_history(history)[-self.max_history_items :]
        scores: defaultdict[str, float] = defaultdict(float)
        for position, history_id in enumerate(reversed(history_ids), start=1):
            weight = 1.0 / np.log2(position + 1.0)
            for neighbor_id, similarity in self.item_neighbors.get(history_id, {}).items():
                scores[neighbor_id] += similarity * weight

        out = dict(scores)
        self._history_score_cache[history] = out
        return out

    @staticmethod
    def _parse_history(history: str) -> list[str]:
        if not isinstance(history, str) or not history.strip():
            return []
        return history.split()

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
