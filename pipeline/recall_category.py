from __future__ import annotations

import numpy as np
import pandas as pd


class CategoryRecallScorer:
    """Score candidates by matching their category with the user's click history."""

    def __init__(self, category_weight: float = 0.4, subcategory_weight: float = 0.6):
        self.category_weight = category_weight
        self.subcategory_weight = subcategory_weight
        self.news_category_table: pd.DataFrame | None = None

    def fit(self, news: pd.DataFrame) -> "CategoryRecallScorer":
        table = news[["news_id", "category", "subcategory"]].copy()
        table["news_id"] = table["news_id"].astype(str)
        table["category"] = table["category"].fillna("unknown").astype(str)
        table["subcategory"] = table["subcategory"].fillna("unknown").astype(str)
        self.news_category_table = table.drop_duplicates("news_id")
        return self

    def score_candidates(self, candidates: pd.DataFrame) -> pd.DataFrame:
        if self.news_category_table is None:
            raise RuntimeError("CategoryRecallScorer must be fit before scoring.")

        prepared = candidates[["impression_id", "candidate_news_id", "history"]].copy()
        prepared["impression_id"] = prepared["impression_id"].astype(str)
        prepared["candidate_news_id"] = prepared["candidate_news_id"].astype(str)
        prepared["history"] = prepared["history"].fillna("").astype(str)

        impression_features = prepared[["impression_id", "history"]].drop_duplicates("impression_id")
        category_counts, subcategory_counts = self._build_history_category_counts(impression_features)

        scored = prepared[["impression_id", "candidate_news_id"]].merge(
            self.news_category_table,
            left_on="candidate_news_id",
            right_on="news_id",
            how="left",
        )
        scored["category"] = scored["category"].fillna("unknown")
        scored["subcategory"] = scored["subcategory"].fillna("unknown")

        scored = scored.merge(category_counts, on=["impression_id", "category"], how="left")
        scored = scored.merge(subcategory_counts, on=["impression_id", "subcategory"], how="left")
        scored["history_category_count"] = scored["history_category_count"].fillna(0.0)
        scored["history_subcategory_count"] = scored["history_subcategory_count"].fillna(0.0)
        scored["history_len_for_category"] = scored["history_len_for_category"].fillna(0.0)
        scored["history_len_for_subcategory"] = scored["history_len_for_subcategory"].fillna(0.0)

        category_denom = scored["history_len_for_category"].replace(0, 1)
        subcategory_denom = scored["history_len_for_subcategory"].replace(0, 1)
        scored["category_recall_score"] = scored["history_category_count"] / category_denom
        scored["subcategory_recall_score"] = scored["history_subcategory_count"] / subcategory_denom
        scored["category_recall_combined_score"] = (
            self.category_weight * scored["category_recall_score"]
            + self.subcategory_weight * scored["subcategory_recall_score"]
        )
        scored["recalled_by_category"] = (
            scored["category_recall_combined_score"] > 0
        ).astype(np.int8)

        keep_cols = [
            "impression_id",
            "candidate_news_id",
            "category_recall_score",
            "subcategory_recall_score",
            "category_recall_combined_score",
            "recalled_by_category",
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
                ["impression_id", "category_recall_combined_score", "_original_order"],
                ascending=[True, False, True],
            )
            .groupby("impression_id", sort=False)
            .head(top_n)
        )
        return selected.drop(columns=["_original_order"]).reset_index(drop=True)

    def _build_history_category_counts(
        self, impressions: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        exploded = impressions.copy()
        exploded["history_news_id"] = exploded["history"].str.split()
        exploded = exploded.explode("history_news_id").dropna(subset=["history_news_id"])
        if exploded.empty:
            return (
                pd.DataFrame(
                    columns=[
                        "impression_id",
                        "category",
                        "history_category_count",
                        "history_len_for_category",
                    ]
                ),
                pd.DataFrame(
                    columns=[
                        "impression_id",
                        "subcategory",
                        "history_subcategory_count",
                        "history_len_for_subcategory",
                    ]
                ),
            )

        history_news = exploded.merge(
            self.news_category_table,
            left_on="history_news_id",
            right_on="news_id",
            how="left",
        )
        history_news["category"] = history_news["category"].fillna("unknown")
        history_news["subcategory"] = history_news["subcategory"].fillna("unknown")

        category_counts = (
            history_news.groupby(["impression_id", "category"])
            .size()
            .reset_index(name="history_category_count")
        )
        subcategory_counts = (
            history_news.groupby(["impression_id", "subcategory"])
            .size()
            .reset_index(name="history_subcategory_count")
        )
        history_lengths = (
            history_news.groupby("impression_id")
            .size()
            .reset_index(name="history_len")
        )
        category_counts = category_counts.merge(history_lengths, on="impression_id", how="left")
        subcategory_counts = subcategory_counts.merge(history_lengths, on="impression_id", how="left")
        category_counts = category_counts.rename(columns={"history_len": "history_len_for_category"})
        subcategory_counts = subcategory_counts.rename(
            columns={"history_len": "history_len_for_subcategory"}
        )
        return category_counts, subcategory_counts
