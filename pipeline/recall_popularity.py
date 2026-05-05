from __future__ import annotations

import numpy as np
import pandas as pd


class PopularityRecallScorer:
    """Build popularity-based recall scores from train candidate logs."""

    def __init__(self, recent_window_days: float = 3.0):
        self.recent_window_days = recent_window_days
        self.popularity_table: pd.DataFrame | None = None

    def fit(self, train_candidates: pd.DataFrame, news: pd.DataFrame | None = None) -> "PopularityRecallScorer":
        df = train_candidates.copy()
        df["candidate_news_id"] = df["candidate_news_id"].astype(str)
        df["label"] = df["label"].fillna(0).astype(int)

        popularity = (
            df.groupby("candidate_news_id")["label"]
            .agg(
                popularity_impressions="size",
                popularity_clicks="sum",
            )
            .reset_index()
            .rename(columns={"candidate_news_id": "news_id"})
        )
        popularity["global_popularity_score"] = self._minmax_log_scale(
            popularity["popularity_clicks"]
        )
        popularity["popularity_ctr_score"] = (
            popularity["popularity_clicks"] + 1.0
        ) / (popularity["popularity_impressions"] + 20.0)

        recent = self._build_recent_popularity(df)
        popularity = popularity.merge(recent, on="news_id", how="left")
        popularity["recent_popularity_score"] = popularity["recent_popularity_score"].fillna(0.0)

        if news is not None and not news.empty:
            popularity = self._add_category_popularity(popularity, news)
        else:
            popularity["category_popularity_score"] = 0.0

        popularity["popularity_score"] = (
            0.45 * popularity["global_popularity_score"]
            + 0.35 * popularity["recent_popularity_score"]
            + 0.20 * popularity["category_popularity_score"]
        )
        self.popularity_table = popularity
        return self

    def score_candidates(self, candidates: pd.DataFrame) -> pd.DataFrame:
        if self.popularity_table is None:
            raise RuntimeError("PopularityRecallScorer must be fit before scoring.")

        scored = candidates[["impression_id", "candidate_news_id"]].copy()
        scored["impression_id"] = scored["impression_id"].astype(str)
        scored["candidate_news_id"] = scored["candidate_news_id"].astype(str)
        scored = scored.merge(
            self.popularity_table,
            left_on="candidate_news_id",
            right_on="news_id",
            how="left",
        ).drop(columns=["news_id"])

        fill_cols = [
            "popularity_impressions",
            "popularity_clicks",
            "global_popularity_score",
            "popularity_ctr_score",
            "recent_popularity_score",
            "category_popularity_score",
            "popularity_score",
        ]
        scored[fill_cols] = scored[fill_cols].fillna(0.0)
        scored["recalled_by_popularity"] = (scored["popularity_score"] > 0).astype(np.int8)
        if "label" in candidates.columns:
            scored["label"] = candidates["label"].astype(int).to_numpy()
        return scored

    def select_top_n(self, candidates: pd.DataFrame, top_n: int) -> pd.DataFrame:
        scored = self.score_candidates(candidates)
        scored["_original_order"] = np.arange(len(scored))
        selected = (
            scored.sort_values(
                ["impression_id", "popularity_score", "_original_order"],
                ascending=[True, False, True],
            )
            .groupby("impression_id", sort=False)
            .head(top_n)
        )
        return selected.drop(columns=["_original_order"]).reset_index(drop=True)

    def _build_recent_popularity(self, train_candidates: pd.DataFrame) -> pd.DataFrame:
        if "time" not in train_candidates.columns:
            return pd.DataFrame(columns=["news_id", "recent_popularity_score"])

        clicked = train_candidates[train_candidates["label"] == 1].copy()
        if clicked.empty:
            return pd.DataFrame(columns=["news_id", "recent_popularity_score"])

        clicked["time"] = pd.to_datetime(clicked["time"], errors="coerce")
        clicked = clicked.dropna(subset=["time"])
        if clicked.empty:
            return pd.DataFrame(columns=["news_id", "recent_popularity_score"])

        latest_time = clicked["time"].max()
        age_days = (latest_time - clicked["time"]).dt.total_seconds() / 86400.0
        window = max(float(self.recent_window_days), 1e-6)
        clicked["_recent_weight"] = np.exp(-age_days / window)
        recent = (
            clicked.groupby("candidate_news_id")["_recent_weight"]
            .sum()
            .reset_index(name="recent_popularity_raw")
            .rename(columns={"candidate_news_id": "news_id"})
        )
        recent["recent_popularity_score"] = self._minmax_log_scale(
            recent["recent_popularity_raw"]
        )
        return recent[["news_id", "recent_popularity_score"]]

    def _add_category_popularity(self, popularity: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
        news_features = news[["news_id", "category"]].copy()
        news_features["news_id"] = news_features["news_id"].astype(str)
        news_features["category"] = news_features["category"].fillna("unknown").astype(str)

        out = popularity.merge(news_features, on="news_id", how="left")
        out["category"] = out["category"].fillna("unknown")
        category_max = out.groupby("category")["popularity_clicks"].transform("max").replace(0, np.nan)
        out["category_popularity_score"] = (
            np.log1p(out["popularity_clicks"]) / np.log1p(category_max)
        ).fillna(0.0)
        return out.drop(columns=["category"])

    @staticmethod
    def _minmax_log_scale(values: pd.Series) -> pd.Series:
        scaled = np.log1p(values.astype(float))
        max_value = scaled.max()
        if not np.isfinite(max_value) or max_value <= 0:
            return pd.Series(np.zeros(len(values)), index=values.index)
        return scaled / max_value
