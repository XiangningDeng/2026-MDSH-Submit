from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.config import CATEGORICAL_COLUMNS, FEATURE_COLUMNS


def prepare_news_features(news: pd.DataFrame) -> pd.DataFrame:
    news = news.copy()
    news["news_id"] = news["news_id"].astype(str)
    news["category"] = news["category"].fillna("unknown").astype(str)
    news["subcategory"] = news["subcategory"].fillna("unknown").astype(str)
    news["title"] = news["title"].fillna("").astype(str)
    news["abstract"] = news["abstract"].fillna("").astype(str)
    news["title_word_count"] = news["title"].str.split().str.len().astype(np.int16)
    news["abstract_word_count"] = news["abstract"].str.split().str.len().astype(np.int16)
    news["title_char_count"] = news["title"].str.len().astype(np.int16)
    news["abstract_char_count"] = news["abstract"].str.len().astype(np.int16)
    return news[
        [
            "news_id",
            "category",
            "subcategory",
            "title_word_count",
            "abstract_word_count",
            "title_char_count",
            "abstract_char_count",
        ]
    ].copy()


def prepare_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    df = candidates.copy()
    df["impression_id"] = df["impression_id"].astype(str)
    df["user_id"] = df["user_id"].fillna("unknown").astype(str)
    df["candidate_news_id"] = df["candidate_news_id"].astype(str)
    df["history"] = df["history"].fillna("").astype(str)
    if "label" in df.columns:
        df["label"] = df["label"].astype(np.int8)
    return df


def build_impression_features(candidates: pd.DataFrame) -> pd.DataFrame:
    impressions = candidates[["impression_id", "time", "history"]].drop_duplicates("impression_id").copy()
    histories = impressions["history"].str.split()
    impressions["history_len"] = histories.str.len().fillna(0).astype(np.int16)
    impressions["history_unique_len"] = histories.apply(
        lambda x: len(set(x)) if isinstance(x, list) else 0
    ).astype(np.int16)
    timestamps = pd.to_datetime(impressions["time"], errors="coerce")
    impressions["hour"] = timestamps.dt.hour.fillna(-1).astype(np.int8)
    impressions["weekday"] = timestamps.dt.dayofweek.fillna(-1).astype(np.int8)
    impressions["is_weekend"] = impressions["weekday"].isin([5, 6]).astype(np.int8)
    return impressions


def build_history_match_tables(
    impression_features: pd.DataFrame, news_features: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    exploded = impression_features[["impression_id", "history"]].copy()
    exploded["history_news_id"] = exploded["history"].str.split()
    exploded = exploded.explode("history_news_id").dropna(subset=["history_news_id"])
    if exploded.empty:
        return (
            pd.DataFrame(columns=["impression_id", "category", "history_category_count"]),
            pd.DataFrame(columns=["impression_id", "subcategory", "history_subcategory_count"]),
        )

    history_news = exploded.merge(
        news_features[["news_id", "category", "subcategory"]],
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
    return category_counts, subcategory_counts


def build_popularity_features(train_candidates: pd.DataFrame) -> pd.DataFrame:
    popularity = (
        train_candidates.groupby("candidate_news_id")["label"]
        .agg(news_train_impressions="size", news_train_clicks="sum")
        .reset_index()
        .rename(columns={"candidate_news_id": "news_id"})
    )
    popularity["news_train_ctr"] = (
        popularity["news_train_clicks"] + 1.0
    ) / (popularity["news_train_impressions"] + 20.0)
    return popularity


def assemble_features(
    candidates: pd.DataFrame,
    impression_features: pd.DataFrame,
    news_features: pd.DataFrame,
    popularity_features: pd.DataFrame,
    category_counts: pd.DataFrame,
    subcategory_counts: pd.DataFrame,
    tfidf_scores: pd.DataFrame | None = None,
    popularity_scores: pd.DataFrame | None = None,
    category_scores: pd.DataFrame | None = None,
    itemcf_scores: pd.DataFrame | None = None,
    entity_embedding_scores: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = candidates.merge(
        impression_features.drop(columns=["history"]),
        on="impression_id",
        how="left",
    )
    df = df.merge(news_features, left_on="candidate_news_id", right_on="news_id", how="left")
    df = df.merge(popularity_features, on="news_id", how="left")
    df = df.merge(category_counts, on=["impression_id", "category"], how="left")
    df = df.merge(subcategory_counts, on=["impression_id", "subcategory"], how="left")
    if tfidf_scores is not None:
        score_cols = ["impression_id", "candidate_news_id", "tfidf_score"]
        df = df.merge(tfidf_scores[score_cols], on=["impression_id", "candidate_news_id"], how="left")
    else:
        df["tfidf_score"] = 0.0
    if popularity_scores is not None:
        score_cols = [
            "impression_id",
            "candidate_news_id",
            "global_popularity_score",
            "recent_popularity_score",
            "category_popularity_score",
            "popularity_score",
            "recalled_by_popularity",
        ]
        df = df.merge(
            popularity_scores[score_cols],
            on=["impression_id", "candidate_news_id"],
            how="left",
        )
    else:
        df["global_popularity_score"] = 0.0
        df["recent_popularity_score"] = 0.0
        df["category_popularity_score"] = 0.0
        df["popularity_score"] = 0.0
        df["recalled_by_popularity"] = 0
    if category_scores is not None:
        score_cols = [
            "impression_id",
            "candidate_news_id",
            "category_recall_score",
            "subcategory_recall_score",
            "category_recall_combined_score",
            "recalled_by_category",
        ]
        df = df.merge(
            category_scores[score_cols],
            on=["impression_id", "candidate_news_id"],
            how="left",
        )
    else:
        df["category_recall_score"] = 0.0
        df["subcategory_recall_score"] = 0.0
        df["category_recall_combined_score"] = 0.0
        df["recalled_by_category"] = 0
    if itemcf_scores is not None:
        score_cols = [
            "impression_id",
            "candidate_news_id",
            "itemcf_score",
            "itemcf_rank",
            "recalled_by_itemcf",
        ]
        df = df.merge(
            itemcf_scores[score_cols],
            on=["impression_id", "candidate_news_id"],
            how="left",
        )
    else:
        df["itemcf_score"] = 0.0
        df["itemcf_rank"] = 0
        df["recalled_by_itemcf"] = 0
    if entity_embedding_scores is not None:
        score_cols = [
            "impression_id",
            "candidate_news_id",
            "entity_embedding_score",
            "entity_embedding_rank",
            "recalled_by_entity_embedding",
        ]
        df = df.merge(
            entity_embedding_scores[score_cols],
            on=["impression_id", "candidate_news_id"],
            how="left",
        )
    else:
        df["entity_embedding_score"] = 0.0
        df["entity_embedding_rank"] = 0
        df["recalled_by_entity_embedding"] = 0

    fill_map = {
        "category": "unknown",
        "subcategory": "unknown",
        "title_word_count": 0,
        "abstract_word_count": 0,
        "title_char_count": 0,
        "abstract_char_count": 0,
        "news_train_impressions": 0,
        "news_train_clicks": 0,
        "news_train_ctr": 0.0,
        "history_category_count": 0,
        "history_subcategory_count": 0,
        "tfidf_score": 0.0,
        "global_popularity_score": 0.0,
        "recent_popularity_score": 0.0,
        "category_popularity_score": 0.0,
        "popularity_score": 0.0,
        "recalled_by_popularity": 0,
        "category_recall_score": 0.0,
        "subcategory_recall_score": 0.0,
        "category_recall_combined_score": 0.0,
        "recalled_by_category": 0,
        "itemcf_score": 0.0,
        "itemcf_rank": 0,
        "recalled_by_itemcf": 0,
        "entity_embedding_score": 0.0,
        "entity_embedding_rank": 0,
        "recalled_by_entity_embedding": 0,
    }
    df = df.fillna(fill_map)
    history_len_safe = df["history_len"].replace(0, 1)
    df["history_category_ratio"] = df["history_category_count"] / history_len_safe
    df["history_subcategory_ratio"] = df["history_subcategory_count"] / history_len_safe
    return df


class RankingFeatureBuilder:
    def __init__(
        self,
        feature_columns: list[str] | None = None,
        categorical_columns: list[str] | None = None,
    ):
        self.feature_columns = feature_columns or list(FEATURE_COLUMNS)
        self.categorical_columns = categorical_columns or list(CATEGORICAL_COLUMNS)
        self.news_features: pd.DataFrame | None = None
        self.popularity_features: pd.DataFrame | None = None

    def fit(self, train_candidates: pd.DataFrame, news: pd.DataFrame) -> "RankingFeatureBuilder":
        self.news_features = prepare_news_features(news)
        self.popularity_features = build_popularity_features(prepare_candidates(train_candidates))
        return self

    def transform(
        self,
        candidates: pd.DataFrame,
        tfidf_scores: pd.DataFrame | None = None,
        popularity_scores: pd.DataFrame | None = None,
        category_scores: pd.DataFrame | None = None,
        itemcf_scores: pd.DataFrame | None = None,
        entity_embedding_scores: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if self.news_features is None or self.popularity_features is None:
            raise RuntimeError("Feature builder must be fit before transform.")
        prepared = prepare_candidates(candidates)
        impression_features = build_impression_features(prepared)
        category_counts, subcategory_counts = build_history_match_tables(
            impression_features, self.news_features
        )
        features = assemble_features(
            prepared,
            impression_features,
            self.news_features,
            self.popularity_features,
            category_counts,
            subcategory_counts,
            tfidf_scores,
            popularity_scores,
            category_scores,
            itemcf_scores,
            entity_embedding_scores,
        )
        for col in self.categorical_columns:
            features[col] = features[col].astype("category")
        return features
