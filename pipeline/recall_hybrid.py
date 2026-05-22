from __future__ import annotations

import numpy as np
import pandas as pd


RECALL_SCORE_COLUMNS = {
    "tfidf": "tfidf_score",
    "bm25": "bm25_score",
    "popularity": "popularity_score",
    "category": "category_recall_combined_score",
    "itemcf": "itemcf_score",
    "entity_embedding": "entity_embedding_score",
    "sentence_embedding": "sentence_embedding_score",
}


def select_recall_top_n(
    scores: pd.DataFrame,
    score_col: str,
    top_n: int,
    include_zero_score: bool = False,
) -> pd.DataFrame:
    if top_n <= 0:
        raise ValueError("top_n must be positive.")
    if score_col not in scores.columns:
        raise ValueError(f"Missing recall score column: {score_col}")

    selected = scores[["impression_id", "candidate_news_id", score_col]].copy()
    selected["impression_id"] = selected["impression_id"].astype(str)
    selected["candidate_news_id"] = selected["candidate_news_id"].astype(str)
    selected[score_col] = selected[score_col].fillna(0.0).astype(float)
    selected["_original_order"] = np.arange(len(selected))

    if not include_zero_score:
        selected = selected[selected[score_col] > 0].copy()

    selected = (
        selected.sort_values(
            ["impression_id", score_col, "_original_order"],
            ascending=[True, False, True],
        )
        .groupby("impression_id", sort=False)
        .head(top_n)
        .drop(columns=["_original_order"])
        .reset_index(drop=True)
    )
    return selected


def build_hybrid_recall_pool(
    candidates: pd.DataFrame,
    scores_by_recall: dict[str, pd.DataFrame],
    top_n: int | dict[str, int],
    include_zero_score: bool = False,
) -> tuple[pd.DataFrame, dict]:
    top_n_by_source = _resolve_top_n_by_source(scores_by_recall, top_n)

    candidates = candidates.copy()
    candidates["impression_id"] = candidates["impression_id"].astype(str)
    candidates["candidate_news_id"] = candidates["candidate_news_id"].astype(str)
    before = len(candidates)

    selected_parts = []
    source_stats = {}
    for source, scores in scores_by_recall.items():
        if scores is None:
            continue
        score_col = RECALL_SCORE_COLUMNS[source]
        source_top_n = top_n_by_source[source]
        selected = select_recall_top_n(scores, score_col, source_top_n, include_zero_score)
        selected_parts.append(selected[["impression_id", "candidate_news_id"]])
        source_stats[source] = {
            "top_n": source_top_n,
            "candidates_after": int(len(selected)),
            "avg_candidates_per_impression": _avg_candidates_per_impression(selected),
        }

    if selected_parts:
        selected_keys = (
            pd.concat(selected_parts, ignore_index=True)
            .drop_duplicates(["impression_id", "candidate_news_id"])
            .reset_index(drop=True)
        )
        kept = candidates.merge(
            selected_keys,
            on=["impression_id", "candidate_news_id"],
            how="inner",
        )
    else:
        selected_keys = pd.DataFrame(columns=["impression_id", "candidate_news_id"])
        kept = candidates.iloc[0:0].copy()

    stats = {
        "enabled": True,
        "sources": list(scores_by_recall.keys()),
        "top_n": top_n if isinstance(top_n, int) else None,
        "top_n_by_source": top_n_by_source,
        "include_zero_score": include_zero_score,
        "candidates_before": int(before),
        "candidates_after": int(len(kept)),
        "candidate_keep_rate": round(len(kept) / before, 6) if before else 0.0,
        "avg_candidates_per_impression": _avg_candidates_per_impression(kept),
        "source_stats": source_stats,
    }
    stats.update(_positive_recall_stats(candidates, kept))
    return kept.reset_index(drop=True), stats


def _resolve_top_n_by_source(
    scores_by_recall: dict[str, pd.DataFrame],
    top_n: int | dict[str, int],
) -> dict[str, int]:
    sources = list(scores_by_recall.keys())
    if isinstance(top_n, int):
        if top_n <= 0:
            raise ValueError("top_n must be positive.")
        return {source: top_n for source in sources}

    missing_sources = set(sources) - set(top_n)
    if missing_sources:
        raise ValueError(
            f"Missing top_n quota for recall sources: {sorted(missing_sources)}"
        )

    top_n_by_source = {source: int(top_n[source]) for source in sources}
    invalid_sources = [source for source, value in top_n_by_source.items() if value <= 0]
    if invalid_sources:
        raise ValueError(f"top_n quotas must be positive: {sorted(invalid_sources)}")
    return top_n_by_source


def filter_scores_to_candidates(
    scores: pd.DataFrame | None,
    candidates: pd.DataFrame,
) -> pd.DataFrame | None:
    if scores is None:
        return None
    keys = candidates[["impression_id", "candidate_news_id"]].copy()
    keys["impression_id"] = keys["impression_id"].astype(str)
    keys["candidate_news_id"] = keys["candidate_news_id"].astype(str)
    return scores.merge(keys, on=["impression_id", "candidate_news_id"], how="inner")


def build_hybrid_recall_features(
    candidates: pd.DataFrame,
    scores_by_recall: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    keys = candidates[["impression_id", "candidate_news_id"]].copy()
    keys["impression_id"] = keys["impression_id"].astype(str)
    keys["candidate_news_id"] = keys["candidate_news_id"].astype(str)
    features = keys.copy()

    score_feature_cols = []
    rank_feature_cols = []
    recalled_cols = []
    for source, scores in scores_by_recall.items():
        if scores is None:
            continue
        score_col = RECALL_SCORE_COLUMNS[source]
        source_score_col = f"_{source}_score_for_hybrid"
        rank_col = f"{source}_rank"
        recalled_col = f"recalled_by_{source}"

        source_scores = scores[["impression_id", "candidate_news_id", score_col]].copy()
        source_scores["impression_id"] = source_scores["impression_id"].astype(str)
        source_scores["candidate_news_id"] = source_scores["candidate_news_id"].astype(str)
        source_scores[source_score_col] = source_scores[score_col].fillna(0.0).astype(float)
        source_scores = source_scores.drop(columns=[score_col])
        source_scores["_original_order"] = np.arange(len(source_scores))
        source_scores[rank_col] = (
            source_scores.sort_values(
                ["impression_id", source_score_col, "_original_order"],
                ascending=[True, False, True],
            )
            .groupby("impression_id", sort=False)
            .cumcount()
            + 1
        ).astype(np.int32)
        source_scores.loc[source_scores[source_score_col] <= 0, rank_col] = 0
        source_scores[recalled_col] = (source_scores[source_score_col] > 0).astype(np.int8)
        source_scores = source_scores.drop(columns=["_original_order"])

        features = features.merge(
            source_scores,
            on=["impression_id", "candidate_news_id"],
            how="left",
        )
        score_feature_cols.append(source_score_col)
        rank_feature_cols.append(rank_col)
        recalled_cols.append(recalled_col)

    for col in score_feature_cols + rank_feature_cols + recalled_cols:
        if col in features.columns:
            features[col] = features[col].fillna(0)

    if recalled_cols:
        features["recalled_by_num_sources"] = features[recalled_cols].sum(axis=1).astype(np.int8)
    else:
        features["recalled_by_num_sources"] = 0
    features["recall_source_overlap_count"] = features["recalled_by_num_sources"]

    if score_feature_cols:
        features["max_recall_score"] = features[score_feature_cols].max(axis=1).astype(float)
        features["mean_recall_score"] = features[score_feature_cols].mean(axis=1).astype(float)
    else:
        features["max_recall_score"] = 0.0
        features["mean_recall_score"] = 0.0

    if rank_feature_cols:
        rank_matrix = features[rank_feature_cols].replace(0, np.nan)
        features["best_recall_rank"] = rank_matrix.min(axis=1).fillna(0).astype(np.int32)
        features["mean_recall_rank"] = rank_matrix.mean(axis=1).fillna(0.0).astype(float)
    else:
        features["best_recall_rank"] = 0
        features["mean_recall_rank"] = 0.0

    output_cols = [
        "impression_id",
        "candidate_news_id",
        "recalled_by_tfidf",
        "recalled_by_bm25",
        "recalled_by_popularity",
        "recalled_by_category",
        "recalled_by_itemcf",
        "recalled_by_entity_embedding",
        "recalled_by_sentence_embedding",
        "recalled_by_num_sources",
        "recall_source_overlap_count",
        "max_recall_score",
        "mean_recall_score",
        "tfidf_rank",
        "bm25_rank",
        "popularity_rank",
        "category_rank",
        "itemcf_rank",
        "entity_embedding_rank",
        "sentence_embedding_rank",
        "best_recall_rank",
        "mean_recall_rank",
    ]
    for col in output_cols:
        if col not in features.columns:
            features[col] = 0
    return features[output_cols]


def _avg_candidates_per_impression(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return round(float(df.groupby("impression_id", sort=False).size().mean()), 4)


def _positive_recall_stats(before: pd.DataFrame, after: pd.DataFrame) -> dict:
    if "label" not in before.columns:
        return {
            "positive_keep_rate": None,
            "hit_rate": None,
        }

    positive_before = before[before["label"].astype(int) == 1][
        ["impression_id", "candidate_news_id"]
    ].copy()
    positive_total = len(positive_before)
    if positive_total == 0:
        return {
            "positive_keep_rate": 0.0,
            "hit_rate": 0.0,
        }

    kept_keys = after[["impression_id", "candidate_news_id"]].copy()
    positives_kept = positive_before.merge(
        kept_keys,
        on=["impression_id", "candidate_news_id"],
        how="inner",
    )
    positive_impressions = positive_before["impression_id"].nunique()
    hit_impressions = positives_kept["impression_id"].nunique()
    return {
        "positive_keep_rate": round(len(positives_kept) / positive_total, 6),
        "hit_rate": round(hit_impressions / positive_impressions, 6)
        if positive_impressions
        else 0.0,
    }
