from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline.config import PipelineConfig
from pipeline.data_prepare import load_train_valid_data
from pipeline.recall_bm25 import BM25RecallScorer
from pipeline.recall_category import CategoryRecallScorer
from pipeline.recall_entity_embedding import EntityEmbeddingRecallScorer
from pipeline.recall_itemcf import ItemCFRecallScorer
from pipeline.recall_popularity import PopularityRecallScorer
from pipeline.recall_sentence_embedding import SentenceEmbeddingRecallScorer
from pipeline.recall_tfidf import TfidfRecallScorer


RECALL_SCORE_COLUMNS = {
    "tfidf": "tfidf_score",
    "bm25": "bm25_score",
    "popularity": "popularity_score",
    "category": "category_recall_combined_score",
    "itemcf": "itemcf_score",
    "entity_embedding": "entity_embedding_score",
    "sentence_embedding": "sentence_embedding_score",
}


def sample_impressions(df: pd.DataFrame, max_impressions: int | None) -> pd.DataFrame:
    if max_impressions is None or max_impressions <= 0:
        return df
    keep_ids = df["impression_id"].drop_duplicates().head(max_impressions)
    return df[df["impression_id"].isin(keep_ids)].reset_index(drop=True)


def format_duration(seconds: float) -> str:
    if seconds >= 60:
        return f"{seconds / 60:.2f} min"
    return f"{seconds:.2f} sec"


def cache_path(config: PipelineConfig, recall_name: str, split: str, max_impressions: int | None) -> Path:
    limit = "full" if max_impressions is None or max_impressions <= 0 else str(max_impressions)
    return config.cache_dir / f"{split}_{recall_name}_recall_scores_{limit}.csv"


def load_cached_scores(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, dtype={"impression_id": str, "candidate_news_id": str})


def write_scores(scores: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(path, index=False)


def score_recall(
    recall_name: str,
    config: PipelineConfig,
    train_candidates: pd.DataFrame,
    valid_candidates: pd.DataFrame,
    news: pd.DataFrame,
    max_train_impressions: int | None,
    max_valid_impressions: int | None,
) -> pd.DataFrame:
    valid_cache = cache_path(config, recall_name, "valid", max_valid_impressions)
    cached = load_cached_scores(valid_cache)
    if cached is not None:
        return cached

    train_cache = cache_path(config, recall_name, "train", max_train_impressions)
    train_scores = load_cached_scores(train_cache)

    if recall_name == "tfidf":
        if TfidfRecallScorer.artifacts_exist(config.cache_dir):
            scorer = TfidfRecallScorer.load_artifacts(config.cache_dir, config.tfidf_params)
        else:
            scorer = TfidfRecallScorer(config.tfidf_params).fit(news)
            scorer.save_artifacts(config.cache_dir)
        if train_scores is None:
            train_scores = scorer.score_candidates(train_candidates)
            write_scores(train_scores, train_cache)
        valid_scores = scorer.score_candidates(valid_candidates)
    elif recall_name == "bm25":
        scorer = BM25RecallScorer(config.bm25_params).fit(news)
        if train_scores is None:
            train_scores = scorer.score_candidates(train_candidates)
            write_scores(train_scores, train_cache)
        valid_scores = scorer.score_candidates(valid_candidates)
    elif recall_name == "popularity":
        scorer = PopularityRecallScorer().fit(train_candidates, news)
        if train_scores is None:
            train_scores = scorer.score_candidates(train_candidates)
            write_scores(train_scores, train_cache)
        valid_scores = scorer.score_candidates(valid_candidates)
    elif recall_name == "category":
        scorer = CategoryRecallScorer().fit(news)
        if train_scores is None:
            train_scores = scorer.score_candidates(train_candidates)
            write_scores(train_scores, train_cache)
        valid_scores = scorer.score_candidates(valid_candidates)
    elif recall_name == "itemcf":
        scorer = ItemCFRecallScorer().fit(train_candidates)
        if train_scores is None:
            train_scores = scorer.score_candidates(train_candidates)
            write_scores(train_scores, train_cache)
        valid_scores = scorer.score_candidates(valid_candidates)
    elif recall_name == "entity_embedding":
        scorer = EntityEmbeddingRecallScorer(
            [config.train_entity_embedding_path, config.valid_entity_embedding_path]
        ).fit(news)
        if train_scores is None:
            train_scores = scorer.score_candidates(train_candidates)
            write_scores(train_scores, train_cache)
        valid_scores = scorer.score_candidates(valid_candidates)
    elif recall_name == "sentence_embedding":
        scorer = SentenceEmbeddingRecallScorer(config.sentence_embedding_path).fit(news)
        if train_scores is None:
            train_scores = scorer.score_candidates(train_candidates)
            write_scores(train_scores, train_cache)
        valid_scores = scorer.score_candidates(valid_candidates)
    else:
        raise ValueError(f"Unknown recall: {recall_name}")

    write_scores(valid_scores, valid_cache)
    return valid_scores


def select_top_n(
    scores: pd.DataFrame,
    score_col: str,
    top_n: int,
    include_zero_score: bool = False,
) -> pd.DataFrame:
    ranked = scores[["impression_id", "candidate_news_id", score_col, "label"]].copy()
    if not include_zero_score:
        ranked = ranked[ranked[score_col].fillna(0.0) > 0].copy()
    ranked["_original_order"] = range(len(ranked))
    return (
        ranked.sort_values(
            ["impression_id", score_col, "_original_order"],
            ascending=[True, False, True],
        )
        .groupby("impression_id", sort=False)
        .head(top_n)
        .drop(columns=["_original_order"])
        .reset_index(drop=True)
    )


def evaluate_selected(
    selected: pd.DataFrame,
    candidates: pd.DataFrame,
    recall_name: str,
    top_n: int,
) -> dict:
    positives = candidates[candidates["label"] == 1][["impression_id", "candidate_news_id"]]
    selected_keys = selected[["impression_id", "candidate_news_id"]]
    kept_positive = positives.merge(selected_keys, on=["impression_id", "candidate_news_id"], how="inner")

    positive_impressions = positives["impression_id"].nunique()
    hit_impressions = kept_positive["impression_id"].nunique()
    total_positives = len(positives)

    return {
        "recall": recall_name,
        "top_n": top_n,
        "impressions_total": int(candidates["impression_id"].nunique()),
        "positive_impressions": int(positive_impressions),
        "candidate_rows_total": int(len(candidates)),
        "candidate_rows_kept": int(len(selected)),
        "candidate_keep_rate": round(len(selected) / len(candidates), 6) if len(candidates) else 0.0,
        "positive_items_total": int(total_positives),
        "positive_items_kept": int(len(kept_positive)),
        "hit_rate": round(hit_impressions / positive_impressions, 6) if positive_impressions else 0.0,
        "positive_keep_rate": round(len(kept_positive) / total_positives, 6) if total_positives else 0.0,
        "avg_candidates_kept_per_impression": round(
            len(selected) / candidates["impression_id"].nunique(), 4
        )
        if candidates["impression_id"].nunique()
        else 0.0,
    }


def score_diagnostics(scores: pd.DataFrame, recall_name: str, score_col: str) -> dict:
    values = scores[score_col].fillna(0.0)
    return {
        "recall": recall_name,
        "score_col": score_col,
        "rows": int(len(scores)),
        "nonzero_score_rate": round(float((values != 0).mean()), 6) if len(scores) else 0.0,
        "positive_score_rate": round(float((values > 0).mean()), 6) if len(scores) else 0.0,
        "min_score": float(values.min()) if len(scores) else 0.0,
        "mean_score": float(values.mean()) if len(scores) else 0.0,
        "max_score": float(values.max()) if len(scores) else 0.0,
    }


def evaluate_union(
    selected_by_recall: dict[str, pd.DataFrame],
    candidates: pd.DataFrame,
    top_n: int,
) -> dict:
    frames = []
    for recall_name, selected in selected_by_recall.items():
        frame = selected[["impression_id", "candidate_news_id"]].copy()
        frame["recall"] = recall_name
        frames.append(frame)
    if not frames:
        return {}

    union = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(["impression_id", "candidate_news_id"])
        .reset_index(drop=True)
    )
    return evaluate_selected(union, candidates, "union", top_n)


def pairwise_overlap(selected_by_recall: dict[str, pd.DataFrame], top_n: int) -> list[dict]:
    names = list(selected_by_recall)
    records = []
    for i, left_name in enumerate(names):
        left = selected_by_recall[left_name][["impression_id", "candidate_news_id"]].drop_duplicates()
        for right_name in names[i + 1 :]:
            right = selected_by_recall[right_name][["impression_id", "candidate_news_id"]].drop_duplicates()
            intersection = left.merge(right, on=["impression_id", "candidate_news_id"], how="inner")
            union_size = len(pd.concat([left, right], ignore_index=True).drop_duplicates())
            records.append(
                {
                    "left_recall": left_name,
                    "right_recall": right_name,
                    "top_n": top_n,
                    "left_rows": int(len(left)),
                    "right_rows": int(len(right)),
                    "intersection_rows": int(len(intersection)),
                    "jaccard": round(len(intersection) / union_size, 6) if union_size else 0.0,
                }
            )
    return records


def run_recall_eval(
    config: PipelineConfig,
    recalls: list[str],
    top_ns: list[int],
    max_train_impressions: int | None,
    max_valid_impressions: int | None,
    include_zero_score: bool = False,
) -> dict:
    started_at = time.perf_counter()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.cache_dir.mkdir(parents=True, exist_ok=True)

    train_candidates, valid_candidates, news = load_train_valid_data(config)
    train_candidates = sample_impressions(train_candidates, max_train_impressions)
    valid_candidates = sample_impressions(valid_candidates, max_valid_impressions)

    recall_scores: dict[str, pd.DataFrame] = {}
    diagnostics = []
    timings = {}

    for recall_name in recalls:
        section_started = time.perf_counter()
        scores = score_recall(
            recall_name,
            config,
            train_candidates,
            valid_candidates,
            news,
            max_train_impressions,
            max_valid_impressions,
        )
        recall_scores[recall_name] = scores
        diagnostics.append(score_diagnostics(scores, recall_name, RECALL_SCORE_COLUMNS[recall_name]))
        timings[f"{recall_name}_score"] = format_duration(time.perf_counter() - section_started)

    records = []
    union_records = []
    overlap_records = []
    for top_n in top_ns:
        selected_by_recall = {}
        for recall_name, scores in recall_scores.items():
            selected = select_top_n(
                scores,
                RECALL_SCORE_COLUMNS[recall_name],
                top_n,
                include_zero_score=include_zero_score,
            )
            selected_by_recall[recall_name] = selected
            records.append(evaluate_selected(selected, valid_candidates, recall_name, top_n))

        union_record = evaluate_union(selected_by_recall, valid_candidates, top_n)
        if union_record:
            union_records.append(union_record)
        overlap_records.extend(pairwise_overlap(selected_by_recall, top_n))

    metrics_df = pd.DataFrame(records)
    union_df = pd.DataFrame(union_records)
    overlap_df = pd.DataFrame(overlap_records)
    diagnostics_df = pd.DataFrame(diagnostics)

    metrics_df.to_csv(config.output_dir / "recall_metrics.csv", index=False)
    union_df.to_csv(config.output_dir / "recall_union_metrics.csv", index=False)
    overlap_df.to_csv(config.output_dir / "recall_pairwise_overlap.csv", index=False)
    diagnostics_df.to_csv(config.output_dir / "recall_score_diagnostics.csv", index=False)

    result = {
        "recalls": recalls,
        "top_ns": top_ns,
        "include_zero_score": include_zero_score,
        "output_dir": str(config.output_dir),
        "metrics_path": str(config.output_dir / "recall_metrics.csv"),
        "union_metrics_path": str(config.output_dir / "recall_union_metrics.csv"),
        "pairwise_overlap_path": str(config.output_dir / "recall_pairwise_overlap.csv"),
        "score_diagnostics_path": str(config.output_dir / "recall_score_diagnostics.csv"),
        "timings": timings,
    }
    result["timings"]["total"] = format_duration(time.perf_counter() - started_at)
    with (config.output_dir / "recall_eval_summary.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate recall-stage topN coverage.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-dir", default="outputs/recall_eval")
    parser.add_argument("--cache-dir", default="outputs/pipeline_cache")
    parser.add_argument(
        "--recalls",
        nargs="+",
        choices=list(RECALL_SCORE_COLUMNS),
        default=list(RECALL_SCORE_COLUMNS),
        help="Recall sources to evaluate.",
    )
    parser.add_argument(
        "--top-ns",
        nargs="+",
        type=int,
        default=[20, 50, 100],
        help="topN values for recall-stage evaluation.",
    )
    parser.add_argument("--max-train-impressions", type=int, default=None)
    parser.add_argument("--max-valid-impressions", type=int, default=None)
    parser.add_argument(
        "--include-zero-score",
        action="store_true",
        help="Allow zero-score candidates to fill topN. Default evaluates only score > 0.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
        project_root=Path(args.project_root).resolve(),
        output_dir=Path(args.output_dir),
        cache_dir=Path(args.cache_dir),
    )
    result = run_recall_eval(
        config=config,
        recalls=args.recalls,
        top_ns=args.top_ns,
        max_train_impressions=args.max_train_impressions,
        max_valid_impressions=args.max_valid_impressions,
        include_zero_score=args.include_zero_score,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
