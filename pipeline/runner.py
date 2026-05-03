from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from pipeline.config import PipelineConfig
from pipeline.data_prepare import load_train_valid_data
from pipeline.feature_builder import RankingFeatureBuilder
from pipeline.metrics import evaluate_predictions, write_metrics
from pipeline.rank_lgbm import LightGBMRanker, coordinate_search
from pipeline.recall_tfidf import TfidfRecallScorer


class RecommendationPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def _top_k(self, predictions: pd.DataFrame) -> pd.DataFrame:
        return (
            predictions.sort_values(["impression_id", "score"], ascending=[True, False])
            .groupby("impression_id", sort=False)
            .head(self.config.top_k)
            .reset_index(drop=True)
        )

    def _sample_impressions(self, df: pd.DataFrame, max_impressions: int | None) -> pd.DataFrame:
        if max_impressions is None or max_impressions <= 0:
            return df
        keep_ids = df["impression_id"].drop_duplicates().head(max_impressions)
        return df[df["impression_id"].isin(keep_ids)].reset_index(drop=True)

    @staticmethod
    def _apply_recall_top_k(
        candidates: pd.DataFrame,
        tfidf_scores: pd.DataFrame,
        recall_top_k: int | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        before = len(candidates)
        if recall_top_k is None or recall_top_k <= 0:
            return candidates, tfidf_scores, {
                "recall_top_k": None,
                "candidates_before": before,
                "candidates_after": before,
                "candidate_keep_rate": 1.0,
                "impressions_total": int(candidates["impression_id"].nunique()),
                "impressions_fully_kept": int(candidates["impression_id"].nunique()),
            }

        score_cols = ["impression_id", "candidate_news_id", "tfidf_score"]
        scored = candidates.merge(
            tfidf_scores[score_cols],
            on=["impression_id", "candidate_news_id"],
            how="left",
        )
        scored["tfidf_score"] = scored["tfidf_score"].fillna(0.0)
        scored["_original_order"] = range(len(scored))
        scored["_tfidf_rank"] = (
            scored.sort_values(
                ["impression_id", "tfidf_score", "_original_order"],
                ascending=[True, False, True],
            )
            .groupby("impression_id", sort=False)
            .cumcount()
            + 1
        )

        kept = (
            scored[scored["_tfidf_rank"] <= recall_top_k]
            .sort_values("_original_order")
            .drop(columns=["tfidf_score", "_original_order", "_tfidf_rank"])
            .reset_index(drop=True)
        )
        kept_scores = tfidf_scores.merge(
            kept[["impression_id", "candidate_news_id"]],
            on=["impression_id", "candidate_news_id"],
            how="inner",
        )

        counts = candidates.groupby("impression_id", sort=False).size()
        stats = {
            "recall_top_k": recall_top_k,
            "candidates_before": before,
            "candidates_after": len(kept),
            "candidate_keep_rate": round(len(kept) / before, 6) if before else 0.0,
            "impressions_total": int(counts.shape[0]),
            "impressions_fully_kept": int((counts <= recall_top_k).sum()),
        }
        return kept, kept_scores, stats

    def _tfidf_score_cache_path(self, split: str, max_impressions: int | None) -> str:
        limit = "full" if max_impressions is None or max_impressions <= 0 else str(max_impressions)
        return str(self.config.cache_dir / f"{split}_tfidf_scores_{limit}.csv")

    @staticmethod
    def _load_or_score_tfidf(
        scorer: TfidfRecallScorer,
        candidates: pd.DataFrame,
        cache_path: str,
    ) -> pd.DataFrame:
        path = pd.io.common.stringify_path(cache_path)
        try:
            return pd.read_csv(path, dtype={"impression_id": str, "candidate_news_id": str})
        except FileNotFoundError:
            scores = scorer.score_candidates(candidates)
            scores.to_csv(path, index=False)
            return scores

    def run(
        self,
        mode: str = "eval",
        tune_lgbm: bool = False,
        max_train_impressions: int | None = None,
        max_valid_impressions: int | None = None,
    ) -> dict:
        started_at = time.perf_counter()
        timings: dict[str, str] = {}

        def mark(name: str, section_started_at: float) -> float:
            now = time.perf_counter()
            timings[name] = self._format_duration(now - section_started_at)
            return now

        if mode not in {"eval", "inference"}:
            raise ValueError("mode must be either 'eval' or 'inference'")

        np.random.seed(self.config.seed)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        section_started_at = time.perf_counter()
        train_candidates, valid_candidates, news = load_train_valid_data(self.config)
        train_candidates = self._sample_impressions(train_candidates, max_train_impressions)
        valid_candidates = self._sample_impressions(valid_candidates, max_valid_impressions)
        section_started_at = mark("data_prepare", section_started_at)

        needs_tfidf = self.config.use_tfidf_score or (
            self.config.recall_top_k is not None and self.config.recall_top_k > 0
        )
        if needs_tfidf:
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)
            if TfidfRecallScorer.artifacts_exist(self.config.cache_dir):
                tfidf = TfidfRecallScorer.load_artifacts(self.config.cache_dir, self.config.tfidf_params)
            else:
                tfidf = TfidfRecallScorer(self.config.tfidf_params).fit(news)
                tfidf.save_artifacts(self.config.cache_dir)

            train_tfidf_scores = self._load_or_score_tfidf(
                tfidf,
                train_candidates,
                self._tfidf_score_cache_path("train", max_train_impressions),
            )
            valid_tfidf_scores = self._load_or_score_tfidf(
                tfidf,
                valid_candidates,
                self._tfidf_score_cache_path("valid", max_valid_impressions),
            )
            valid_tfidf_scores.to_csv(self.config.output_dir / "tfidf_recall_scores.csv", index=False)
        else:
            train_tfidf_scores = None
            valid_tfidf_scores = None
        section_started_at = mark("tfidf_recall", section_started_at)

        full_valid_candidates = valid_candidates
        if self.config.recall_top_k is not None and self.config.recall_top_k > 0:
            train_candidates, train_tfidf_scores, train_recall_stats = self._apply_recall_top_k(
                train_candidates,
                train_tfidf_scores,
                self.config.recall_top_k,
            )
            valid_candidates, valid_tfidf_scores, valid_recall_stats = self._apply_recall_top_k(
                valid_candidates,
                valid_tfidf_scores,
                self.config.recall_top_k,
            )
        else:
            train_recall_stats = {
                "recall_top_k": None,
                "candidates_before": len(train_candidates),
                "candidates_after": len(train_candidates),
                "candidate_keep_rate": 1.0,
                "impressions_total": int(train_candidates["impression_id"].nunique()),
                "impressions_fully_kept": int(train_candidates["impression_id"].nunique()),
            }
            valid_recall_stats = {
                "recall_top_k": None,
                "candidates_before": len(valid_candidates),
                "candidates_after": len(valid_candidates),
                "candidate_keep_rate": 1.0,
                "impressions_total": int(valid_candidates["impression_id"].nunique()),
                "impressions_fully_kept": int(valid_candidates["impression_id"].nunique()),
            }

        feature_builder = RankingFeatureBuilder(
            self.config.feature_columns,
            self.config.categorical_columns,
        ).fit(train_candidates, news)
        train_features = feature_builder.transform(train_candidates, train_tfidf_scores)
        valid_features = feature_builder.transform(valid_candidates, valid_tfidf_scores)
        section_started_at = mark("feature_build", section_started_at)

        lgbm_params = dict(self.config.lgbm_params)
        if tune_lgbm:
            lgbm_params, search_results = coordinate_search(
                lgbm_params,
                self.config.lgbm_search_space,
                train_features,
                valid_features,
                self.config.feature_columns,
                self.config.categorical_columns,
            )
            search_results.to_csv(self.config.output_dir / "lgbm_search_results.csv", index=False)
            section_started_at = mark("lgbm_tuning", section_started_at)

        ranker = LightGBMRanker(
            lgbm_params,
            self.config.feature_columns,
            self.config.categorical_columns,
        ).train(train_features, valid_features, verbose_eval=True)
        section_started_at = mark("lgbm_train", section_started_at)

        valid_scores = ranker.predict(valid_features)
        reranked_predictions = valid_features[["impression_id", "candidate_news_id"]].copy()
        if "label" in valid_features.columns:
            reranked_predictions["label"] = valid_features["label"].astype(int)
        reranked_predictions["score"] = valid_scores

        if self.config.recall_top_k is None or self.config.recall_top_k <= 0:
            predictions = reranked_predictions
        else:
            predictions = full_valid_candidates[
                ["impression_id", "candidate_news_id", "label"]
            ].copy()
            predictions["score"] = -1e9
            predictions = predictions.merge(
                reranked_predictions[["impression_id", "candidate_news_id", "score"]],
                on=["impression_id", "candidate_news_id"],
                how="left",
                suffixes=("", "_reranked"),
            )
            predictions["score"] = predictions["score_reranked"].fillna(predictions["score"])
            predictions = predictions.drop(columns=["score_reranked"])

        reranked_predictions.to_csv(
            self.config.output_dir / "reranked_candidates.csv",
            index=False,
        )
        predictions[["impression_id", "candidate_news_id", "score"]].to_csv(
            self.config.output_dir / "prediction.txt",
            index=False,
        )
        section_started_at = mark("lgbm_predict_and_write", section_started_at)

        top_k = self._top_k(predictions)
        top_k.to_csv(self.config.output_dir / f"top{self.config.top_k}_recommendations.csv", index=False)

        ranker.save_params(self.config.output_dir / "best_params.json")
        ranker.save_model(self.config.model_path)
        ranker.feature_importance().to_csv(
            self.config.output_dir / "feature_importance.csv",
            index=False,
        )

        result = {
            "prediction_path": str(self.config.output_dir / "prediction.txt"),
            "top_k_path": str(self.config.output_dir / f"top{self.config.top_k}_recommendations.csv"),
            "model_path": str(self.config.model_path),
            "recall": {
                "use_tfidf_score": self.config.use_tfidf_score,
                "train": train_recall_stats,
                "valid": valid_recall_stats,
            },
        }
        if mode == "eval":
            metrics = evaluate_predictions(predictions)
            write_metrics(metrics, self.config.output_dir / "metrics.json")
            result["metrics"] = metrics
        section_started_at = mark("metrics_and_artifacts", section_started_at)

        timings["total"] = self._format_duration(time.perf_counter() - started_at)
        result["timings"] = timings

        with (self.config.output_dir / "run_summary.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if seconds >= 60:
            return f"{seconds / 60:.2f} min"
        return f"{seconds:.2f} sec"
