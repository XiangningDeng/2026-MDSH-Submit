from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TF-IDF recall + LightGBM ranking pipeline.")
    parser.add_argument("--mode", choices=["eval", "inference"], default="eval")
    parser.add_argument("--project-root", default=".", help="Repository root containing data/train and data/valid.")
    parser.add_argument("--output-dir", default="outputs/pipeline")
    parser.add_argument("--cache-dir", default="outputs/pipeline_cache")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--recall-top-k",
        type=int,
        default=None,
        help="Optional TF-IDF recall cutoff per impression before LightGBM reranking.",
    )
    parser.add_argument(
        "--no-tfidf-score",
        action="store_true",
        help="Exclude tfidf_score from LightGBM features for ablation.",
    )
    parser.add_argument(
        "--max-train-impressions",
        type=int,
        default=None,
        help="Optional quick-test limit on the number of train impressions.",
    )
    parser.add_argument(
        "--max-valid-impressions",
        type=int,
        default=None,
        help="Optional quick-test limit on the number of validation impressions.",
    )
    parser.add_argument(
        "--tune-lgbm",
        action="store_true",
        help="Run the same coordinate-search style tuning used in the LightGBM baseline notebook.",
    )
    args = parser.parse_args()

    from pipeline.config import PipelineConfig
    from pipeline.runner import RecommendationPipeline

    config = PipelineConfig(
        project_root=Path(args.project_root).resolve(),
        output_dir=Path(args.output_dir),
        cache_dir=Path(args.cache_dir),
        top_k=args.top_k,
        recall_top_k=args.recall_top_k,
        use_tfidf_score=not args.no_tfidf_score,
    )
    result = RecommendationPipeline(config).run(
        mode=args.mode,
        tune_lgbm=args.tune_lgbm,
        max_train_impressions=args.max_train_impressions,
        max_valid_impressions=args.max_valid_impressions,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
