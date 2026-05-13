from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run recall features + LightGBM ranking pipeline.")
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
        "--hybrid-recall-top-n",
        type=int,
        default=None,
        help="Optional topN per recall source for multi-recall candidate generation before LightGBM reranking.",
    )
    parser.add_argument(
        "--hybrid-recalls",
        nargs="+",
        choices=[
            "tfidf",
            "bm25",
            "popularity",
            "category",
            "itemcf",
            "entity_embedding",
            "sentence_embedding",
        ],
        default=["tfidf", "entity_embedding", "category"],
        help="Recall sources to union when --hybrid-recall-top-n is set.",
    )
    parser.add_argument(
        "--hybrid-include-zero-score",
        action="store_true",
        help="Allow zero-score candidates to fill each recall source topN.",
    )
    parser.add_argument(
        "--use-tfidf-score",
        action="store_true",
        help="Include tfidf_score in LightGBM features for ablation.",
    )
    parser.add_argument(
        "--use-bm25-score",
        action="store_true",
        help="Include BM25 recall scores in LightGBM features for ablation.",
    )
    parser.add_argument(
        "--use-popularity-score",
        action="store_true",
        help="Include popularity recall scores in LightGBM features for ablation.",
    )
    parser.add_argument(
        "--use-category-score",
        action="store_true",
        help="Include category recall scores in LightGBM features for ablation.",
    )
    parser.add_argument(
        "--use-itemcf-score",
        action="store_true",
        help="Include ItemCF recall scores in LightGBM features for ablation.",
    )
    parser.add_argument(
        "--use-entity-embedding-score",
        action="store_true",
        help="Include entity embedding recall scores in LightGBM features for ablation.",
    )
    parser.add_argument(
        "--use-sentence-embedding-score",
        action="store_true",
        help="Include sentence embedding recall scores in LightGBM features for ablation.",
    )
    parser.add_argument(
        "--use-hybrid-recall-features",
        action="store_true",
        help="Include multi-recall interaction features in LightGBM without candidate cutoff.",
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
    parser.add_argument(
        "--ranker",
        choices=["binary", "lambdarank"],
        default="binary",
        help="LightGBM training objective: binary classifier baseline or LambdaRank ranker.",
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
        hybrid_recall_top_n=args.hybrid_recall_top_n,
        hybrid_recall_sources=args.hybrid_recalls,
        hybrid_include_zero_score=args.hybrid_include_zero_score,
        use_tfidf_score=args.use_tfidf_score,
        use_bm25_score=args.use_bm25_score,
        use_popularity_score=args.use_popularity_score,
        use_category_score=args.use_category_score,
        use_itemcf_score=args.use_itemcf_score,
        use_entity_embedding_score=args.use_entity_embedding_score,
        use_sentence_embedding_score=args.use_sentence_embedding_score,
        use_hybrid_recall_features=args.use_hybrid_recall_features,
        ranker=args.ranker,
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
