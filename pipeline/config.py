from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


SEED = 42

NEWS_COLUMNS = [
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]

BEHAVIORS_COLUMNS = ["impression_id", "user_id", "time", "history", "impressions"]

TFIDF_PARAMS = {
    "max_features": 30_000,
    "ngram_range": (1, 2),
    "sublinear_tf": True,
    "min_df": 2,
    "stop_words": "english",
    "token_pattern": r"(?u)\b\w+\b",
}

BM25_PARAMS = {
    "vectorizer": {
        "max_features": 50_000,
        "ngram_range": (1, 1),
        "min_df": 2,
        "stop_words": "english",
        "token_pattern": r"(?u)\b\w+\b",
    },
    "k1": 1.5,
    "b": 0.75,
}

LIGHTGBM_BEST_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "n_estimators": 400,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.7,
    "max_depth": -1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "n_jobs": -1,
    "bagging_freq": 1,
    "verbosity": -1,
}

LGBM_SEARCH_SPACE = {
    "num_leaves": [31, 63, 127],
    "min_child_samples": [20, 50, 100],
    "learning_rate": [0.03, 0.05, 0.1],
    "feature_fraction": [0.7, 0.8, 0.9],
    "bagging_fraction": [0.7, 0.8, 0.9],
    "max_depth": [6, 8, -1],
}

FEATURE_COLUMNS = [
    "tfidf_score",
    "bm25_score",
    "bm25_rank",
    "recalled_by_bm25",
    "global_popularity_score",
    "recent_popularity_score",
    "category_popularity_score",
    "popularity_score",
    "recalled_by_popularity",
    "category_recall_score",
    "subcategory_recall_score",
    "category_recall_combined_score",
    "recalled_by_category",
    "itemcf_score",
    "itemcf_rank",
    "recalled_by_itemcf",
    "entity_embedding_score",
    "entity_embedding_rank",
    "recalled_by_entity_embedding",
    "sentence_embedding_score",
    "sentence_embedding_rank",
    "recalled_by_sentence_embedding",
    "recalled_by_tfidf",
    "recalled_by_num_sources",
    "recall_source_overlap_count",
    "max_recall_score",
    "mean_recall_score",
    "tfidf_rank",
    "popularity_rank",
    "category_rank",
    "best_recall_rank",
    "mean_recall_rank",
    "history_len",
    "history_unique_len",
    "hour",
    "weekday",
    "is_weekend",
    "title_word_count",
    "abstract_word_count",
    "title_char_count",
    "abstract_char_count",
    "news_train_impressions",
    "news_train_clicks",
    "news_train_ctr",
    "history_category_count",
    "history_subcategory_count",
    "history_category_ratio",
    "history_subcategory_ratio",
    "category",
    "subcategory",
]

CATEGORICAL_COLUMNS = ["category", "subcategory"]

RECALL_SOURCES = [
    "tfidf",
    "bm25",
    "popularity",
    "category",
    "itemcf",
    "entity_embedding",
    "sentence_embedding",
]

HYBRID_RECALL_FEATURE_COLUMNS = [
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


@dataclass
class PipelineConfig:
    project_root: Path = field(default_factory=lambda: Path.cwd())
    output_dir: Path = field(default_factory=lambda: Path("outputs/pipeline"))
    cache_dir: Path = field(default_factory=lambda: Path("outputs/pipeline_cache"))
    top_k: int = 10
    recall_top_k: int | None = None
    use_tfidf_score: bool = False
    use_bm25_score: bool = False
    use_popularity_score: bool = False
    use_category_score: bool = False
    use_itemcf_score: bool = False
    use_entity_embedding_score: bool = False
    use_sentence_embedding_score: bool = False
    use_hybrid_recall_features: bool = False
    hybrid_recall_top_n: int | None = None
    hybrid_recall_sources: list[str] = field(
        default_factory=lambda: ["tfidf", "entity_embedding", "category"]
    )
    hybrid_include_zero_score: bool = False
    seed: int = SEED
    tfidf_params: dict = field(default_factory=lambda: dict(TFIDF_PARAMS))
    bm25_params: dict = field(default_factory=lambda: dict(BM25_PARAMS))
    lgbm_params: dict = field(default_factory=lambda: dict(LIGHTGBM_BEST_PARAMS))
    lgbm_search_space: dict = field(default_factory=lambda: dict(LGBM_SEARCH_SPACE))
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    categorical_columns: list[str] = field(default_factory=lambda: list(CATEGORICAL_COLUMNS))

    @property
    def train_news_path(self) -> Path:
        return self.project_root / "data/train/news.tsv"

    @property
    def valid_news_path(self) -> Path:
        return self.project_root / "data/valid/news.tsv"

    @property
    def train_behaviors_path(self) -> Path:
        return self.project_root / "data/train/behaviors.tsv"

    @property
    def valid_behaviors_path(self) -> Path:
        return self.project_root / "data/valid/behaviors.tsv"

    @property
    def shared_train_candidates_path(self) -> Path:
        return self.project_root / "outputs/shared/train_candidates.csv"

    @property
    def shared_valid_candidates_path(self) -> Path:
        return self.project_root / "outputs/shared/valid_candidates.csv"

    @property
    def model_path(self) -> Path:
        return self.output_dir / "lightgbm_model.txt"

    @property
    def train_entity_embedding_path(self) -> Path:
        return self.project_root / "data/train/entity_embedding.vec"

    @property
    def valid_entity_embedding_path(self) -> Path:
        return self.project_root / "data/valid/entity_embedding.vec"

    @property
    def sentence_embedding_path(self) -> Path:
        return (
            self.cache_dir
            / "sentence_embeddings_sentence-transformers_all-MiniLM-L6-v2.npz"
        )

    def __post_init__(self) -> None:
        unknown_sources = set(self.hybrid_recall_sources) - set(RECALL_SOURCES)
        if unknown_sources:
            raise ValueError(f"Unknown hybrid recall sources: {sorted(unknown_sources)}")

        if not self.use_tfidf_score:
            self.feature_columns = [
                col for col in self.feature_columns if col != "tfidf_score"
            ]
        if not self.use_bm25_score:
            bm25_cols = {
                "bm25_score",
                "bm25_rank",
                "recalled_by_bm25",
            }
            self.feature_columns = [
                col for col in self.feature_columns if col not in bm25_cols
            ]
        if not self.use_popularity_score:
            popularity_cols = {
                "global_popularity_score",
                "recent_popularity_score",
                "category_popularity_score",
                "popularity_score",
                "recalled_by_popularity",
            }
            self.feature_columns = [
                col for col in self.feature_columns if col not in popularity_cols
            ]
        if not self.use_category_score:
            category_recall_cols = {
                "category_recall_score",
                "subcategory_recall_score",
                "category_recall_combined_score",
                "recalled_by_category",
            }
            self.feature_columns = [
                col for col in self.feature_columns if col not in category_recall_cols
            ]
        if not self.use_itemcf_score:
            itemcf_cols = {
                "itemcf_score",
                "itemcf_rank",
                "recalled_by_itemcf",
            }
            self.feature_columns = [
                col for col in self.feature_columns if col not in itemcf_cols
            ]
        if not self.use_entity_embedding_score:
            entity_embedding_cols = {
                "entity_embedding_score",
                "entity_embedding_rank",
                "recalled_by_entity_embedding",
            }
            self.feature_columns = [
                col for col in self.feature_columns if col not in entity_embedding_cols
            ]
        if not self.use_sentence_embedding_score:
            sentence_embedding_cols = {
                "sentence_embedding_score",
                "sentence_embedding_rank",
                "recalled_by_sentence_embedding",
            }
            self.feature_columns = [
                col for col in self.feature_columns if col not in sentence_embedding_cols
            ]
        if self.use_hybrid_recall_features:
            for col in HYBRID_RECALL_FEATURE_COLUMNS:
                if col not in self.feature_columns:
                    self.feature_columns.append(col)
        else:
            hybrid_recall_cols = set(HYBRID_RECALL_FEATURE_COLUMNS)
            self.feature_columns = [
                col for col in self.feature_columns if col not in hybrid_recall_cols
            ]
