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


@dataclass
class PipelineConfig:
    project_root: Path = field(default_factory=lambda: Path.cwd())
    output_dir: Path = field(default_factory=lambda: Path("outputs/pipeline"))
    cache_dir: Path = field(default_factory=lambda: Path("outputs/pipeline_cache"))
    top_k: int = 10
    recall_top_k: int | None = None
    use_tfidf_score: bool = False
    use_popularity_score: bool = False
    use_category_score: bool = False
    use_itemcf_score: bool = False
    use_entity_embedding_score: bool = False
    seed: int = SEED
    tfidf_params: dict = field(default_factory=lambda: dict(TFIDF_PARAMS))
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

    def __post_init__(self) -> None:
        if not self.use_tfidf_score:
            self.feature_columns = [
                col for col in self.feature_columns if col != "tfidf_score"
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
