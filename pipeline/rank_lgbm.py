from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

try:
    from lightgbm import LGBMClassifier, LGBMRanker, early_stopping, log_evaluation
except ImportError as exc:
    raise ImportError("Please install lightgbm first, for example: pip install lightgbm") from exc


class LightGBMRanker:
    def __init__(
        self,
        params: dict,
        feature_columns: list[str],
        categorical_columns: list[str],
        ranker_type: str = "binary",
    ):
        self.params = dict(params)
        self.feature_columns = list(feature_columns)
        self.categorical_columns = list(categorical_columns)
        self.ranker_type = ranker_type
        self.model: LGBMClassifier | LGBMRanker | None = None
        self.training_summary: dict = {"ranker": ranker_type}
        if self.ranker_type not in {"binary", "lambdarank"}:
            raise ValueError("ranker_type must be either 'binary' or 'lambdarank'.")

    def train(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame | None = None,
        verbose_eval: bool = True,
    ) -> "LightGBMRanker":
        callbacks = [early_stopping(stopping_rounds=30, first_metric_only=True)]
        if verbose_eval:
            callbacks.append(log_evaluation(period=50))

        if self.ranker_type == "lambdarank":
            return self._train_lambdarank(train_df, valid_df, callbacks)

        self.model = LGBMClassifier(**self.params)
        fit_kwargs = {
            "X": train_df[self.feature_columns],
            "y": train_df["label"],
            "eval_metric": "auc",
            "categorical_feature": self.categorical_columns,
            "callbacks": callbacks,
        }
        if valid_df is not None and "label" in valid_df.columns:
            fit_kwargs["eval_set"] = [(valid_df[self.feature_columns], valid_df["label"])]
        self.model.fit(**fit_kwargs)
        self.training_summary = {
            "ranker": "binary",
            "train_rows": int(len(train_df)),
            "valid_rows": int(len(valid_df)) if valid_df is not None else None,
        }
        return self

    def _train_lambdarank(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame | None,
        callbacks: list,
    ) -> "LightGBMRanker":
        train_rank_df, train_group, train_stats = prepare_lambdarank_frame(
            train_df,
            drop_invalid_groups=True,
        )
        if train_rank_df.empty:
            raise ValueError("No valid LambdaRank train groups after filtering.")

        self.model = LGBMRanker(**self.params)
        fit_kwargs = {
            "X": train_rank_df[self.feature_columns],
            "y": train_rank_df["label"],
            "group": train_group,
            "eval_metric": "ndcg",
            "eval_at": [5, 10],
            "categorical_feature": self.categorical_columns,
            "callbacks": callbacks,
        }

        valid_stats = None
        if valid_df is not None and "label" in valid_df.columns:
            valid_rank_df, valid_group, valid_stats = prepare_lambdarank_frame(
                valid_df,
                drop_invalid_groups=True,
            )
            if not valid_rank_df.empty:
                fit_kwargs["eval_set"] = [
                    (valid_rank_df[self.feature_columns], valid_rank_df["label"])
                ]
                fit_kwargs["eval_group"] = [valid_group]

        self.model.fit(**fit_kwargs)
        self.training_summary = {
            "ranker": "lambdarank",
            "train": train_stats,
            "valid_eval": valid_stats,
        }
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("LightGBM model must be trained before predict.")
        if self.ranker_type == "lambdarank":
            return pd.Series(
                self.model.predict(df[self.feature_columns]),
                index=df.index,
                name="score",
            )
        return pd.Series(
            self.model.predict_proba(df[self.feature_columns])[:, 1],
            index=df.index,
            name="score",
        )

    def feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("LightGBM model must be trained before feature_importance.")
        return pd.DataFrame(
            {
                "feature": self.feature_columns,
                "importance_gain": self.model.booster_.feature_importance(importance_type="gain"),
                "importance_split": self.model.booster_.feature_importance(importance_type="split"),
            }
        ).sort_values("importance_gain", ascending=False)

    def save_model(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("LightGBM model must be trained before save_model.")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.booster_.save_model(str(path))

    def save_params(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.params, f, ensure_ascii=False, indent=2)


def prepare_lambdarank_frame(
    df: pd.DataFrame,
    drop_invalid_groups: bool = True,
) -> tuple[pd.DataFrame, list[int], dict]:
    required = {"impression_id", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"LambdaRank frame missing columns: {missing}")

    sort_cols = ["impression_id"]
    if "candidate_pos" in df.columns:
        sort_cols.append("candidate_pos")
    else:
        sort_cols.append("candidate_news_id")

    ranked = df.sort_values(sort_cols, kind="mergesort").copy()
    group_stats = (
        ranked.groupby("impression_id", sort=False)["label"]
        .agg(group_size="size", positive_count="sum")
        .reset_index()
    )
    group_stats["negative_count"] = (
        group_stats["group_size"] - group_stats["positive_count"]
    )
    invalid_mask = (
        (group_stats["group_size"] < 2)
        | (group_stats["positive_count"] <= 0)
        | (group_stats["negative_count"] <= 0)
    )
    invalid_ids = set(group_stats.loc[invalid_mask, "impression_id"])

    if drop_invalid_groups and invalid_ids:
        ranked = ranked[~ranked["impression_id"].isin(invalid_ids)].copy()

    group_sizes = (
        ranked.groupby("impression_id", sort=False).size().astype(int).tolist()
        if not ranked.empty
        else []
    )
    summary = {
        "rows_before": int(len(df)),
        "groups_before": int(group_stats.shape[0]),
        "dropped_groups": int(len(invalid_ids)) if drop_invalid_groups else 0,
        "dropped_rows": int(len(df) - len(ranked)) if drop_invalid_groups else 0,
        "rows_after": int(len(ranked)),
        "groups_after": int(len(group_sizes)),
        "groups_size_lt_2": int((group_stats["group_size"] < 2).sum()),
        "groups_zero_positive": int((group_stats["positive_count"] <= 0).sum()),
        "groups_zero_negative": int((group_stats["negative_count"] <= 0).sum()),
    }
    return ranked.reset_index(drop=True), group_sizes, summary


def coordinate_search(
    base_params: dict,
    search_space: dict,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    ranker_type: str = "binary",
) -> tuple[dict, pd.DataFrame]:
    best_params = dict(base_params)
    records = []

    def train_and_auc(params: dict) -> float:
        model = LightGBMRanker(params, feature_columns, categorical_columns, ranker_type)
        model.train(train_df, valid_df, verbose_eval=False)
        if ranker_type == "lambdarank":
            scores = model.model.best_score_.get("valid_0", {})
            return float(scores.get("ndcg@10", scores.get("ndcg@5", 0.0)))
        return float(model.model.best_score_["valid_0"]["auc"])

    current_best_auc = train_and_auc(best_params)
    for param_name, candidates in search_space.items():
        round_best_value = best_params[param_name]
        round_best_auc = current_best_auc
        for value in candidates:
            trial_params = dict(best_params)
            trial_params[param_name] = value
            trial_auc = train_and_auc(trial_params)
            metric_name = "valid_ndcg" if ranker_type == "lambdarank" else "valid_auc"
            records.append({"param_name": param_name, "trial_value": value, metric_name: trial_auc})
            if trial_auc > round_best_auc:
                round_best_auc = trial_auc
                round_best_value = value
        best_params[param_name] = round_best_value
        current_best_auc = round_best_auc

    metric_name = "valid_ndcg" if ranker_type == "lambdarank" else "valid_auc"
    search_results = pd.DataFrame(records).sort_values(
        [metric_name, "param_name"], ascending=[False, True]
    )
    return best_params, search_results.reset_index(drop=True)
