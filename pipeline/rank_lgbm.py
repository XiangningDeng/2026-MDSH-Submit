from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

try:
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
except ImportError as exc:
    raise ImportError("Please install lightgbm first, for example: pip install lightgbm") from exc


class LightGBMRanker:
    def __init__(
        self,
        params: dict,
        feature_columns: list[str],
        categorical_columns: list[str],
    ):
        self.params = dict(params)
        self.feature_columns = list(feature_columns)
        self.categorical_columns = list(categorical_columns)
        self.model: LGBMClassifier | None = None

    def train(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame | None = None,
        verbose_eval: bool = True,
    ) -> "LightGBMRanker":
        callbacks = [early_stopping(stopping_rounds=30, first_metric_only=True)]
        if verbose_eval:
            callbacks.append(log_evaluation(period=50))

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
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("LightGBM model must be trained before predict.")
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


def coordinate_search(
    base_params: dict,
    search_space: dict,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> tuple[dict, pd.DataFrame]:
    best_params = dict(base_params)
    records = []

    def train_and_auc(params: dict) -> float:
        model = LightGBMRanker(params, feature_columns, categorical_columns)
        model.train(train_df, valid_df, verbose_eval=False)
        return float(model.model.best_score_["valid_0"]["auc"])

    current_best_auc = train_and_auc(best_params)
    for param_name, candidates in search_space.items():
        round_best_value = best_params[param_name]
        round_best_auc = current_best_auc
        for value in candidates:
            trial_params = dict(best_params)
            trial_params[param_name] = value
            trial_auc = train_and_auc(trial_params)
            records.append(
                {"param_name": param_name, "trial_value": value, "valid_auc": trial_auc}
            )
            if trial_auc > round_best_auc:
                round_best_auc = trial_auc
                round_best_value = value
        best_params[param_name] = round_best_value
        current_best_auc = round_best_auc

    search_results = pd.DataFrame(records).sort_values(
        ["valid_auc", "param_name"], ascending=[False, True]
    )
    return best_params, search_results.reset_index(drop=True)

