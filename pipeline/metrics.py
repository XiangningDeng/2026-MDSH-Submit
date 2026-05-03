from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def auc_binary(y_true: np.ndarray, y_score: np.ndarray):
    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    pos_ranks = ranks[y_true == 1].sum()
    return float((pos_ranks - pos * (pos + 1) / 2.0) / (pos * neg))


def mrr_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(-y_score)
    sorted_y = y_true[order]
    pos_idx = np.where(sorted_y == 1)[0]
    if len(pos_idx) == 0:
        return 0.0
    return float(1.0 / (pos_idx[0] + 1))


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    order = np.argsort(-y_score)[:k]
    gains = y_true[order].astype(float)
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float(np.sum(gains * discounts))
    ideal = np.sort(y_true)[::-1][:k].astype(float)
    idcg = float(np.sum(ideal * discounts[: len(ideal)]))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_predictions(predictions: pd.DataFrame) -> dict:
    required = {"impression_id", "label", "score"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction dataframe missing columns: {missing}")

    auc_list, mrr_list, ndcg5_list, ndcg10_list = [], [], [], []
    for _, group in predictions.groupby("impression_id", sort=False):
        y = group["label"].to_numpy(dtype=int)
        s = group["score"].to_numpy(dtype=float)
        auc_val = auc_binary(y, s)
        if auc_val is not None:
            auc_list.append(auc_val)
        mrr_list.append(mrr_binary(y, s))
        ndcg5_list.append(ndcg_at_k(y, s, 5))
        ndcg10_list.append(ndcg_at_k(y, s, 10))

    return {
        "impressions_total": int(predictions["impression_id"].nunique()),
        "impressions_with_auc": len(auc_list),
        "AUC": float(np.mean(auc_list)) if auc_list else None,
        "MRR": float(np.mean(mrr_list)) if mrr_list else None,
        "nDCG@5": float(np.mean(ndcg5_list)) if ndcg5_list else None,
        "nDCG@10": float(np.mean(ndcg10_list)) if ndcg10_list else None,
    }


def write_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

