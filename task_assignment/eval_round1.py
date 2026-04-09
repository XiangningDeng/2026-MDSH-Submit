import argparse
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


def mrr_binary(y_true: np.ndarray, y_score: np.ndarray):
    order = np.argsort(-y_score)
    sorted_y = y_true[order]
    pos_idx = np.where(sorted_y == 1)[0]
    if len(pos_idx) == 0:
        return 0.0
    return float(1.0 / (pos_idx[0] + 1))


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int):
    order = np.argsort(-y_score)[:k]
    gains = y_true[order].astype(float)
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float(np.sum(gains * discounts))

    ideal = np.sort(y_true)[::-1][:k].astype(float)
    idcg = float(np.sum(ideal * discounts[: len(ideal)]))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def load_candidates(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    required = {"impression_id", "candidate_news_id", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Candidate file missing columns: {missing}")
    return df[list(required)].copy()


def load_predictions(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, sep="\t", header=None)
    if {"impression_id", "candidate_news_id", "score"}.issubset(df.columns):
        out = df[["impression_id", "candidate_news_id", "score"]].copy()
    else:
        if df.shape[1] < 3:
            raise ValueError(
                "Prediction file must have 3 columns: impression_id,candidate_news_id,score"
            )
        out = df.iloc[:, :3].copy()
        out.columns = ["impression_id", "candidate_news_id", "score"]
    out["impression_id"] = out["impression_id"].astype(str)
    out["candidate_news_id"] = out["candidate_news_id"].astype(str)
    out["score"] = out["score"].astype(float)
    return out


def evaluate(merged: pd.DataFrame):
    auc_list, mrr_list, ndcg5_list, ndcg10_list = [], [], [], []
    group_count = 0
    auc_valid_count = 0

    for _, g in merged.groupby("impression_id", sort=False):
        y = g["label"].to_numpy(dtype=int)
        s = g["score"].to_numpy(dtype=float)
        group_count += 1
        auc_val = auc_binary(y, s)
        if auc_val is not None:
            auc_list.append(auc_val)
            auc_valid_count += 1
        mrr_list.append(mrr_binary(y, s))
        ndcg5_list.append(ndcg_at_k(y, s, 5))
        ndcg10_list.append(ndcg_at_k(y, s, 10))

    return {
        "impressions_total": group_count,
        "impressions_with_auc": auc_valid_count,
        "AUC": float(np.mean(auc_list)) if auc_list else None,
        "MRR": float(np.mean(mrr_list)) if mrr_list else None,
        "nDCG@5": float(np.mean(ndcg5_list)) if ndcg5_list else None,
        "nDCG@10": float(np.mean(ndcg10_list)) if ndcg10_list else None,
    }


def maybe_update_leaderboard(path: Path, method_name: str, metrics: dict):
    row = {
        "method": method_name,
        "auc": metrics["AUC"],
        "mrr": metrics["MRR"],
        "ndcg5": metrics["nDCG@5"],
        "ndcg10": metrics["nDCG@10"],
        "impressions_total": metrics["impressions_total"],
        "impressions_with_auc": metrics["impressions_with_auc"],
    }
    if path.exists():
        df = pd.read_csv(path)
        df = df[df["method"] != method_name]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Round1 unified evaluator.")
    parser.add_argument(
        "--candidates",
        default="outputs/shared/valid_candidates.csv",
        help="Shared valid candidate file from shared_preprocessing.py",
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Method predictions with columns: impression_id,candidate_news_id,score",
    )
    parser.add_argument("--method-name", required=True)
    parser.add_argument(
        "--out-metrics",
        default=None,
        help="Output metrics.json path. Default: outputs/<method>/metrics.json",
    )
    parser.add_argument(
        "--leaderboard",
        default="outputs/leaderboard_round1.csv",
        help="Leaderboard CSV path",
    )
    args = parser.parse_args()

    candidates = load_candidates(Path(args.candidates))
    preds = load_predictions(Path(args.predictions))
    candidates["impression_id"] = candidates["impression_id"].astype(str)
    candidates["candidate_news_id"] = candidates["candidate_news_id"].astype(str)
    candidates["label"] = candidates["label"].astype(int)

    merged = candidates.merge(
        preds, on=["impression_id", "candidate_news_id"], how="left", validate="one_to_one"
    )
    merged["score"] = merged["score"].fillna(-1e9)

    metrics = evaluate(merged)

    out_metrics = (
        Path(args.out_metrics)
        if args.out_metrics
        else Path("outputs") / args.method_name / "metrics.json"
    )
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    with out_metrics.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    maybe_update_leaderboard(Path(args.leaderboard), args.method_name, metrics)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Wrote metrics to: {out_metrics}")
    print(f"Updated leaderboard: {args.leaderboard}")


if __name__ == "__main__":
    main()

