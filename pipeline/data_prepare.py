from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline.config import BEHAVIORS_COLUMNS, NEWS_COLUMNS, PipelineConfig


def parse_impressions(impressions: str, default_label: int | None = None) -> list[tuple[str, int | None]]:
    if not isinstance(impressions, str) or not impressions.strip():
        return []

    pairs: list[tuple[str, int | None]] = []
    for token in impressions.strip().split():
        if "-" in token:
            news_id, label = token.rsplit("-", 1)
            try:
                pairs.append((news_id, int(label)))
            except ValueError:
                pairs.append((token, default_label))
        else:
            pairs.append((token, default_label))
    return pairs


def load_news(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", header=None, names=NEWS_COLUMNS)


def load_all_news(config: PipelineConfig) -> pd.DataFrame:
    frames = []
    for path in [config.train_news_path, config.valid_news_path]:
        if path.exists():
            frames.append(load_news(path))
    if not frames:
        raise FileNotFoundError("No news.tsv files found under data/train or data/valid.")
    news = pd.concat(frames, ignore_index=True).drop_duplicates("news_id")
    news["news_id"] = news["news_id"].astype(str)
    return news.reset_index(drop=True)


def expand_behaviors(path: Path, has_labels: bool = True) -> pd.DataFrame:
    behaviors = pd.read_csv(path, sep="\t", header=None, names=BEHAVIORS_COLUMNS, dtype=str)
    rows = []
    default_label = 0 if has_labels else None

    for _, row in behaviors.iterrows():
        for pos, (news_id, label) in enumerate(parse_impressions(row["impressions"], default_label)):
            record = {
                "impression_id": row["impression_id"],
                "user_id": row["user_id"],
                "time": row["time"],
                "history": row["history"] if isinstance(row["history"], str) else "",
                "candidate_news_id": news_id,
                "candidate_pos": pos,
            }
            if has_labels:
                record["label"] = int(label or 0)
            rows.append(record)

    out = pd.DataFrame(rows)
    if out.empty:
        columns = [
            "impression_id",
            "user_id",
            "time",
            "history",
            "candidate_news_id",
            "candidate_pos",
        ]
        if has_labels:
            columns.append("label")
        return pd.DataFrame(columns=columns)

    out["impression_id"] = out["impression_id"].astype(str)
    out["candidate_news_id"] = out["candidate_news_id"].astype(str)
    if has_labels:
        out["label"] = out["label"].astype(int)
    return out


def load_candidates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"impression_id": str, "candidate_news_id": str})
    required = {
        "impression_id",
        "user_id",
        "time",
        "history",
        "candidate_news_id",
        "label",
        "candidate_pos",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Candidate file missing columns: {missing}")
    df["history"] = df["history"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    return df


def load_train_valid_data(config: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = [
        config.train_behaviors_path,
        config.valid_behaviors_path,
        config.train_news_path,
        config.valid_news_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required MIND files: " + ", ".join(missing))

    news = load_all_news(config)
    if config.shared_train_candidates_path.exists():
        train_candidates = load_candidates(config.shared_train_candidates_path)
    else:
        train_candidates = expand_behaviors(config.train_behaviors_path, has_labels=True)

    if config.shared_valid_candidates_path.exists():
        valid_candidates = load_candidates(config.shared_valid_candidates_path)
    else:
        valid_candidates = expand_behaviors(config.valid_behaviors_path, has_labels=True)
    return train_candidates, valid_candidates, news
