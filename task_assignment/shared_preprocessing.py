import argparse
from pathlib import Path

import pandas as pd


BEHAVIORS_COLUMNS = ["impression_id", "user_id", "time", "history", "impressions"]


def parse_impressions(impressions: str):
    if not isinstance(impressions, str) or not impressions.strip():
        return []
    pairs = []
    for token in impressions.strip().split():
        if "-" not in token:
            continue
        news_id, label = token.rsplit("-", 1)
        try:
            y = int(label)
        except ValueError:
            continue
        pairs.append((news_id, y))
    return pairs


def expand_behaviors(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=BEHAVIORS_COLUMNS, dtype=str)
    rows = []
    for _, row in df.iterrows():
        pairs = parse_impressions(row["impressions"])
        for pos, (news_id, label) in enumerate(pairs):
            rows.append(
                {
                    "impression_id": row["impression_id"],
                    "user_id": row["user_id"],
                    "time": row["time"],
                    "history": row["history"] if isinstance(row["history"], str) else "",
                    "candidate_news_id": news_id,
                    "label": int(label),
                    "candidate_pos": pos,
                }
            )
    out = pd.DataFrame(rows)
    out["impression_id"] = out["impression_id"].astype(str)
    out["candidate_news_id"] = out["candidate_news_id"].astype(str)
    out["label"] = out["label"].astype(int)
    return out


def write_df(df: pd.DataFrame, out_stem: Path):
    csv_path = out_stem.with_suffix(".csv")
    parquet_path = out_stem.with_suffix(".parquet")
    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(parquet_path, index=False)
        parquet_note = f", {parquet_path}"
    except Exception:
        parquet_note = " (parquet skipped: pyarrow/fastparquet not installed)"
    print(f"Wrote: {csv_path}{parquet_note}")


def main():
    parser = argparse.ArgumentParser(
        description="Expand MIND behaviors.tsv into impression-candidate tables."
    )
    parser.add_argument("--train", default="data/train/behaviors.tsv")
    parser.add_argument("--valid", default="data/valid/behaviors.tsv")
    parser.add_argument(
        "--out-dir", default="outputs/shared", help="Output directory for expanded files"
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = expand_behaviors(Path(args.train))
    valid_df = expand_behaviors(Path(args.valid))

    write_df(train_df, out_dir / "train_candidates")
    write_df(valid_df, out_dir / "valid_candidates")

    print(
        f"Done. train rows={len(train_df):,}, valid rows={len(valid_df):,}. "
        f"Use valid_candidates as shared evaluation input."
    )


if __name__ == "__main__":
    main()

