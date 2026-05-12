from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline.config import NEWS_COLUMNS
from pipeline.data_prepare import load_news


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build_news_text(news: pd.DataFrame) -> list[str]:
    title = news["title"].fillna("").astype(str)
    abstract = news["abstract"].fillna("").astype(str)
    return (title + ". " + abstract).str.strip().tolist()


def safe_model_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("_")


def load_all_news(project_root: Path) -> pd.DataFrame:
    frames = []
    for path in [
        project_root / "data/train/news.tsv",
        project_root / "data/valid/news.tsv",
        project_root / "data/test/news.tsv",
    ]:
        if path.exists():
            frames.append(load_news(path))
    if not frames:
        raise FileNotFoundError("No news.tsv files found under data/train, data/valid, or data/test.")

    news = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("news_id")
        .reset_index(drop=True)
    )
    news["news_id"] = news["news_id"].astype(str)
    for column in NEWS_COLUMNS:
        if column not in news.columns:
            news[column] = ""
    return news


def generate_embeddings(
    project_root: Path,
    output_path: Path,
    model_name: str,
    batch_size: int,
    device: str | None,
    normalize_embeddings: bool,
    max_seq_length: int | None,
) -> dict:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required. Install it with: "
            "pip install -U sentence-transformers"
        ) from exc

    news = load_all_news(project_root)
    texts = build_news_text(news)
    news_ids = news["news_id"].astype(str).to_numpy()

    model_kwargs = {}
    if device:
        model_kwargs["device"] = device
    model = SentenceTransformer(model_name, **model_kwargs)
    if max_seq_length is not None and max_seq_length > 0:
        model.max_seq_length = max_seq_length

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    ).astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        news_ids=news_ids,
        embeddings=embeddings,
        model_name=np.asarray(model_name),
        normalize_embeddings=np.asarray(normalize_embeddings),
        max_seq_length=np.asarray(model.max_seq_length),
    )

    metadata = {
        "model_name": model_name,
        "news_count": int(len(news_ids)),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "normalize_embeddings": bool(normalize_embeddings),
        "max_seq_length": int(model.max_seq_length),
        "output_path": str(output_path),
    }
    with output_path.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate sentence-transformer news embeddings for sentence embedding recall."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--output-path",
        default=None,
        help="Output .npz path. Defaults to outputs/pipeline_cache/sentence_embeddings_<model>.npz.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--device",
        default=None,
        help="Optional sentence-transformers device, for example cuda, cpu, or mps.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Do not L2-normalize embeddings. The recall scorer normalizes after loading if needed.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="Optional model.max_seq_length override.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output_path = (
        Path(args.output_path)
        if args.output_path
        else project_root
        / "outputs/pipeline_cache"
        / f"sentence_embeddings_{safe_model_name(args.model_name)}.npz"
    )
    if not output_path.is_absolute():
        output_path = project_root / output_path

    metadata = generate_embeddings(
        project_root=project_root,
        output_path=output_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
        normalize_embeddings=not args.no_normalize,
        max_seq_length=args.max_seq_length,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
