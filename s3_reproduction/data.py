from __future__ import annotations

import glob
import re
import unicodedata
from pathlib import Path

import pandas as pd
import numpy as np


def clean_text(value: object) -> str:
    text = unicodedata.normalize("NFC", str(value))
    return re.sub(r"\s+", " ", text).strip()


def _sample(frame: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    if limit and len(frame) > limit:
        return frame.sample(n=limit, random_state=seed)
    return frame


def load_visfd(root: Path, max_documents: int | None, seed: int) -> list[str]:
    frame = pd.read_csv(root / "ViSFD" / "ViSFD.csv", usecols=["comment"])
    frame = _sample(frame, max_documents, seed)
    return [text for text in frame["comment"].map(clean_text) if text]


def load_vietnamese_news(root: Path, max_documents: int | None, seed: int) -> list[str]:
    files = sorted(glob.glob(str(root / "Vietnamese-News" / "all" / "*.parquet")))
    if not files:
        raise FileNotFoundError("Không tìm thấy shard Parquet của Vietnamese-News")
    # Avoid loading all 2.4M rows merely to make a small sample. Randomize shard
    # order, collect enough rows, then sample deterministically from that pool.
    if max_documents:
        rng = np.random.default_rng(seed)
        ordered = [files[index] for index in rng.permutation(len(files))]
        chunks: list[pd.DataFrame] = []
        rows = 0
        for path in ordered:
            chunk = pd.read_parquet(path, columns=["text"])
            chunks.append(chunk)
            rows += len(chunk)
            if rows >= max_documents:
                break
        frame = _sample(pd.concat(chunks, ignore_index=True), max_documents, seed)
    else:
        frame = pd.concat((pd.read_parquet(path, columns=["text"]) for path in files), ignore_index=True)
    return [text for text in frame["text"].map(clean_text) if text]


def load_dataset(name: str, root: Path, max_documents: int | None, seed: int) -> list[str]:
    if name == "visfd":
        return load_visfd(root, max_documents, seed)
    if name == "vietnamese-news":
        return load_vietnamese_news(root, max_documents, seed)
    raise ValueError(f"Dataset không hỗ trợ: {name}")
