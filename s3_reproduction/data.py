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


CURATED_REPO_ID = "VTSNLP/vietnamese_curated_dataset"
CURATED_N_SHARDS = 132


def load_curated_frame(max_documents: int, seed: int) -> pd.DataFrame:
    """Shared shard-fetch/sample logic behind load_vietnamese_curated, kept
    here (not inlined) so validate_curated.py can rebuild the exact same
    sample -- including the `domain` column, which load_vietnamese_curated
    itself discards -- for external validation without duplicating the
    sampling logic and risking it drifting out of sync.
    """
    from huggingface_hub import hf_hub_download

    if not max_documents:
        raise ValueError(
            "load_curated_frame không hỗ trợ tải toàn bộ 12.17 triệu tài liệu "
            "(34.65GB, 132 shard) -- truyền max_documents > 0"
        )
    rng = np.random.default_rng(seed)
    shard_order = rng.permutation(CURATED_N_SHARDS)
    chunks: list[pd.DataFrame] = []
    rows = 0
    for shard_index in shard_order:
        filename = f"data/train-{shard_index:05d}-of-{CURATED_N_SHARDS:05d}.parquet"
        local_path = hf_hub_download(repo_id=CURATED_REPO_ID, filename=filename, repo_type="dataset")
        chunk = pd.read_parquet(local_path, columns=["text", "domain"])
        chunks.append(chunk)
        rows += len(chunk)
        if rows >= max_documents:
            break
    return _sample(pd.concat(chunks, ignore_index=True), max_documents, seed)


def load_vietnamese_curated(root: Path, max_documents: int | None, seed: int) -> list[str]:
    """VTSNLP/vietnamese_curated_dataset on the HF Hub: 12,169,131 documents /
    34.65GB across 132 parquet shards (C4 + OSCAR + Wikipedia + Binhvq news,
    deduplicated and quality-filtered with NeMo Curator). ~3x the size of
    Vietnamese-News, so unlike load_vietnamese_news this does NOT assume the
    shards are already sitting in dataset/ -- it pulls only as many shards as
    needed to cover max_documents straight from the Hub (huggingface_hub caches
    each shard locally after its first download; needs HF_TOKEN in env for
    reasonable rate limits, same as the smoke-test downloads in REPRODUCE.md).

    `root` is accepted only for interface symmetry with the other loaders;
    this one has no local-file dependency.
    """
    frame = load_curated_frame(max_documents, seed)
    return [text for text in frame["text"].map(clean_text) if text]


UTS_BANK_REPO_ID = "undertheseanlp/UTS2017_Bank"


def load_uts_bank_frame() -> pd.DataFrame:
    """Shared fetch behind load_uts_bank, kept separate so validate_uts_bank.py
    can rebuild the exact same (text, label) frame -- including `label`, which
    load_uts_bank itself discards -- for external validation.

    No max_documents/seed/sampling: the whole `classification` config (train +
    test, 2,471 rows) is small enough to just use in full and it's already a
    fixed, deterministic set -- unlike load_curated_frame there's no shard
    subsampling to reproduce.
    """
    from huggingface_hub import hf_hub_download

    frames = []
    for split in ("train", "test"):
        path = hf_hub_download(
            repo_id=UTS_BANK_REPO_ID, filename=f"data/classification/{split}.jsonl", repo_type="dataset"
        )
        frames.append(pd.read_json(path, lines=True))
    return pd.concat(frames, ignore_index=True)


def load_uts_bank(root: Path, max_documents: int | None, seed: int) -> list[str]:
    """undertheseanlp/UTS2017_Bank `classification` config: 2,471 Vietnamese
    banking customer-feedback texts, each single-labeled with one of 14 real
    aspect categories (CUSTOMER_SUPPORT, CARD, LOAN, SECURITY, ...) -- unlike
    ViSFD's multi-label {ASPECT#sentiment} tags or VTSNLP's `domain`, this is
    the smallest and most imbalanced label set used so far (some categories
    have under 15 examples), worth keeping in mind when reading validation
    AUCs against it. `root`/`seed` accepted for interface symmetry only; the
    full set is always used, no sampling.
    """
    frame = load_uts_bank_frame()
    if max_documents:
        frame = _sample(frame, max_documents, seed)
    return [text for text in frame["text"].map(clean_text) if text]


def load_dataset(name: str, root: Path, max_documents: int | None, seed: int) -> list[str]:
    if name == "visfd":
        return load_visfd(root, max_documents, seed)
    if name == "vietnamese-news":
        return load_vietnamese_news(root, max_documents, seed)
    if name == "vietnamese-curated":
        return load_vietnamese_curated(root, max_documents, seed)
    if name == "uts-bank":
        return load_uts_bank(root, max_documents, seed)
    raise ValueError(f"Dataset không hỗ trợ: {name}")
