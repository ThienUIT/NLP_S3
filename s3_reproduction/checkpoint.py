from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
from sklearn.decomposition import FastICA


@dataclass
class ModelCheckpoint:
    """A fitted FastICA decomposition plus the vocabulary/topics needed to reuse it.

    Deliberately excludes the encoder and raw document embeddings: re-encoding
    text is the slow part of this pipeline (see CafeBERTEncoder), while the
    decomposition itself is a few small matrices that fit and save in well
    under a second. To score new documents, encode them with the same model
    named in `metadata["encoder"]` and call `ica.transform(embeddings)`.

    `topics` holds each axis' positive-pole words, `topics_negative` the
    negative pole -- paper §3.1 treats both as meaningful ("negative
    definition" of a topic), not noise to discard.
    """

    ica: FastICA
    vocabulary: list[str]
    topics: list[list[str]]
    metadata: dict[str, Any]
    topics_negative: list[list[str]] = field(default_factory=list)


def save_model(
    ica: FastICA,
    vocabulary: list[str],
    topics: list[list[str]],
    metadata: dict[str, Any],
    output_dir: Path,
    topics_negative: list[list[str]] | None = None,
    timestamp: datetime | None = None,
) -> Path:
    """Save a fitted decomposition as artifacts/.../models/model_n<N>_HHMM_DDMMYY.joblib.

    n_topics is embedded in the filename (from `metadata["n_topics"]`) because
    a single `--n-topics 10 20 30 40 50` invocation fits several models within
    the same minute -- without it, later n_topics values silently overwrite
    earlier ones sharing the same HHMM stamp.
    """
    timestamp = timestamp or datetime.now()
    output_dir.mkdir(parents=True, exist_ok=True)
    n_topics = metadata.get("n_topics")
    stem = f"model_n{n_topics}_{timestamp.strftime('%H%M_%d%m%y')}" if n_topics is not None \
        else f"model_{timestamp.strftime('%H%M_%d%m%y')}"
    path = output_dir / f"{stem}.joblib"
    checkpoint = ModelCheckpoint(
        ica=ica, vocabulary=vocabulary, topics=topics,
        metadata=metadata, topics_negative=topics_negative or [],
    )
    joblib.dump(checkpoint, path)
    return path


def save_latest(
    ica: FastICA,
    vocabulary: list[str],
    topics: list[list[str]],
    metadata: dict[str, Any],
    output_dir: Path,
    topics_negative: list[list[str]] | None = None,
) -> Path:
    """Overwrite <output_dir>/latest.joblib -- the most recent fit for this backend/dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "latest.joblib"
    checkpoint = ModelCheckpoint(
        ica=ica, vocabulary=vocabulary, topics=topics,
        metadata=metadata, topics_negative=topics_negative or [],
    )
    joblib.dump(checkpoint, path)
    return path


def load_model(path: Path) -> ModelCheckpoint:
    return joblib.load(path)
