"""Full vocabulary ranking for a fitted axis, with scores -- not just a
top-N word list. A checkpoint only keeps top_n=10 words per pole (see
checkpoint.py); this re-derives the ranking for the WHOLE vocabulary on
demand from the cached word_embeddings.npy, so 6-10 words (too generic to
tell what an axis "means") can become 20-50 words with their actual score,
enough for a human or an LLM to name the axis with confidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import FastICA

from .checkpoint import load_model

WordScore = tuple[str, float]


def rank_vocabulary(
    ica: FastICA, vocabulary: list[str], embeddings: np.ndarray, top_k: int = 20,
) -> tuple[list[list[WordScore]], list[list[WordScore]]]:
    """Score every vocabulary word against every axis using the paper's
    §3.1 point 6 'combined' importance (Wjt^3 / ||Wj||) and return the
    top_k highest (positive pole) and lowest (negative pole) per axis,
    each word paired with its actual score -- not just ranked words.
    """
    projected = ica.transform(embeddings)  # (n_vocab, n_topics); .transform() re-centers internally
    norm = np.linalg.norm(projected, axis=1, keepdims=True).clip(min=1e-12)
    combined = (projected**3 / norm).T  # (n_topics, n_vocab)
    words = np.asarray(vocabulary)
    positive: list[list[WordScore]] = []
    negative: list[list[WordScore]] = []
    for row in combined:
        pos_idx = np.argsort(row)[-top_k:][::-1]
        neg_idx = np.argsort(row)[:top_k]
        positive.append(list(zip(words[pos_idx].tolist(), row[pos_idx].round(4).tolist())))
        negative.append(list(zip(words[neg_idx].tolist(), row[neg_idx].round(4).tolist())))
    return positive, negative


def format_word_scores(pairs: list[WordScore]) -> str:
    return ", ".join(f"{word} ({score:+.3f})" for word, score in pairs)


def format_report(
    positive: list[list[WordScore]],
    negative: list[list[WordScore]],
    metadata: dict[str, Any] | None = None,
) -> str:
    lines: list[str] = []
    if metadata:
        lines.append(
            f"# Topic report -- {metadata.get('dataset', '?')} / {metadata.get('backend', '?')} "
            f"/ n_topics={metadata.get('n_topics', len(positive))}"
        )
        lines.append("")
    for index, (pos, neg) in enumerate(zip(positive, negative)):
        lines.append(f"## Topic {index}")
        lines.append("")
        lines.append(f"**Cực dương (+, {len(pos)} từ):** {format_word_scores(pos)}")
        lines.append("")
        lines.append(f"**Cực âm (-, {len(neg)} từ):** {format_word_scores(neg)}")
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Xuất báo cáo đầy đủ từ khoá + điểm số cho từng trục của một checkpoint"
    )
    parser.add_argument("checkpoint", type=Path, help="Đường dẫn tới file .joblib trong models/")
    parser.add_argument("--top-k", type=int, default=20, help="Số từ mỗi cực (mặc định 20)")
    parser.add_argument("--output", type=Path, default=None, help="Mặc định: <checkpoint>.report.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = load_model(args.checkpoint)
    dataset_dir = args.checkpoint.parent.parent  # models/<file>.joblib -> .../<backend>/<dataset>/
    vocabulary = json.loads((dataset_dir / "vocabulary.json").read_text(encoding="utf-8"))
    embeddings = np.load(dataset_dir / "word_embeddings.npy")
    positive, negative = rank_vocabulary(checkpoint.ica, vocabulary, embeddings, args.top_k)
    report = format_report(positive, negative, checkpoint.metadata)
    output = args.output or args.checkpoint.with_suffix(".report.md")
    output.write_text(report, encoding="utf-8")
    print(f"Đã ghi {output}")


if __name__ == "__main__":
    main()
