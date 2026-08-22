"""External validation for undertheseanlp/UTS2017_Bank: correlate S3's axes
against the dataset's own 14 real banking aspect labels (CUSTOMER_SUPPORT,
CARD, LOAN, SECURITY, ...). Reuses validate_visfd.py's correlate_axes()/
combined_axes_auc() directly -- only the label frame differs (one boolean
column per label value, single-label like validate_curated.py's `domain`,
not ViSFD's multi-label {ASPECT#sentiment} tags).

Smallest, most imbalanced label set used in this project so far (2,471 rows
total, some categories under 15 examples) -- correlate_axes already skips
any axis/label pair with fewer than 10 positive or negative examples, so
several of the 14 will show up as "không đủ mẫu".

Usage:
    python -m s3_reproduction.validate_uts_bank artifacts/turftopic/uts-bank-e5/models/model_n14_....joblib --combined
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .checkpoint import load_model
from .data import clean_text, load_uts_bank_frame
from .validate_visfd import auc_tier, combined_axes_auc, correlate_axes


def build_label_frame() -> tuple[pd.DataFrame, list[str]]:
    """Rebuilds the exact (text, label) set load_uts_bank used -- train+test
    concatenated, no sampling -- then applies the identical clean+drop-empty
    filter so rows align positionally with the cached document_embeddings.npy.
    """
    frame = load_uts_bank_frame().copy()
    frame["text"] = frame["text"].map(clean_text)
    frame = frame[frame["text"] != ""].reset_index(drop=True)
    labels = sorted(frame["label"].dropna().unique().tolist())
    for label in labels:
        frame[label] = frame["label"] == label
    return frame, labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="So khớp trục S3 với 14 nhãn khía cạnh ngân hàng thật của UTS2017_Bank")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--top-words", type=int, default=6)
    parser.add_argument("--combined", action="store_true")
    parser.add_argument("--export-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = load_model(args.checkpoint)
    dataset_dir = args.checkpoint.parent.parent
    document_embeddings = np.load(dataset_dir / "document_embeddings.npy")

    frame, labels = build_label_frame()
    n_docs = min(len(frame), len(document_embeddings))
    frame = frame.iloc[:n_docs].reset_index(drop=True)
    document_embeddings = document_embeddings[:n_docs]
    if len(frame) != len(document_embeddings):
        print(
            f"CẢNH BÁO: dựng lại {len(frame)} dòng nhưng document_embeddings.npy có "
            f"{len(document_embeddings)} -- có thể lệch thứ tự, kết quả chỉ tham khảo."
        )

    doc_topic = checkpoint.ica.transform(document_embeddings)
    results = correlate_axes(doc_topic, frame, labels)
    combined = combined_axes_auc(doc_topic, frame, labels) if args.combined else None
    print(f"n_topics={doc_topic.shape[1]}  n_docs={n_docs}  n_labels={len(labels)}\n")

    print("=== Nhãn gốc -> trục khớp nhất, xếp theo AUC ===")
    n_reliable = 0
    n_checked = 0
    for label in labels:
        rows = results[results["aspect"] == label]
        if rows.empty:
            print(f"{label}: không đủ mẫu để kiểm định")
            continue
        n_checked += 1
        best = rows.loc[rows["auc"].idxmax()]
        tier = auc_tier(best["auc"])
        n_reliable += tier == "tốt"
        words = checkpoint.topics[int(best["axis"])][: args.top_words] if best["mean_diff"] > 0 else checkpoint.topics_negative[int(best["axis"])][: args.top_words]
        combined_str = ""
        if combined is not None:
            match = combined[combined["aspect"] == label]
            if not match.empty:
                c = match.iloc[0]
                combined_str = f"  | gộp {doc_topic.shape[1]} trục: AUC={c['combined_auc']:.3f} ({auc_tier(c['combined_auc']):4s})"
        pole = "cực dương" if best["mean_diff"] > 0 else "cực âm"
        print(
            f"{label:18s} -> Topic {int(best['axis']):>2}  AUC={best['auc']:.3f} ({tier:4s})  "
            f"r={best['r']:+.3f}  {pole}  n={best['n_mentioned']:>4}  -- {words}{combined_str}"
        )

    print(f"\n-> {n_reliable}/{n_checked} nhãn đủ mẫu có AUC >= 0.75 (1 trục); "
          f"{len(labels) - n_checked}/{len(labels)} nhãn không đủ mẫu để kiểm (quá ít ví dụ)")
    if combined is not None:
        n_reliable_combined = (combined["combined_auc"] >= 0.75).sum()
        print(f"-> {n_reliable_combined}/{len(combined)} nhãn đủ mẫu có AUC >= 0.75 khi gộp tất cả trục")

    if args.export_csv:
        results.rename(columns={"aspect": "label"}).sort_values(
            ["label", "auc"], ascending=[True, False]
        ).to_csv(args.export_csv, index=False)
        print(f"Đã ghi toàn bộ bảng axis x label vào {args.export_csv}")


if __name__ == "__main__":
    main()
