"""External validation for VTSNLP/vietnamese_curated_dataset: correlate S3's
axes against the corpus' own `domain` labels (single content category per
document, assigned during NeMo Curator curation -- Health, Sports, Business,
Arts_and_Entertainment, ... 25 values). Same idea as validate_visfd.py's
aspect-label check, reusing its correlate_axes()/combined_axes_auc() machinery
directly -- only the label frame is built differently here (one boolean
column per domain value from a single categorical column, instead of parsed
{ASPECT#sentiment} tags).

Usage:
    python -m s3_reproduction.validate_curated artifacts/turftopic/vietnamese-curated-e5/models/model_n30_....joblib
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .checkpoint import load_model
from .data import clean_text, load_curated_frame
from .validate_visfd import auc_tier, combined_axes_auc, correlate_axes


def build_domain_frame(max_documents: int, seed: int) -> tuple[pd.DataFrame, list[str]]:
    """Rebuilds the exact sample load_vietnamese_curated used (same seed,
    same shard order, same sampling) -- including `domain`, which the
    training loader discards -- then applies the identical clean+drop-empty
    filter so rows align positionally with the cached document_embeddings.npy.
    """
    frame = load_curated_frame(max_documents, seed).copy()
    frame["text"] = frame["text"].map(clean_text)
    frame = frame[frame["text"] != ""].reset_index(drop=True)
    domains = sorted(frame["domain"].dropna().unique().tolist())
    for domain in domains:
        frame[domain] = frame["domain"] == domain
    return frame, domains


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="So khớp trục S3 với nhãn domain gốc của VTSNLP curated")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--top-words", type=int, default=6)
    parser.add_argument(
        "--max-documents", type=int, default=None,
        help="Giá trị --max-documents CHÍNH XÁC dùng lúc train checkpoint này -- "
             "bắt buộc vì pandas .sample(n=...) không cho kết quả lồng nhau khi n đổi, "
             "không thể suy ra từ metadata['documents'] (đã bị lọc bớt dòng rỗng)",
    )
    parser.add_argument("--combined", action="store_true", help="Thêm AUC khi gộp tất cả trục (logistic regression, 5-fold CV)")
    parser.add_argument("--export-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = load_model(args.checkpoint)
    dataset_dir = args.checkpoint.parent.parent
    document_embeddings = np.load(dataset_dir / "document_embeddings.npy")

    # cli.py started saving max_documents/seed in metadata after this checkpoint
    # format existed; older checkpoints need it passed explicitly since
    # metadata["documents"] (post-empty-filter count) can't be reused for
    # pandas .sample(n=...), which isn't consistent across different n.
    max_documents = args.max_documents or checkpoint.metadata.get("max_documents")
    if max_documents is None:
        raise SystemExit(
            "--max-documents bắt buộc (giá trị đã dùng lúc train checkpoint này, "
            f"ví dụ 20000) -- metadata['documents']={checkpoint.metadata.get('documents')} "
            "là số dòng SAU khi lọc rỗng, không dùng lại được để tái tạo đúng mẫu."
        )
    seed = checkpoint.metadata.get("seed", 42)
    frame, domains = build_domain_frame(max_documents, seed)
    n_docs = min(len(frame), len(document_embeddings))
    frame = frame.iloc[:n_docs].reset_index(drop=True)
    document_embeddings = document_embeddings[:n_docs]
    if len(frame) != len(document_embeddings):
        print(
            f"CẢNH BÁO: dựng lại mẫu VTSNLP được {len(frame)} dòng nhưng "
            f"document_embeddings.npy có {len(document_embeddings)} -- có thể lệch thứ tự, kết quả chỉ tham khảo."
        )

    doc_topic = checkpoint.ica.transform(document_embeddings)
    results = correlate_axes(doc_topic, frame, domains)
    print(f"n_topics={doc_topic.shape[1]}  n_docs={n_docs}  n_domains={len(domains)}\n")

    combined = combined_axes_auc(doc_topic, frame, domains) if args.combined else None

    print("=== Domain gốc -> trục khớp nhất, xếp theo AUC ===")
    n_reliable = 0
    for domain in domains:
        rows = results[results["aspect"] == domain]
        if rows.empty:
            print(f"{domain}: không đủ mẫu để kiểm định")
            continue
        best = rows.loc[rows["auc"].idxmax()]
        tier = auc_tier(best["auc"])
        n_reliable += tier == "tốt"
        words = checkpoint.topics[int(best["axis"])][: args.top_words] if best["mean_diff"] > 0 else checkpoint.topics_negative[int(best["axis"])][: args.top_words]
        combined_str = ""
        if combined is not None:
            match = combined[combined["aspect"] == domain]
            if not match.empty:
                c = match.iloc[0]
                combined_str = f"  | gộp {doc_topic.shape[1]} trục: AUC={c['combined_auc']:.3f} ({auc_tier(c['combined_auc']):4s})"
        pole = "cực dương" if best["mean_diff"] > 0 else "cực âm"
        print(
            f"{domain:28s} -> Topic {int(best['axis']):>2}  AUC={best['auc']:.3f} ({tier:4s})  "
            f"r={best['r']:+.3f}  {pole}  n={best['n_mentioned']:>5}  -- {words}{combined_str}"
        )

    print(f"\n-> {n_reliable}/{len(domains)} domain có AUC >= 0.75 khi chỉ dùng 1 trục tốt nhất")
    if combined is not None:
        n_reliable_combined = (combined["combined_auc"] >= 0.75).sum()
        print(f"-> {n_reliable_combined}/{len(combined)} domain có AUC >= 0.75 khi gộp tất cả trục")

    if args.export_csv:
        results.rename(columns={"aspect": "domain"}).sort_values(
            ["domain", "auc"], ascending=[True, False]
        ).to_csv(args.export_csv, index=False)
        print(f"Đã ghi toàn bộ bảng axis x domain vào {args.export_csv}")


if __name__ == "__main__":
    main()
